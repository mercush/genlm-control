import asyncio
import numpy as np
from dataclasses import dataclass

from genlm_control.potential import Potential
from genlm_control.util import load_async_trie
from genlm_control.constant import EOT


@dataclass
class SubTokenMarginal:
    """
    Parameterizes a subtoken potential in which
    * complete(token) = logw_next[token]
    * prefix(subtoken) = marginal over subtoken completions of complete
    """

    subtoken_ws: np.array
    prefix2node: dict
    complete2node: dict

    __slots__ = (
        "subtoken_ws",
        "prefix2node",
        "complete2node",
    )

    def prefix(self, subtoken_ctx):
        node = self.prefix2node.get(tuple(subtoken_ctx), None)
        if node is None:
            return float("-inf")
        return np.log(self.subtoken_ws[node])

    def complete(self, subtoken_ctx):
        node = self.complete2node.get(tuple(subtoken_ctx), None)
        if node is None:
            return float("-inf")
        return np.log(self.subtoken_ws[node])


class SubtokenPotential(Potential):
    def __init__(self, potential, trie=None):
        self.potential = potential
        self.trie = trie or load_async_trie(potential.decode_eos)

        trie = self.trie.trie
        self.prefix2node = {}
        stack = [(trie.root, [])]

        while stack:
            node, prefix = stack.pop()
            self.prefix2node[tuple(prefix)] = node
            for subtoken, child in trie.children[node].items():
                if subtoken is not None:
                    stack.append((child, prefix + [subtoken]))

        self.complete2node = {tuple(x): i for x, i in self.trie.trie.word2leaf.items()}

        self.cache = {}

        units = set()
        for token in self.potential.decode:
            for subtoken in token:
                units.add(subtoken)

        vocab = list(units)
        vocab.append(self.potential.eos)

        super().__init__(vocab, eos=EOT)

    async def _get_subtoken_weights(self, token_ctx):
        token_ctx_hash = tuple(token_ctx)
        if token_ctx_hash not in self.cache:
            token_logws = await self.potential.logw_next(token_ctx)
            trie_ws = await self.trie.weight_sum(np.exp(token_logws.weights))
            self.cache[token_ctx_hash] = SubTokenMarginal(
                trie_ws, self.prefix2node, self.complete2node
            )
        return self.cache[token_ctx_hash]

    async def prefix(self, subtoken_ctx, token_ctx):
        subtoken = await self._get_subtoken_weights(tuple(token_ctx))
        return subtoken.prefix(tuple(subtoken_ctx))

    async def complete(self, subtoken_ctx, token_ctx):
        subtoken = await self._get_subtoken_weights(tuple(token_ctx))
        return subtoken.complete(tuple(subtoken_ctx))

    async def score(self, subtoken_ctx, token_ctx):
        if subtoken_ctx and subtoken_ctx[-1] is self.eos:
            return await self.complete(subtoken_ctx, token_ctx)
        else:
            return await self.prefix(subtoken_ctx, token_ctx)

    async def logw_next(self, subtoken_ctx, token_ctx):
        # TODO: optimize this.
        logws = self.alloc_logws()
        subtoken_ctx_hash = tuple(subtoken_ctx)
        subtoken = await self._get_subtoken_weights(subtoken_ctx_hash)

        prefix_logw = subtoken.prefix(subtoken_ctx_hash)
        for i, x in enumerate(self.decode):
            logws[i] = subtoken.prefix(subtoken_ctx_hash + (x,)) - prefix_logw
        logws[-1] = subtoken.complete(subtoken_ctx_hash) - prefix_logw

        return self.make_lazy_weights(logws)

    async def batch_prefix(self, subtoken_ctxs, tokens_ctxs):
        assert len(subtoken_ctxs) == len(tokens_ctxs)
        Ws = await asyncio.gather(
            *[
                self.prefix(subtoken_ctx, token_ctx)
                for subtoken_ctx, token_ctx in zip(subtoken_ctxs, tokens_ctxs)
            ]
        )
        return np.array(Ws)

    async def batch_complete(self, subtoken_ctxs, tokens_ctxs):
        assert len(subtoken_ctxs) == len(tokens_ctxs)
        Ws = await asyncio.gather(
            *[
                self.complete(subtoken_ctx, token_ctx)
                for subtoken_ctx, token_ctx in zip(subtoken_ctxs, tokens_ctxs)
            ]
        )
        return np.array(Ws)

    async def batch_score(self, subtoken_ctxs, tokens_ctxs):
        assert len(subtoken_ctxs) == len(tokens_ctxs)
        return await asyncio.gather(
            *[
                self.score(subtoken_ctx, token_ctx)
                for subtoken_ctx, token_ctx in zip(subtoken_ctxs, tokens_ctxs)
            ]
        )

    async def batch_logw_next(self, subtoken_ctxs, tokens_ctxs):
        assert len(subtoken_ctxs) == len(tokens_ctxs)
        return await asyncio.gather(
            *[
                self.logw_next(subtoken_ctx, token_ctx)
                for subtoken_ctx, token_ctx in zip(subtoken_ctxs, tokens_ctxs)
            ]
        )

    # TODO: Implement these
    def __mul__(self, other):
        raise NotImplementedError

    def to_autobatched(self):
        raise NotImplementedError

    def coerce(self, other):
        raise NotImplementedError

    def to_multiprocess(self):
        raise NotImplementedError

    def clear_cache(self):
        self.cache.clear()

    async def cleanup(self):  # TODO: move to genlm_backend
        if task := getattr(self.trie, "_task", None):
            if not task.done() and not task.cancelled():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
