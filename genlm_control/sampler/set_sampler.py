import numpy as np
from genlm_grammar import Float
from arsenal.maths import sample_dict, logsumexp
from arsenal.datastructures import LocatorMaxHeap
from abc import ABC, abstractmethod

from genlm_control.util import load_async_trie
from genlm_control.constant import EOS


class SetSampler(ABC):
    """Base class for set samplers.

    A set sampler samples a weighted set of tokens. The weight associated with each token is given as:

        target.logw_next(token | context) - log_inclusion_probability

    where log_inclusion_probability is the log of the probability the token was included in the sampled set.

    Attributes:
        target (Potential): The target potential with respect to which the set's weights are computed.
        token_type (TokenType): The type of tokens in the set.
    """

    def __init__(self, target):
        self.target = target
        self.token_type = self.target.token_type

    @abstractmethod
    async def sample_set(self, context: list) -> Float.chart:
        pass

    async def trace_swor(self, context):
        from genlm_control.tracer import TraceSWOR

        tracer = TraceSWOR()
        logP = Float.chart()
        while tracer.root.mass > 0:
            with tracer:
                tokens, logp = await self.sample_set(context, draw=tracer)
                for t, logw in tokens.items():
                    if t in logP:
                        logP[t] = logsumexp([logP[t], logw + logp])
                    else:
                        logP[t] = logw + logp

        return logP


class TrieSetSampler(SetSampler):
    """
    A set sampler that uses a trie to sample a weighted set of tokens.
    """

    def __init__(self, iterable_potential, item_potential, f):
        if not iterable_potential.token_type.is_iterable_of(item_potential.token_type):
            raise ValueError(
                "The token type of the iterable_potential must be an iterable of the token type of the item_potential. "
                f"Got {iterable_potential.token_type} and {item_potential.token_type}."
            )
        super().__init__(
            iterable_potential * item_potential.coerce(iterable_potential, f=f)
        )
        self.iterable_potential = iterable_potential
        self.item_potential = item_potential
        self.f = f
        self.trie_executor = load_async_trie(
            self.iterable_potential.decode_eos, backend="parallel"
        )
        self.trie = self.trie_executor.trie

    async def sample_set(self, context):
        raise NotImplementedError("Subclasses must implement sample_set")


class EagerSetSampler(TrieSetSampler):
    async def sample_set(self, context, draw=sample_dict):
        logws = await self.iterable_potential.logw_next(context)
        item_ws = await self.trie_executor.weight_sum(logws.exp().weights)

        tokens = Float.chart()
        curr = self.trie.root
        subtokens = []
        logp, logw = 0, 0

        while True:
            children = self.trie.children[curr]
            item_w_curr = item_ws[curr]
            item_ws1 = Float.chart(
                {a: item_ws[c] / item_w_curr for a, c in children.items()}
            )

            if None in item_ws1:
                leaf = children[None]
                token = self.trie.leaf2word[leaf]
                tokens[token] = logws[token] + logw - logp

            item_logws2 = await self.item_potential.logw_next(
                self.f(context) + subtokens
            )
            item_ws2 = item_logws2.exp().materialize()
            w_next = (item_ws1 * item_ws2).trim()

            if not w_next:
                break

            ps = w_next.normalize()
            b = draw(ps)
            logp += np.log(ps[b])
            logw += item_logws2[b]

            if b is EOS:
                assert not subtokens, "subtokens should be empty at EOS"
                tokens[EOS] = logws[EOS] + logw - logp
                break

            subtokens.append(b)
            curr = children[b]

        return tokens, logp


class TopKSetSampler(TrieSetSampler):
    def __init__(self, iterable_potential, item_potential, f, K):
        super().__init__(iterable_potential, item_potential, f=f)
        self.K = K

    async def sample_set(self, context, draw=sample_dict):
        logws = await self.iterable_potential.logw_next(context)
        max_logws = await self.trie_executor.weight_max(logws.weights)

        k = 0
        tokens = Float.chart()
        async for token, logw in self._lazy_enum(context, max_logws):
            tokens[token] = logw
            k += 1
            if k >= self.K:
                break

        # This step is expensive!!
        if self.K and len(tokens) == self.K:
            # Get the distribution over wildcard tokens
            W_wc = Float.chart(
                {token: w for token, w in logws.exp().items() if token not in tokens}
            )

            # if W_wc is non-empty, sample a wildcard token to ensure absolute continuity
            if W_wc:
                P_wc = W_wc.normalize()
                wildcard = draw(P_wc)
                logp_wc = np.log(P_wc[wildcard])
                w_guide_wc = await self.item_potential.logw_next_seq(
                    self.f(context), self.f([wildcard])
                )
                tokens[wildcard] = np.log(W_wc[wildcard]) + w_guide_wc - logp_wc

        return tokens, logp_wc

    async def _lazy_enum(self, context, max_logws):
        agenda = LocatorMaxHeap()

        W = Float.chart()

        # initial conditions
        (token, node) = ((), self.trie.root)
        agenda[token, node, False] = max_logws[node]
        W[node] = 0

        children = self.trie.children

        curr_priority = float("inf")
        prev_best = float("inf")
        while agenda:
            (token, node, done), score = agenda.popitem()

            assert score <= curr_priority, [score, curr_priority]
            curr_priority = score

            # terminal state
            if done:
                value = W[node] + max_logws[node]
                assert prev_best >= value
                prev_best = value
                yield (self.trie.leaf2word[node], value)
                continue

            logws = await self.item_potential.logw_next(self.f(context) + list(token))
            # Our heuristic won't work if the item potential assigns positive log-weights to any tokens
            assert all(logw <= 0 for logw in logws.values()), (
                "All item potential logws must be <= 0"
            )

            for x, y in children[node].items():
                if x is None:
                    W_y = W[node]
                    W[y] = W_y
                    agenda[token, y, True] = W_y + max_logws[y]
                else:
                    W_y = W[node] + logws[x]
                    if W_y == float("-inf"):
                        continue
                    W[y] = W_y
                    agenda[(*token, x), y, False] = W_y + max_logws[y]
