import torch
import numpy as np
from abc import ABC, abstractmethod
from arsenal.maths import logsumexp, sample_dict

from genlm_grammar import Float
from genlm_control.constant import EOS
from genlm_control.util import load_trie, sample_categorical, LazyWeights


class TokenSampler(ABC):
    """Abstract base class for properly weighted samplers of tokens.

    Args:
        target: The target potential that samples with be properly weighted with respect to.
    """

    def __init__(self, target):
        self.target = target
        self.decode = target.decode
        self.decode_eos = target.decode_eos
        self.encode = target.encode
        self.encode_eos = target.encode_eos

    @abstractmethod
    async def logw_next(self, context):
        pass

    @abstractmethod
    async def batch_logw_next(self, contexts):
        pass

    async def sample_token(self, context, draw=None):
        W, logp = await self.logw_next(context, draw=draw)

        draw = draw or (lambda ws: self.decode_eos[sample_categorical(ws.weights)])

        logps = W.normalize()
        token = draw(logps.exp())

        return token, W.sum(), logp + logps[token]

    async def batch_sample_token(self, contexts, draw=None):
        Ws, logps = await self.batch_logw_next(contexts, draw=draw)

        draw = draw or (lambda ws: self.decode_eos[sample_categorical(ws.weights)])

        results = []
        for W, logp in zip(Ws, logps):
            logps = W.normalize()
            token = draw(logps.exp())
            results.append([token, W.sum(), logp + logps[token]])

        tokens, log_ws, log_ps = zip(*results)

        return tokens, log_ws, log_ps

    def make_lazy_weights(self, weights, **kwargs):
        return LazyWeights(
            weights=weights, encode=self.encode_eos, decode=self.decode_eos, **kwargs
        )

    async def trace_swor(self, context):
        from genlm_control.tracer import TraceSWOR

        tracer = TraceSWOR()
        logP_unnorm = np.full(len(self.decode_eos), float("-inf"))
        while tracer.root.mass > 0:
            with tracer:
                token, Z, log_p = await self.sample_token(context, draw=tracer)
            token_id = self.encode_eos[token]
            logP_unnorm[token_id] = logsumexp([logP_unnorm[token_id], Z + log_p])

        return self.make_lazy_weights(logP_unnorm)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.target!r})"


class DirectTokenSampler(TokenSampler):
    def __init__(self, p):
        super().__init__(target=p)
        self.p = p

    async def logw_next(self, context, draw=None):
        return await self.p.logw_next(context), 0

    async def batch_logw_next(self, contexts, draw=None):
        return await self.p.batch_logw_next(contexts), np.zeros(len(contexts))


class IncrementalTokenSampler(TokenSampler):
    def __init__(self, p, guide, f, g, **kwargs):
        super().__init__(target=p * guide.lift(p, f=f, g=g))
        self.p = p
        self.guide = guide
        self.f = f
        self.g = g

        self.trie = load_trie(self.decode_eos, **kwargs)
        self.p_idxs = [p.encode_eos[x] for x in self.decode_eos]

    async def logw_next(self, context, verbosity=0, draw=None):
        draw = draw or sample_dict
        p_ws = await self.p.logw_next(context)
        mass = self.trie.mass_sum(torch.tensor(np.exp(p_ws.weights[self.p_idxs])))
        Ws, logps = await self._batch_traverse_trie([context], [mass], draw)
        return Ws[0], logps[0]

    async def batch_logw_next(self, contexts, verbosity=0, draw=None):
        draw = draw or sample_dict
        ws_batch = await self.p.batch_logw_next(contexts)
        stacked_weights = np.stack([w.weights for w in ws_batch])
        masses = self.trie.batch_mass_sum(np.exp(stacked_weights[:, self.p_idxs]))
        return await self._batch_traverse_trie(contexts, masses, draw)

    async def _batch_traverse_trie(self, contexts, masses, draw):
        batch_size = len(contexts)
        paths = [[] for _ in range(batch_size)]
        currs = [self.trie.root] * batch_size

        inc_logps = np.zeros(batch_size)
        g_logws = np.zeros(batch_size)

        batch_Ws = np.full((batch_size, len(self.decode_eos)), float("-inf"))
        active = np.ones(batch_size, dtype=bool)
        while np.any(active):
            active_indices = np.where(active)[0]
            active_contexts = [self.f(contexts[i]) + paths[i] for i in active_indices]
            if not active_contexts:
                break

            W_guide_batch = await self.guide.batch_logw_next(active_contexts)
            W_guide_batch = [w.exp().materialize() for w in W_guide_batch]

            for batch_idx, active_idx in enumerate(active_indices):
                mass = masses[active_idx]
                curr = currs[active_idx]
                mass_curr = mass[curr]
                children_curr = self.trie.children[curr]

                W_p_bytes = Float.chart()
                for a, c in children_curr.items():
                    W_p_bytes[a] = mass[c] / mass_curr

                if None in W_p_bytes:
                    log_m = np.log(mass[children_curr[None]])
                    token = self.encode[self.g(paths[active_idx])]
                    batch_Ws[active_idx, token] = (
                        log_m + g_logws[active_idx] - inc_logps[active_idx]
                    )

                W_guide_bytes = W_guide_batch[batch_idx]

                W_next = (W_p_bytes * W_guide_bytes).trim()

                if not W_next:
                    active[active_idx] = False
                    continue

                P_next = W_next.normalize()
                b = draw(P_next)
                inc_logps[active_idx] += np.log(P_next[b])
                g_logws[active_idx] += np.log(W_guide_bytes[b])

                paths[active_idx].append(b)
                currs[active_idx] = children_curr[b]

                if b is EOS:
                    children_curr = self.trie.children[currs[active_idx]]
                    batch_Ws[active_idx, -1] = (
                        np.log(masses[active_idx][children_curr[None]])
                        + g_logws[active_idx]
                        - inc_logps[active_idx]
                    )
                    active[active_idx] = False

        return [self.make_lazy_weights(Ws) for Ws in batch_Ws], inc_logps
