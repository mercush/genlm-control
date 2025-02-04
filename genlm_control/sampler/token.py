import torch
import numpy as np
from abc import ABC, abstractmethod
from arsenal.maths import logsumexp, sample_dict

from genlm_grammar import Float
from genlm_control.util import load_trie
from genlm_control.constant import EOS


class TokenSampler:
    def __init__(self, p):
        self.p = p

        self.decode = p.decode
        self.decode_eos = p.decode_eos
        self.encode = p.encode
        self.encode_eos = p.encode_eos

    async def sample_token(self, context, draw=sample_dict):
        W = await self.p.logw_next(context)
        logps = W.normalize()
        token = draw(logps.exp())
        return token, W.sum(), logps[token]

    async def batch_sample_token(self, contexts, draw=sample_dict):
        Ws = await self.p.batch_logw_next(contexts)
        tokens = []
        log_ws = []
        log_ps = []
        for W in Ws:
            logP = W.normalize()
            token = draw(logP.exp())
            tokens.append(token)
            log_ws.append(W.sum())
            log_ps.append(logP[token])
        return tokens, log_ws, log_ps

    def make_lazy_weights(self, Ws, **kwargs):
        return self.p.make_lazy_weights(Ws, **kwargs)

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

    # async def batch_trace_swor(self, contexts):
    #    from genlm_control.tracer import TraceSWOR

    #    tracer = TraceSWOR()
    #    logP_unnorms = np.full((len(contexts), len(self.decode_eos)), float("-inf"))
    #    while tracer.root.mass > 0:
    #        with tracer:
    #            tokens, Zs, log_ps = await self.batch_sample_token(
    #                contexts, draw=tracer
    #            )
    #        for i, (token, Z, log_p) in enumerate(zip(tokens, Zs, log_ps)):
    #            token_id = self.encode_eos[token]
    #            logP_unnorms[i, token_id] = logsumexp(
    #                [logP_unnorms[i, token_id], Z + log_p]
    #            )
    #
    #    return [self.make_lazy_weights(logP_unnorm) for logP_unnorm in logP_unnorms]


class SetTokenSampler(ABC, TokenSampler):
    @abstractmethod
    async def logw_next(self, context):
        pass

    @abstractmethod
    async def batch_logw_next(self, contexts):
        pass

    async def sample_token(self, context, draw=sample_dict):
        W, logp = await self.logw_next(context, draw=draw)
        logps = W.normalize()
        token = draw(logps.exp())
        return token, W.sum(), logp + logps[token]

    async def batch_sample_token(self, contexts, draw=sample_dict):
        Ws, logps = await self.batch_logw_next(contexts, draw=draw)
        tokens = []
        log_ws = []
        log_ps = []
        for W, logp in zip(Ws, logps):
            logP = W.normalize()
            token = draw(logP.exp())
            tokens.append(token)
            log_ws.append(W.sum())
            log_ps.append(logp + logP[token])
        return tokens, log_ws, log_ps


class TopKTokenSampler(SetTokenSampler):
    def __init__(self, p, guide, f, K):
        super().__init__(p)
        self.K = K
        self.guide = guide
        self.f = f
        self.trie = load_trie(p.decode_eos)

    def sample_token(self, context):
        pass


class IncrementalTokenSampler(SetTokenSampler):
    # TODO: make relationship between p and guide explicit
    def __init__(self, p, guide, f, g):
        super().__init__(p)
        self.guide = guide
        self.f = f
        self.g = g
        self.trie = load_trie(self.decode_eos)

    def __repr__(self):
        return f"IncrementalTokenSampler(p={self.p!r}, guide={self.guide!r}, f={self.f!r}, g={self.g!r})"

    async def logw_next(self, context, verbosity=0, draw=sample_dict):
        ws = (await self.p.logw_next(context)).weights
        mass = self.trie.mass_sum(  # TODO: check dtype in genlm-backend
            torch.tensor(np.exp(ws), dtype=torch.float32)
        )

        path = []
        curr = self.trie.root
        children_curr = self.trie.children[curr]

        inc_logp = 0
        g_logw = 0

        Ws = np.full((len(self.decode_eos),), float("-inf"))
        while True:
            mass_curr = mass[curr]

            W_p_bytes = Float.chart()
            for a, c in children_curr.items():
                W_p_bytes[a] = mass[c] / mass_curr

            if None in W_p_bytes:
                p_logw = np.log(mass[children_curr[None]])
                Ws[self.p.encode[self.g(path)]] = p_logw + g_logw - inc_logp

            W_guide_bytes = (
                (await self.guide.logw_next(self.f(context) + path)).exp().materialize()
            )

            W_next = (W_p_bytes * W_guide_bytes).trim()

            if not W_next:
                break

            P_next = W_next.normalize()
            b = draw(P_next)
            inc_logp += np.log(P_next[b])
            g_logw += np.log(W_guide_bytes[b])

            path.append(b)
            curr = children_curr[b]
            children_curr = self.trie.children[curr]

            if b is EOS:
                assert None in children_curr
                assert len(children_curr) == 1
                p_logw = np.log(mass[children_curr[None]])
                Ws[-1] = p_logw + g_logw - inc_logp
                break

        return self.make_lazy_weights(Ws), inc_logp

    async def batch_logw_next(self, contexts, verbosity=0, draw=sample_dict):
        ws_batch = await self.p.batch_logw_next(contexts)
        masses = self.trie.batch_mass_sum(
            torch.tensor(np.exp([w.weights for w in ws_batch]), dtype=torch.float32)
        )

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
                    token = self.p.encode[self.g(paths[active_idx])]
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
                    assert None in children_curr
                    assert len(children_curr) == 1
                    p_logw = np.log(masses[active_idx][children_curr[None]])
                    batch_Ws[active_idx, -1] = (
                        p_logw + g_logws[active_idx] - inc_logps[active_idx]
                    )
                    active[active_idx] = False

        return [self.make_lazy_weights(Ws) for Ws in batch_Ws], inc_logps
