import torch
import numpy as np
from abc import ABC, abstractmethod
from arsenal.maths import logsumexp, sample_dict

from genlm_grammar import Float
from genlm_control.util import load_trie


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

    async def batch_trace_swor(self, contexts):
        from genlm_control.tracer import TraceSWOR

        tracer = TraceSWOR()
        logP_unnorms = np.full((len(contexts), len(self.decode_eos)), float("-inf"))
        while tracer.root.mass > 0:
            with tracer:
                tokens, Zs, log_ps = await self.batch_sample_token(
                    contexts, draw=tracer
                )
            for i, (token, Z, log_p) in enumerate(zip(tokens, Zs, log_ps)):
                token_id = self.encode_eos[token]
                logP_unnorms[i, token_id] = logsumexp(
                    [logP_unnorms[i, token_id], Z + log_p]
                )

        return [self.make_lazy_weights(logP_unnorm) for logP_unnorm in logP_unnorms]


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

        inc_logp = 0
        g_logw = 0

        Ws = np.full((len(self.decode) + 1,), float("-inf"))
        while True:
            children_curr = self.trie.children[curr]
            mass_curr = mass[curr]

            W_p_bytes = Float.chart()
            for a, c in children_curr.items():
                W_p_bytes[a] = mass[c] / mass_curr

            W_guide_bytes = (
                (await self.guide.logw_next(self.f(context) + path)).exp().materialize()
            )

            if None in W_p_bytes:
                log_m = np.log(mass[children_curr[None]])
                Ws[self.p.encode[self.g(path)]] = log_m + g_logw - inc_logp

            W_next = (W_p_bytes * W_guide_bytes).trim()

            if not W_next:
                break

            P_next = W_next.normalize()
            b = draw(P_next)
            inc_logp += np.log(P_next[b])
            g_logw += np.log(W_guide_bytes[b])

            path.append(b)
            curr = children_curr[b]

        return self.make_lazy_weights(Ws), inc_logp

    async def batch_logw_next(self, contexts, verbosity=0, draw=sample_dict):
        ws_batch = await self.p.batch_logw_next(contexts)
        ws_batch = [w.weights for w in ws_batch]
        masses = self.trie.batch_mass_sum(
            torch.tensor(np.exp(ws_batch), dtype=torch.float32)
        )

        batch_size = len(contexts)
        paths = [[] for _ in range(batch_size)]
        currs = [self.trie.root] * batch_size

        inc_logps = np.zeros(batch_size)
        g_logws = np.zeros(batch_size)

        batch_Ws = np.full((batch_size, len(self.decode)), float("-inf"))
        active = np.ones(batch_size, dtype=bool)

        while np.any(active):
            # Process only active contexts
            active_contexts = [
                self.f(ctx) + paths[i] for i, ctx in enumerate(contexts) if active[i]
            ]
            if not active_contexts:
                break

            W_guide_batch = await self.guide.batch_logw_next(active_contexts)
            W_guide_batch = [w.exp().materialize() for w in W_guide_batch]

            active_idx = 0
            for i in range(batch_size):
                if not active[i]:
                    continue

                children_curr = self.trie.children[currs[i]]
                mass_curr = masses[i][currs[i]]

                W_p_bytes = Float.chart()
                for a, c in children_curr.items():
                    W_p_bytes[a] = masses[i][c] / mass_curr

                W_guide_bytes = W_guide_batch[active_idx]

                if None in W_p_bytes:
                    log_m = np.log(masses[i][children_curr[None]])
                    batch_Ws[i, self.p.encode[self.g(paths[i])]] = (
                        log_m + g_logws[i] - inc_logps[i]
                    )

                W_next = (W_p_bytes * W_guide_bytes).trim()

                if not W_next:
                    active[i] = False
                    continue

                P_next = W_next.normalize()
                b = draw(P_next)
                inc_logps[i] += np.log(P_next[b])
                g_logws[i] += np.log(W_guide_bytes[b])

                paths[i].append(b)
                currs[i] = children_curr[b]
                active_idx += 1

        return [self.make_lazy_weights(Ws) for Ws in batch_Ws], inc_logps
