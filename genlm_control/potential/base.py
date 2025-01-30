import asyncio
import numpy as np
from abc import ABC, abstractmethod
from arsenal.maths import logsumexp, sample_dict

from genlm_control.util import LazyWeights
from genlm_control.sampler import WeightedSample
from genlm_control.operators import PotentialOperators


class Potential(ABC, PotentialOperators):
    def __init__(self, vocabulary):
        self.decode = vocabulary
        self.encode = {}
        for i, x in enumerate(vocabulary):
            if x in self.encode:
                raise ValueError(f"Duplicate token {x!r} found in vocabulary")
            self.encode[x] = i
        self.decode_eos = self.decode + [EOS]
        self.encode_eos = {**self.encode, **{EOS: len(self.decode)}}

    @abstractmethod
    async def complete(self, context):
        pass

    @abstractmethod
    async def prefix(self, context):
        pass

    async def score(self, context):
        if context and context[-1] is EOS:
            return await self.complete(context[:-1])
        return await self.prefix(context)

    async def logp_next(self, context):
        # logp_next(x)[eos] = complete(x) - prefix(x)
        # logp_next(x)[y] = prefix(x + y) - prefix(x)
        extended = [[*context, token] for token in self.decode]
        w_complete, W_prefix = await asyncio.gather(
            self.complete(context), self.batch_prefix([context] + extended)
        )

        if W_prefix[0] == float("-inf"):
            raise ValueError(f"Context {context!r} is not in the potential's domain.")

        W = np.zeros(len(self.decode_eos))
        W[:-1] = W_prefix[1:] - W_prefix[0]
        W[-1] = w_complete - W_prefix[0]

        return self.make_lazy_weights(W)

    async def logp_next_seq(self, context, extension):
        return (await self.score(context + extension)) - (await self.prefix(context))

    async def batch_logp_next(self, contexts):
        N = len(contexts)
        V = len(self.decode)

        all_extended = [
            [*context, token] for context in contexts for token in self.decode
        ]

        complete_ws, prefix_ws = await asyncio.gather(
            self.batch_complete(contexts), self.batch_prefix(contexts + all_extended)
        )

        assert (
            len(complete_ws) == N
        ), f"Expected {N} complete scores, got {len(complete_ws)}"
        assert len(prefix_ws) == N * (
            V + 1
        ), f"Expected {N * (V + 1)} prefix scores, got {len(prefix_ws)}"

        context_ws = prefix_ws[:N]
        extended_ws = prefix_ws[N:]

        logp_nexts = []
        for n in range(N):
            if context_ws[n] == float("-inf"):
                raise ValueError(
                    f"Context {contexts[n]!r} is not in the potential's domain."
                )

            W = np.zeros(len(self.decode_eos))
            W[:-1] = extended_ws[n * V : (n + 1) * V] - context_ws[n]  # decode
            W[-1] = complete_ws[n] - context_ws[n]  # eos

            logp_nexts.append(self.make_lazy_weights(W))

        return logp_nexts

    async def batch_complete(self, contexts):
        return np.array(
            await asyncio.gather(*[self.complete(context) for context in contexts])
        )

    async def batch_prefix(self, contexts):
        return np.array(
            await asyncio.gather(*[self.prefix(context) for context in contexts])
        )

    async def batch_score(self, contexts):
        return np.array(
            await asyncio.gather(*[self.score(context) for context in contexts])
        )

    async def batch_logp_next_seq(self, context, extensions):
        prefix = await self.prefix(context)
        if prefix == float("-inf"):
            raise ValueError(f"Context {context!r} is not in the potential's support.")
        scores = await self.batch_score(
            [context + extension for extension in extensions]
        )
        return scores - prefix

    async def sample(
        self,
        n_samples,
        generator=None,
        critic=None,
        resampler=None,
        context=[],
        max_tokens=25,
        draw=sample_dict,
    ):
        """Generates samples that are properly weighted with respect to self * critic."""
        if not generator:
            generator = self

        diff = self - generator

        particles = [WeightedSample(list(context)) for _ in range(n_samples)]
        n_tokens = 0
        while not all(a.finished for a in particles):
            import ipdb

            ipdb.set_trace()
            Ws = await generator.batch_logp_next([a.context for a in particles])
            for a, W in zip(particles, Ws):
                Z = logsumexp(W.weights)
                token = draw(Ws.normalize().materialize())
                a.context.append(token)
                a.log_w += Z

            if diff:
                corrections = diff.batch_score([a.context for a in particles])
                for a, correction in zip(particles, corrections):
                    a.log_w += correction

            if critic and resampler:
                twist_amts = critic.batch_score([a.context for a in particles])
                for a, twist_amt in zip(particles, twist_amts):
                    a.twist(twist_amt)

            if resampler:
                particles = resampler(particles)

            # if transformer: # TODO: for rejuvenation or more complex moves
            #    particles = transformer(particles)

            n_tokens += 1
            if n_tokens >= max_tokens:
                break

        if critic and not resampler:
            twist_amts = critic.batch_score([a.context for a in particles])
            for a, twist_amt in zip(particles, twist_amts):
                a.twist(twist_amt)

        return context, -1 # log_w

    def make_lazy_weights(self, weights, log=True):
        return LazyWeights(
            weights=weights, encode=self.encode_eos, decode=self.decode_eos, log=log
        )

    ################
    # Move to util #
    ################

    async def assert_properties(
        self, context, rtol=1e-3, atol=1e-5, top=None, verbosity=0
    ):
        logp_next = await self.logp_next(context)
        _top_logps = logp_next.materialize(top=top)

        context_w = await self.prefix(context)
        top_logp_next = logp_next.materialize(top=top)

        extensions = [list(context) + [x] for x in top_logp_next]
        extension_ws = await self.batch_score(extensions)

        errors, valids = [], []
        for i, token in enumerate(top_logp_next):
            want = extension_ws[i] - context_w
            have = top_logp_next[token]
            is_inf = want == float("-inf") and have == float("-inf")
            diff = {
                "token": repr(token),
                "expected": want,
                "actual": have,
                "abs_diff": 0 if is_inf else abs(want - have),
                "rel_diff": 0 if is_inf else abs((want - have) / want),
            }

            if np.isclose(want, have, rtol=rtol, atol=atol):
                valids.append(diff)
            else:
                errors.append(diff)

        if valids and verbosity > 0:
            print("\033[92mProperties satisfied for tokens:\033[0m\n")
            for valid in valids:
                print(
                    (
                        f"Token:    \033[94m{valid['token']}\033[0m\n"
                        f"Expected: \033[92m{valid['expected']:.6f}\033[0m (score(context + [{valid['token']!r}]) - prefix(context))\n"
                        f"Actual:   \033[93m{valid['actual']:.6f}\033[0m (logp_next(context)[{valid['token']!r}])\n"
                        f"Abs Diff: \033[96m{valid['abs_diff']:.6f} <= {atol=}\033[0m\n"
                        f"Rel Diff: \033[95m{valid['rel_diff']:.6f} <= {rtol=}\033[0m\n\n"
                    )
                )

        if errors:
            error_msg = (
                "\033[91mPotential properties not satisfied for tokens:\033[0m\n\n"
            )
            for error in errors:
                error_msg += (
                    f"Token:    \033[93m{error['token']}\033[0m\n"
                    f"Expected: \033[92m{error['expected']:.6f}\033[0m (score(context + [{error['token']!r}]) - prefix(context))\n"
                    f"Actual:   \033[91m{error['actual']:.6f}\033[0m (logp_next(context)[{error['token']!r}])\n"
                    f"Abs Diff: \033[95m{error['abs_diff']:.6f} > {atol=}\033[0m\n"
                    f"Rel Diff: \033[95m{error['rel_diff']:.6f} > {rtol=}\033[0m\n\n"
                )
            raise AssertionError(error_msg)


class eos:
    """Special sentinel token for end-of-sequence."""

    def __repr__(self):
        return "<EOS>"

    def __radd__(self, other):
        if isinstance(other, str):
            return other + "<EOS>"
        elif isinstance(other, bytes):
            return other + b"\x00"  # Using null byte as it's rarely used in text
        elif isinstance(other, (list, tuple)):
            return type(other)(list(other) + [self])


EOS = eos()
