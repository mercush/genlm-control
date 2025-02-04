import pytest
import numpy as np
from itertools import chain
from arsenal.maths import logsumexp
from tqdm import tqdm

from genlm_control.sampler.token import TokenSampler, IncrementalTokenSampler
from genlm_control.potential.base import Potential
from genlm_control.potential.built_in import BoolFSA

# These tests check that the token sampler is properly
# weighted with respect to the unnormalized local product p
# it aims to sample from.

# For non-batched methods, we use sampling without replacement
# to compute E[wf(x)], where (x,w) ~ sampler, exactly. This then
# allows us to check that E[wf(x)] = \sum_x p(x)f(x).

# For batched methods, we use a monte carlo estimate to compute
# E[wf(x)] for each context, and check that this
# estimate is close to \sum_x p(x)f(x). We need a monte carlo
# method because the tracer cannot handle batching.


class UniformPotential(Potential):
    async def prefix(self, context):
        return 0

    async def complete(self, context):
        return 0


async def batch_monte_carlo_estimate(sampler, contexts, N):
    Ws = np.full((len(contexts), len(sampler.decode_eos)), -np.inf)
    for _ in tqdm(range(N)):
        tokens, log_ws, _ = await sampler.batch_sample_token(contexts)
        for i, (token, log_w) in enumerate(zip(tokens, log_ws)):
            token_id = sampler.encode_eos[token]
            Ws[i, token_id] = logsumexp([Ws[i, token_id], log_w - np.log(N)])

    return [sampler.make_lazy_weights(W) for W in Ws]


@pytest.mark.asyncio
async def test_direct():
    p = UniformPotential(["a", "ac", "acc", "b", "bc", "c"])
    sampler = TokenSampler(p)

    have = await sampler.trace_swor([])
    want = await p.logw_next([])
    have.assert_equal_unordered(want)

    N = 10000
    contexts = [[], ["acc"]]
    wants = await p.batch_logw_next(contexts)
    haves = await batch_monte_carlo_estimate(sampler, contexts, N)
    for have, want in zip(haves, wants):
        have.exp().assert_equal_unordered(want.exp(), atol=0.05, rtol=0.05)


@pytest.mark.asyncio
async def test_incremental():
    p = UniformPotential(["a", "ac", "acc", "b", "bc", "c"])
    guide = BoolFSA.from_regex(r"(a|b)c+", to_bytes=False)

    sampler = IncrementalTokenSampler(
        p, guide, f=lambda x: list(chain(*x)), g=lambda x: "".join(x)
    )
    product = p * guide.lift(p)

    have = await sampler.trace_swor([])
    want = await product.logw_next([])
    have.assert_equal_unordered(want)

    # EOS is available here
    have = await sampler.trace_swor(["acc"])
    want = await product.logw_next(["acc"])
    have.assert_equal_unordered(want)

    N = 10000
    contexts = [[], ["acc"]]
    wants = await product.batch_logw_next(contexts)
    haves = await batch_monte_carlo_estimate(sampler, contexts, N)
    for have, want in zip(haves, wants):
        have.exp().assert_equal_unordered(want.exp(), atol=0.05, rtol=0.05)
