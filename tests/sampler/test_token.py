import pytest
from itertools import chain

from genlm_control.sampler.token import TokenSampler, IncrementalTokenSampler
from genlm_control.potential.base import Potential
from genlm_control.potential.built_in import BoolFSA


class UniformPotential(Potential):
    async def prefix(self, context):
        return 0

    async def complete(self, context):
        return 0


@pytest.mark.asyncio
async def test_direct():
    p = UniformPotential(["a", "ac", "acc", "b", "bc", "c"])
    sampler = TokenSampler(p)

    have = await sampler.trace_swor([])
    want = await p.logw_next([])
    have.assert_equal_unordered(want)

    # haves = await sampler.batch_trace_swor([[], ['a']])
    # wants = await p.batch_logw_next([[], ['a']])
    # for have, want in zip(haves, wants):


#     have.assert_equal_unordered(want)


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

    # haves = await sampler.batch_trace_swor([[], ['acc']])
    # wants = await product.batch_logw_next([[], ['acc']])
    # for have, want in zip(haves, wants):
    #    have.assert_equal_unordered(want)
