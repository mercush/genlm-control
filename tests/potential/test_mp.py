import pytest
import numpy as np

from genlm_control.potential.base import Potential, EOS
from genlm_control.potential.mp import mp


class SimplePotential(Potential):
    async def complete(self, context):
        return -float(len(context))

    async def prefix(self, context):
        return -0.5 * float(len(context))


@pytest.fixture
def V():
    return [b"a", b"b", b"c"]


@pytest.fixture
def mp_potential(V):
    return mp(SimplePotential, (V,), num_workers=2)


@pytest.fixture
def regular_potential(V):
    return SimplePotential(V)


@pytest.mark.asyncio
async def test_mp_score(mp_potential, regular_potential):
    seq = [b"b", b"c"]

    mp_score = await mp_potential.score(seq)
    regular_score = await regular_potential.score(seq)
    assert mp_score == regular_score == -1.0

    seq_terminated = seq + [EOS]
    mp_score = await mp_potential.score(seq_terminated)
    regular_score = await regular_potential.score(seq_terminated)
    assert mp_score == regular_score == -2.0


@pytest.mark.asyncio
async def test_mp_batch_score(mp_potential, regular_potential):
    contexts = [[b"a"], [b"a", b"b"], [b"a", b"b", EOS]]

    have = await mp_potential.batch_score(contexts)
    want = await regular_potential.batch_score(contexts)
    np.testing.assert_array_equal(have, want)


@pytest.mark.asyncio
async def test_mp_prefix_complete(mp_potential, regular_potential):
    context = [b"b", b"c"]

    have = await mp_potential.prefix(context)
    want = await regular_potential.prefix(context)
    assert have == want == -1.0

    have = await mp_potential.complete(context)
    want = await regular_potential.complete(context)
    assert have == want == -2.0


@pytest.mark.asyncio
async def test_mp_batch_prefix_complete(mp_potential, regular_potential):
    contexts = [[b"a"], [b"a", b"b"]]

    have = await mp_potential.batch_prefix(contexts)
    want = await regular_potential.batch_prefix(contexts)
    np.testing.assert_array_equal(have, want)

    have = await mp_potential.batch_complete(contexts)
    want = await regular_potential.batch_complete(contexts)
    np.testing.assert_array_equal(have, want)


@pytest.mark.asyncio
async def test_mp_logp_next(mp_potential, regular_potential):
    seq = [b"b", b"c"]
    have = await mp_potential.logp_next(seq)
    want = await regular_potential.logp_next(seq)
    np.testing.assert_array_equal(have.weights, want.weights)


@pytest.mark.asyncio
async def test_mp_batch_logp_next(mp_potential, regular_potential):
    contexts = [[b"a"], [b"a", b"b"], [b"a", b"b", EOS]]
    haves = await mp_potential.batch_logp_next(contexts)
    wants = await regular_potential.batch_logp_next(contexts)
    for have, want in zip(haves, wants):
        np.testing.assert_array_equal(have.weights, want.weights)


@pytest.mark.asyncio
async def test_mp_logp_next_seq(mp_potential, regular_potential):
    context = [b"b"]
    extension = [b"c"]

    have = await mp_potential.logp_next_seq(context, extension)
    want = await regular_potential.logp_next_seq(context, extension)

    np.testing.assert_array_equal(have, want)


@pytest.mark.asyncio
async def test_mp_batch_logp_next_seq(mp_potential, regular_potential):
    context = [b"a"]
    extensions = [[b"c", b"a"], [b"d"], [b"e", b"a", b"g"]]
    haves = await mp_potential.batch_logp_next_seq(context, extensions)
    wants = await regular_potential.batch_logp_next_seq(context, extensions)

    for have, want in zip(haves, wants):
        np.testing.assert_array_equal(have, want)


def test_cleanup(mp_potential):
    assert mp_potential.pool is not None
    mp_potential.__del__()
    assert mp_potential.pool is None
