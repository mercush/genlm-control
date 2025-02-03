import pytest
import asyncio
import numpy as np

from genlm_control.potential.base import Potential, EOS


class SimplePotential(Potential):
    async def complete(self, context):
        return -float(len(context))  # Scoring based on length

    async def prefix(self, context):
        return -0.5 * float(len(context))  # Different scoring for prefixes


@pytest.fixture
def potential():
    return SimplePotential([b"a", b"b", b"c"])


@pytest.mark.asyncio
async def test_score(potential):
    context = [b"b", b"c"]

    have = await potential.score(context)
    want = await potential.prefix(context)
    assert want == -1.0
    assert have == want

    have = await potential.score(context + [EOS])
    want = await potential.complete(context)
    assert want == -2.0
    assert have == want

    have = await potential.score([])
    want = await potential.prefix([])
    assert want == 0.0
    assert have == want


@pytest.mark.asyncio
async def test_logw_next_seq(potential):
    context = [b"b", b"c"]
    extension = b"a"
    have = await potential.logw_next_seq(context, extension)
    want = await potential.score(context + [extension]) - await potential.prefix(
        context
    )
    assert have == want


@pytest.mark.asyncio
async def test_logw_next(potential):
    context = [b"b", b"c"]
    have = (await potential.logw_next(context)).materialize()
    for token in potential.decode_eos:
        want = await potential.score(context + [token]) - await potential.prefix(
            context
        )
        assert have[token] == want


@pytest.mark.asyncio
async def test_batch_score(potential):
    seq1 = [b"a"]
    seq2 = [b"a", b"b"]
    seq3 = [b"a", b"b", EOS]

    have = await potential.batch_score([seq1, seq2, seq3])
    want = await asyncio.gather(
        potential.score(seq1), potential.score(seq2), potential.score(seq3)
    )

    np.testing.assert_array_equal(have, want)
    np.testing.assert_array_equal(have, [-0.5, -1.0, -2.0])


@pytest.mark.asyncio
async def test_batch_logw_next(potential):
    seq1 = [b"a"]
    seq2 = [b"b", b"c"]

    haves = await potential.batch_logw_next([seq1, seq2])
    wants = await asyncio.gather(potential.logw_next(seq1), potential.logw_next(seq2))

    for want, have in zip(haves, wants):
        np.testing.assert_array_equal(have.weights, want.weights)


@pytest.mark.asyncio
async def test_batch_logw_next_seq(potential):
    context = [b"b", b"c"]
    extensions = [b"a", b"b"]
    haves = await potential.batch_logw_next_seq(context, extensions)
    for i, extension in enumerate(extensions):
        want = await potential.logw_next_seq(context, extensions[i])
        assert haves[i] == want


@pytest.mark.asyncio
async def test_empty(potential):
    with pytest.raises(ValueError):
        await potential.batch_logw_next_seq([b"b", b"c"], [])

    with pytest.raises(ValueError):
        await potential.batch_logw_next([])

    with pytest.raises(ValueError):
        await potential.batch_score([])

    with pytest.raises(ValueError):
        await potential.batch_prefix([])

    with pytest.raises(ValueError):
        await potential.batch_complete([])


@pytest.mark.asyncio
async def test_properties(potential):
    await potential.assert_logw_next_consistency([b"b", b"c"], verbosity=1)
    await potential.assert_autoreg_fact([b"b", b"c"], verbosity=1)
    await potential.assert_batch_consistency(
        [[b"b", b"c"], [b"a"]], extensions=[b"a", b"b"], verbosity=1
    )
