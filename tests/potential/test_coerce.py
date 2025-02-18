import pytest
import numpy as np
from genlm_control.typing import Atomic
from genlm_control.constant import EOS
from genlm_control.potential.coerce import Coerced
from genlm_control.potential.base import Potential


class MockPotential(Potential):
    """A simple mock potential for testing the Lifted potential."""

    def __init__(self):  # individual bytes
        super().__init__([b"a"[0], b"b"[0], b"c"[0]])

    async def complete(self, context):
        return len(context)

    async def prefix(self, context):
        return len(context) / 2


@pytest.fixture
def mock_potential():
    return MockPotential()


@pytest.fixture
def target_vocab():  # bytes
    return [b"aa", b"bb", b"aab", b"dd"]


@pytest.fixture
def coerced(mock_potential, target_vocab):
    def f(seq):
        return [b for bs in seq for b in bs]

    return Coerced(mock_potential, target_vocab, f=f)


def test_token_type(coerced):
    assert coerced.token_type == Atomic(bytes)


@pytest.mark.asyncio
async def test_coerced_initialization(coerced, mock_potential):
    assert coerced.potential == mock_potential
    assert set(coerced.decode) == {b"aa", b"bb", b"aab"}


@pytest.mark.asyncio
async def test_coerced_prefix(coerced):
    result = await coerced.prefix([b"aa", b"bb"])
    assert result == 2


@pytest.mark.asyncio
async def test_coerced_complete(coerced):
    result = await coerced.complete([b"aa", b"bb"])
    assert result == 4


@pytest.mark.asyncio
async def test_coerced_score(coerced):
    result = await coerced.score([b"aa", b"bb", EOS])
    assert result == 4


@pytest.mark.asyncio
async def test_coerced_logw_next(mock_potential, coerced):
    have = await coerced.logw_next([b"aa", b"bb"])
    want = await mock_potential.batch_logw_next_seq(b"aabb", coerced.decode_eos)
    for i, x in enumerate(coerced.decode_eos):
        assert have[x] == want[i], [have[x], want[i], x]


@pytest.mark.asyncio
async def test_coerced_batch_operations(coerced):
    sequences = [[b"aa"], [b"bb"]]

    have = await coerced.batch_complete(sequences)
    want = np.array([await coerced.complete(sequence) for sequence in sequences])
    np.testing.assert_array_equal(have, want)

    have = await coerced.batch_prefix(sequences)
    want = np.array([await coerced.prefix(sequence) for sequence in sequences])
    np.testing.assert_array_equal(have, want)

    have = await coerced.batch_score(sequences)
    want = np.array([await coerced.score(sequence) for sequence in sequences])
    np.testing.assert_array_equal(have, want)

    haves = await coerced.batch_logw_next(sequences)
    wants = [await coerced.logw_next(sequence) for sequence in sequences]
    for have, want in zip(haves, wants):
        have.assert_equal(want)

    have = await coerced.batch_logw_next_seq([b"aa"], sequences)
    want = np.array(
        [await coerced.logw_next_seq([b"aa"], sequences) for sequence in sequences]
    )
    np.testing.assert_array_equal(have, want)


@pytest.mark.asyncio
async def test_coerced_invalid_vocab():
    def f(seq):
        return [b for bs in seq for b in bs]  # list of bytes -> list of ints

    with pytest.raises(ValueError):
        Coerced(MockPotential(), [b"xx", b"yy"], f=f)


@pytest.mark.asyncio
async def test_coerced_custom(mock_potential):
    coerced = Coerced(
        mock_potential,
        target_vocab=[b"aa", b"bb"],
        f=lambda seq: [item[0] for item in seq],  # Take first byte of each token
    )

    assert coerced.token_type == Atomic(bytes)

    assert len(coerced.decode) == 2
    assert set(coerced.decode) == {b"a", b"b"}

    have = await coerced.complete([b"aa", b"bb"])
    want = await mock_potential.complete(b"ab")
    assert have == want

    have = await coerced.prefix([b"aa", b"bb"])
    want = await mock_potential.prefix(b"ab")
    assert have == want

    have = await coerced.score([b"aa", b"bb", EOS])
    want = await mock_potential.score(b"ab" + EOS)
    assert have == want
