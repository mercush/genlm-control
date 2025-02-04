import pytest
import asyncio
import time
import numpy as np
from genlm_control.potential import Potential


class MockPotential(Potential):
    """Mock potential for testing with controlled delays"""

    def __init__(self):
        super().__init__(list(range(256)))
        self.delay = 0.1  # 100ms delay per operation

    async def complete(self, context):
        time.sleep(self.delay)
        return np.log(len(context))

    async def prefix(self, context):
        time.sleep(self.delay)
        return np.log(len(context) / 2)

    async def batch_complete(self, contexts):
        time.sleep(self.delay)  # Single delay for batch
        return [np.log(len(context)) for context in contexts]

    async def batch_prefix(self, contexts):
        time.sleep(self.delay)  # Single delay for batch
        return [np.log(len(context) / 2) for context in contexts]


@pytest.mark.asyncio
async def test_correctness():
    """Test that autobatched results match sequential results"""
    potential = MockPotential()
    autobatched = potential.to_autobatched()

    sequences = [b"hello", b"world", b"test", b"batch", b"foo"]

    want = await asyncio.gather(*(potential.complete(seq) for seq in sequences))
    have = await asyncio.gather(*(autobatched.complete(seq) for seq in sequences))
    assert want == have, [want, have]

    want = await asyncio.gather(*(potential.prefix(seq) for seq in sequences))
    have = await asyncio.gather(*(autobatched.prefix(seq) for seq in sequences))
    assert want == have, [want, have]

    want = await asyncio.gather(*(potential.score(seq) for seq in sequences))
    have = await asyncio.gather(*(autobatched.score(seq) for seq in sequences))
    assert want == have, [want, have]

    want = await asyncio.gather(
        *(potential.logw_next_seq([b"h"], seq) for seq in sequences)
    )
    have = await asyncio.gather(
        *(autobatched.logw_next_seq([b"h"], seq) for seq in sequences)
    )
    assert want == have, [want, have]

    wants = await asyncio.gather(*(potential.logw_next(seq) for seq in sequences))
    haves = await asyncio.gather(*(autobatched.logw_next(seq) for seq in sequences))
    for have, want in zip(haves, wants):
        have.assert_equal(want)

    await autobatched.cleanup()


@pytest.mark.asyncio
async def test_performance():
    """Test that autobatched operations are faster than sequential"""
    potential = MockPotential()
    autobatched = potential.to_autobatched()

    sequences = [b"hello", b"world", b"test", b"batch", b"foo"]

    start = time.perf_counter()
    await asyncio.gather(*(potential.complete(seq) for seq in sequences))
    sequential_time = time.perf_counter() - start

    start = time.perf_counter()
    await asyncio.gather(*(autobatched.complete(seq) for seq in sequences))
    autobatched_time = time.perf_counter() - start

    print(sequential_time, autobatched_time)

    assert autobatched_time < sequential_time / 2

    await autobatched.cleanup()


@pytest.mark.asyncio
async def test_error_handling():
    """Test that errors in batch processing are properly propagated"""

    class ErrorPotential(MockPotential):
        async def batch_complete(self, contexts):
            raise ValueError("Test error")

    potential = ErrorPotential()
    autobatched = potential.to_autobatched()

    with pytest.raises(ValueError, match="Test error"):
        await autobatched.complete(b"test")

    await autobatched.cleanup()
