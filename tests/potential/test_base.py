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
def V():
    return [b"a", b"b", b"c"]

@pytest.fixture
def potential(V):
    return SimplePotential(V)

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

@pytest.mark.asyncio
async def test_assert_properties(potential):
    context = [b"b", b"c"]
    await potential.assert_properties(context)

@pytest.mark.asyncio
async def test_batch_score(potential):
    seq1 = [b"a"]
    seq2 = [b"a", b"b"] 
    seq3 = [b"a", b"b", EOS]
    
    have = await potential.batch_score([seq1, seq2, seq3])
    want = await asyncio.gather(
        potential.score(seq1),
        potential.score(seq2), 
        potential.score(seq3)
    )
    
    np.testing.assert_array_equal(have, want)
    np.testing.assert_array_equal(have, [-0.5, -1.0, -2.0])

@pytest.mark.asyncio
async def test_batch_logp_next(potential):
    seq1 = [b"a"] 
    seq2 = [b"b", b"c"]  
    
    haves = await potential.batch_logp_next([seq1, seq2])
    wants = await asyncio.gather(
        potential.logp_next(seq1),
        potential.logp_next(seq2)
    )

    for want, have in zip(haves, wants):
        np.testing.assert_array_equal(have.weights, want.weights)