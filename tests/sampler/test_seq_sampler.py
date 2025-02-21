import pytest
import numpy as np

from genlm_control.sampler.sequence import Importance, SMC
from genlm_control.sampler.token import DirectTokenSampler

from hypothesis import strategies as st, settings, given
from conftest import (
    weighted_set,
    weighted_sequence,
    double_weighted_sequence,
    WeightedSet,
)


@pytest.mark.asyncio
@settings(deadline=None)
@given(weighted_set(weighted_sequence))
async def test_importance(S):
    sequences, weights = zip(*S)

    p = WeightedSet(sequences, weights)
    unit_sampler = DirectTokenSampler(p)

    n_particles = 100
    sampler = Importance(unit_sampler, n_particles)

    sequences = await sampler.infer()
    assert len(sequences) == n_particles
    assert np.isclose(sequences.log_ml, np.log(sum(weights)), atol=1e-3, rtol=1e-5)


@pytest.mark.asyncio
@settings(deadline=None)
@given(weighted_set(double_weighted_sequence))
async def test_importance_with_critic(S):
    sequences, weights1, weights2 = zip(*S)

    p = WeightedSet(sequences, weights1)
    unit_sampler = DirectTokenSampler(p)
    critic = WeightedSet(sequences, weights2)

    n_particles = 10
    sampler = Importance(unit_sampler, n_particles=n_particles, critic=critic)
    sequences = await sampler.infer()

    target = p * critic
    for seq, logw in sequences:
        logZ = sum([(await p.logw_next(seq[:n])).sum() for n in range(len(seq))])
        logeps = await target.prefix([])
        assert np.isclose(logw, logZ + logeps + await critic.score(seq))


@pytest.mark.asyncio
@settings(deadline=None)
@given(weighted_set(weighted_sequence), st.floats(min_value=0, max_value=1))
async def test_smc(S, ess_threshold):
    sequences, weights = zip(*S)

    p = WeightedSet(sequences, weights)
    unit_sampler = DirectTokenSampler(p)

    n_particles = 100
    sampler = SMC(unit_sampler, n_particles, ess_threshold)

    sequences = await sampler.infer()
    assert len(sequences) == n_particles
    assert np.isclose(sequences.log_ml, np.log(sum(weights)), atol=1e-3, rtol=1e-5)


@pytest.mark.asyncio
@settings(deadline=None)
@given(weighted_set(double_weighted_sequence), st.floats(min_value=0, max_value=1))
async def _test_smc_with_critic(S, ess_threshold):
    sequences, weights1, weights2 = zip(*S)

    p = WeightedSet(sequences, weights1)
    unit_sampler = DirectTokenSampler(p)
    critic = WeightedSet(sequences, weights2)

    n_particles = 100
    sampler = SMC(unit_sampler, n_particles, ess_threshold=ess_threshold, critic=critic)

    sequences = await sampler.infer()

    intersection_ws = [w1 * w2 for w1, w2 in zip(weights1, weights2)]
    assert len(sequences) == n_particles
    assert np.isclose(
        sequences.log_ml, np.log(sum(intersection_ws)), atol=1e-3, rtol=1e-5
    )


@st.composite
def smc_params(draw, item_sampler, max_seq_len=5, max_size=5):
    S = draw(weighted_set(item_sampler, max_seq_len, max_size))
    stop_point = draw(st.integers(min_value=1, max_value=max_seq_len))
    return S, stop_point


@pytest.mark.asyncio
@settings(deadline=None)
@given(smc_params(double_weighted_sequence))
async def test_smc_weights(params):
    S, stop_point = params
    sequences, weights1, weights2 = zip(*S)

    p = WeightedSet(sequences, weights1)
    unit_sampler = DirectTokenSampler(p)
    critic = WeightedSet(sequences, weights2)

    n_particles = 100
    sampler = SMC(
        unit_sampler,
        critic=critic,
        n_particles=n_particles,
        ess_threshold=0,
        max_tokens=stop_point,
    )

    sequences = await sampler.infer()

    target = p * critic
    for seq, logw in sequences:
        logZ = sum([(await p.logw_next(seq[:n])).sum() for n in range(len(seq))])
        twist = await critic.score(seq)
        logeps = await target.prefix([])
        assert np.isclose(logw, logZ + logeps + twist)
