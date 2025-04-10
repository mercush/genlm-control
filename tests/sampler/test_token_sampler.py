import pytest
import tempfile
import numpy as np

from genlm.control.sampler import DirectTokenSampler, SetTokenSampler, EagerSetSampler
from conftest import (
    mock_params,
    iter_item_params,
    MockPotential,
    trace_swor,
    mock_vocab,
)

from hypothesis import given, settings, strategies as st


@pytest.mark.asyncio
@settings(deadline=None)
@given(mock_params())
async def test_direct_token_sampler(params):
    vocab, next_token_ws, context = params
    mock_potential = MockPotential(vocab, np.log(next_token_ws))
    sampler = DirectTokenSampler(mock_potential)

    try:
        have = await trace_swor(sampler, context)
        want = await sampler.target.logw_next(context)
        have.assert_equal(want, atol=1e-5, rtol=1e-5)
    finally:
        await sampler.cleanup()


@pytest.mark.asyncio
@settings(deadline=None)
@given(iter_item_params())
async def test_set_token_sampler(params):
    iter_vocab, iter_next_token_ws, item_vocab, item_next_token_ws, context = params

    mock_iter = MockPotential(iter_vocab, np.log(iter_next_token_ws))
    mock_item = MockPotential(item_vocab, np.log(item_next_token_ws))

    sampler = SetTokenSampler(
        set_sampler=EagerSetSampler(
            iter_potential=mock_iter,
            item_potential=mock_item,
        )
    )

    try:
        have = await trace_swor(sampler, context)
        want = await sampler.target.logw_next(context)
        have.assert_equal(want, atol=1e-5, rtol=1e-5)
    finally:
        await sampler.cleanup()


@st.composite
def mock_vocab_and_logws(draw, max_w=1e3):
    vocab = draw(mock_vocab())
    ws = draw(
        st.lists(
            st.floats(1e-5, max_w),
            min_size=len(vocab) + 1,
            max_size=len(vocab) + 1,
        )
    )
    ws2 = draw(
        st.lists(
            st.floats(1e-5, max_w),
            min_size=len(vocab) + 1,
            max_size=len(vocab) + 1,
        )
    )
    logws = [np.log(w) if w > 0 else -np.inf for w in ws]
    logws2 = [np.log(w) if w > 0 else -np.inf for w in ws2]
    return vocab, logws, logws2


@pytest.mark.asyncio
@settings(deadline=None)
@given(mock_vocab_and_logws())
async def test_smc_token_sampler(params):
    vocab, logws, logws_critic = params
    mock_potential = MockPotential(vocab, logws)
    sequences = await DirectTokenSampler(mock_potential).smc(
        n_particles=10,
        ess_threshold=0.5,
        max_tokens=10,
    )
    assert len(sequences) == 10
    assert all(len(seq) <= 10 for seq in sequences)

    mock_critic = MockPotential(vocab, logws_critic)
    sequences = await DirectTokenSampler(mock_potential).smc(
        n_particles=10,
        ess_threshold=0.5,
        max_tokens=10,
        critic=mock_critic,
    )
    assert len(sequences) == 10
    assert all(len(seq) <= 10 for seq in sequences)

    with tempfile.NamedTemporaryFile() as tmp:
        sequences = await DirectTokenSampler(mock_potential).smc(
            n_particles=10,
            ess_threshold=0.5,
            max_tokens=10,
            json_path=tmp.name,
        )
        assert len(sequences) == 10
        assert all(len(seq) <= 10 for seq in sequences)

        sequences = await DirectTokenSampler(mock_potential).smc(
            n_particles=10,
            ess_threshold=0.5,
            max_tokens=10,
            critic=mock_critic,
            json_path=tmp.name,
        )
        assert len(sequences) == 10
        assert all(len(seq) <= 10 for seq in sequences)
