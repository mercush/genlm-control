import pytest
import numpy as np

from genlm_control.sampler import EagerSetSampler, TopKSetSampler
from conftest import iter_item_params, MockPotential

from hypothesis import given, strategies as st, settings


@pytest.mark.asyncio
@settings(deadline=None)
@given(iter_item_params())
async def test_eager_set_sampler(params):
    iter_vocab, iter_next_token_ws, item_vocab, item_next_token_ws, context = params

    mock_iter = MockPotential(iter_vocab, np.log(iter_next_token_ws))
    mock_item = MockPotential(item_vocab, np.log(item_next_token_ws))

    eager_set_sampler = EagerSetSampler(
        iter_potential=mock_iter,
        item_potential=mock_item,
    )

    try:
        have = await eager_set_sampler.trace_swor(context)
        want = await eager_set_sampler.target.logw_next(context)
        have.assert_equal(want, atol=1e-5, rtol=1e-5)
    finally:
        await eager_set_sampler.cleanup()


@pytest.mark.asyncio
@settings(deadline=None)
@given(iter_item_params(max_item_w=1), st.integers(1, 30))
async def test_topk_set_sampler(params, K):
    iter_vocab, iter_next_token_ws, item_vocab, item_next_token_ws, context = params

    mock_iter = MockPotential(iter_vocab, np.log(iter_next_token_ws))
    mock_item = MockPotential(item_vocab, np.log(item_next_token_ws))

    topk_set_sampler = TopKSetSampler(
        iter_potential=mock_iter, item_potential=mock_item, K=K
    )

    try:
        have = await topk_set_sampler.trace_swor(context)
        want = await topk_set_sampler.target.logw_next(context)
        have.assert_equal(want, atol=1e-5, rtol=1e-5)
    finally:
        await topk_set_sampler.cleanup()
