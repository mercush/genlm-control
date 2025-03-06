import pytest
import numpy as np

from genlm_control.sampler import DirectTokenSampler, SetTokenSampler, EagerSetSampler
from conftest import mock_params, iter_item_params, MockPotential

from hypothesis import given, settings


@pytest.mark.asyncio
@settings(deadline=None)
@given(mock_params())
async def test_direct_token_sampler(params):
    vocab, next_token_ws, context = params
    mock_potential = MockPotential(vocab, np.log(next_token_ws))
    sampler = DirectTokenSampler(mock_potential)

    try:
        have = await sampler.trace_swor(context)
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
        have = await sampler.trace_swor(context)
        want = await sampler.target.logw_next(context)
        have.assert_equal(want, atol=1e-5, rtol=1e-5)
    finally:
        await sampler.cleanup()
