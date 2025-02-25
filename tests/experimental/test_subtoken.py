import pytest
import numpy as np
from arsenal.maths import logsumexp
from hypothesis import given, settings, strategies as st
from conftest import mock_params, mock_vocab_and_ws, MockPotential

from genlm_control.constant import EOT, EOS
from genlm_control.experimental.subtoken import SubtokenPotential


@pytest.mark.asyncio
@settings(deadline=None, max_examples=5)
@given(mock_vocab_and_ws())
async def test_init(params):
    vocab, next_token_ws = params
    p = MockPotential(vocab, np.log(next_token_ws))
    p_st = SubtokenPotential(p)

    await p_st.cleanup()

    assert all(
        all(subtoken in p_st.decode_eos for subtoken in token) for token in p.decode_eos
    )

    # Check that all tokens are in complete2node.
    assert all(tuple(v) in p_st.complete2node for v in p.decode_eos)
    # Check that all prefixes are in prefix2node.
    assert (
        all(
            all(tuple(v[:i]) in p_st.prefix2node for i in range(0, len(v) + 1))
            for v in p.decode
        )
        and tuple(p.eos) in p_st.prefix2node
    )

    assert p_st.decode_eos[-1] is EOT
    assert p_st.prefix2node[()] is p_st.trie.trie.root


@pytest.mark.asyncio
@settings(deadline=None)
@given(mock_params())
async def test_weights(params):
    vocab, next_token_ws, context = params
    potential = MockPotential(vocab, np.log(next_token_ws))
    subtoken_potential = SubtokenPotential(potential)

    try:
        logws_token = await potential.logw_next(context)
        for token, want in logws_token.items():
            have = await subtoken_potential.complete(token, context)
            assert np.isclose(have, want, atol=1e-5, rtol=1e-5), token

        marginal = {(): float("-inf")}
        for token, logw in logws_token.items():
            if token is EOS:
                marginal[()] = logsumexp([marginal[()], logw])
                marginal[tuple(token)] = logw
                continue

            for i in range(0, len(token) + 1):
                subtoken = tuple(token[:i])
                if subtoken not in marginal:
                    marginal[subtoken] = float("-inf")
                marginal[subtoken] = logsumexp([marginal[subtoken], logw])

        for subtoken, want in marginal.items():
            have = await subtoken_potential.prefix(subtoken, context)
            assert np.isclose(have, want, atol=1e-5, rtol=1e-5), subtoken
    finally:
        await subtoken_potential.cleanup()


@st.composite
def mock_params_with_subtoken_ctx(draw):
    vocab, next_token_ws, context = draw(mock_params())
    token = draw(st.sampled_from(vocab))
    prefix_idx = draw(st.integers(0, len(token)))
    subtoken_ctx = token[:prefix_idx]
    return vocab, next_token_ws, context, subtoken_ctx


@pytest.mark.asyncio
@settings(deadline=None, max_examples=1)
@given(mock_params_with_subtoken_ctx())
async def test_properties(params):
    vocab, next_token_ws, context, subtoken_ctx = params
    potential = MockPotential(vocab, np.log(next_token_ws))
    subtoken_potential = SubtokenPotential(potential)

    try:
        await subtoken_potential.assert_autoreg_fact(
            subtoken_ctx, method_args=(context,)
        )
        await subtoken_potential.assert_logw_next_consistency(
            subtoken_ctx, method_args=(context,)
        )
    finally:
        await subtoken_potential.cleanup()
