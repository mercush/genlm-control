import pytest
import numpy as np
from arsenal.maths import logsumexp
from hypothesis import given, strategies as st, settings

from genlm_control.potential.built_in import PromptedLLM


@pytest.fixture(
    scope="module",
    params=[
        # pytest.param(
        #    ("vllm", {"engine_opts": {"dtype": "float"}}),
        #    marks=pytest.mark.skipif(
        #        not torch.cuda.is_available(), reason="vllm requires CUDA"
        #    ),
        # ),
        ("hf", {"hf_opts": {"torch_dtype": "float"}}),
    ],
)
def llm_config(request):
    return request.param


@pytest.fixture(scope="module")
def llm(llm_config):
    backend, opts = llm_config
    return PromptedLLM.from_name("gpt2", backend=backend, **opts)


@pytest.mark.asyncio
@given(st.text(min_size=1))
async def test_prompt_setting(llm, pre_prompt):
    pre_prompt_ids = llm.model.tokenizer.encode(pre_prompt)

    # Test ids setter
    llm.prompt_ids = pre_prompt_ids
    assert llm.prompt_ids == pre_prompt_ids
    assert b"".join(llm.prompt).decode() == pre_prompt

    # Test str setter
    llm.set_prompt_from_str(pre_prompt)
    assert b"".join(llm.prompt).decode() == pre_prompt
    assert llm.prompt_ids == pre_prompt_ids


@pytest.mark.asyncio
@settings(deadline=None)
@given(st.text(min_size=1), st.text(min_size=1))
async def test_scoring(llm, pre_prompt, context_str):
    pre_prompt_ids = llm.model.tokenizer.encode(pre_prompt)
    context = llm.tokenize(context_str)

    llm.prompt_ids = pre_prompt_ids

    have = await llm.prefix(context)
    want = await reference_scorer(llm, context)
    assert np.isclose(have, want), [have, want]

    have = await llm.complete(context)
    want = await reference_scorer(llm, context, eos=True)
    assert np.isclose(have, want), [have, want]


@pytest.mark.asyncio
@settings(deadline=None)
@given(
    st.text(min_size=1), st.text(min_size=1), st.lists(st.text(min_size=1), min_size=1)
)
async def test_properties(llm, pre_prompt, context, contexts):
    pre_prompt_ids = llm.model.tokenizer.encode(pre_prompt)
    context = llm.tokenize(context)

    llm.prompt_ids = pre_prompt_ids

    await llm.assert_logw_next_consistency(context, top=10, rtol=1e-3, atol=1e-3)
    await llm.assert_autoreg_fact(context, rtol=1e-3, atol=1e-3)

    contexts = [llm.tokenize(context) for context in contexts]
    await llm.assert_batch_consistency(contexts, rtol=1e-3, atol=1e-3)


@st.composite
def eos_test_params(draw):
    # Probably can decrase the size of these ranges for faster tests.
    eos_token_ids = draw(
        st.lists(
            st.integers(min_value=0, max_value=50256),
            min_size=1,
            max_size=3,
            unique=True,
        )
    )
    valid_ids = st.integers(min_value=0, max_value=50256).filter(
        lambda x: x not in eos_token_ids
    )
    context_ids = draw(st.lists(valid_ids, min_size=1, max_size=5))
    prompt_ids = draw(
        st.lists(st.integers(min_value=0, max_value=50256), min_size=1, max_size=5)
    )
    return eos_token_ids, context_ids, prompt_ids


@pytest.mark.asyncio
@settings(deadline=None)
@given(eos_test_params())
async def test_eos_tokens(llm, params):
    eos_token_ids, context_ids, prompt_ids = params
    llm.prompt_ids = prompt_ids
    eos_tokens = [llm.token_maps.decode[x] for x in eos_token_ids]
    new_llm = llm.spawn_new_eos(eos_tokens=eos_tokens)

    assert new_llm.prompt_ids == prompt_ids  # check prompt_ids is not changed
    assert new_llm.token_maps.eos_idxs == eos_token_ids
    assert set(new_llm.token_maps.decode) - set(eos_tokens) == set(new_llm.decode)

    context = new_llm.decode_tokens(context_ids)
    have = await new_llm.complete(context)
    want = await reference_scorer(new_llm, context, eos=True)
    assert np.isclose(have, want), [have, want]


async def reference_scorer(llm, context, eos=False):
    """Compute the log probability of the context given the prompt."""
    context_ids = llm.encode_tokens(context)

    logps = await llm.model.next_token_logprobs(llm.prompt_ids)
    total_logp = logps[context_ids[0]].item()

    for i in range(1, len(context_ids)):
        logps = await llm.model.next_token_logprobs(llm.prompt_ids + context_ids[:i])
        total_logp += logps[context_ids[i]].item()

    if eos:
        logps = await llm.model.next_token_logprobs(llm.prompt_ids + context_ids)
        eos_logp = float("-inf")
        for i in llm.token_maps.eos_idxs:
            eos_logp = logsumexp([eos_logp, logps[i].item()])
        total_logp += eos_logp

    return total_logp
