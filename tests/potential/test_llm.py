import pytest
import torch
import numpy as np

from genlm_control.potential.built_in import PromptedLLM

backends = [
    pytest.param(
        ("vllm", {"engine_opts": {"dtype": "float"}}),
        marks=pytest.mark.skipif(
            not torch.cuda.is_available(), reason="vllm requires CUDA"
        ),
    ),
    ("hf", {"hf_opts": {"torch_dtype": "float"}}),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", backends)
async def test_backend(backend):
    backend, opts = backend
    llm = PromptedLLM.from_name("gpt2", backend=backend, **opts)

    pre_prompt = "hello"
    llm.prompt = pre_prompt
    assert llm.prompt_ids == llm.model.tokenizer.encode(pre_prompt)
    assert b"".join(llm.prompt).decode() == pre_prompt

    context = llm.tokenize(" world, it's me")
    have = await llm.prefix(context)
    want = await reference_scorer(llm, context)
    assert np.isclose(have, want), [have, want]

    have = await llm.complete(context)
    want = await reference_scorer(llm, context, eos=True)
    assert np.isclose(have, want), [have, want]

    await llm.assert_properties(context, top=10, verbosity=1)


async def reference_scorer(llm, context, eos=False):
    """Compute the log probability of the context given the prompt."""
    context_ids = llm.encode_tokens(context)

    logps = await llm.model.next_token_logprobs(llm.prompt_ids + context_ids[:0])
    total_logp = logps[context_ids[0]].item()

    for i in range(1, len(context_ids)):
        logps = await llm.model.next_token_logprobs(llm.prompt_ids + context_ids[:i])
        total_logp += logps[context_ids[i]].item()

    if eos:
        logps = await llm.model.next_token_logprobs(llm.prompt_ids + context_ids)
        for i in llm.token_maps.eos_idxs:
            total_logp += logps[i].item()

    return total_logp
