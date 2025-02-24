import pytest
from genlm_control import InferenceEngine
from genlm_control.potential import PromptedLLM, BoolFSA
from genlm_control.sampler import direct_token_sampler, eager_token_sampler


@pytest.fixture(scope="module")
def llm():
    return PromptedLLM.from_name("gpt2", backend="hf", temperature=0.5)


@pytest.fixture(scope="module")
def sentence_wfsa():
    return BoolFSA.from_regex(r".*\.$")


async def assert_engine_run(engine, n_particles, max_tokens, ess_threshold):
    sequences = await engine(
        n_particles=n_particles, ess_threshold=ess_threshold, max_tokens=max_tokens
    )

    assert len(sequences) == n_particles
    assert all(len(seq) <= max_tokens for seq in sequences)

    print(sequences)

    return sequences


@pytest.mark.asyncio
async def test_with_llm(llm):
    mtl_llm = llm.spawn_new_eos([b"."])
    mtl_llm.set_prompt_from_str("Montreal is")

    sampler = direct_token_sampler(mtl_llm)
    engine = InferenceEngine(sampler)

    sequences = await assert_engine_run(
        engine, n_particles=10, max_tokens=25, ess_threshold=0.5
    )

    assert all(b"." not in seq for seq, _ in sequences)

    await engine.cleanup()


@pytest.mark.asyncio
async def test_with_product_llm(llm):
    mtl_llm = llm.spawn_new_eos([b"."])
    mtl_llm.set_prompt_from_str("Montreal is")

    nyc_llm = mtl_llm.spawn()
    nyc_llm.set_prompt_from_str("NYC is")

    sampler = direct_token_sampler(mtl_llm * nyc_llm)
    engine = InferenceEngine(sampler)

    await assert_engine_run(engine, n_particles=10, max_tokens=25, ess_threshold=0.5)

    await engine.cleanup()


@pytest.mark.asyncio
async def test_with_llm_and_critic(llm):
    mtl_llm = llm.spawn_new_eos([b"."])
    mtl_llm.set_prompt_from_str("Montreal is")

    nyc_llm = mtl_llm.spawn()
    nyc_llm.set_prompt_from_str("NYC is")

    sampler = direct_token_sampler(mtl_llm)
    engine = InferenceEngine(sampler, critic=nyc_llm)

    await assert_engine_run(engine, n_particles=10, max_tokens=25, ess_threshold=0.5)

    await engine.cleanup()


@pytest.mark.asyncio
async def test_with_llm_and_fsa(llm):
    mtl_llm = llm.spawn_new_eos([b"."])
    mtl_llm.set_prompt_from_str("Montreal is")

    sampler = direct_token_sampler(mtl_llm)

    best_fsa = BoolFSA.from_regex(r"\sthe\s(best|greatest).+").coerce(
        mtl_llm, f=b"".join
    )

    engine = InferenceEngine(sampler, critic=best_fsa)
    await assert_engine_run(engine, n_particles=10, max_tokens=25, ess_threshold=0.5)

    nyc_llm = mtl_llm.spawn()
    nyc_llm.set_prompt_from_str("NYC is")
    engine = InferenceEngine(sampler, critic=best_fsa * nyc_llm)
    await assert_engine_run(engine, n_particles=10, max_tokens=25, ess_threshold=0.5)

    await engine.cleanup()


@pytest.mark.asyncio
async def test_with_llm_and_fsa_eager_sampler(llm):
    mtl_llm = llm.spawn_new_eos([b"."])
    mtl_llm.set_prompt_from_str("Montreal is")

    best_fsa = BoolFSA.from_regex(r"\sthe\s(best|greatest).+")

    sampler = eager_token_sampler(mtl_llm, best_fsa)
    engine = InferenceEngine(sampler)

    await assert_engine_run(engine, n_particles=10, max_tokens=25, ess_threshold=0.5)

    nyc_llm = mtl_llm.spawn()
    nyc_llm.set_prompt_from_str("NYC is")

    engine = InferenceEngine(sampler, critic=nyc_llm)

    await assert_engine_run(engine, n_particles=10, max_tokens=25, ess_threshold=0.5)

    await engine.cleanup()
