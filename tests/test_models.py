import pytest
import asyncio
from unittest.mock import Mock
from genlm_control.potential import Potential
from genlm_control.models import SequenceSampler, BatchSubModel, Proposal, Twist
from genlm_control.constant import EOS

TOKEN_WEIGHT = -0.5
CRITIC_SCORE = 10
PREFIX_WEIGHT_GEN = 2
PREFIX_WEIGHT_CRITIC = 3


class MockPotential(Potential):
    def __init__(self):
        super().__init__(list(range(256)))

    async def prefix(self, context):
        return PREFIX_WEIGHT_GEN

    async def complete(self, context):
        return 0


class MockCritic(Potential):
    def __init__(self):
        super().__init__(list(range(256)))

    async def prefix(self, context):
        return PREFIX_WEIGHT_CRITIC

    async def complete(self, context):
        return CRITIC_SCORE

    async def batch_score(self, contexts):
        return [CRITIC_SCORE] * len(contexts)


class MockGenerator:
    target = MockPotential()

    async def batch_sample_token(self, contexts):
        # Returns (tokens, log_weights, log probs [unused here])
        return [1] * len(contexts), [TOKEN_WEIGHT] * len(contexts), None


@pytest.fixture
def generator():
    return MockGenerator()


@pytest.fixture
def critic():
    return MockCritic()


@pytest.mark.asyncio
async def test_sequence_sampler_basic():
    generator = MockGenerator()
    sampler = SequenceSampler(max_tokens=3, generator=generator)

    await sampler.start()

    # Test initial state
    assert sampler.max_tokens == 3
    assert sampler.context == []
    assert sampler.twister is None

    # Test one step
    await sampler.step()
    assert len(sampler.context) == 1
    assert sampler.max_tokens == 2
    assert sampler.weight == TOKEN_WEIGHT + PREFIX_WEIGHT_GEN

    # Test next step
    await sampler.step()
    assert len(sampler.context) == 2
    assert sampler.max_tokens == 1
    assert sampler.weight == 2 * TOKEN_WEIGHT + PREFIX_WEIGHT_GEN

    await sampler.cleanup()


@pytest.mark.asyncio
async def test_sequence_sampler_with_critic():
    generator = MockGenerator()
    critic = MockCritic()
    sampler = SequenceSampler(max_tokens=3, generator=generator, critic=critic)

    # Test that twist is initialized
    assert sampler.twister is not None

    await sampler.start()

    # Test one step
    await sampler.step()
    assert len(sampler.context) == 1
    assert (
        sampler.weight
        == TOKEN_WEIGHT + CRITIC_SCORE + PREFIX_WEIGHT_GEN + PREFIX_WEIGHT_CRITIC
    )

    # Test next step
    await sampler.step()
    assert len(sampler.context) == 2
    assert (
        sampler.weight
        == 2 * TOKEN_WEIGHT + CRITIC_SCORE + PREFIX_WEIGHT_GEN + PREFIX_WEIGHT_CRITIC
    )

    await sampler.cleanup()


@pytest.mark.asyncio
async def test_sequence_sampler_eos():
    generator = MockGenerator()

    # Create a generator that returns EOS token
    async def mock_batch_sample_token(*args):
        return [EOS], [-0.5], None

    generator.batch_sample_token = mock_batch_sample_token
    sampler = SequenceSampler(max_tokens=3, generator=generator)

    await sampler.step()
    assert sampler.context[-1] is EOS
    assert sampler.finished

    await sampler.cleanup()


@pytest.mark.asyncio
async def test_batch_sub_model():
    batch_size = None

    class TestBatchSubModel(BatchSubModel):
        async def batch_forward(self, models):
            nonlocal batch_size
            batch_size = len(models)
            for model in models:
                model.processed = True

    batch_model = TestBatchSubModel()

    # Create mock models
    model1 = Mock()
    model2 = Mock()

    # Process models
    await asyncio.gather(batch_model(model1), batch_model(model2))

    assert model1.processed
    assert model2.processed
    assert batch_size == 2

    await batch_model.cleanup()


@pytest.mark.asyncio
async def test_proposal():
    generator = MockGenerator()
    proposal = Proposal(generator)

    # Create mock models
    model1 = Mock()
    model1.context = []
    model2 = Mock()
    model2.context = []

    # Process models
    await asyncio.gather(proposal(model1), proposal(model2))

    assert len(model1.context) == 1
    assert len(model2.context) == 1

    assert model1.score.call_count == 1
    assert model2.score.call_count == 1

    await proposal.cleanup()


@pytest.mark.asyncio
async def test_twister():
    critic = MockCritic()
    twist = Twist(critic)

    # Create mock models
    model1 = Mock()
    model1.context = []
    model2 = Mock()
    model2.context = []

    # Process models
    await asyncio.gather(twist(model1), twist(model2))

    model1.twist.assert_called_once()
    model2.twist.assert_called_once()

    await twist.cleanup()
