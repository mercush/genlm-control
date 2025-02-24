import pytest
import numpy as np
from genlm_control.potential import Product, Potential
from genlm_control.typing import Atomic


class SimplePotential(Potential):
    def __init__(self, vocabulary, scale=1.0):
        super().__init__(vocabulary)
        self.scale = scale

    async def complete(self, context):
        return -float(len(context)) * self.scale

    async def prefix(self, context):
        return -0.5 * float(len(context)) * self.scale


@pytest.fixture
def vocab():
    return [b"a", b"b", b"c"]


@pytest.fixture
def p1(vocab):
    return SimplePotential(vocab, scale=1.0)


@pytest.fixture
def p2(vocab):
    return SimplePotential(vocab, scale=2.0)


@pytest.fixture
def product(p1, p2):
    return Product(p1, p2)


def test_initialization(vocab, p1, p2):
    # Test successful initialization
    product = Product(p1, p2)
    assert product.token_type == Atomic(bytes)
    assert len(product.decode) == len(vocab)
    assert set(product.decode) == set(vocab)

    # Test mismatched token types
    class DifferentPotential(SimplePotential):
        def __init__(self):
            super().__init__([1, 2, 3])  # Different token type (int)

    with pytest.raises(
        ValueError, match="Potentials in product must have the same token type"
    ):
        Product(p1, DifferentPotential())

    # Test non-overlapping vocabularies
    p3 = SimplePotential([b"d", b"e", b"f"])
    with pytest.raises(
        ValueError, match="Potentials in product must share a common vocabulary"
    ):
        Product(p1, p3)


@pytest.mark.asyncio
async def test_prefix(product):
    context = [b"a", b"b"]
    result = await product.prefix(context)
    # Should be sum of both potentials' prefix values
    expected = -0.5 * len(context) * (1.0 + 2.0)
    assert result == expected


@pytest.mark.asyncio
async def test_complete(product):
    context = [b"a", b"b"]
    result = await product.complete(context)
    # Should be sum of both potentials' complete values
    expected = -len(context) * (1.0 + 2.0)
    assert result == expected


@pytest.mark.asyncio
async def test_logw_next(product):
    context = [b"a", b"b"]
    result = await product.logw_next(context)

    # Test that weights are properly combined
    weights = result.weights
    assert len(weights) == len(product.decode_eos)

    # Test individual token weights
    for token in product.decode:
        extended = context + [token]
        score = await product.score(extended)
        prefix_score = await product.prefix(context)
        expected_weight = score - prefix_score
        assert np.isclose(result.weights[product.encode_eos[token]], expected_weight)


@pytest.mark.asyncio
async def test_batch_operations(product):
    contexts = [[b"a"], [b"a", b"b"]]

    # Test batch_complete
    complete_results = await product.batch_complete(contexts)
    expected = [-3.0, -6.0]  # Combined scales (1.0 + 2.0) * -len(context)
    np.testing.assert_array_almost_equal(complete_results, expected)

    # Test batch_prefix
    prefix_results = await product.batch_prefix(contexts)
    expected = [-1.5, -3.0]  # Combined scales (1.0 + 2.0) * -0.5 * len(context)
    np.testing.assert_array_almost_equal(prefix_results, expected)


@pytest.mark.asyncio
async def test_properties(product):
    # Test the inherited property checks
    await product.assert_logw_next_consistency([b"b", b"c"], verbosity=1)
    await product.assert_autoreg_fact([b"b", b"c"], verbosity=1)
    await product.assert_batch_consistency([[b"b", b"c"], [b"a"]], verbosity=1)
