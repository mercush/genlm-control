import pytest
from genlm_control.potential.base import Potential
from genlm_control.potential.lifted import Lifted
from genlm_control.product import Product
from genlm_control.potential.autobatch import AutoBatchedPotential
from genlm_control.potential.mp import MPPotential


class SimplePotential(Potential):
    """A simple potential for testing operators."""

    def __init__(self, vocabulary):
        super().__init__(vocabulary)

    async def complete(self, context):
        return 0

    async def prefix(self, context):
        return 0

    def spawn(self):
        return SimplePotential(self.decode)


@pytest.fixture
def vocab():
    return [b"a"[0], b"b"[0], b"c"[0]]


@pytest.fixture
def p1(vocab):
    return SimplePotential(vocab)


@pytest.fixture
def p2(vocab):
    return SimplePotential(vocab)


@pytest.mark.asyncio
async def test_product_operator(p1, p2):
    have = p1 * p2
    want = Product(p1, p2)
    assert have.p1 == want.p1
    assert have.p2 == want.p2
    assert have.decode == want.decode


@pytest.mark.asyncio
async def test_lift_operator(p1):
    target_vocab = [b"aa", b"bb", b"cc"]

    # Test with default transformations
    def f(seq):
        return [x for xs in seq for x in xs]

    def g(x):
        return bytes(x)

    lifted = p1.lift(SimplePotential(target_vocab), f=f, g=g)
    assert set(lifted.decode) == set(target_vocab)

    # Test with custom transformations
    def f(seq):
        return [xs[0] for xs in seq]

    def g(x):
        return bytes(x)

    have = p1.lift(SimplePotential(target_vocab), f=f, g=g)
    want = Lifted(p1, target_vocab, f=f, g=g)
    assert have.potential == want.potential
    assert have.decode == want.decode


@pytest.mark.asyncio
async def test_to_autobatched(p1):
    have = p1.to_autobatched()
    want = AutoBatchedPotential(p1)
    assert have.potential == want.potential

    await have.cleanup()
    await want.cleanup()


@pytest.mark.asyncio
async def test_to_multiprocess(p1):
    num_workers = 2
    have = p1.to_multiprocess(num_workers=num_workers)
    want = MPPotential(p1.spawn, (), num_workers=num_workers)
    assert have.decode == want.decode


@pytest.mark.asyncio
async def test_operator_chaining(p1, p2):
    have = (p1 * p2).to_autobatched()
    want = AutoBatchedPotential(Product(p1, p2))
    assert have.potential.p1 == want.potential.p1
    assert have.potential.p2 == want.potential.p2
    assert have.decode == want.decode

    await have.cleanup()
    await want.cleanup()

    V = [b"aa", b"bb", b"cc"]

    def f(seq):
        return [x for xs in seq for x in xs]

    def g(x):
        return bytes(x)

    have = (p1 * p2).lift(SimplePotential(V), f=f, g=g)
    want = Lifted(Product(p1, p2), V, f=f, g=g)
    assert have.potential.p1 == want.potential.p1
    assert have.potential.p2 == want.potential.p2
    assert have.decode == want.decode
