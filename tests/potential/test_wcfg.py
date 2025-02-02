import pytest
import numpy as np
from genlm_grammar import CFG, Float
from genlm_control.potential.built_in import WCFG, BoolCFG
from genlm_control.constant import EOS


@pytest.fixture
def byte_wcfg():
    c = CFG(Float, S="S", V={b"a"[0], b"b"[0]})
    c.add(3.0, "S", "A", "B")
    c.add(2.0, "S", "A", "B", "B")
    c.add(1.0, "A", b"a"[0])
    c.add(1.0, "B", b"b"[0])
    return c


@pytest.mark.asyncio
async def test_wcfg_complete(byte_wcfg):
    pot = WCFG(byte_wcfg)

    log_weight = await pot.complete(b"ab")
    assert np.isclose(log_weight, np.log(3))

    log_weight = await pot.complete(b"abb")
    assert np.isclose(log_weight, np.log(2))

    log_weight = await pot.complete(b"a")
    assert log_weight == float("-inf")


@pytest.mark.asyncio
async def test_wcfg_prefix(byte_wcfg):
    pot = WCFG(byte_wcfg)

    log_weight = await pot.prefix(b"ab")
    assert np.isclose(log_weight, np.log(5))

    log_weight = await pot.complete(b"abb")
    assert np.isclose(log_weight, np.log(2))

    log_weight = await pot.prefix(b"a")
    assert np.isclose(log_weight, np.log(5))


@pytest.mark.asyncio
async def test_bcfg_complete(byte_wcfg):
    pot = BoolCFG(byte_wcfg)

    log_weight = await pot.complete(b"ab")
    assert log_weight == 0

    log_weight = await pot.complete(b"abb")
    assert log_weight == 0

    log_weight = await pot.complete(b"a")
    assert log_weight == float("-inf")


@pytest.mark.asyncio
async def test_bcfg_prefix(byte_wcfg):
    pot = BoolCFG(byte_wcfg)

    log_weight = await pot.prefix(b"ab")
    assert log_weight == 0

    log_weight = await pot.complete(b"abb")
    assert log_weight == 0

    log_weight = await pot.prefix(b"a")
    assert log_weight == 0


@pytest.mark.asyncio
async def test_properties(byte_wcfg):
    pot = WCFG(byte_wcfg)

    extensions = [b"a", b"b", [*b"b", EOS]]

    await pot.assert_logw_next_consistency(b"ab", verbosity=1)
    await pot.assert_autoreg_fact(b"ab", verbosity=1)
    await pot.assert_batch_consistency(
        [b"a", b"ab"], extensions=extensions, verbosity=1
    )

    await pot.assert_logw_next_consistency(b"", verbosity=1)
    await pot.assert_autoreg_fact(b"", verbosity=1)
    await pot.assert_batch_consistency([b""], extensions=extensions, verbosity=1)

    pot = BoolCFG(byte_wcfg)

    await pot.assert_logw_next_consistency(b"ab", verbosity=1)
    await pot.assert_autoreg_fact(b"ab", verbosity=1)
    await pot.assert_batch_consistency(
        [b"a", b"ab"], extensions=extensions, verbosity=1
    )

    await pot.assert_logw_next_consistency(b"", verbosity=1)
    await pot.assert_autoreg_fact(b"", verbosity=1)
    await pot.assert_batch_consistency([b""], extensions=extensions, verbosity=1)


@pytest.mark.asyncio
async def test_wcfg_from_string():
    grammar = """
    3.0: S -> A B
    2.0: S -> A B B
    1.0: A -> a
    1.0: B -> b
    """
    pot = WCFG.from_string(grammar)

    log_weight = await pot.complete(b"ab")
    assert np.isclose(log_weight, np.log(3))

    log_weight = await pot.complete(b"abb")
    assert np.isclose(log_weight, np.log(2))

    log_weight = await pot.complete(b"a")
    assert log_weight == float("-inf")


@pytest.mark.asyncio
async def test_bcfg_from_lark():
    lark_grammar = """
    start: A B | A B B
    A: "a"
    B: "b"
    """
    pot = BoolCFG.from_lark(lark_grammar)

    log_weight = await pot.complete(b"ab")
    assert log_weight == 0

    log_weight = await pot.complete(b"abb")
    assert log_weight == 0

    log_weight = await pot.complete(b"a")
    assert log_weight == float("-inf")


@pytest.mark.asyncio
async def test_sample(byte_wcfg):
    # Mean importance weight provides an unbiased estimate of
    # the normalizing constant of the model.
    pot = WCFG(byte_wcfg)
    contexts, log_ws = await pot.sample(n_samples=100)
    total_weight = pot.model(b"")  # total weight of all sequences
    assert np.isclose(np.mean(np.exp(log_ws)), total_weight)

    pot = BoolCFG(byte_wcfg)
    contexts, log_ws = await pot.sample(n_samples=100)
    total_weight = 2  # two valid sequences
    assert np.isclose(np.mean(np.exp(log_ws)), total_weight)
