import pytest
import numpy as np
from genlm.grammar import CFG, Float, Boolean
from genlm.control.potential.built_in import WCFG, BoolCFG


@pytest.fixture
def byte_wcfg():
    c = CFG(Float, S="S", V={b"a"[0], b"b"[0]})
    c.add(3.0, "S", "A", "B")
    c.add(2.0, "S", "A", "B", "B")
    c.add(1.0, "A", b"a"[0])
    c.add(1.0, "B", b"b"[0])
    return c


def test_wcfg_init_wrong_semiring():
    # Test initialization with non-Float semiring
    c = CFG(Boolean, S="S", V={b"a"[0], b"b"[0]})
    with pytest.raises(ValueError):
        WCFG(c)


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

    # Test empty string handling
    log_weight = await pot.prefix(b"")
    assert log_weight == 0

    log_weight = await pot.prefix(b"ab")
    assert log_weight == 0

    log_weight = await pot.complete(b"abb")
    assert log_weight == 0

    log_weight = await pot.prefix(b"a")
    assert log_weight == 0


@pytest.mark.asyncio
async def test_properties(byte_wcfg):
    pot = WCFG(byte_wcfg)

    await pot.assert_logw_next_consistency(b"ab")
    await pot.assert_autoreg_fact(b"ab")
    await pot.assert_batch_consistency([b"a", b"ab"])

    await pot.assert_logw_next_consistency(b"")
    await pot.assert_autoreg_fact(b"")
    await pot.assert_batch_consistency([b""])

    pot = BoolCFG(byte_wcfg)

    await pot.assert_logw_next_consistency(b"ab")
    await pot.assert_autoreg_fact(b"ab")
    await pot.assert_batch_consistency([b"a", b"ab"])

    await pot.assert_logw_next_consistency(b"")
    await pot.assert_autoreg_fact(b"")
    await pot.assert_batch_consistency([b""])


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

    pot_spawned = pot.spawn()
    assert pot_spawned.cfg == pot.cfg


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

    pot_spawned = pot.spawn()
    assert pot_spawned.cfg == pot.cfg


def test_wcfg_repr():
    c = CFG(Float, S="S", V={b"a"[0]})
    pot = WCFG(c)
    repr(pot)
    pot._repr_html_()


def test_bcfg_repr():
    c = CFG(Boolean, S="S", V={b"a"[0]})
    pot = BoolCFG(c)
    repr(pot)
    pot._repr_html_()


def test_wcfg_clear_cache():
    c = CFG(Float, S="S", V={b"a"[0]})
    pot = WCFG(c)
    pot.clear_cache()


def test_bcfg_clear_cache():
    c = CFG(Boolean, S="S", V={b"a"[0]})
    pot = BoolCFG(c)
    pot.clear_cache()
