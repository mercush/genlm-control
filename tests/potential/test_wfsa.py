import pytest
import numpy as np
from genlm_grammar import WFSA as BaseWFSA, Float
from genlm_control.potential.built_in import WFSA, BoolFSA


@pytest.fixture
def float_wfsa():
    """Creates a simple WFSA in float semiring"""
    m = BaseWFSA(Float)
    m.add_I(0, 1.0)
    m.add_arc(0, b"a"[0], 1, 2)
    m.add_arc(1, b"b"[0], 2, 1)
    m.add_arc(1, b"c"[0], 2, 1)
    m.add_arc(1, b"d"[0], 3, 1)  # dead end
    m.add_F(2, 1.0)
    return m


@pytest.mark.asyncio
async def test_wfsa(float_wfsa):
    pot = WFSA(float_wfsa)

    log_weight = await pot.complete(b"ab")
    assert np.isclose(log_weight, np.log(2))

    log_weight = await pot.complete(b"ac")
    assert np.isclose(log_weight, np.log(2))

    log_weight = await pot.complete(b"a")
    assert log_weight == float("-inf")

    log_weight = await pot.prefix(b"a")
    assert np.isclose(log_weight, np.log(4))

    log_weight = await pot.prefix(b"c")
    assert log_weight == float("-inf")

    log_weight = await pot.prefix(b"ab")
    assert np.isclose(log_weight, np.log(2))

    await pot.assert_logw_next_consistency(b"a")
    await pot.assert_autoreg_fact(b"a")

    await pot.assert_logw_next_consistency(b"")
    await pot.assert_autoreg_fact(b"")


@pytest.mark.asyncio
async def test_wfsa_regex():
    pot = WFSA.from_regex("a(b|c)")

    log_weight = await pot.complete(b"ab")
    assert np.isclose(log_weight, np.log(0.5))

    log_weight = await pot.complete(b"ac")
    assert np.isclose(log_weight, np.log(0.5))

    log_weight = await pot.complete(b"a")
    assert log_weight == float("-inf")

    log_weight = await pot.prefix(b"a")
    assert np.isclose(log_weight, 0)

    log_weight = await pot.prefix(b"c")
    assert log_weight == float("-inf")

    log_weight = await pot.prefix(b"ab")
    assert np.isclose(log_weight, np.log(0.5))

    await pot.assert_logw_next_consistency(b"a")
    await pot.assert_autoreg_fact(b"a")

    await pot.assert_logw_next_consistency(b"")
    await pot.assert_autoreg_fact(b"")


@pytest.mark.asyncio
async def test_bool_fsa(float_wfsa):
    pot = BoolFSA(float_wfsa)

    log_weight = await pot.complete(b"ab")
    assert log_weight == 0

    log_weight = await pot.complete(b"ac")
    assert log_weight == 0

    log_weight = await pot.complete(b"a")
    assert log_weight == float("-inf")

    log_weight = await pot.prefix(b"a")
    assert log_weight == 0

    log_weight = await pot.prefix(b"c")
    assert log_weight == float("-inf")

    log_weight = await pot.prefix(b"ab")
    assert log_weight == 0

    await pot.assert_logw_next_consistency(b"a")
    await pot.assert_autoreg_fact(b"a")

    await pot.assert_logw_next_consistency(b"")
    await pot.assert_autoreg_fact(b"")
