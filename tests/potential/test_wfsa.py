import pytest
import numpy as np
from genlm_grammar import WFSA as BaseWFSA, Float
from genlm_control.potential.base import EOS
from genlm_control.potential.built_in import WFSA, BoolFSA

@pytest.fixture
def float_wfsa():
    """Creates a simple WFSA in float semiring"""
    m = BaseWFSA(Float)
    m.add_I(0, 1.0)
    m.add_arc(0, b'a'[0], 1, 1.0)
    m.add_arc(1, b'b'[0], 2, 0.5)
    m.add_arc(1, b'c'[0], 2, 0.5) 
    m.add_arc(1, b'd'[0], 3, 0.5) # dead end
    m.add_F(2, 1.0)
    return m

@pytest.mark.asyncio
async def test_wfsa_complete(float_wfsa):
    pot_F = WFSA(float_wfsa)
    pot_regex = WFSA.from_regex("a(b|c)")

    import ipdb; ipdb.set_trace()

    await _test_wfsa_complete(pot_F)
    await _test_wfsa_complete(pot_regex)

async def _test_wfsa_complete(pot):
    log_weight = await pot.complete(b'ab')
    assert np.isclose(log_weight, np.log(0.5))

    log_weight = await pot.complete(b'ac')
    assert np.isclose(log_weight, np.log(0.5))

    log_weight = await pot.complete(b'a')
    assert log_weight == float('-inf')

async def test_wfsa_prefix(float_wfsa):    
    pot = WFSA(float_wfsa)
    have = await pot.prefix(b"a")
    want = np.log(0.5)
    assert np.isclose(have, want)

@pytest.mark.asyncio
async def test_properties(float_wfsa):
    pot = WFSA(float_wfsa)
    await pot.assert_properties(b'ab')

    pot = WFSA.from_regex("a(b|c)")
    await pot.assert_properties(b'ab')

@pytest.mark.asyncio
async def test_bool_fsa(float_wfsa):
    pot = BoolFSA(float_wfsa)

    log_weight = await pot.complete(b'ab')
    assert log_weight == 0

    log_weight = await pot.complete(b'ac')
    assert log_weight == 0

    log_weight = await pot.complete(b'a')
    assert log_weight == float('-inf')

    log_weight = await pot.prefix(b'a')
    assert log_weight == 0

    log_weight = await pot.prefix(b'c')
    assert log_weight == float('-inf')

    log_weight = await pot.prefix(b'ab')
    assert log_weight == 0

    await pot.assert_properties(b'a')

# This test is redundant with assert_properties, since if both prefix and complete are correct,
# then passing assert_properties indicates that logp_next is also correct. We leave it in 
# for extra coverage though.
@pytest.mark.asyncio
async def test_wfsa_logp_next(float_wfsa):
    pot = WFSA(float_wfsa)

    inf = float('-inf')
    
    vocabulary = [b'a'[0], b'b'[0], b'c'[0], EOS]
    for input_seq, want in [
        (b'',   [0, inf, inf, inf]),
        (b'a',  [inf, np.log(0.5), np.log(0.5), inf]),
        (b'ab', [inf, inf, inf, 0]),
        (b'ac', [inf, inf, inf, 0])
    ]:
        have = (await pot.logp_next(input_seq)).materialize()
        for i, v in enumerate(vocabulary):
            assert have[v] == want[i], [v, want, have]
