import re
import pytest
import numpy as np
from genlm_grammar import WFSA as BaseWFSA, Float
from genlm_control.potential.built_in import WFSA, BoolFSA
from hypothesis import strategies as st, given


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

    await pot.assert_batch_consistency([b"", b"ab", b"ac"])


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

    await pot.assert_batch_consistency([b"", b"ab", b"ac"])


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

    await pot.assert_batch_consistency([b"", b"ab", b"ac"])


@st.composite
def regex_pattern(draw, max_depth=3):
    """Composite strategy to generate nested regex patterns"""

    def pattern_strategy(depth):
        if depth <= 0:
            # Base case: single escaped character
            char = draw(st.characters(blacklist_categories=("Cs",)))
            return re.escape(char)

        # Choose which type of pattern to generate
        pattern_type = draw(
            st.sampled_from(
                [
                    "simple",
                    "alternation",
                    "concatenation",
                    "optional",
                    "kleene",
                    "plus",
                    "quantified",
                ]
            )
        )

        if pattern_type == "simple":
            return pattern_strategy(0)

        # Generate sub-pattern(s)
        if pattern_type in ("alternation", "concatenation"):
            num_patterns = draw(st.integers(min_value=2, max_value=3))
            patterns = [pattern_strategy(depth - 1) for _ in range(num_patterns)]

            if pattern_type == "alternation":
                return f"({'|'.join(patterns)})"
            else:  # concatenation
                return f"({''.join(patterns)})"

        # Single sub-pattern with operator
        sub_pattern = pattern_strategy(depth - 1)
        if pattern_type == "optional":
            return f"({sub_pattern})?"
        elif pattern_type == "kleene":
            return f"({sub_pattern})*"
        elif pattern_type == "plus":
            return f"({sub_pattern})+"
        else:  # quantified
            quantifier = draw(st.sampled_from(["+", "*", "?", "{1,3}"]))
            return f"({sub_pattern}){quantifier}"

    return pattern_strategy(max_depth)


@pytest.mark.asyncio
@given(regex_pattern(max_depth=3), st.data())
async def test_bool_fsa_with_generated_regex(pattern, data):
    """Test that BoolFSA accepts strings that match its regex pattern"""
    pot = BoolFSA.from_regex(pattern)

    matching_str = data.draw(st.from_regex(pattern, fullmatch=True))
    byte_string = matching_str.encode("utf-8")

    log_weight = await pot.complete(byte_string)
    assert log_weight == 0, [matching_str, pattern]

    for prefix in range(len(byte_string)):
        log_weight = await pot.prefix(byte_string[:prefix])
        assert log_weight == 0, [matching_str, byte_string[:prefix]]
