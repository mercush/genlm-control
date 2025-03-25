import pytest
from genlm_control.potential.built_in.json import JsonSchema
import json
from typing import Any
from dataclasses import dataclass
from hypothesis import given, strategies as st, assume, example, settings
from hypothesis_jsonschema import from_schema


@pytest.mark.asyncio
async def test_validates_a_list_of_integers():
    potential = JsonSchema({"type": "array", "items": {"type": "integer"}})

    assert await potential.prefix(b"[1,2,3") == 0.0
    assert await potential.prefix(b'["hello world"') == -float("inf")
    assert await potential.prefix(b"{") == -float("inf")


@pytest.mark.asyncio
async def test_rejects_as_prefix_when_no_valid_continuation():
    potential = JsonSchema({"type": "object"})

    assert await potential.prefix(b"}") == -float("inf")


@pytest.mark.asyncio
async def test_whitespace_is_valid_prefix_and_invalid_complete():
    potential = JsonSchema({"type": "object"})

    assert await potential.prefix(b"\t") == 0.0
    assert await potential.complete(b"\t") == -float("inf")


@pytest.mark.asyncio
@pytest.mark.parametrize("schema", [{"type": "array", "items": {"type": "integer"}}])
@pytest.mark.parametrize(
    "context",
    [b"[1,2,3", json.dumps(list(range(20))).encode("utf-8")],
)
async def test_consistency_properties(schema, context):
    potential = JsonSchema(schema)
    await potential.assert_autoreg_fact(context)


@pytest.mark.asyncio
async def test_will_error_on_impossible_unicode_prefixes():
    potential = JsonSchema({"type": "object"})
    assert await potential.prefix([190] * 5) == -float("inf")


@st.composite
def json_schema(draw):
    type = draw(
        st.sampled_from(
            [
                "null",
                "boolean",
                "integer",
                "number",
                "string",
                "object",
                "array",
            ]
        )
    )

    # TODO: Add some bounds in for some of these?
    if type in ("null", "boolean", "integer", "number", "string"):
        return {"type": type}

    if type == "object":
        result = {"type": "object"}
        result["properties"] = draw(
            st.dictionaries(
                st.from_regex("[A-Za-z0-9_]+"),
                json_schema(),
            )
        )
        if result["properties"]:
            result["required"] = draw(
                st.lists(st.sampled_from(sorted(result["properties"])), unique=True)
            )
        result["additionalProperties"] = draw(st.booleans())
        return result

    assert type == "array"
    result = {"type": "array", "items": draw(json_schema())}
    min_contains = draw(st.integers(0, 10))
    if min_contains > 0:
        result["minContains"] = min_contains
    if draw(st.booleans()):
        max_contains = draw(st.integers(min_contains, 20))
        result["maxContains"] = max_contains
    return result


@dataclass(frozen=True)
class JSONSChemaPotentialProblem:
    schema: Any
    document: bytes
    prefix: bytes


@st.composite
def json_schema_potential_problem(draw):
    schema = draw(json_schema())
    value = draw(from_schema(schema))
    text = json.dumps(
        value,
        # Inverted so that this shrinks to True, as ascii-only
        # JSON is simpler.
        ensure_ascii=not draw(st.booleans()),
        # Similarly inverted so as to shrink to True, on the
        # theory that this means that if keys are out of
        # order in a shrunk example then it really matters.
        sort_keys=not draw(st.booleans()),
        indent=draw(st.one_of(st.none(), st.integers(0, 4), st.text(alphabet=" \t"))),
    )

    document = text.encode("utf-8")
    assert document
    assume(len(document) > 1)
    i = draw(st.integers(1, len(document) - 1))
    prefix = document[:i]
    assume(prefix.strip())

    return JSONSChemaPotentialProblem(schema=schema, document=document, prefix=prefix)


@pytest.mark.asyncio
@example(
    JSONSChemaPotentialProblem(
        schema={"type": "string"},
        document=b'"0\xc2\x80\xc2\x80"',
        prefix=b'"0\xc2\x80\xc2',
    )
)
@example(
    JSONSChemaPotentialProblem(
        schema={
            "type": "string",
        },
        document=b'"000000000\\u001f\xc2\x80\xc2\x80"',
        prefix=b'"000000000\\u001f\xc2\x80\xc2\x80',
    ),
)
@example(
    JSONSChemaPotentialProblem(
        schema={
            "type": "string",
        },
        document=b'"000\\u001f\xc2\x80\xc2\x80\xc2\x80"',
        prefix=b'"000\\u001f\xc2\x80\xc2\x80\xc2',
    ),
)
@given(json_schema_potential_problem())
@settings(max_examples=200)
async def test_always_returns_correctly_on_valid_documents(problem):
    potential = JsonSchema(problem.schema)

    assert await potential.prefix(problem.prefix) == 0.0
    assert await potential.prefix(problem.document) == 0.0
    if await potential.complete(problem.prefix) > -float("inf"):
        # This can sometimes happen because e.g. numeric literals can have
        # a prefix that is also a valid JSON value. We check here that the
        # prefix is actually valid JSON and if so allow it.
        json.loads(problem.prefix)
    assert await potential.complete(problem.document) == 0.0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "format",
    [
        "ipv4",
        "date-time",
        "date",
        "date-time",
        # duration not present in Draft 7 which we're currently using.
        # "duration",
        "email",
        "hostname",
        "idn-hostname",
        "ipv4",
        "ipv6",
        "json-pointer",
        "relative-json-pointer",
        "time",
        "uri",
        "uri-reference",
    ],
)
async def test_validates_formats(format):
    potential = JsonSchema({"format": format, "type": "string"})
    assert await potential.prefix(b'"hello world"') == -float("inf")


@pytest.mark.asyncio
async def test_validates_regex_format():
    potential = JsonSchema({"format": "regex", "type": "string"})
    assert await potential.prefix(b'"["') == -float("inf")


@pytest.mark.asyncio
async def test_will_not_allow_nonsense_after_json():
    potential = JsonSchema({"type": "object"})
    assert await potential.complete(b"{} hello world") == -float("inf")
