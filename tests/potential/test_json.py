import pytest
from genlm_control.potential.built_in.json import JsonSchema
import json


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
