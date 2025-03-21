import json_stream

from jsonschema import validators, Draft7Validator, ValidationError
from collections.abc import Sequence, Mapping
from genlm_control.potential.base import Potential


def is_sequence(checker, instance):
    return isinstance(instance, Sequence) and not isinstance(
        instance, (str, bytes, bytearray, Mapping)
    )


def is_object(checker, instance):
    return isinstance(instance, Mapping)


type_checker = Draft7Validator.TYPE_CHECKER

custom_type_checker = type_checker.redefine_many(
    {
        "array": is_sequence,
        "object": is_object,
    }
)

LazyCompatibleValidator = validators.extend(
    Draft7Validator, type_checker=custom_type_checker
)


class OutOfBytes(Exception):
    pass


class JustOneBlockIterable:
    """Provides a single value (intended to be bytes from a context)
    and then signals if the reader tried to read past it. This allows
    us to distinguish invalid JSON from incomplete JSON by seeing if
    the reader tried to read more than it had or failed early."""

    def __init__(self, block):
        self.__block = block
        self.read_past_first_block = False

    def __iter__(self):
        yield self.__block
        self.read_past_first_block = True


UTF8_START_BYTE_MASKS = [
    (0b00000000, 0b10000000),
    (0b11000000, 0b11100000),
    (0b11100000, 0b11110000),
    (0b11110000, 0b11111000),
]


def is_utf8_start_byte(n: int) -> bool:
    """Checks if this is a byte that can appear at the
    start of a UTF-8 character."""
    assert 0 <= n < 256
    for prefix, mask in UTF8_START_BYTE_MASKS:
        if n & mask == prefix:
            return True
    return False


class JsonSchema(Potential):
    def __init__(self, schema):
        super().__init__(
            list(range(256)),
        )
        self.schema = schema
        self.validator = LazyCompatibleValidator(self.schema)

    def __check_context(self, context):
        context = bytes(context)

        # JSON documents have to be valid UTF-8, but we might be
        # in the middle of generating a UTF-8 character. If so, we
        # only consider the prefix that is valid UTF-8, but need
        # to signal at the end that this is a valid prefix and not
        # a valid complete document.
        incomplete_utf8_at_end = False
        try:
            try:
                context.decode("utf-8")
            except UnicodeDecodeError:
                for i in range(1, min(5, len(context))):
                    if is_utf8_start_byte(context[-i]):
                        context = context[:-i]
                        context.decode("utf-8")
                        incomplete_utf8_at_end = True
                        break
                else:
                    raise
        except UnicodeDecodeError:
            raise ValueError("Invalid UTF-8")

        # Feeding just whitespace to json-stream causes it to raise
        # StopIteration, and this is always a valid start to a JSON
        # document of any schema, and never a valid JSON value.
        if not context.strip():
            raise OutOfBytes()

        iterable = JustOneBlockIterable(context)
        try:
            x = json_stream.load(iterable, persistent=True)
            self.validator.validate(x)
            if hasattr(x, "read_all"):
                x.read_all()
        except ValueError:
            if iterable.read_past_first_block:
                raise OutOfBytes()
            else:
                raise
        if incomplete_utf8_at_end:
            raise OutOfBytes()

    async def complete(self, context) -> float:
        # TODO:
        # 1. Create some sort of caching for the validator, so
        #    we can reuse ones from previous calls.
        # 2. Use a Lark JSON grammar as a prefilter to rule out any
        #    bytes that can't be included next in valid JSON.

        try:
            self.__check_context(context)
        except (ValueError, ValidationError, OutOfBytes):
            return -float("inf")

        return 0.0

    async def prefix(self, context) -> float:
        # TODO:
        # 1. Create some sort of caching for the validator, so
        #    we can reuse ones from previous calls.
        # 2. Use a Lark JSON grammar as a prefilter to rule out any
        #    bytes that can't be included next in valid JSON.
        try:
            self.__check_context(context)
        except (ValueError, ValidationError):
            return -float("inf")
        except OutOfBytes:
            pass

        return 0.0
