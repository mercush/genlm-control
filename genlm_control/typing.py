from dataclasses import dataclass
from collections.abc import Sequence as SequenceABC


@dataclass
class TokenType:
    """Base class representing the type of a token"""

    def check(self, value):
        """Check if a value matches this type"""
        raise NotImplementedError

    def is_iterable_of(self, element_type: "TokenType") -> bool:
        """Check if this type can be interpreted as an iterable of element_type.

        Args:
            element_type: The type to check if this is an iterable of

        Examples:
            >>> Sequence(Atomic(int)).is_iterable_of(Atomic(int))
            True
            >>> Atomic(bytes).is_iterable_of(Atomic(int))
            True
        """
        if isinstance(self, Sequence):
            return self.element_type is element_type

        if isinstance(self, Atomic):
            # Special cases for built-in iterables
            if (
                self.type is bytes
                and isinstance(element_type, Atomic)
                and element_type.type is int
            ):
                return True
            if (
                self.type is str
                and isinstance(element_type, Atomic)
                and element_type.type is str
            ):
                return True

        return False


@dataclass
class Atomic(TokenType):
    """Represents a simple type like int or str"""

    type: type  # The Python type (int, str, etc.)

    def check(self, value):
        return isinstance(value, self.type)


@dataclass
class Sequence(TokenType):
    """Represents a list/sequence of another type"""

    element_type: TokenType  # The type of elements in the sequence

    def check(self, value):
        return isinstance(value, (list, tuple)) and all(
            self.element_type.check(x) for x in value
        )


def infer_type(value):
    """Infer the TokenType from a value.

    Args:
        value: A sample value to infer type from

    Returns:
        TokenType: The inferred type

    Examples:
        >>> infer_type(42)
        Atomic(type=int)
        >>> infer_type([1, 2, 3])
        Sequence(element_type=Atomic(type=int))
        >>> infer_type([[1, 2], [3, 4]])
        Sequence(element_type=Sequence(element_type=Atomic(type=int)))
    """
    if isinstance(value, SequenceABC) and not isinstance(value, (str, bytes)):
        if not value:
            raise ValueError("Cannot infer type from empty sequence")
        # Recursively infer type of first element
        element_type = infer_type(value[0])
        # Verify all elements match this type
        if not all(element_type.check(x) for x in value):
            raise ValueError("Inconsistent types in sequence")
        return Sequence(element_type)

    return Atomic(type(value))


def infer_vocabulary_type(vocabulary):
    """Infer the TokenType from a vocabulary.

    Args:
        vocabulary: List of tokens to infer type from

    Returns:
        TokenType: The inferred type

    Raises:
        ValueError: If vocabulary is empty or contains inconsistent types

    Examples:
        >>> infer_vocabulary_type([1, 2, 3])
        Atomic(type=int)
        >>> infer_vocabulary_type([[1, 2], [3, 4]])
        Sequence(element_type=Atomic(type=int))
    """
    if not vocabulary:
        raise ValueError("Cannot infer type from empty vocabulary")

    token_type = infer_type(vocabulary[0])

    if not all(token_type.check(x) for x in vocabulary):
        raise ValueError("Inconsistent types in vocabulary")

    return token_type


def infer_transformations(source_type, target_type):
    """Infer the transformations required to convert source to target."""
    # Same types - use identity functions
    if source_type == target_type:
        return lambda x: x, lambda x: x

    # Handle Sequence -> Sequence conversions
    if isinstance(source_type, Sequence) and isinstance(target_type, Sequence):
        s_elem_type = source_type.element_type
        t_elem_type = target_type.element_type

        # Special case: List[bytes] -> List[int]
        if (
            isinstance(s_elem_type, Atomic)
            and s_elem_type.type is bytes
            and isinstance(t_elem_type, Atomic)
            and t_elem_type.type is int
        ):
            return (
                lambda seq: [list(b) for b in seq],
                lambda seq: [bytes(ints) for ints in seq],
            )

        # Recursive case for nested sequences
        f_elem, g_elem = infer_transformations(s_elem_type, t_elem_type)
        return (
            lambda seq: [f_elem(x) for x in seq],
            lambda seq: [g_elem(x) for x in seq],
        )

    # Handle Atomic -> Atomic conversions
    if isinstance(source_type, Atomic) and isinstance(target_type, Atomic):
        s_type = source_type.type
        t_type = target_type.type

        # Special case for bytes <-> str
        if s_type is str and t_type is bytes:
            return (
                lambda x: x.encode(errors="replace"),
                lambda x: x.decode(errors="replace"),
            )

        if s_type is bytes and t_type is str:
            return (
                lambda x: x.decode(errors="replace"),
                lambda x: x.encode(errors="replace"),
            )

        # Direct type conversions for basic types
        basic_types = (str, int, bool, float)
        if t_type in basic_types and s_type in basic_types:
            return lambda x: t_type(x), lambda x: s_type(x)

    # Handle Sequence -> Atomic conversions
    if isinstance(source_type, Sequence) and isinstance(target_type, Atomic):
        s_elem_type = source_type.element_type
        t_type = target_type.type

        # List of chars -> string
        if (
            isinstance(s_elem_type, Atomic)
            and s_elem_type.type is str
            and t_type is str
        ):
            return lambda x: "".join(x), lambda x: list(x)

        # List of ints -> bytes
        if (
            isinstance(s_elem_type, Atomic)
            and s_elem_type.type is int
            and t_type is bytes
        ):
            return lambda x: bytes(x), lambda x: list(x)

    # Handle Atomic -> Sequence conversions
    if isinstance(source_type, Atomic) and isinstance(target_type, Sequence):
        s_type = source_type.type
        t_elem_type = target_type.element_type

        # String -> list of chars/bytes
        if s_type is str:
            if isinstance(t_elem_type, Atomic):
                if t_elem_type.type is str:
                    return lambda x: list(x), lambda x: "".join(x)
                if t_elem_type.type is bytes:
                    return (
                        lambda x: [c.encode() for c in x],
                        lambda x: "".join(b.decode() for b in x),
                    )

        # Bytes -> list of ints/bytes
        if s_type is bytes:
            if isinstance(t_elem_type, Atomic):
                if t_elem_type.type is int:
                    return lambda x: list(x), lambda x: bytes(x)
                if t_elem_type.type is bytes:
                    # Split into chunks of size 1
                    return lambda x: [bytes([b]) for b in x], lambda x: b"".join(x)

    raise TypeError(
        f"Cannot infer transformations between {source_type} and {target_type}"
    )
