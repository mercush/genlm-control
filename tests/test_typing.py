import pytest
from genlm_control.typing import (
    Atomic,
    Sequence,
    infer_type,
    infer_token_transformations,
    infer_sequence_transformation,
    infer_vocabulary_type,
)


def check_roundtrip(source_type, target_type, source_value):
    f, g = infer_token_transformations(source_type, target_type)
    converted = f(source_value)
    roundtrip = g(converted)
    assert roundtrip == source_value, (
        f"Roundtrip failed: {source_value} -> {converted} -> {roundtrip}"
    )


def test_atomic_type_check():
    int_type = Atomic(int)
    assert int_type.check(42)
    assert not int_type.check("42")

    str_type = Atomic(str)
    assert str_type.check("hello")
    assert not str_type.check(b"hello")

    bytes_type = Atomic(bytes)
    assert bytes_type.check(b"hello")
    assert not bytes_type.check("hello")


def test_sequence_type_check():
    int_seq = Sequence(Atomic(int))
    assert int_seq.check([1, 2, 3])
    assert not int_seq.check([1, "2", 3])
    assert not int_seq.check(42)

    nested_int_seq = Sequence(Sequence(Atomic(int)))
    assert nested_int_seq.check([[1, 2], [3, 4]])
    assert not nested_int_seq.check([[1, 2], 3])

    bytes_seq = Sequence(Atomic(bytes))
    assert bytes_seq.check([b"hello", b"world"])
    assert not bytes_seq.check(["hello", "world"])


def test_atomic_inference():
    assert infer_type(42) == Atomic(int)
    assert infer_type("hello") == Atomic(str)
    assert infer_type(b"hello") == Atomic(bytes)
    assert infer_type(3.14) == Atomic(float)
    assert infer_type(True) == Atomic(bool)


def test_sequence_inference():
    assert infer_type([1, 2, 3]) == Sequence(Atomic(int))
    assert infer_type(["a", "b"]) == Sequence(Atomic(str))
    assert infer_type([[1, 2], [3, 4]]) == Sequence(Sequence(Atomic(int)))
    assert infer_type([b"AB", b"CD"]) == Sequence(Atomic(bytes))


def test_empty_sequence_error():
    with pytest.raises(ValueError):
        infer_type([])


def test_inconsistent_sequence_error():
    with pytest.raises(ValueError):
        infer_type([1, "2", 3])


def test_identity_transformations():
    int_type = Atomic(int)
    f, g = infer_token_transformations(int_type, int_type)
    assert f(42) == 42
    assert g(42) == 42


def test_atomic_token_transformations():
    """Test conversions between atomic types"""
    cases = [
        (Atomic(str), Atomic(int), "42", 42),
        (Atomic(str), Atomic(float), "3.14", 3.14),
        (Atomic(str), Atomic(bool), "True", True),
        (Atomic(str), Atomic(bytes), "hello", b"hello"),
        (Atomic(int), Atomic(float), 42, 42.0),
        (Atomic(int), Atomic(bool), 1, True),
        (Atomic(bytes), Atomic(str), b"hello", "hello"),
    ]

    for source_type, target_type, source_val, target_val in cases:
        f, g = infer_token_transformations(source_type, target_type)
        assert f(source_val) == target_val
        check_roundtrip(source_type, target_type, source_val)


def test_sequence_token_transformations():
    """Test conversions between sequence types"""
    cases = [
        # List[str] -> List[int]
        (Sequence(Atomic(str)), Sequence(Atomic(int)), ["1", "2", "3"], [1, 2, 3]),
        # List[bytes] -> List[int]
        (Sequence(Atomic(bytes)), Sequence(Atomic(int)), [b"A", b"B"], [[65], [66]]),
        # Nested sequences
        (
            Sequence(Sequence(Atomic(str))),
            Sequence(Sequence(Atomic(int))),
            [["1", "2"], ["3", "4"]],
            [[1, 2], [3, 4]],
        ),
    ]

    for source_type, target_type, source_val, target_val in cases:
        f, g = infer_token_transformations(source_type, target_type)
        assert f(source_val) == target_val
        check_roundtrip(source_type, target_type, source_val)


def test_sequence_atomic_transformations():
    """Test conversions between sequences and atomic types"""
    cases = [
        # List[str] -> str (join)
        (Sequence(Atomic(str)), Atomic(str), ["h", "i"], "hi"),
        # List[int] -> bytes
        (Sequence(Atomic(int)), Atomic(bytes), [65, 66], b"AB"),
        # str -> List[str] (chars)
        (Atomic(str), Sequence(Atomic(str)), "hi", ["h", "i"]),
        # bytes -> List[int]
        (Atomic(bytes), Sequence(Atomic(int)), b"AB", [65, 66]),
    ]

    for source_type, target_type, source_val, target_val in cases:
        f, g = infer_token_transformations(source_type, target_type)
        assert f(source_val) == target_val
        check_roundtrip(source_type, target_type, source_val)


def test_invalid_transformations():
    """Test that invalid transformations raise TypeError"""
    invalid_cases = [
        (Atomic(int), Atomic(dict)),  # No conversion path
        (Sequence(Atomic(dict)), Sequence(Atomic(int))),  # Invalid element type
        (Atomic(int), Sequence(Atomic(dict))),  # Invalid sequence conversion
    ]

    for source_type, target_type in invalid_cases:
        with pytest.raises(TypeError):
            infer_token_transformations(source_type, target_type)


def test_edge_cases():
    # Empty sequences
    str_seq = Sequence(Atomic(str))
    int_seq = Sequence(Atomic(int))
    f, g = infer_token_transformations(str_seq, int_seq)
    assert f([]) == []
    assert g([]) == []

    check_roundtrip(Atomic(str), Atomic(bytes), "Hello 世界")


def test_is_iterable_of():
    assert Sequence(Atomic(int)).is_iterable_of(Atomic(int))
    assert Sequence(Atomic(str)).is_iterable_of(Atomic(str))
    assert not Sequence(Atomic(int)).is_iterable_of(Atomic(str))

    assert Atomic(bytes).is_iterable_of(Atomic(int))
    assert Atomic(str).is_iterable_of(Atomic(str))

    assert not Atomic(int).is_iterable_of(Atomic(int))
    assert not Atomic(bytes).is_iterable_of(Atomic(str))
    assert not Atomic(str).is_iterable_of(Atomic(int))

    nested_seq = Sequence(Sequence(Atomic(int)))
    assert nested_seq.is_iterable_of(Sequence(Atomic(int)))
    assert not nested_seq.is_iterable_of(Atomic(int))


def test_sequence_transformation():
    """Test the infer_sequence_transformation function"""
    cases = [
        # List[str] -> List[bytes]
        (
            Sequence(Atomic(str)),
            Atomic(bytes),
            ["AB", "CD"],
            [b"A", b"B", b"C", b"D"],
        ),
        # List[List[int]] -> List[int] (flattening)
        (
            Sequence(Atomic(int)),
            Atomic(int),
            [[1, 2], [3, 4]],
            [1, 2, 3, 4],
        ),
        # Bytes -> int
        (
            Atomic(bytes),
            Atomic(int),
            [b"AB", b"CD"],
            [65, 66, 67, 68],
        ),
    ]

    for source_type, target_type, source_val, target_val in cases:
        f = infer_sequence_transformation(source_type, target_type)
        result = f(source_val)
        assert result == target_val, f"Expected {target_val}, got {result}"


def test_vocabulary_type_inference():
    """Test the infer_vocabulary_type function"""
    assert infer_vocabulary_type([1, 2, 3]) == Atomic(int)
    assert infer_vocabulary_type(["a", "b"]) == Atomic(str)
    assert infer_vocabulary_type([[1, 2], [3, 4]]) == Sequence(Atomic(int))

    # Test empty vocabulary
    with pytest.raises(ValueError):
        infer_vocabulary_type([])

    # Test inconsistent types
    with pytest.raises(ValueError):
        infer_vocabulary_type([1, "2", 3])


def test_error_conditions():
    """Test various error conditions"""
    # Test invalid sequence transformations
    with pytest.raises(TypeError):
        infer_sequence_transformation(Atomic(dict), Atomic(int))

    # Test transformation of invalid types
    with pytest.raises(TypeError):
        infer_token_transformations(Atomic(complex), Atomic(int))

    # Test inconsistent nested sequences
    with pytest.raises(ValueError):
        infer_type([[1, 2], [3, "4"]])
