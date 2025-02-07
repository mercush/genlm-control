import pytest
from genlm_control.typing import Atomic, Sequence, infer_type, infer_transformations


def check_roundtrip(source_type, target_type, source_value):
    f, g = infer_transformations(source_type, target_type)
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
    f, g = infer_transformations(int_type, int_type)
    assert f(42) == 42
    assert g(42) == 42


def test_atomic_transformations():
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
        f, g = infer_transformations(source_type, target_type)
        assert f(source_val) == target_val
        check_roundtrip(source_type, target_type, source_val)


def test_sequence_transformations():
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
        f, g = infer_transformations(source_type, target_type)
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
        f, g = infer_transformations(source_type, target_type)
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
            infer_transformations(source_type, target_type)


def test_edge_cases():
    # Empty sequences
    str_seq = Sequence(Atomic(str))
    int_seq = Sequence(Atomic(int))
    f, g = infer_transformations(str_seq, int_seq)
    assert f([]) == []
    assert g([]) == []

    check_roundtrip(Atomic(str), Atomic(bytes), "Hello 世界")
