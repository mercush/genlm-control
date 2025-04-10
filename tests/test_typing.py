import pytest
from genlm.control.typing import (
    Atomic,
    Sequence,
    infer_type,
    infer_vocabulary_type,
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


def test_atomic_convert():
    int_type = Atomic(int)
    assert int_type.convert(42) == 42
    assert int_type.convert("42") == 42
    repr(int_type)


def test_sequence_convert():
    int_seq = Sequence(Atomic(int))
    assert int_seq.convert([1, 2, 3]) == (1, 2, 3)
    assert int_seq.convert(["1", "2", "3"]) == (1, 2, 3)
    repr(int_seq)
