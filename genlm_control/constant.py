class EndOfSequence:
    """Special sentinel token for end-of-sequence."""

    def __repr__(self):
        return "EOS"

    def __radd__(self, other):
        if isinstance(other, (str, bytes)):
            return [*list(other), self]
        elif isinstance(other, (list, tuple)):
            return type(other)(list(other) + [self])
        else:
            raise TypeError(f"Cannot concatenate {type(other)} with {type(self)}")

    def __iter__(self):
        return iter([self])


EOS = EndOfSequence()
