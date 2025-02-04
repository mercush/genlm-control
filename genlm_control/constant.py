class EndOfSequence:
    """Special sentinel token for end-of-sequence."""

    def __repr__(self):
        return "<EOS>"

    def __radd__(self, other):
        if isinstance(other, str):
            return other + "<EOS>"
        elif isinstance(other, bytes):
            return other + b"\x00"  # Using null byte as it's rarely used in text
        elif isinstance(other, (list, tuple)):
            return type(other)(list(other) + [self])

    def __iter__(self):
        return iter([self])


EOS = EndOfSequence()
