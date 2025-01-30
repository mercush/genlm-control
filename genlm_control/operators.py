class PotentialOperators:
    """
    A mixin class that provides operator overloading methods for composing potential instances.

    This class is intended to be inherited by other potential classes.
    """

    def __mul__(self, other):
        """Direct product. Vocabularies must be compatible."""
        from genlm_control.product import Product

        return Product(self, other)

    def lift(self, other):
        """Lift `self` so that it operations on `other`'s vocabulary."""
        from genlm_control.potential.lifted import Lifted

        return Lifted(self, other.decode)

    def __sub__(self, other):
        if self is other:
            return
        raise NotImplementedError()
