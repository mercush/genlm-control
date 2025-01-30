from genlm_control.product import Product
from genlm_control.potential.base import Potential
from genlm_control.potential.lifted import Lifted


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


# Product(x, y) = Product(y, x)
# Product(x, lift(y)) = Product(lift(y), x)

# Sample x is properly weighted w.r.t. generator, we want it to be
# properly weighted w.r.t. product.


def subtract(x, y):
    """
    Decomposes x and y into their constituent potentials and returns two lists:
    1. Potentials in x but not in y
    2. Potentials in y but not in x

    Args:
        x: A Potential instance
        y: A Potential instance

    Returns:
        tuple: (a_only, b_only) where each is a list of Potentials
    """

    def decompose(p):
        if isinstance(p, Lifted):
            return decompose(p.potential)
        elif isinstance(p, Product):
            return [*decompose(p.p1), *decompose(p.p2)]
        elif isinstance(p, Potential):
            return [p]
        raise ValueError(f"Cannot decompose {p}")

    xs = decompose(x)
    ys = decompose(y)

    x_only = [p for p in xs if p not in ys]
    y_only = [p for p in ys if p not in xs]

    return x_only, y_only
