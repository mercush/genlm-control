import asyncio
import warnings
from genlm_control.potential.base import Potential


class Product(Potential):
    """
    Combine two potential instances via element-wise multiplication (sum in log space).

    This class creates a new potential that is the element-wise product of two potentials.
    For example,
    ```
    prefix(xs) = p1.prefix(xs) + p2.prefix(xs)
    ```
    and
    ```
    logw_next(x | xs) = p1.logw_next(x | xs) + p2.logw_next(x | xs)
    ```

    The new potential's vocabulary is the intersection of the two potentials' vocabularies.

    This class inherits all methods from [`Potential`][genlm_control.potential.base.Potential],
    see there for method documentation.

    Attributes:
        p1 (Potential): The first potential instance.
        p2 (Potential): The second potential instance.
        token_type (str): The type of tokens that this product potential operates on.
        decode (list): The common vocabulary shared between the two potentials.

    Warning:
        Be careful when taking products of potentials with minimal vocabulary overlap.
        The resulting potential will only operate on tokens present in both vocabularies.
    """

    def __init__(self, p1, p2):
        """Initialize a Product potential.

        Args:
            p1 (Potential): First potential
            p2 (Potential): Second potential
        """
        self.p1 = p1
        self.p2 = p2

        if self.p1.token_type == self.p2.token_type:
            self.token_type = self.p1.token_type
        else:
            raise ValueError(
                "Potentials in product must have the same token type. "
                f"Got {self.p1.token_type} and {self.p2.token_type}."
            )

        common_vocab = list(set(p1.decode) & set(p2.decode))
        if not common_vocab:
            raise ValueError("Potentials in product must share a common vocabulary")

        # Check for small vocabulary overlap
        threshold = 0.1
        for potential, name in [(p1, "p1"), (p2, "p2")]:
            overlap_ratio = len(common_vocab) / len(potential.decode)
            if overlap_ratio < threshold:
                warnings.warn(
                    f"Common vocabulary ({len(common_vocab)} tokens) is less than {threshold * 100}% "
                    f"of {name}'s ({p1!r}) vocabulary ({len(potential.decode)} tokens). "
                    "This Product potential only operates on this relatively small subset of tokens.",
                    RuntimeWarning,
                )

        super().__init__(common_vocab, token_type=self.token_type)

        # For fast products of weights
        self.v1_idxs = [p1.encode_eos[token] for token in self.decode_eos]
        self.v2_idxs = [p2.encode_eos[token] for token in self.decode_eos]

    async def prefix(self, context):
        w1, w2 = await asyncio.gather(self.p1.prefix(context), self.p2.prefix(context))
        return w1 + w2

    async def complete(self, context):
        w1, w2 = await asyncio.gather(
            self.p1.complete(context), self.p2.complete(context)
        )
        return w1 + w2

    async def logw_next_seq(self, context, extension):
        W1, W2 = await asyncio.gather(
            self.p1.logw_next_seq(context, extension),
            self.p2.logw_next_seq(context, extension),
        )
        return W1 + W2

    async def batch_complete(self, contexts):
        W1, W2 = await asyncio.gather(
            self.p1.batch_complete(contexts), self.p2.batch_complete(contexts)
        )
        return W1 + W2

    async def batch_prefix(self, contexts):
        W1, W2 = await asyncio.gather(
            self.p1.batch_prefix(contexts), self.p2.batch_prefix(contexts)
        )
        return W1 + W2

    async def batch_logw_next_seq(self, context, extensions):
        W1, W2 = await asyncio.gather(
            self.p1.batch_logw_next_seq(context, extensions),
            self.p2.batch_logw_next_seq(context, extensions),
        )
        return W1 + W2

    async def logw_next(self, context):
        W1, W2 = await asyncio.gather(
            self.p1.logw_next(context), self.p2.logw_next(context)
        )
        return self.make_lazy_weights(
            W1.weights[self.v1_idxs] + W2.weights[self.v2_idxs]
        )

    async def batch_logw_next(self, contexts):
        Ws1, Ws2 = await asyncio.gather(
            self.p1.batch_logw_next(contexts), self.p2.batch_logw_next(contexts)
        )
        return [
            self.make_lazy_weights(
                Ws1[n].weights[self.v1_idxs] + Ws2[n].weights[self.v2_idxs]
            )
            for n in range(len(contexts))
        ]

    def spawn(self, p1_opts=None, p2_opts=None):
        return Product(
            self.p1.spawn(**(p1_opts or {})),
            self.p2.spawn(**(p2_opts or {})),
        )

    def __repr__(self):
        return f"Product({self.p1!r}, {self.p2!r})"
