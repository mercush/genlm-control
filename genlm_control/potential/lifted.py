from genlm_control.potential.base import Potential
from itertools import chain

# single state transducer


class Lifted(Potential):
    """Represents a potential lifted to operate on a target vocabulary.

    Lifts a potential with vocabulary elements of type T
    to a potential with vocabulary elements of type Iterable[T].

    Args:
        potential: The base potential to lift
        target_vocab: The target vocabulary to lift the potential to

    Raises:
        ValueError: If no valid tokens are found in the target vocabulary
    """

    def __init__(self, potential, target_vocab):
        self.potential = potential

        valid_tokens = []
        for token in target_vocab:
            if set(token) <= set(potential.decode):  # use flatten
                valid_tokens.append(token)

        if not valid_tokens:
            raise ValueError("No valid tokens found in target vocabulary")

        super().__init__(valid_tokens)

    @staticmethod
    def _flatten(context):  # one to many
        # int -> list[str]
        # int -> list[int]
        return list(chain.from_iterable(context))

    def _batch_flatten(self, contexts):
        return [self._flatten(context) for context in contexts]

    async def complete(self, context):
        return await self.potential.complete(context=self._flatten(context))

    async def prefix(self, context):
        return await self.potential.prefix(context=self._flatten(context))

    async def score(self, context):
        return await self.potential.score(context=self._flatten(context))

    async def logw_next(self, context):
        Ws = await self.potential.batch_logw_next_seq(
            context=self._flatten(context), extensions=self.decode_eos
        )
        return self.make_lazy_weights(Ws)

    async def logw_next_seq(self, context, extension):
        return await self.potential.logw_next_seq(
            context=self._flatten(context), extension=self._flatten(extension)
        )

    async def batch_complete(self, contexts):
        return await self.potential.batch_complete(
            contexts=self._batch_flatten(contexts)
        )

    async def batch_prefix(self, contexts):
        return await self.potential.batch_prefix(contexts=self._batch_flatten(contexts))

    async def batch_score(self, contexts):
        return await self.potential.batch_score(contexts=self._batch_flatten(contexts))

    async def batch_logw_next_seq(self, context, extensions):
        return await self.potential.batch_logw_next_seq(
            context=self._flatten(context), extensions=self._batch_flatten(extensions)
        )

    def __repr__(self):
        return f"Lifted({self.potential!r})"


# TODO: Add docstrings to the lifted methods.
Lifted.complete.__doc__ = Potential.complete.__doc__
Lifted.prefix.__doc__ = Potential.prefix.__doc__
Lifted.score.__doc__ = Potential.score.__doc__
Lifted.logw_next.__doc__ = Potential.logw_next.__doc__
Lifted.logw_next_seq.__doc__ = Potential.logw_next_seq.__doc__
Lifted.batch_complete.__doc__ = Potential.batch_complete.__doc__
Lifted.batch_prefix.__doc__ = Potential.batch_prefix.__doc__
Lifted.batch_score.__doc__ = Potential.batch_score.__doc__
Lifted.batch_logw_next_seq.__doc__ = Potential.batch_logw_next_seq.__doc__
