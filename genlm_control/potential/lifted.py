from genlm_control.potential.base import Potential


class Lifted(Potential):
    """Lifts a potential with vocabulary elements of type T
    to a potential with vocabulary elements of type Iterable[T]."""

    supported_target_types = {
        # type | flatten function
        str: lambda x: "".join(x),
        # list:  lambda x: sum(x, []),
        # tuple: lambda x: sum(x, ()),
        bytes: lambda x: b"".join(x),
    }

    def __init__(self, potential, target_vocab):
        self.potential = potential

        valid_tokens = []
        for token in target_vocab:
            if type(token) not in self.supported_target_types:
                raise TypeError(f"Unsupported target token type: {type(token)}")
            if set(token) <= set(potential.decode):
                valid_tokens.append(token)

        if not valid_tokens:
            raise ValueError("No valid tokens found in target vocabulary")

        super().__init__(valid_tokens)

    def _flatten(self, context):
        if not context:
            return context

        context_type = type(context[0])
        if context_type not in self.supported_target_types:
            raise TypeError(
                f"Cannot flatten context with elements of type {context_type}."
            )

        return self.supported_target_types[context_type](context)

    def _batch_flatten(self, contexts):
        return [self._flatten(context) for context in contexts]

    async def complete(self, context):
        return await self.potential.complete(context=self._flatten(context))

    async def prefix(self, context):
        return await self.potential.prefix(context=self._flatten(context))

    async def score(self, context):
        return await self.potential.score(context=self._flatten(context))

    async def logp_next(self, context):
        Ws = await self.potential.batch_logp_next_seq(
            context=self._flatten(context), extensions=self.decode_eos
        )
        return self.make_lazy_weights(Ws)

    async def logp_next_seq(self, context, extension):
        return await self.potential.logp_next_seq(
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

    async def batch_logp_next_seq(self, context, extensions):
        return await self.potential.batch_logp_next_seq(
            context=self._flatten(context), extensions=self._batch_flatten(extensions)
        )

    def __repr__(self):
        return f"Lifted({self.potential!r})"


class Lowered(Potential):
    # Tim's char LM algorithm?
    pass
