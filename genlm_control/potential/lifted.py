from genlm_control.potential import Potential


class Lifted(Potential):
    """Represents a potential lifted to operate on a target vocabulary."""

    def __init__(self, potential, target_vocab, f, g, f_seq):
        self.potential = potential
        self.f = f
        self.g = g
        self.f_seq = f_seq

        valid_tokens = []
        for target_token in target_vocab:
            base_token = f(target_token)
            if set(base_token) <= set(potential.decode):
                valid_tokens.append(g(base_token))

        if not valid_tokens:
            raise ValueError("No valid tokens found in target vocabulary")

        super().__init__(valid_tokens)

    def _batch_f_seq(self, contexts):
        return [self.f_seq(context) for context in contexts]

    async def complete(self, context):
        return await self.potential.complete(context=self.f_seq(context))

    async def prefix(self, context):
        return await self.potential.prefix(context=self.f_seq(context))

    async def logw_next(self, context):
        Ws = await self.potential.batch_logw_next_seq(
            context=self.f_seq(context), extensions=self.decode_eos
        )
        return self.make_lazy_weights(Ws)

    async def logw_next_seq(self, context, extension):
        return await self.potential.logw_next_seq(
            context=self.f_seq(context), extension=self.f_seq(extension)
        )

    async def batch_complete(self, contexts):
        return await self.potential.batch_complete(contexts=self._batch_f_seq(contexts))

    async def batch_prefix(self, contexts):
        return await self.potential.batch_prefix(contexts=self._batch_f_seq(contexts))

    async def batch_logw_next_seq(self, context, extensions):
        return await self.potential.batch_logw_next_seq(
            context=self.f_seq(context), extensions=self._batch_f_seq(extensions)
        )

    def __repr__(self):
        return f"Lifted({self.potential!r})"
