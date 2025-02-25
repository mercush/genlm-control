from arsenal.maths import logsumexp
from genlm_control.util import fast_sample_lazyweights
from hfppl import SubModel


class TokenSampler(SubModel):
    """Base class for sampling a token from a potential's vocabulary.

    Args:
        target: The potential that samples are properly weighted with respect to.
    """

    def __init__(self, target):
        super().__init__()
        self.target = target
        self.token_type = self.target.token_type

    async def start_weight(self):
        return await self.target.prefix([])

    async def forward(self):
        parent = self.parent  # For some reason, need to hold onto this reference.
        token, logw, logp = await self.sample(parent.token_ctx)
        parent.score(logw)
        parent.logp += logp
        return token

    async def sample(self, context, draw):
        raise NotImplementedError("Subclasses must implement sample method")

    async def trace_swor(self, context):
        from genlm_control.tracer import TraceSWOR

        tracer = TraceSWOR()
        logP = self.target.alloc_logws()
        while tracer.root.mass > 0:
            with tracer:
                token, logw, logp = await self.sample(context, draw=tracer)
                token_id = self.target.encode_eos[token]
                logP[token_id] = logsumexp([logP[token_id], logw + logp])

        return self.target.make_lazy_weights(logP)

    async def cleanup(self):
        pass


class DirectTokenSampler(TokenSampler):
    """Samples individual tokens directly from a potential's logw_next function.

    Samples are properly weighted with respect to `potential.logw_next(token | context)`.

    Args:
        potential (Potential): The potential function to sample from

    Warning:
        Only use this sampler if the potential's `logw_next` method is efficient. This is the case
        for potentials like `PromptedLLM`, but for custom potentials with a large vocabulary size,
        the default implementation of `logw_next` generally will not be efficient, and thus this
        sampler will be slow.
    """

    def __init__(self, potential):
        super().__init__(target=potential)
        self.potential = potential

    async def sample(self, context, draw=None):
        logws = await self.potential.logw_next(context)
        logps = logws.normalize()
        if draw is None:
            # fast sampling from logps using gumbel-max trick
            token = fast_sample_lazyweights(logps)
        else:
            token = draw(logps.exp().materialize())
        return token, logws.sum(), logps[token]

    async def cleanup(self):
        pass


class SetTokenSampler(TokenSampler):
    """Samples individual tokens by sampling a set of tokens and then selecting one.

    Samples are properly weighted with respect to `set_sampler.target.logw_next(token | context)`.

    Args:
        set_sampler (SetSampler): The set sampler to sample from
    """

    def __init__(self, set_sampler):
        super().__init__(set_sampler.target)
        self.set_sampler = set_sampler

    async def sample(self, context, draw=None):
        logws, logp = await self.set_sampler.sample_set(context, draw=draw)
        logps = logws.normalize()
        if draw is None:
            token = fast_sample_lazyweights(logps)
        else:
            token = draw(logps.exp().materialize())
        return token, logws.sum(), logp + logps[token]

    async def cleanup(self):
        await self.set_sampler.cleanup()
