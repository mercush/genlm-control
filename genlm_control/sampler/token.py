from arsenal.maths import sample_dict, logsumexp


class TokenSampler:
    """Base class for sampling a token from a potential's vocabulary.

    Args:
        target: The potential that samples are properly weighted with respect to.
        unit_type: The type of unit being sampled
        draw: The sampling function to use
    """

    def __init__(self, target):
        super().__init__()
        self.target = target

    async def sample(self, context, draw=sample_dict):
        raise NotImplementedError("Subclasses must implement sample method")

    async def trace_swor(self, context):
        from genlm_control.tracer import TraceSWOR

        tracer = TraceSWOR()
        logws = self.target.alloc_logws()
        while tracer.root.mass > 0:
            with tracer:
                token, logw, logp = await self.sample(context, draw=tracer)
                token_id = self.target.encode_eos[token]
                logws[token_id] = logsumexp([logws[token_id], logw + logp])

        return self.target.make_lazy_weights(logws)

    async def cleanup(self):
        pass


class DirectTokenSampler(TokenSampler):
    """Samples individual tokens directly from a potential's logw_next function.

    Samples are properly weighted with respect to the potential.

    Args:
        potential (Potential): The potential function to sample from
        draw: The sampling function to use (defaults to sample_dict)
    """

    def __init__(self, potential):
        super().__init__(target=potential)
        self.potential = potential

    async def sample(self, context, draw=sample_dict):
        logws = await self.potential.logw_next(context)
        logps = logws.normalize()
        token = draw(logps.exp().materialize())
        return token, logws.sum(), logps[token]


class SetTokenSampler(TokenSampler):
    """Samples individual tokens by sampling a set of tokens and then selecting one.

    Samples are properly weighted with respect to the set sampler's target potential.

    Args:
        set_sampler (SetSampler): The set sampler to sample from
        draw: The sampling function to use (defaults to sample_dict)
    """

    def __init__(self, set_sampler):
        super().__init__(set_sampler.target)
        self.set_sampler = set_sampler

    async def sample(self, context, draw=sample_dict):
        logws, logp_set = await self.set_sampler.sample_set(context, draw=draw)
        logps = logws.normalize()
        token = draw(logps.exp().materialize())
        return token, logws.sum(), logp_set + logps[token]

    async def cleanup(self):
        await self.set_sampler.cleanup()
