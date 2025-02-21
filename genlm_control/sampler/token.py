from hfppl import SubModel
from arsenal.maths import sample_dict, logsumexp


class TokenSampler(SubModel):
    """Base class for sampling a token from a potential's vocabulary.

    Args:
        target: The potential that samples are properly weighted with respect to.
        unit_type: The type of unit being sampled
        draw: The sampling function to use
    """

    def __init__(self, target, unit_type, draw):
        super().__init__()
        self.target = target
        self.unit_type = unit_type
        self.draw = draw

    async def sample(self, context, draw=sample_dict):
        raise NotImplementedError("Subclasses must implement sample method")

    async def forward(self):
        token, logw = await self.sample(self.parent.context, draw=self.draw)
        self.parent.score(logw)
        return token

    async def trace_swor(self, context):
        from genlm_control.tracer import TraceSWOR

        tracer = TraceSWOR()
        logP = self.target.alloc_logws()
        while tracer.root.mass > 0:
            with tracer:
                token, logw = await self.sample(context, draw=tracer)
                token_id = self.target.encode_eos[token]
                logP[token_id] = logsumexp([logP[token_id], logw])

        return self.target.make_lazy_weights(logP)


class DirectTokenSampler(TokenSampler):
    """Samples individual tokens directly from a potential's logw_next function.

    Samples are properly weighted with respect to the potential.logw_next(token | context).

    Args:
        potential (Potential): The potential function to sample from
        draw: The sampling function to use (defaults to sample_dict)
    """

    def __init__(self, potential, draw=sample_dict):
        super().__init__(target=potential, unit_type=potential.token_type, draw=draw)
        self.potential = potential

    async def sample(self, context, draw=sample_dict):
        logws = await self.potential.logw_next(context)
        token = draw(logws.normalize().exp().materialize())
        return token, logws.sum()


class SetTokenSampler(TokenSampler):
    """Samples individual tokens by sampling a set of tokens and then selecting one.

    Samples are properly weighted with respect to the set sampler's target.

    Args:
        set_sampler (SetSampler): The set sampler to sample from
        draw: The sampling function to use (defaults to sample_dict)
    """

    def __init__(self, set_sampler, draw=sample_dict):
        super().__init__(set_sampler.target, set_sampler.token_type, draw=draw)
        self.set_sampler = set_sampler

    async def sample(self, context, draw=sample_dict):
        logws = await self.set_sampler.sample_set(context)
        token = draw(logws.normalize().exp().materialize())
        return token, logws.sum()
