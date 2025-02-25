from arsenal.maths import logsumexp
from genlm_control.util import fast_sample_lazyweights
from hfppl import SubModel
from genlm_control.sampler.set import SetSampler


class TokenSampler(SubModel):
    """Base class for sampling a token from a potential's vocabulary.

    `TokenSampler`s generate properly weighted samples with respect to a `target` potential.

    Given a context of tokens $x_1, \\ldots, x_{n-1}$ in the target potential's vocabulary,
    a `TokenSampler` samples a token $x_n$ and weight $w$ from the potential's vocabulary.

    The sampled token and weight are properly weighted with respect to
    $$
    \\textsf{target.logw_next}(x_n | x_1, \\ldots, x_{n-1})
    $$

    Args:
        target (Potential): The potential that samples are properly weighted with respect to.
    """

    def __init__(self, target):
        super().__init__()
        self.target = target
        self.token_type = self.target.token_type

    async def start_weight(self):
        """Compute the weight of the empty sequence under the target potential."""
        return await self.target.prefix([])

    async def forward(self):
        parent = self.parent  # For some reason, need to hold onto this reference.
        token, logw, logp = await self.sample(parent.token_ctx)
        parent.score(logw)
        parent.logp += logp
        return token

    async def sample(self, context, draw):
        """Sample a token and weight from the potential's vocabulary."""
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
    """Samples individual tokens directly from the log-normalized `logw_next` function
    of a potential.

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
        """Sample a token and weight that are properly weighted with respect to the target potential's `logw_next` method.

        Given a context of tokens $x_1, \\ldots, x_{n-1}$ in the target potential's vocabulary,
        this method samples a token $x_n$ and weight $w$ from the potential's vocabulary.

        The sampled token and weight are properly weighted with respect to
        $$
        \\textsf{target.logw_next}(x_n | x_1, \\ldots, x_{n-1})
        $$

        The returned weight corresponds to the log normalizing constant of $\\textsf{target.logw_next}(x_n | x_1, \\ldots, x_{n-1})$.

        Returns:
            (token, weight, logp): A tuple containing the sampled token, weight, and log-probability of the sampled token.
        """
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
    """Samples individual tokens by sampling a weighted set of tokens and then selecting one
    proportional to its weight.

    This class wraps a `SetSampler`.

    Args:
        set_sampler (SetSampler): The set sampler to sample from
    """

    def __init__(self, set_sampler):
        assert isinstance(set_sampler, SetSampler)
        super().__init__(set_sampler.target)
        self.set_sampler = set_sampler

    async def sample(self, context, draw=None):
        """Sample a token and weight by sampling a weighted set of tokens from the `set_sampler`
        and then selecting one proportional to its weight.

        Given a context of tokens $x_1, \\ldots, x_{n-1}$ in the vocabulary of the set sampler's target potential,
        this method samples a token $x_n$ from the potential's vocabulary and a weight.

        The sampled token and weight are properly weighted with respect to
        $$
        \\textsf{set_sampler.target.logw_next}(x_n | x_1, \\ldots, x_{n-1})
        $$

        The returned weight corresponds to the sum of the weights of the sampled set.

        Args:
            context (list[int]): A sequence of tokens in the vocabulary of the set sampler's target potential.

        Returns:
            (token, weight, logp): A tuple containing the sampled token, weight, and log-probability of the random
                choices made in sampling that token.

        Note:
            For properly weighted sampling, the `set_sampler` must assign correct weights to each token. See
            `SetSampler` for more details.
        """
        logws, logp = await self.set_sampler.sample_set(context, draw=draw)
        logps = logws.normalize()
        if draw is None:
            token = fast_sample_lazyweights(logps)
        else:
            token = draw(logps.exp().materialize())
        return token, logws.sum(), logp + logps[token]

    async def cleanup(self):
        """Clean up the sampler.

        This method should be called when the sampler is no longer needed.
        """
        await self.set_sampler.cleanup()
