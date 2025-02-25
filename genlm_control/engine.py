from genlm_control.potential import Potential
from genlm_control.sampler.token import TokenSampler
from genlm_control.sampler.sequence import SequenceModel, _unpack_particles, Sequences

from hfppl import smc_standard


class InferenceEngine:
    def __init__(self, unit_sampler, critic=None):
        if not isinstance(unit_sampler, TokenSampler):
            raise ValueError("`unit_sampler` must be a TokenSampler")

        if critic:
            if not isinstance(critic, Potential):
                raise ValueError("`critic` must be a Potential")
            if not unit_sampler.token_type == critic.token_type:
                raise ValueError(
                    "`critic` must have the same token type as the `unit_sampler`. "
                    f"Got {unit_sampler.token_type} and {critic.token_type}."
                    f"\nMaybe you forgot to coerce the critic to the token type of the unit sampler? See `Coerce`."
                    if critic.token_type.is_iterable_of(unit_sampler.token_type)
                    else ""
                )

        self.unit_sampler = unit_sampler
        self.critic = critic
        self.model = SequenceModel(
            unit_sampler=unit_sampler, critic=critic, max_tokens=float("inf")
        )

    async def __call__(self, n_particles, ess_threshold, max_tokens):
        try:
            original_max_tokens = self.model.max_tokens
            self.model.max_tokens = max_tokens
            particles = await smc_standard(
                model=self.model,
                n_particles=n_particles,
                ess_threshold=ess_threshold,
            )
        finally:
            self.model.max_tokens = original_max_tokens

        return Sequences(*_unpack_particles(particles))

    async def cleanup(self):
        await self.unit_sampler.cleanup()
