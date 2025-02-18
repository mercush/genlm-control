from hfppl import Model
import numpy as np
from genlm_grammar import Float
from genlm_control.constant import EOS
from arsenal.maths import logsumexp


class SequenceSampler(Model):
    def __init__(self, unit_sampler, critic=None, max_tokens=float("inf"), eos=EOS):
        super().__init__()
        self.context = []
        self.unit_sampler = unit_sampler
        self.max_tokens = max_tokens
        self.critic = critic
        self.eos = eos

        if self.critic:
            self.target = unit_sampler.target * self.critic
        else:
            self.target = unit_sampler.target

    async def start(self):
        # Init weight with empty sequence weight under target.
        self.score(await self.target.prefix([]))

    async def step(self):
        unit, logw, _ = await self.unit_sampler.sample(self.context)

        self.score(logw)
        self.context.append(unit)

        if self.critic:
            twist_amt = await self.critic.score(self.context)
            self.twist(twist_amt)

        self.max_tokens -= 1
        if self.max_tokens == 0 or self.context[-1] is self.eos:
            self.finish()
            return

    def immutable_properties(self):
        return set(["unit_sampler", "critic"])


class ParticleApproximation:
    def __init__(self, particles):
        self.particles = list(particles)
        self.size = len(particles)
        self.log_weights = np.array([p.weight for p in self.particles])
        self.log_total = logsumexp(self.log_weights)

        # log-marginal likelihood estimate (Note: need to exponentiate to have
        # an unbiased estimate of the true marginal likelihood).
        self.log_ml = np.log(np.mean(np.exp(self.log_weights)))

        # log-normalized weights
        self.log_normalized_weights = self.log_weights - self.log_total

        # Compute the effective sample size
        self.log_ess = -logsumexp(2 * self.log_normalized_weights)
        self.ess = np.exp(self.log_ess)

    @property
    def posterior(self):
        posterior = Float.chart()
        for p, prob in zip(self.particles, np.exp(self.log_normalized_weights)):
            if type(p.context[0]) is str:

                def joint_fct(x):
                    return "".join(x)
            elif type(p.context[0]) is bytes:

                def joint_fct(x):
                    return b"".join(x)
            else:

                def joint_fct(x):
                    return x

            posterior[joint_fct(p.context)] += prob

        return posterior.normalize().sort_descending()

    def __len__(self):
        return self.size

    def __iter__(self):
        return iter(self.particles)

    def __getitem__(self, i):
        return self.particles[i]

    def __str__(self):
        return str(self.posterior)

    def _repr_html_(self):
        return self.posterior._repr_html_()

    def show(self):
        for p in sorted(self, reverse=True):
            print(p)
