import torch
import numpy as np
from genlm_grammar import Float, Log
from arsenal.maths import logsumexp


# class InferenceEngine:
#    def __init__(self, unit_sampler, critic=None):
#        self.model = SequenceSampler(unit_sampler, critic)

#    async def __call__(
#        self,
#        n_particles,
#        ess_threshold=0.5,
#        max_tokens=float("inf"),
#        visualization_dir=None,
#        jsonl_file=None,
#    ):
#        self.model.max_tokens = max_tokens

#        particles = await smc_standard(
#            self.model,
#            n_particles,
#            ess_threshold,
#            visualization_dir,
#            jsonl_file,
#        )

#        return ParticleApproximation(particles)


class LazyWeights:
    def __init__(self, weights, encode, decode, log=True):
        assert len(weights) == len(decode)
        assert len(encode) == len(decode)

        self.weights = weights
        self.encode = encode
        self.decode = decode
        self.is_log = log

    def __getitem__(self, token):
        if token not in self.encode:
            return float("-inf") if self.is_log else 0
        return self.weights[self.encode[token]]

    def __len__(self):
        return len(self.weights)

    def __array__(self):
        raise NotImplementedError(
            "LazyWeights cannot be converted to a numpy array. "
            "If you want to combine multiple LazyWeights, use their weights attribute directly."
        )

    def keys(self):
        return self.decode

    def values(self):
        return self.weights

    def items(self):
        return zip(self.keys(), self.values())

    def normalize(self):
        if self.is_log:
            return self.spawn(self.weights - logsumexp(self.weights))
        else:
            return self.spawn(self.weights / np.sum(self.weights))

    def __mul__(self, other):
        if self.is_log:
            assert other.is_log
            return self.spawn(self.weights + other.weights)
        else:
            return self.spawn(self.weights * other.weights)

    def __add__(self, other):
        if self.is_log:
            assert other.is_log
            max_ab = np.maximum(self.weights, other.weights)
            weights = max_ab + np.log1p(np.exp(-np.abs(self.weights - other.weights)))
            return self.spawn(weights)
        else:
            return self.spawn(self.weights + other.weights)

    def spawn(self, new_weights, log=None):
        if log is None:
            log = self.is_log
        return LazyWeights(
            weights=new_weights, encode=self.encode, decode=self.decode, log=log
        )

    def materialize(self, top=None):
        weights = self.weights
        if top is not None:
            top_ws = weights.argsort()[-int(top) :]
        else:
            top_ws = weights.argsort()

        semiring = Log if self.is_log else Float

        chart = semiring.chart()
        for i in reversed(top_ws):
            chart[self.decode[i]] = weights[i]

        return chart

    def __repr__(self):
        return repr(self.materialize())

    def exp(self):
        assert self.is_log, "Exponentiating non-log weights"
        return self.spawn(np.exp(self.weights), log=False)

    def log(self):
        assert not self.is_log, "Taking the logarithm of non-log weights"
        return self.spawn(np.log(self.weights), log=True)

    def sum(self):
        if self.is_log:
            return logsumexp(self.weights)
        else:
            return np.sum(self.weights)

    def assert_equal(self, other, **kwargs):
        assert self.decode == other.decode
        np.testing.assert_allclose(self.weights, other.weights, **kwargs)

    def assert_equal_unordered(self, other, **kwargs):
        assert set(self.decode) == set(other.decode), "keys do not match"

        for x in self.decode:
            have, want = self[x], other[x]
            assert np.isclose(have, want, **kwargs), f"{x}: {have} != {want}"

    def sample(self):
        lws = self.normalize()

        if self.is_log:
            ps = np.exp(lws.weights)
        else:
            ps = lws.weights

        nonzero_idx = ps.nonzero()[0]
        nonzero_probs = ps[nonzero_idx]
        chosen_idx = nonzero_idx[np.random.multinomial(1, pvals=nonzero_probs).argmax()]

        return self.decode[chosen_idx], ps[chosen_idx]


def load_trie(V, backend=None, **kwargs):
    if backend is None:
        backend = "parallel" if torch.cuda.is_available() else "sequential"

    if backend == "parallel":
        from genlm_backend.trie import ParallelTokenCharacterTrie

        return ParallelTokenCharacterTrie(V, **kwargs)
    else:
        from genlm_backend.trie import TokenCharacterTrie

        return TokenCharacterTrie(V, **kwargs)


def load_async_trie(V, backend=None, **kwargs):
    from genlm_backend.trie import AsyncTokenCharacterTrie

    return AsyncTokenCharacterTrie(load_trie(V, backend, **kwargs))


def sample_categorical(pvals):
    # Helper function to sample from a categorical distribution
    # Much faster than np.random.multinomial for pvals with many zeros
    # TODO: test
    nonzero_idx = pvals.nonzero()[0]
    nonzero_probs = pvals[nonzero_idx]
    chosen_idx = nonzero_idx[np.random.multinomial(1, pvals=nonzero_probs).argmax()]
    return chosen_idx
