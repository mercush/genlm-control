import numpy as np
from genlm_grammar import Float, Log
from arsenal.maths import logsumexp


class LazyWeights:
    def __init__(self, weights, encode, decode, log=True):
        assert len(weights) == len(decode)
        assert len(encode) == len(decode)

        self.weights = weights
        self.encode = encode
        self.decode = decode
        self.log = log

    def __getitem__(self, token):
        if token not in self.encode:
            return float("-inf") if self.log else 0
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
        if self.log:
            return self.spawn(self.weights - logsumexp(self.weights))
        else:
            return self.spawn(self.weights / np.sum(self.weights))

    def __mul__(self, other):
        if self.log:
            assert self.other.log
            return self.spawn(self.weights + other.weights)
        else:
            return self.spawn(self.weights * other.weights)

    def __add__(self, other):
        if self.log:
            assert self.other.log
            max_ab = np.maximum(self.weights, other.weights)
            weights = max_ab + np.log1p(np.exp(-np.abs(self.weights - other.weights)))
            return self.spawn(weights)
        else:
            return self.spawn(self.weights + other.weights)

    def spawn(self, new_weights):
        return LazyWeights(
            weights=new_weights, encode=self.encode, decode=self.decode, log=self.log
        )

    def materialize(self, top=None):
        weights = self.weights
        if top is not None:
            top_ws = weights.argsort()[-int(top) :]
        else:
            top_ws = weights.argsort()

        if self.log:
            chart = Log.chart()
        else:
            chart = Float.chart()

        for i in reversed(top_ws):
            chart[self.decode[i]] = weights[i]

        return chart

    def __repr__(self):
        return repr(self.materialize())
