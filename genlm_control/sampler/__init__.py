from .set import EagerSetSampler, TopKSetSampler
from .token import DirectTokenSampler, SetTokenSampler
from .sequence import SMC, Importance, SequenceModel

__all__ = [
    "EagerSetSampler",
    "TopKSetSampler",
    "DirectTokenSampler",
    "SetTokenSampler",
    "Importance",
    "SMC",
    "SequenceModel",
]


def direct_token_sampler(potential):
    return DirectTokenSampler(potential)


def eager_token_sampler(iter_potential, item_potential):
    return SetTokenSampler(EagerSetSampler(iter_potential, item_potential))


def topk_token_sampler(iter_potential, item_potential, K):
    return SetTokenSampler(TopKSetSampler(iter_potential, item_potential, K))
