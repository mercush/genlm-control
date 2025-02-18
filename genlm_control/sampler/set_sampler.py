import numpy as np
from genlm_grammar import Float
from arsenal.maths import sample_dict
from arsenal.datastructures import LocatorMaxHeap
from abc import ABC, abstractmethod

from genlm_control.util import load_async_trie
from genlm_control.constant import EOS


class SetSampler(ABC):
    """Base class for set samplers.

    A set sampler samples a weighted set of tokens. The weight associated with each token is given as:

        target.logw_next(token | context) - log_inclusion_probability

    where log_inclusion_probability is the log of the probability the token was included in the sampled set.

    Attributes:
        target (Potential): The target potential with respect to which the set's weights are computed.
        token_type (TokenType): The type of tokens in the set.
    """

    def __init__(self, target):
        self.target = target
        self.token_type = self.target.token_type

    @abstractmethod
    async def sample_set(self, context: list) -> Float.chart:
        pass


class TrieSetSampler(SetSampler):
    """
    A set sampler that uses a trie to sample a weighted set of tokens.
    """

    def __init__(self, iterable_potential, item_potential, f):
        assert iterable_potential.token_type.is_iterable_of(item_potential.token_type)
        super().__init__(iterable_potential * item_potential.coerce(f=f))
        self.iterable_potential = iterable_potential
        self.item_potential = item_potential
        self.f = f
        self.trie = load_async_trie(self.iterable_potential.decode_eos)

    async def sample_set(self, context):
        raise NotImplementedError("Subclasses must implement sample_set")


class EagerSetSampler(TrieSetSampler):
    async def sample_set(self, context, draw=sample_dict):
        logws = await self.iterable_potential.logw_next(context)
        item_ws = await self.trie.mass_sum(logws.exp().weights)

        tokens = Float.chart()
        curr = self.trie.root
        subtokens = []
        logp, logw = 0, 0

        while True:
            children = self.trie.children[curr]
            item_w_curr = item_ws[curr]
            item_ws1 = Float.chart(
                {a: item_ws[c] / item_w_curr for a, c in children.items()}
            )

            if None in item_ws1:
                leaf = children[None]
                token = self.trie.leaf_to_word[leaf]
                tokens[token] = np.log(item_ws[leaf]) + logw - logp

            item_logws2 = await self.item_potential.logw_next(
                self.f(context) + subtokens
            )
            item_ws2 = item_logws2.exp().weights
            w_next = (item_ws1 * item_ws2).trim()

            if not w_next:
                break

            ps = w_next.normalize()
            b = draw(ps)
            logp += np.log(ps[b])
            logw += item_logws2[b]

            if b is EOS:
                assert not subtokens, "subtokens should be empty at EOS"
                children = self.trie.children[curr]
                assert None in children
                node_idx = children[None]
                tokens[EOS] = np.log(item_ws1[node_idx]) + logw - logp
                break

            subtokens.append(b)
            curr = children[b]

        return tokens


class TopKSetSampler(TrieSetSampler):
    def __init__(self, potential, guide, f, K):
        super().__init__(potential, guide, f)
        self.K = K

    async def sample_set(self, context, draw=sample_dict):
        logws = await self.potential.logw_next(context)
        mass = await self.trie.mass_max(logws.exp().weights)

        k = 0
        tokens = Float.chart()
        async for token, logw in self._lazy_enum(context, mass):
            tokens[token] = logw
            k += 1
            if k >= self.K:
                break

        if not self.K and len(tokens) == self.K:
            # compute distribution over wildcard tokens
            W_wc = Float.chart()
            for token, token_id in self.target.encode.items():
                if token in tokens:
                    continue
                w = mass[self.trie.token_id_to_leaf[token_id]]
                if w == 0:
                    continue
                W_wc[token] = w

            # if W_wc is non-empty, sample a wildcard token to ensure absolute continuity
            if W_wc:
                P_wc = W_wc.normalize()
                wildcard = draw(P_wc)
                logp_wc = np.log(P_wc[wildcard])
                w_guide_wc = self.guide.logw_next_seq(
                    self.f(context), self.f([wildcard])
                )
                tokens[wildcard] = np.log(W_wc[wildcard]) + w_guide_wc - logp_wc

        return tokens

    async def _lazy_enum(self, context, max_mass):
        agenda = LocatorMaxHeap()

        P = Float.chart()

        # initial conditions
        (token, node) = ([], self.trie.root)
        agenda[token, node, False] = 0
        P[node] = 0

        children = self.trie.children

        curr_priority = 0
        prev_best = 0
        while agenda:
            (token, node, done), score = agenda.popitem()

            assert score <= curr_priority
            curr_priority = score

            # terminal state
            if done:
                value = P[node] + max_mass[node]
                assert prev_best >= value
                prev_best = value
                yield (self.g(token), value)
                continue

            # Efficiently compute guide.p(x | context + token) for x ∈ guide.V.
            # These are individual characters that are aligned with the trie.
            p = await self.guide.logw_next(self.f(context) + token)

            for x, y in children[node].items():
                if x is None:
                    P_y = P[node]
                    P[y] = P_y
                    agenda[token, y, True] = P_y + max_mass[y]

                else:
                    P_y = P[node] + p[x]
                    if P_y == float("-inf"):
                        continue
                    P[y] = P_y
                    agenda[[*token, x], y, False] = P_y + max_mass[y]
