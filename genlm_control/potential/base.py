import asyncio
import numpy as np
from abc import ABC, abstractmethod

from genlm_control.constant import EOS, EndOfSequence
from genlm_control.util import LazyWeights
from genlm_control.typing import infer_vocabulary_type
from genlm_control.potential.operators import PotentialOps
from genlm_control.potential.testing import PotentialTests


class Potential(ABC, PotentialOps, PotentialTests):
    """Abstract base class for potentials.

    A Potential represents a weighted language over a vocabulary. Subclasses must minimally implement methods
    to assess the weight of a sequence as a member of the language (`complete`) and as a prefix of the language (`prefix`).

    Other methods come with default implementations, but may be overridden by subclasses.
    """

    def __init__(self, vocabulary, token_type=None, eos=None):
        """
        Initialize the potential.

        Args:
            vocabulary (list): List of tokens that make up the vocabulary.
            token_type (TokenType, optional): Optional TokenType of all elements of the vocabulary.
                If None, will be inferred from vocabulary.
            eos (EndOfSequence, optional): Special token to use as end-of-sequence. Defaults to `EOS`.
                In general, this should not be set by users.

        Raises:
            ValueError: If vocabulary is empty.
            TypeError: If vocabulary contains tokens which are not of `token_type`.
        """
        if not vocabulary:
            raise ValueError("Vocabulary cannot be empty")

        if token_type is None:
            token_type = infer_vocabulary_type(vocabulary)

        if not all(token_type.check(x) for x in vocabulary):
            raise TypeError(f"Tokens in vocabulary must be of type {token_type}.")

        if eos is None:
            eos = EOS
        else:
            assert isinstance(eos, EndOfSequence)

        self.token_type = token_type
        self.decode = vocabulary
        self.encode = {}
        for i, x in enumerate(vocabulary):
            if x in self.encode:
                raise ValueError(f"Duplicate token {x!r} found in vocabulary")
            self.encode[x] = i
        self.decode_eos = self.decode + [eos]
        self.encode_eos = {**self.encode, **{eos: len(self.decode)}}

    ####################
    # Instance methods #
    ####################

    @abstractmethod
    async def complete(self, context):
        """Assess the weight of `context` as a member of the language.

        Args:
            context (list): Sequence of tokens to score.

        Returns:
            (float): Log weight of the context under the language.
        """
        pass

    @abstractmethod
    async def prefix(self, context):
        """Assess the weight of `context` as a prefix of the language.

        Args:
            context (list): Sequence of tokens to score.

        Returns:
            (float): Log weight of the context as a prefix of the language.
        """
        pass

    async def score(self, context):
        """Assess the weight of `context` based on EOS-termination.

        Dispatches to `complete` if `context` ends with `EOS`, otherwise to `prefix`.

        Args:
            context (list): Sequence of tokens to score.

        Returns:
            (float): Log weight of the context, either as a prefix or complete sequence.
        """
        return (await self.batch_score([context]))[0]

    async def logw_next(self, context):
        """Compute the next-token weights of each token in the vocabulary and the special EOS token given `context`.

        Args:
            context (list): Sequence of tokens to condition on.

        Returns:
            (LazyWeights): Weights of each token in the vocabulary and EOS.
        """
        ctx_log_w = await self.prefix(context)

        if ctx_log_w == float("-inf"):
            raise ValueError(f"Context {context!r} has weight zero under `prefix`.")

        scores = await self.batch_score([[*context, x] for x in self.decode_eos])
        logws = scores - ctx_log_w

        return self.make_lazy_weights(logws)

    ###################
    # Batched methods #
    ###################

    async def batch_complete(self, contexts):
        """Batched equivalent to `complete`.

        Assess the weight of each context as a complete sequence of the language.

        Args:
            contexts (list): List of sequences of tokens.

        Returns:
            (np.array): Array of log weights for each context.
        """
        if not contexts:
            raise ValueError("Contexts must be non-empty.")

        return np.array(
            await asyncio.gather(*[self.complete(context) for context in contexts])
        )

    async def batch_prefix(self, contexts):
        """Batched equivalent to `prefix`.

        Assess the weight of each context as a prefix of the language.

        Args:
            contexts (list): List of sequences of tokens.

        Returns:
            (np.array): Array of log weights for each context.
        """
        if not contexts:
            raise ValueError("Contexts must be non-empty.")

        return np.array(
            await asyncio.gather(*[self.prefix(context) for context in contexts])
        )

    async def batch_score(self, contexts):
        """Batched equivalent to `score`.

        Assess the weight of each context based on EOS-termination.

        Args:
            contexts (list): List of sequences of tokens.

        Returns:
            (np.array): Array of log weights for each context.
        """
        if not contexts:
            raise ValueError("Contexts must be non-empty.")

        complete, prefix = [], []
        complete_indices, prefix_indices = [], []

        for i, context in enumerate(contexts):
            if context and context[-1] is EOS:
                complete.append(context[:-1])
                complete_indices.append(i)
            else:
                prefix.append(context)
                prefix_indices.append(i)

        complete_scores = (
            await self.batch_complete(complete) if complete else np.array([])
        )
        prefix_scores = await self.batch_prefix(prefix) if prefix else np.array([])

        results = np.empty(len(contexts))
        if len(complete_scores) > 0:
            results[complete_indices] = complete_scores
        if len(prefix_scores) > 0:
            results[prefix_indices] = prefix_scores

        return results

    async def batch_logw_next(self, contexts):
        """Batched equivalent to `logw_next`.

        Computes the next-token weights of each token in the vocabulary and EOS
        given each context in the batch.

        Args:
            contexts (list): List of sequences of tokens.

        Returns:
            (list): List of LazyWeights objects, one for each context.

        Raises:
            ValueError: If any context has zero weight (log weight of -inf) under `prefix`.
        """
        if not contexts:
            raise ValueError("Contexts must be non-empty.")

        num_contexts = len(contexts)
        vocab_size = len(self.decode)

        # Slow!!
        extended_contexts = [
            list(context) + [x] for context in contexts for x in self.decode
        ]

        complete_scores, all_prefix_scores = await asyncio.gather(
            self.batch_complete(contexts),
            self.batch_prefix(contexts + extended_contexts),
        )

        base_scores = all_prefix_scores[:num_contexts]
        extension_scores = all_prefix_scores[num_contexts:]

        batch_logws = []
        for i in range(num_contexts):
            base_score = base_scores[i]
            if base_score == float("-inf"):
                raise ValueError(
                    f"Context {contexts[i]!r} has weight zero under `prefix`."
                )

            logws = np.zeros(len(self.decode_eos))
            start = i * vocab_size
            logws[:-1] = extension_scores[start : start + vocab_size] - base_score
            logws[-1] = complete_scores[i] - base_score

            batch_logws.append(self.make_lazy_weights(logws))

        return batch_logws

    #############
    # Utilities #
    #############

    def make_lazy_weights(self, weights, log=True):
        """Helper method to create a LazyWeights object over the potential's vocabulary and EOS.

        Args:
            weights (np.array): Array of weights.
            log (bool, optional): Whether the weights are in log space. Defaults to True.

        Returns:
            (LazyWeights): LazyWeights object.
        """
        return LazyWeights(
            weights=weights, encode=self.encode_eos, decode=self.decode_eos, log=log
        )

    def alloc_logws(self, default=float("-inf")):
        """Allocate a new array of log weights for the potential's vocabulary and EOS.

        Args:
            default (float, optional): Default log weight. Defaults to -inf.

        Returns:
            (np.array): Array of length `len(self.decode_eos)` filled with `default`.
        """
        return np.full((len(self.decode_eos),), default)

    def spawn(self):
        """
        Spawn a fresh instance of the potential.

        This method is not required by default, but may be implemented by subclasses
        to support CPU-parallelism using (`MultiProcPotential`)[genlm_control.potential.multi_proc.MultiProcPotential].
        """
        raise NotImplementedError(
            "Potential.spawn() must be implemented by subclasses."
        )
