import asyncio
import numpy as np
from abc import ABC, abstractmethod
from arsenal.maths import logsumexp

from genlm_control.constant import EOS
from genlm_control.util import LazyWeights
from genlm_control.operators import PotentialOps
from genlm_control.typing import infer_vocabulary_type
from genlm_control.potential.testing import PotentialTests


class Potential(ABC, PotentialOps, PotentialTests):
    """Abstract base class for potentials.

    A Potential represents a weighted language over a vocabulary.

    Subclasses must minimally implement methods to assess the weight of a sequence as a member of the language (`complete`) and
    as a prefix of the language (`prefix`).
    """

    def __init__(self, vocabulary, token_type=None):
        """
        Initialize the potential.

        Args:
            vocabulary (list): List of tokens that make up the vocabulary.
            token_type (TokenType, optional): Optional TokenType of all elements of the vocabulary.
                If None, will be inferred from vocabulary.

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

        self.token_type = token_type
        self.decode = vocabulary
        self.encode = {}
        for i, x in enumerate(vocabulary):
            if x in self.encode:
                raise ValueError(f"Duplicate token {x!r} found in vocabulary")
            self.encode[x] = i
        self.decode_eos = self.decode + [EOS]
        self.encode_eos = {**self.encode, **{EOS: len(self.decode)}}

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
        """Compute the weights each token in the vocabulary and the special EOS token given `context`.

        The log weight of a token x is computed as:
        $$
        w(x \mid \text{context}) = \text{score}(\text{context} + x) - \text{prefix}(\text{context})
        $$

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

    async def logw_next_seq(self, context, extension):
        """Assess the weight of `extension` given `context`.

        `extension` may optionally include the special EOS token at the end.

        Args:
            context (list): Sequence of tokens to condition on.
            extension (list): Sequence of tokens to score.

        Returns:
            (LazyWeights): Log weight of `extension` given `context`.
        """
        return (await self.batch_logw_next_seq(context, [extension]))[0]

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

        Computes the log weights for each token in the vocabulary and EOS
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

        extended_contexts = [[*context, x] for context in contexts for x in self.decode]

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

    async def batch_logw_next_seq(self, context, extensions):
        """Batched equivalent to `logw_next_seq`.

        Args:
            context (list): Sequence of tokens to condition on.
            extensions (list): List of sequences of tokens to score.

        Returns:
            (np.array): Array of log weights for each extension.

        Raises:
            ValueError: If context has zero weight (log weight of -inf) under `prefix`.
        """
        if not extensions:
            raise ValueError("Extensions must be non-empty.")

        prefix = await self.prefix(context)
        if prefix == float("-inf"):
            raise ValueError(f"Context {context!r} has weight zero under `prefix`.")

        scores = await self.batch_score(
            [[*context, *extension] for extension in extensions]
        )
        return scores - prefix

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

    def spawn(self):
        """
        Spawn a fresh instance of the potential.

        This method is not required by default, but may be implemented by subclasses
        to support CPU-parallelism using multiprocessing.
        """
        raise NotImplementedError(
            "Potential.spawn() must be implemented by subclasses."
        )

    async def sample(self, context=None, max_tokens=float("inf"), n_samples=1):
        """Generate properly weighted samples from the potential.

        Args:
            context (list, optional): Initial context. Defaults to None.
            max_tokens (int, optional): Maximum number of tokens to generate. Defaults to float("inf").
            n_samples (int, optional): Number of samples to generate. Defaults to 1.

        Returns:
            tuple[list[list], np.ndarray]: Tuple of (sequences, log weights), where sequences is a list of
                token sequences and log weights is an array of corresponding log weights.

        Raises:
            ValueError: If n_samples < 1 or max_tokens < 1
        """
        if n_samples < 1:
            raise ValueError("n_samples must be at least 1")
        if max_tokens < 1:
            raise ValueError("max_tokens must be at least 1")

        context = [] if context is None else list(context)
        contexts = [context.copy() for _ in range(n_samples)]
        log_ws = np.zeros(n_samples, dtype=np.float64)
        active = np.ones(n_samples, dtype=bool)

        while np.any(active):
            active_idxs = np.where(active)[0]
            logw_nexts = await self.batch_logw_next([contexts[i] for i in active_idxs])

            for i, logw_next in enumerate(logw_nexts):
                idx = active_idxs[i]
                W = logw_next.weights
                Z = logsumexp(W)
                p_next = np.exp(W - Z)

                # Handle numerical precision issues
                p_next = p_next / np.sum(p_next)

                next_token = np.random.choice(self.decode_eos, p=p_next)
                contexts[idx].append(next_token)
                log_ws[idx] += Z

                print(contexts[idx], log_ws[idx])

                if next_token is EOS or len(contexts[idx]) >= max_tokens:
                    active[idx] = False

        return contexts, log_ws
