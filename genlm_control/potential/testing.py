import asyncio
import numpy as np
from genlm_control.constant import EOS


class PotentialTests:
    """A mixin class providing testing utilities for validating Potential implementations.

    This class provides methods to verify the mathematical consistency and correctness
    of Potential implementations through various assertions:

    - logw_next consistency: Verifies that token-level log weights are consistent with
      prefix and complete scores.
    - Autoregressive factorization: Validates that complete scores factor correctly as
      a sum of log token weights.
    - Batch consistency: Ensures batch operations produce identical results to
      their non-batch counterparts.

    All Potential instances inherit from this class to gain access to these
    testing utilities.
    """

    colors = {
        "green": "\033[92m",
        "yellow": "\033[93m",
        "blue": "\033[94m",
        "magenta": "\033[95m",
        "cyan": "\033[96m",
        "red": "\033[91m",
        "reset": "\033[0m",
    }

    async def assert_logw_next_consistency(
        self, context, rtol=1e-3, atol=1e-5, top=None, verbosity=0
    ):
        """
        Assert that logw_next is consistent with prefix and complete.

        For context $x_1, \ldots, x_n$, this checks (in log space) whether:
        $$
        \\texttt{logw_next}(x_i \mid x_{<i}) = \\texttt{score}(x_{1:i}) - \\texttt{prefix}(x_{<i})
        $$

        Args:
            context (list[bytes]): Context to test.
            rtol (float): Relative tolerance for floating point comparison.
            atol (float): Absolute tolerance for floating point comparison.
            top (int): Top-k tokens to test.
            verbosity (int): Verbosity level.

        Raises:
            AssertionError: If the logw_next is not consistent with prefix and complete.
        """
        top_logw_next = (await self.logw_next(context)).materialize(top=top)
        tokens = list(top_logw_next.keys())
        extended = [[*context, x] for x in tokens]

        context_w = await self.prefix(context)
        extended_ws = await self.batch_score(extended)

        wants = np.array([top_logw_next[x] for x in tokens])
        haves = extended_ws - context_w

        errors, valids = [], []
        for i, (want, have) in enumerate(zip(wants, haves)):
            abs_diff, rel_diff = self._compute_diff(want, have)
            info = (want, have, abs_diff, rel_diff, tokens[i])
            (valids if abs_diff <= atol and rel_diff <= rtol else errors).append(info)

        if valids and verbosity > 0:
            print(
                f"{self.colors['green']}logw_next consistency with context={context!r} satisfied for tokens:{self.colors['reset']}\n"
            )
            for valid in valids:
                want, have, abs_diff, rel_diff, token = valid
                print(
                    self._format_diff(want, have, abs_diff, rel_diff, atol, rtol, token)
                )

        if errors:
            error_msg = f"{self.colors['red']}logw_next consistency with context={context!r} not satisfied for tokens:{self.colors['reset']}\n\n"
            for error in errors:
                want, have, abs_diff, rel_diff, token = error
                error_msg += self._format_diff(
                    want, have, abs_diff, rel_diff, atol, rtol, token
                )
            raise AssertionError(error_msg)

    async def assert_autoreg_fact(self, context, rtol=1e-3, atol=1e-5, verbosity=0):
        """
        Assert that complete factors as an autoregressive sum of logw_nexts.

        For context $x_1, \\ldots, x_n$, this checks (in log space) whether:
        $$
        \\texttt{complete}(x_1, \\ldots, x_n) = -\\texttt{prefix}(\epsilon) + \\texttt{logw_next}(x_{1:n})[EOS] + \\sum_{i=1}^n \\texttt{logw_next}(x_{<i})[x_i]
        $$

        Args:
            context (list[bytes]): Context to test.
            rtol (float): Relative tolerance for floating point comparison.
            atol (float): Absolute tolerance for floating point comparison.
            verbosity (int): Verbosity level.

        Raises:
            AssertionError: If the autoregressive factorization is not satisfied.
        """
        want = (await self.complete(context)) - (await self.prefix([]))

        logw_next_results = await asyncio.gather(
            *[self.logw_next(context[:i]) for i in range(len(context))],
            self.logw_next(context),
        )

        have = (
            sum(logw_next_results[i][context[i]] for i in range(len(context)))
            + logw_next_results[-1][EOS]
        )

        abs_diff, rel_diff = self._compute_diff(want, have)
        if abs_diff > atol or rel_diff > rtol:
            error_msg = (
                f"{self.colors['red']}Factorization not satisfied for context {context}:{self.colors['reset']}\n"
                + self._format_diff(want, have, abs_diff, rel_diff, atol, rtol)
            )
            raise AssertionError(error_msg)

        if verbosity > 0:
            print(
                f"{self.colors['green']}Factorization property satisfied for context {context}:{self.colors['reset']}\n"
            )
            print(self._format_diff(want, have, abs_diff, rel_diff, atol, rtol))

    async def assert_batch_consistency(self, contexts, extensions=None, verbosity=0):
        """
        Assert that batch results are equal to non-batch results.

        Args:
            contexts (list[list[bytes]]): Contexts to test.
            extensions (list[bytes], optional): Extensions to test logw_next_seq methods.
                Defaults to None, in which case the logw_next_seq methods are not tested.
            verbosity (int): Verbosity level.

        Raises:
            AssertionError: If the batch results are not equal to the non-batch results.
        """
        batch_logw_nexts = await self.batch_logw_next(contexts)
        batch_scores = await self.batch_score(contexts)

        for i, context in enumerate(contexts):
            logw_next = await self.logw_next(context)
            try:
                np.testing.assert_array_equal(
                    batch_logw_nexts[i].weights, logw_next.weights
                )
                if verbosity > 0:
                    print(
                        f"{self.colors['green']}Batch logw_next consistency satisfied for context {context}:{self.colors['reset']}"
                    )
                    print(
                        f"{self.colors['green']}Non-batched: {logw_next.weights}\n"
                        + f"{self.colors['green']}Batched:     {batch_logw_nexts[i].weights}{self.colors['reset']}\n"
                    )
            except AssertionError:
                raise AssertionError(
                    f"{self.colors['red']}Batch logw_next mismatch for context {context}:{self.colors['reset']}\n"
                    + f"{self.colors['green']}Non-batched: {logw_next.weights}\n"
                    + f"{self.colors['red']}Batched:     {batch_logw_nexts[i].weights}{self.colors['reset']}"
                )

            score = await self.score(context)
            if batch_scores[i] != score:
                raise AssertionError(
                    f"{self.colors['red']}Batch score mismatch for context {context}:{self.colors['reset']}\n"
                    + f"{self.colors['green']}Non-batched: {score}\n"
                    + f"{self.colors['red']}Batched:     {batch_scores[i]}{self.colors['reset']}"
                )
            elif verbosity > 0:
                print(
                    f"{self.colors['green']}Batch score consistency satisfied for context {context}:{self.colors['reset']}"
                )
                print(
                    f"{self.colors['green']}Non-batched: {score}\n"
                    + f"{self.colors['green']}Batched:     {batch_scores[i]}{self.colors['reset']}\n"
                )

            if extensions:
                batch_logw_next_seqs = await self.batch_logw_next_seq(
                    context, extensions
                )
                for j, extension in enumerate(extensions):
                    logw_next_seq = await self.logw_next_seq(context, extension)
                    if batch_logw_next_seqs[j] != logw_next_seq:
                        raise AssertionError(
                            f"{self.colors['red']}Batch logw_next_seq mismatch for context {context} and extension {extension}:{self.colors['reset']}\n"
                            + f"{self.colors['green']}Non-batched: {logw_next_seq}{self.colors['reset']}\n"
                            + f"{self.colors['red']}Batched:    {batch_logw_next_seqs[j]}{self.colors['reset']}"
                        )
                    elif verbosity > 0:
                        print(
                            f"{self.colors['green']}Batch logw_next_seq consistency satisfied for context {context} and extension {extension}:{self.colors['reset']}"
                        )
                        print(
                            f"{self.colors['green']}Non-batched: {logw_next_seq}\n"
                            + f"{self.colors['green']}Batched:    {batch_logw_next_seqs[j]}{self.colors['reset']}\n"
                        )

    def _compute_diff(self, want, have):
        is_inf = want == float("-inf") and have == float("-inf")
        abs_diff = 0 if is_inf else abs(want - have)
        if want == 0:
            rel_diff = 0 if have == 0 else float("inf")
        else:
            rel_diff = 0 if is_inf else abs((want - have) / want)
        return abs_diff, rel_diff

    def _format_diff(self, want, have, abs_diff, rel_diff, atol, rtol, token=None):
        abs_diff_str = (
            f"{self.colors['cyan']}Abs Diff: {abs_diff:.6f} <= {atol=}\033[0m"
        )
        rel_diff_str = (
            f"{self.colors['magenta']}Rel Diff: {rel_diff:.6f} <= {rtol=}\033[0m"
        )

        want_str = f"{self.colors['green']}Expected: {want:.6f}{self.colors['reset']}"
        have_clr = (
            self.colors["yellow"]
            if abs_diff <= atol and rel_diff <= rtol
            else self.colors["red"]
        )
        have_str = f"{have_clr}Actual:   {have:.6f}{self.colors['reset']}"

        if abs_diff <= atol:
            abs_diff_str = f"{self.colors['green']}Abs Diff: {abs_diff:.6f} <= {atol=}{self.colors['reset']}"
        else:
            abs_diff_str = f"{self.colors['red']}Abs Diff: {abs_diff:.6f} > {atol=}{self.colors['reset']}"

        if rel_diff <= rtol:
            rel_diff_str = f"{self.colors['green']}Rel Diff: {rel_diff:.6f} <= {rtol=}{self.colors['reset']}"
        else:
            rel_diff_str = f"{self.colors['red']}Rel Diff: {rel_diff:.6f} > {rtol=}{self.colors['reset']}"

        token_str = (
            f"{self.colors['blue']}Token:    {token}{self.colors['reset']}\n"
            if token
            else ""
        )
        return f"{token_str}{want_str}\n{have_str}\n{abs_diff_str}\n{rel_diff_str}\n\n"
