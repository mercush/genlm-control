import string
import numpy as np

from genlm_grammar import Float
from genlm_grammar.lark_interface import interegular_to_wfsa

from genlm_control.potential.base import Potential, EOS


class WFSA(Potential):
    """
    A weighted finite state automaton (WFSA) potential.

    This class wraps a `genlm_grammar.WFSA` and provides methods for computing the log-weight of a context,
    the prefix log-weight of a context, and the log-weights of the next token given a context.

    Attributes:
        wfsa (genlm_grammar.WFSA): The weighted finite state automaton used for potential calculations.
            Any output weights will be converted to log space.
    """

    def __init__(self, wfsa):
        """
        Initializes the WFSA potential.

        Args:
            wfsa (genlm_grammar.WFSA): The weighted finite state automaton.

        Raises:
            ValueError: If the semiring of the provided WFSA is not Float.
        """
        self.wfsa = wfsa
        if wfsa.R is not Float:
            raise ValueError("Float semiring is required for WFSA potentials")
        self.cache = {(): wfsa.epsremove.start}
        super().__init__(vocabulary=list(self.wfsa.alphabet))

    @classmethod
    def from_regex(cls, pattern, charset=None, to_bytes=True):
        """
        Create a WFSA from a regex pattern.

        Args:
            pattern (str): The regex pattern to convert into a WFSA.
            charset (set): The character set to use for negative character classes.
                Defaults to characters in string.printable.
            to_bytes (bool): Whether to convert the WFSA transitions to bytes.
                Defaults to True. When set to False, the WFSA transitions will be strings.

        Returns:
            (WFSA): An instance of the WFSA class.

        Note:
            The transition weights are automatically normalized to form a probability distribution.
            For each state, the weights of all outgoing transitions (including final state transitions)
            sum to 1.0. This means if a state has n possible transitions, each transition will have
            weight 1/n. To create a WFSA from a regex with non-probabilistic transitions, use `BoolFSA`.
        """
        charset = charset or set(string.printable)
        wfsa = interegular_to_wfsa(pattern, charset=charset)
        if to_bytes:
            wfsa = wfsa.to_bytes()
        return cls(wfsa=wfsa)

    def _consume(self, bs):
        # XXX implement cache eviction
        bs = tuple(bs)

        try:
            return self.cache[bs]
        except KeyError:
            pass

        wfsa = self.wfsa.epsremove
        curr = wfsa.R.chart()
        prev = self._consume(bs[:-1])
        for i in prev:
            for j, w in wfsa.arcs(i, bs[-1]):
                curr[j] += prev[i] * w

        self.cache[bs] = curr

        return curr

    async def complete(self, context):
        """
        Computes the log weight of the context under the weighted language represented by the WFSA.

        For example, if the WFSA accepts "cat" and "car" with weights $w_{cat}$ and $w_{car}$:\n
        - `complete("c")` returns $-\\infty$ since this sequence is not accepted by the WFSA\n
        - `complete("cat")` returns $\\log(w_{cat})$\n
        - `complete("d")` returns $-\\infty$ since this sequence is not accepted by the WFSA

        Args:
            context (list): A sequence of tokens in the WFSA's alphabet.

        Returns:
            (float): Log weight of context under the WFSA.
        """
        w = self.wfsa(context)
        return np.log(w) if w > 0 else float("-inf")

    async def prefix(self, context):
        """
        Computes the prefix log weight of `context` under the WFSA.

        This corresponds to the log of the sum of the weights of all sequences with prefix `context`.

        For example, if the WFSA accepts "cat" and "car" with weights $w_{cat}$ and $w_{car}$:\n
        - `prefix("c")` returns $\\log(w_{cat} + w_{car})$\n
        - `prefix("ca")` returns $\\log(w_{cat})$\n
        - `prefix("d")` returns $-\\infty$ since the WFSA does not accept any sequences with prefix "d"

        Args:
            context (list): A sequence of tokens in the WFSA's alphabet.

        Returns:
            (float): Log weight of `context` as a prefix under the WFSA.
        """
        curr = self._consume(context)
        bkwd = self.wfsa.epsremove.backward
        w = sum(curr[i] * bkwd[i] for i in curr)
        return np.log(w) if w > 0 else float("-inf")

    async def logw_next(self, context):
        """Returns next token log weights given `context`.

        Args:
            context (list): A sequence of tokens in the WFSA's alphabet.

        Returns:
            (LazyWeights): Log-weights for next token and EOS.
        """
        curr = self._consume(context)
        bkwd = self.wfsa.epsremove.backward

        ctx_w = sum(curr[i] * bkwd[i] for i in curr)

        if ctx_w == 0:
            raise ValueError(f"Context {context!r} has zero weight.")

        log_ctx_w = np.log(ctx_w)

        ws = self.wfsa.R.chart()
        for i in curr:
            for b, j, w in self.wfsa.epsremove.arcs(i=i):
                ws[b] += curr[i] * w * bkwd[j]

        ws[EOS] = self.wfsa.R.zero
        for j, w in self.wfsa.epsremove.F:
            ws[EOS] += curr[j] * w

        log_ws = np.array(
            [
                np.log(ws[b]) - log_ctx_w if ws[b] > 0 else float("-inf")
                for b in self.decode_eos
            ]
        )

        return self.make_lazy_weights(log_ws)

    def _repr_svg_(self):
        return self.wfsa._repr_svg_()

    def __repr__(self):
        return f"WFSA(wfsa={self.wfsa!r})"

    def spawn(self):
        cls = type(self)
        return cls(wfsa=self.wfsa)

    def clear_cache(self):
        self.cache = {(): self.wfsa.epsremove.start}


class BoolFSA(WFSA):
    """Boolean FSA potential."""

    async def prefix(self, context):
        """
        Computes whether the context is accepted as a prefix by the FSA.

        Args:
            context (list): A sequence of tokens in the WFSA's alphabet.

        Returns:
            (float): `0` if the context is accepted as a prefix, `-inf` otherwise.
        """
        prefix_w = await super().prefix(context)
        if prefix_w > float("-inf"):
            return 0
        return float("-inf")

    async def complete(self, context):
        """
        Computes whether the context is accepted by the FSA.

        Args:
            context (list): A sequence of tokens in the WFSA's alphabet.

        Returns:
            (float): `0` if the context is accepted, `-inf` otherwise.
        """
        complete_w = await super().complete(context)
        if complete_w > float("-inf"):
            return 0
        return float("-inf")

    async def logw_next(self, context):
        """
        Returns next token log weights given `context`.

        Args:
            context (list): A sequence of tokens in the WFSA's alphabet.

        Returns:
            (LazyWeights): Boolean log-weights for next token.
        """
        logw_next = await super().logw_next(context)
        return logw_next.spawn(
            new_weights=np.where(
                logw_next.weights > float("-inf"), 0, logw_next.weights
            )
        )

    async def batch_logw_next(self, contexts):
        """
        Returns next token log weights for a batch of contexts.

        Args:
            contexts (list): The list of contexts.

        Returns:
            (list): List of log-weights for next token, one per context.
        """
        logw_nexts = await super().batch_logw_next(contexts)
        return [
            logw_next.spawn(
                new_weights=np.where(
                    logw_next.weights > float("-inf"), 0, logw_next.weights
                )
            )
            for logw_next in logw_nexts
        ]

    def __repr__(self):
        return f"BoolFSA(wfsa={self.wfsa!r})"
