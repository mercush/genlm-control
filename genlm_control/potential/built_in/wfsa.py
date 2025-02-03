import string
import numpy as np

from genlm_grammar import Float
from genlm_grammar.lark_interface import interegular_to_wfsa

from genlm_control.potential.base import Potential, EOS


class WFSA(Potential):
    """
    WFSA potential class that models Potentials based on WFSAs. Wraps a `genlm_grammar.WFSA`.

    Attributes:
        wfsa (genlm_grammar.WFSA): The weighted finite state automaton used for potential calculations.
            Any output weights will be converted to log space.
        eos (bytes): End-of-sequence token.
    """

    def __init__(self, wfsa):
        self.wfsa = wfsa
        if wfsa.R is not Float:
            raise ValueError("Float semiring is required for WFSA potentials")
        super().__init__(vocabulary=list(self.wfsa.alphabet))

    @classmethod
    def from_regex(cls, pattern, charset=None, to_bytes=True):
        """
        Create a WFSA from a regex pattern.

        Args:
            pattern (str): The regex pattern to convert into a WFSA.
            eos (bytes|str): The end-of-sequence token to be used in the WFSA.
            charset (set): The character set to use for negative character classes.
                Defaults to characters in string.printable.
            to_bytes (bool): Whether to convert the WFSA transitions to bytes.
                Defaults to True. When set to False, the WFSA transitions will be strings.

        Returns:
            WFSA: An instance of the WFSA class initialized with the generated WFSA.

        Note:
            Uses probabilistic transitions.
        """
        charset = charset or set(string.printable)
        wfsa = interegular_to_wfsa(pattern, charset=charset)
        if to_bytes:
            wfsa = wfsa.to_bytes()
        return cls(wfsa=wfsa)

    def _consume(self, bs, start=None):
        wfsa = self.wfsa.epsremove
        prev = start or wfsa.start
        for b in bs:
            curr = wfsa.R.chart()
            for i in prev:
                for j, w in wfsa.arcs(i, b):
                    curr[j] += prev[i] * w
            prev = curr
        return prev

    async def complete(self, context):
        w = self.wfsa(context)
        return np.log(w) if w > 0 else float("-inf")

    async def prefix(self, context):
        if not context:  # XXX ugh
            return 0
        curr = self._consume(context)
        bkwd = self.wfsa.epsremove.backward
        w = sum(curr[i] * bkwd[i] for i in curr)
        return np.log(w) if w > 0 else float("-inf")

    async def logw_next(self, context):
        """Returns next token log probabilities after consuming context.

        Args:
            context (bytes): Input sequence

        Returns:
            (LazyWeights): Log-probabilities for next token.
        """
        curr = self._consume(context)
        bkwd = self.wfsa.epsremove.backward

        if context:
            ctx_w = sum(curr[i] * bkwd[i] for i in curr)
        else:
            ctx_w = 1  # XXX ugh

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


class BoolFSA(WFSA):
    """Boolean FSA potential."""

    async def prefix(self, context):
        prefix_w = await super().prefix(context)
        if prefix_w > float("-inf"):
            return 0
        return float("-inf")

    async def complete(self, context):
        complete_w = await super().complete(context)
        if complete_w > float("-inf"):
            return 0
        return float("-inf")

    async def logw_next(self, context):
        logw_next = await super().logw_next(context)
        return logw_next.spawn(
            new_weights=np.where(
                logw_next.weights > float("-inf"), 0, logw_next.weights
            )
        )

    async def batch_logw_next(self, contexts):
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
