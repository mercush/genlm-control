import string
import numpy as np

from genlm_grammar import WFSA as BaseWFSA, Float
from genlm_grammar.lark_interface import interegular_to_wfsa

from genlm_control.util import LazyWeights
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

    def _consume_prefix(self, bs):
        wfsa = self.wfsa.epsremove
        prev = wfsa.start
        for b in bs:
            curr = wfsa.R.chart()
            for i in prev:
                for j, w in wfsa.arcs(i, b):
                    curr[j] += prev[i] * w
            prev = curr
        return prev

    async def complete(self, context):
        w = self.wfsa(context)
        return np.log(w) if w > 0 else float('-inf')

    async def prefix(self, context):
        next_state_ws = self._consume_prefix(context)
        w = next_state_ws.sum() # ? 
        return np.log(w) if w > 0 else float('-inf')

    async def logp_next(self, context):
        """Returns next token log probabilities after consuming context.
        
        Args:
            context (bytes): Input sequence
            
        Returns:
            (LazyWeights): Log-probabilities for next token.
        """
        prev = self._consume_prefix(context)
        ws = self.wfsa.R.chart()
        for i in prev:
            for b, _, w in self.wfsa.epsremove.arcs(i=i):
                ws[b] += prev[i] * w

        ws[EOS] = self.wfsa.R.zero
        for j, w in self.wfsa.epsremove.F:
            ws[EOS] += prev[j] * w

        ps = ws.trim().normalize()

        log_ps = np.array([
            np.log(ps[x]) if ps[x] > 0 else float('-inf') 
            for x in self.decode_eos
        ])

        return self.make_lazy_weights(log_ps)

    def _repr_svg_(self):
        return self.wfsa._repr_svg_()

    def __repr__(self):
        return f"WFSA(wfsa={self.wfsa!r})"


class BoolFSA(WFSA):
    """ Boolean FSA potential. """

    async def prefix(self, context):
        prefix_w = await super().prefix(context)
        if prefix_w > float('-inf'):
            return 0
        return float('-inf')

    async def complete(self, context):
        complete_w = await super().complete(context)
        if complete_w > float('-inf'):
            return 0
        return float('-inf')

    async def logp_next(self, context):
        logp_next = await super().logp_next(context)
        return logp_next.spawn(
            new_weights=np.where(
                logp_next.weights > float('-inf'), 0, logp_next.weights
            )
        )

    def __repr__(self):
        return f"BoolFSA(wfsa={self.wfsa!r})"
