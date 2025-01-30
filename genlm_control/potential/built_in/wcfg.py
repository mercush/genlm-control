import numpy as np
from genlm_grammar import CFG, Earley, Float
from genlm_grammar.lark_interface import LarkStuff
from genlm_grammar.cfglm import locally_normalize, _gen_nt

from genlm_control.potential.base import Potential, EOS


class WCFG(Potential):
    """
    Represents a Weighted Context-Free Grammar (WCFG) potential.

    Args:
        cfg (genlm_grammar.CFG): The context-free grammar configuration to use.
        eos (bytes, optional): The end-of-sequence token. If not provided, a default
            value will be used.

    Methods:
        complete(context): Computes the log weight of the context under the WCFG.
        prefix(context): Computes the log prefix weight of the context under the WCFG.
        logp_next(context): Computes the log weights for the next tokens given the context.
        clear_cache(): Clears the internal cache of the grammar model.
    """

    def __init__(self, cfg):
        if cfg.R is not Float:
            raise ValueError("cfg semiring must be Float")
        self.cfg = (
            cfg  # cfg before prefix transform & normalization, augmented with eos
        )
        self.cfg_eos = self._add_eos(cfg, EOS)  # augmented with eos
        self.model = Earley(self.cfg_eos.prefix_grammar)
        super().__init__(vocabulary=list(cfg.V))

    @staticmethod
    def _add_eos(cfg, eos):
        S = _gen_nt("<START>")
        cfg_eos = cfg.spawn(S=S)
        cfg_eos.V.add(eos)
        cfg_eos.add(cfg.R.one, S, cfg.S, eos)
        for r in cfg:
            cfg_eos.add(r.w, r.head, *r.body)
        return cfg_eos

    @classmethod
    def from_string(cls, grammar, to_bytes=True, **kwargs):
        """Create a WCFG from a string.

        Args:
            grammar (str): The string grammar specification to create the WCFG from.
            to_bytes (bool, optional): Whether to convert the WCFG to bytes.
                Defaults to True.
            **kwargs: Additional arguments passed to the WCFG constructor.

        Returns:
            (WCFG): The created WCFG.
        """
        cfg = CFG.from_string(grammar, Float)
        if to_bytes:
            cfg = cfg.to_bytes()
        return cls(cfg, **kwargs)

    async def complete(self, context):
        """
        Compute the log weight of the context under the WCFG.

        Args:
            context (list[bytes]): The context to compute the weight for.

        Returns:
            (float): The log weight of the context under the WCFG.
        """
        w = self.model(list(context) + [EOS])
        return np.log(w) if w > 0 else float("-inf")

    async def prefix(self, context):
        """
        Compute the log prefix weight of the context under the WCFG.

        Args:
            context (list[bytes]): The context to compute the prefix weight for.

        Returns:
            (float): The log prefix weight of the context under the WCFG.
        """
        w = self.model(context)
        return np.log(w) if w > 0 else float("-inf")

    async def logp_next(self, context):
        """
        Compute the log weights for the next tokens given the context.

        Args:
            context (list[bytes]): The context to compute the next token weights for.

        Returns:
            (LazyWeights): The log weights for the next tokens given the context.
        """
        ws = self.model.next_token_weights(self.model.chart(context))
        ps = ws.trim().normalize()
        log_ps = np.array(
            [np.log(ps[x]) if ps[x] > 0 else float("-inf") for x in self.decode_eos]
        )
        return self.make_lazy_weights(log_ps)

    def clear_cache(self):
        """Clear the internal cache of the parser."""
        self.model.clear_cache()

    def __repr__(self):
        return f"WCFG(cfg={self.cfg!r})"

    def _repr_html_(self):
        return self.cfg._repr_html_()


class PCFG(WCFG):
    def __init__(self, cfg, eos=None, **kwargs):
        super().__init__(locally_normalize(cfg, **kwargs), eos=eos)


class BoolCFG(WCFG):
    @classmethod
    def from_lark(cls, lark_string, charset="core"):
        byte_cfg = LarkStuff(lark_string).byte_cfg(charset=charset)
        return cls(byte_cfg)

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

    async def logp_next(self, context):
        logp_next = await super().logp_next(context)
        return logp_next.spawn(
            new_weights=np.where(
                logp_next.weights > float("-inf"), 0, logp_next.weights
            )
        )

    def __repr__(self):
        return f"BoolCFG(cfg={self.cfg!r})"
