import numpy as np
from genlm_grammar import CFG, Earley, Float
from genlm_grammar.lark_interface import LarkStuff
from genlm_grammar.cfglm import _gen_nt

from genlm_control.constant import EOS
from genlm_control.potential.base import Potential


class WCFG(Potential):
    """A weighted context-free grammar potential."""

    def __init__(self, cfg):
        """
        Initialize the WCFG potential.

        Args:
            cfg (genlm_grammar.CFG): The context-free grammar configuration to use.
                The CFG must in the Float semiring.
        """
        if cfg.R is not Float:
            raise ValueError("cfg semiring must be Float")
        self.cfg = cfg  # cfg before prefix transform
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
            to_bytes (bool, optional): Whether to convert the WCFG terminals to indivudual bytes.
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
            context (list): The context to compute the weight for.

        Returns:
            (float): The log weight of the context under the WCFG.
        """
        w = self.model([*context, EOS])
        return np.log(w) if w > 0 else float("-inf")

    async def prefix(self, context):
        """
        Compute the log prefix weight of the context under the WCFG.

        Args:
            context (list): The context to compute the prefix weight for.

        Returns:
            (float): The log prefix weight of the context under the WCFG.
        """
        w = self.model(context)
        return np.log(w) if w > 0 else float("-inf")

    async def logw_next(self, context):
        """
        Compute the next token log weights given the context.

        Args:
            context (list): The context to compute the next token weights for.

        Returns:
            (LazyWeights): The log weights for the next tokens and EOS given the context.
        """
        ws = self.model.next_token_weights(self.model.chart(context))
        ws = ws.trim().normalize()
        log_ws = np.array(
            [np.log(ws[x]) if ws[x] > 0 else float("-inf") for x in self.decode_eos]
        )
        return self.make_lazy_weights(log_ws)

    def clear_cache(self):
        """Clear the internal cache of the parser."""
        self.model.clear_cache()

    def __repr__(self):
        return f"WCFG(cfg={self.cfg!r})"

    def _repr_html_(self):
        return self.cfg._repr_html_()

    def spawn(self):
        """Spawn a new WCFG."""
        return WCFG(self.cfg)


class BoolCFG(WCFG):
    """BoolCFG is a subclass of WCFG that represents a boolean context-free grammar."""

    @classmethod
    def from_lark(cls, lark_string, charset="core"):
        """
        Create a BoolCFG instance from a Lark grammar string.

        The output grammar will be defined at the byte-level.

        Args:
            lark_string (str): The Lark grammar string to parse.
            charset (str): The character set to use. Defaults to "core".
                See `genlm-grammar` documentation for more details.

        Returns:
            BoolCFG: An instance of BoolCFG created from the provided Lark grammar.
        """
        byte_cfg = LarkStuff(lark_string).byte_cfg(charset=charset)
        return cls(byte_cfg)

    async def complete(self, context):
        """
        Checks whether the context is accepted by the CFG.

        Args:
            context (list): The context to check.

        Returns:
            (float): Log weight for whether the context is accepted by the CFG.
        """
        w = self.model([*context, EOS])
        return 0 if w else float("-inf")

    async def prefix(self, context):
        """
        Checks whether the context is accepted as a prefix by the CFG, i.e.,
        whether there exists a completion to the context that is accepted by the CFG.

        Args:
            context (list): The context to check.

        Returns:
            (float): Log weight for whether the context is accepted as a prefix by the CFG.
        """
        w = self.model(context)
        return 0 if w > 0 else float("-inf")

    async def logw_next(self, context):
        """
        Compute the next token log weights given the context.

        Args:
            context (list): The context to compute the next token weights for.

        Returns:
            (LazyWeights): The log weights for the next tokens and EOS given the context.
        """
        ws = self.model.next_token_weights(self.model.chart(context))
        log_ws = np.array([0 if ws[x] > 0 else float("-inf") for x in self.decode_eos])
        return self.make_lazy_weights(log_ws)

    async def batch_logw_next(self, contexts):
        """
        Batch version of `logw_next`.

        Args:
            contexts (list): The list of contexts.

        Returns:
            (list): List of log-weights for next token, one per context.
        """
        Ws = []
        for context in contexts:
            ws = self.model.next_token_weights(self.model.chart(context))
            log_ws = np.array(
                [0 if ws[x] > 0 else float("-inf") for x in self.decode_eos]
            )
            Ws.append(self.make_lazy_weights(log_ws))
        return Ws

    def spawn(self):
        """Spawn a new BoolCFG."""
        return BoolCFG(self.cfg)

    def __repr__(self):
        return f"BoolCFG(cfg={self.cfg!r})"
