import torch
import warnings
import numpy as np
from typing import NamedTuple
from arsenal.maths import logsumexp
#from genlm_backend.llm import load_model_by_name

from genlm_control.util import LazyWeights
from genlm_control.potential.base import Potential

from genlm_backend.llm import AsyncVirtualLM, AsyncTransformer, MockAsyncLM

def load_model_by_name(name, backend, **kwargs): # REMOVE
    if backend == 'vllm':
        model_cls = AsyncVirtualLM
    elif backend == 'hf':
        model_cls = AsyncTransformer 
    elif backend == 'mock':
        model_cls = MockAsyncLM
    else:
        raise ValueError(f"Unknown backend: {backend}. Must be one of ['vllm', 'hf', 'mock']")

    return model_cls.from_name(name, **kwargs)


class TokenMappings(NamedTuple):
    """Container for token mappings between bytes and model IDs in a language model."""
    decode: list[bytes]       # token_id -> bytes
    encode: dict[bytes, int]  # bytes -> token_id
    eos_idxs: list[int]       # IDs of EOS tokens

    @classmethod
    def create(cls, decode, eos_tokens):
        encode = {x: i for i, x in enumerate(decode)}
        if not all(eos in encode for eos in eos_tokens):
            raise ValueError(f"EOS token {eos} not in language model vocabulary")
        eos_idxs = [encode[eos] for eos in eos_tokens]
        return cls(decode=decode, encode=encode, eos_idxs=eos_idxs)


class PromptedLLM(Potential):
    """A potential for a prompted language model.
    
    This class wraps a language model and allows computing next token probabilities
    conditioned on both a context and a fixed prompt prefix.
    
    Args:
        llm (AsyncLM): The language model to use
        prompt (str|list[int], optional): Optional prompt to use as a prompt prefix for all input contexts
        eos_tokens (list[bytes], optional): List of tokens to treat as end-of-sequence tokens.
            Defaults to the EOS token of the language model. 
    """
    def __init__(self, llm, prompt=None, eos_tokens=None):
        self.model = llm    
        self.prompt_ids = []

        if not eos_tokens:
            eos_tokens = [llm.byte_vocab[self.model.tokenizer.eos_token_id]]

        self.token_maps = TokenMappings.create(
            decode=llm.byte_vocab, eos_tokens=eos_tokens
        )

        V = [x for x in self.token_maps.decode if x not in eos_tokens]
        
        super().__init__(vocabulary=V)

        if prompt is not None:
            self.prompt = prompt

    @classmethod
    def from_name(cls, name, backend=None, eos_tokens=None, prompt=None, **kwargs):
        """Create a language model from a name.
        
        Args:
            name: Name of the model to load
            backend (str, optional): `AsyncLM` backend to use:
                * 'vllm' to instantiate an `AsyncVirtualLM`; ideal for GPU usage
                * 'hf' for an `AsyncTransformer`; ideal for CPU usage
                * 'mock' for a `MockAsyncLM`; ideal for testing.
                Defaults to 'vllm' if CUDA is available, otherwise 'hf'.
            prompt (str|list[int], optional): Optional prompt to use as a prompt prefix for all input contexts
            **kwargs: Additional arguments passed to AsyncLM constructor
            
        Returns:
            (AsyncLM): The loaded language model
        """
        backend = backend or ('vllm' if torch.cuda.is_available() else 'hf')
        model = load_model_by_name(name, backend=backend, **kwargs)
        return cls(model, prompt=prompt, eos_tokens=eos_tokens)

    @property
    def prompt(self):
        """Get the current prompt."""
        if not self.prompt_ids:
            return None
        return [self.token_maps.decode[x] for x in self.prompt_ids]

    @prompt.setter
    def prompt(self, value):
        """Set the prompt, accepting either string, list of token IDs, or None.

        If setting a string, the prompt will be tokenized using the language model's tokenizer.
        
        Args:
            value (Union[str, List[int], None]): The prompt to set.
                * None: No prompt
                * str: Tokenized prompt
                * List[int]: List of token IDs
        """
        if value is None:
            self.prompt_ids = []
        elif isinstance(value, str):
            if value.endswith(' '):
                warnings.warn(
                    "Prompt ends with whitespace, which may affect tokenization. "
                    "Consider removing trailing whitespace.",
                    stacklevel=2
                )
            self.prompt_ids = self.model.tokenizer.encode(value)
        elif isinstance(value, (list, tuple)) and all(isinstance(x, int) for x in value):
            self.prompt_ids = list(value)
        else:
            raise ValueError(
                f"Prompt must be None, a string or a list of token IDs, got {type(value)}"
            )

    def tokenize(self, context_str):
        """Tokenize a string to a list of bytes, each corresponding to a token in the vocabulary.

        Tokenization is handled by the language model's tokenizer.
        
        Args:
            context_str (str): The string to encode
            
        Returns:
            (List[bytes]): The encoded string 
        """
        return [self.token_maps.decode[x] for x in self.model.tokenizer.encode(context_str)]

    def encode_tokens(self, tokens):
        """Encode a list of byte tokens to a list of token IDs."""
        return [self.token_maps.encode[x] for x in tokens]

    async def log_probability(self, context):
        """Compute the log probability of the context given the prompt."""
        # TODO: Implement scoring methods for AsyncLMs.
        if not context:
            raise ValueError('context must be non-empty')
        
        context_ids = self.encode_tokens(context)
        logps = await self.model.batch_next_token_logprobs([
            self.prompt_ids + context_ids[:i] for i in range(len(context_ids))
        ])
        target_ids = torch.tensor(context_ids, device=logps.device).unsqueeze(1)
        with torch.no_grad():
            logp = torch.gather(logps, 1, target_ids).sum().item()
        return logp

    async def prefix(self, context):
        return await self.log_probability(context)

    async def complete(self, context):
        """Compute the log probability of the context given the prompt.
        
        Args:
            context (List[bytes]): List of tokens representing the context to score
            
        Returns:
            (torch.Tensor): Log probability of the context given the prompt
        """
        logp_context = await self.log_probability(context)
        logp_next = await self.model.next_token_logprobs(context)
        logp_eos = logsumexp([logp_next.weights[x] for x in self.token_maps.eos_idxs])
        return logp_context

    def _process_logp_next(self, logp_next):
        """Process the log probabilities for the next tokens.

        This function rearranges the log probabilities such that the end-of-sequence (EOS) token's log probability
        is moved to the end of the array. This is necessary for downstream code that assumes the EOS token is at the end.

        Args:
            logp_next (np.array): The log probabilities for the next tokens.

        Returns:
            (LazyWeights|np.array): Processed log probabilities for the next tokens.
        """
        # This is ugly, but it's useful for all potentials to adhere to the convention 
        # of keeping the EOS token at the end of the weights array.
        _logp_next = np.full(len(self.decode) + 1, -np.inf, dtype=logp_next.dtype)
        _logp_next[:len(self.decode)] = logp_next[
            ~np.isin(np.arange(len(logp_next)), self.token_maps.eos_idxs)
        ]
        _logp_next[-1] = logsumexp(logp_next[self.token_maps.eos_idxs])
        return self.make_lazy_weights(_logp_next)
    
    async def logp_next(self, context):
        """Get log probabilities for next tokens given `self.prompt` + `context`.
        
        Args:
            context (List[bytes]): List of tokens representing the context
        Returns:
            (LazyWeights): Log probabilities for next tokens
        """
        logp_next = await self.model.next_token_logprobs(
            self.prompt_ids + self.encode_tokens(context)
        )
        return self._process_logp_next(logp_next.float().cpu().numpy())

    async def batch_logp_next(self, contexts):
        """Get log probabilities for next tokens given `self.prompt` + `context`, for each context.
        
        Args:
            contexts: List of token ID sequences representing contexts
            
        Returns:
            (List[LazyWeights]): Log probabilities for next tokens for each context
        """
        logp_nexts = await self.model.batch_next_token_logprobs([
            self.prompt_ids + self.encode_tokens(context) 
            for context in contexts
        ])
        return np.array([
            self._process_logp_next(logp_next) 
            for logp_next in logp_nexts.float().cpu().numpy()
        ])

    def __repr__(self):
        return f"PromptedLLM(prompt={self.prompt!r})"
