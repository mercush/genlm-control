from genlm.control import SMC
from genlm.control.potential import PromptedLLM, BoolFSA, Potential
from genlm.control.sampler import direct_token_sampler, eager_token_sampler

import torch
import asyncio
from arsenal import timeit
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
)  # For sentiment analysis.

# This file runs through a number of examples of using the `genlm-control` library.
# This is a good starting point for understanding how to build increasingly complex genlm-control programs.
# Refer to the documentation for any details on the methods and classes used below.


async def main():
    # =============== Basic LLM Sampling =============== #

    # Load gpt2 (or any other HuggingFace model) using the HuggingFace backend.
    # Set temperature to 0.5 for less variation in outputs.
    # Setting backend='vllm' will be much faster, but requires a GPU.
    mtl_llm = PromptedLLM.from_name("gpt2", backend="hf", temperature=0.5)

    # Set the fixed prompt prefix for the language model.
    # All language model predictions will be conditioned on this prompt.
    # Under the hood, this converts the string into a sequence of token ids using the model's tokenizer.
    mtl_llm.set_prompt_from_str("Montreal is")

    # Load a sampler that proposes tokens by sampling directly from the LM's distribution.
    token_sampler = direct_token_sampler(mtl_llm)

    # This is the main object that will be used for generation using sequence Monte Carlo (SMC).
    seq_sampler = SMC(token_sampler)

    # Run SMC with 10 particles, a maximum of 10 tokens, and an ESS threshold of 0.5.
    sequences = await seq_sampler(n_particles=10, max_tokens=10, ess_threshold=0.5)
    print("\nBasic sampling result:")
    print(sequences.posterior)

    # Note: Sequences are lists of `bytes` objects because each token in the language model's
    # vocabulary is represented as a bytes object.

    # =============== Prompt Intersection =============== #

    # Spawn a new language model (shallow copy, sharing the same underlying model).
    bos_llm = mtl_llm.spawn()
    bos_llm.set_prompt_from_str("Boston is")

    # Take the product of the two language models.
    # This defines a `Product` potential which is the element-wise product of the two LMs.
    product = mtl_llm * bos_llm

    # Create a sampler that proposes tokens by sampling directly from the product.
    token_sampler = direct_token_sampler(product)

    # Create an SMC sampler.
    seq_sampler = SMC(token_sampler)

    # Run SMC with 10 particles, 10 tokens, and an ESS threshold of 0.5.
    sequences = await seq_sampler(n_particles=10, max_tokens=10, ess_threshold=0.5)
    print("\nPrompt intersection result:")
    print(sequences.posterior)

    # =============== Adding Regex Constraint =============== #

    # Create a regex constraint that matches sequences containing the word "the"
    # followed by either "best" or "worst" and then anything else.
    best_fsa = BoolFSA.from_regex(r"\sthe\s(best|worst).*")

    # By default, BoolFSA's are defined over individual bytes. This means that
    # their `prefix` and `complete` methods are called on byte sequences.
    print("best_fsa.prefix(b'the bes') =", await best_fsa.prefix(b"the bes"))
    print(
        "best_fsa.complete(b'the best city') =",
        await best_fsa.complete(b"the best city"),
    )

    # The following is valid but will be slow!
    # It is slow because it will require calling the fsa on the
    # full product vocabulary at each step of token generation.
    # slow_sampler = direct_token_sampler(
    #    product * best_fsa.coerce(product, f=b''.join)
    # )

    # This sampler is much faster.
    # It will only call the fsa on a subset of the product vocabulary,
    # but maintains the same proper weighting guarantees as the direct sampler.
    token_sampler = eager_token_sampler(product, best_fsa)

    # Create an SMC sampler.
    seq_sampler = SMC(token_sampler)

    # Run SMC with 10 particles, 10 tokens, and an ESS threshold of 0.5.
    sequences = await seq_sampler(n_particles=10, max_tokens=10, ess_threshold=0.5)
    print("\nPrompt intersection with regex constraint result:")
    print(sequences.posterior)

    # =============== Custom Sentiment Analysis Potential =============== #

    # Create our own custom potential for sentiment analysis.
    # Custom potentials must subclass `Potential` and implement the `prefix` and `complete` methods.
    # They can also override other methods, like `batch_prefix`, and `batch_complete` for improved performance.
    # Each Potential needs to specify its vocabulary of tokens; this potential has a vocabulary of individual bytes.
    class SentimentAnalysis(Potential):
        def __init__(self, model, tokenizer, sentiment="POSITIVE"):
            self.model = model
            self.tokenizer = tokenizer

            self.sentiment_idx = model.config.label2id.get(sentiment, None)
            if self.sentiment_idx is None:
                raise ValueError(f"Sentiment {sentiment} not found in model labels")

            super().__init__(
                vocabulary=list(range(256))
            )  # Defined over bytes (ints in the range [0, 255]).

        def _forward(self, contexts):
            strings = [
                bytes(context).decode("utf-8", errors="ignore") for context in contexts
            ]
            inputs = self.tokenizer(strings, return_tensors="pt", padding=True)
            with torch.no_grad():
                logits = self.model(**inputs).logits
            return logits.log_softmax(dim=-1)[:, self.sentiment_idx].cpu().numpy()

        async def prefix(self, context):
            return self._forward([context])[0].item()

        async def complete(self, context):
            return self._forward([context])[0].item()

        async def batch_complete(self, contexts):
            return self._forward(contexts)

        async def batch_prefix(self, contexts):
            return self._forward(contexts)

    # Initialize sentiment analysis potential
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    sentiment_analysis = SentimentAnalysis(
        model=DistilBertForSequenceClassification.from_pretrained(model_name),
        tokenizer=DistilBertTokenizer.from_pretrained(model_name),
        sentiment="POSITIVE",
    )

    # Verify the potential
    print("\nSentiment analysis test:")
    # Check that our potential is working as expected. We should expect higher log weights
    # for more positive sequences.
    print(
        "sentiment_analysis.prefix(b'so good') =",
        await sentiment_analysis.prefix(b"so good"),
    )
    print(
        "sentiment_analysis.prefix(b'so bad') =",
        await sentiment_analysis.prefix(b"so bad"),
    )
    # Check that it satisfies the Potential contract for a given example.
    await sentiment_analysis.assert_logw_next_consistency(b"the best", top=5)
    await sentiment_analysis.assert_autoreg_fact(b"the best")

    # The following is valid but will be slow!
    # slow_sampler = eager_token_sampler(
    #    iter_potential=product, item_potential=best_fsa * sentiment_analysis
    # )

    # This setup will be much faster.
    # It will only call the sentiment analysis potential only once per particle per step,
    # but maintains the same proper weighting guarantees as the eager sampler.
    token_sampler = eager_token_sampler(product, best_fsa)
    critic = sentiment_analysis.coerce(token_sampler.target, f=b"".join)
    seq_sampler = SMC(token_sampler, critic=critic)

    # Run SMC with 10 particles, 10 tokens, and an ESS threshold of 0.5.
    # We also time this inference run for comparison with the next example.
    with timeit("Timing sentiment-guided sampling without autobatching"):
        sequences = await seq_sampler(n_particles=10, max_tokens=10, ess_threshold=0.5)
    print("\nSentiment-guided sampling result:")
    print(sequences.posterior)

    # =============== Optimizing with Autobatching =============== #

    # During inference, all requests to the sentiment analysis potential are
    # made to the instance methods (`prefix`, `complete`). We can take advantage
    # of the fact that we have parallelized batch versions of these methods using
    # the `to_autobatched` method.

    # This creates a new potential that automatically batches concurrent
    # requests to the instance methods (`prefix`, `complete`, `logw_next`)
    # and processes them using the batch methods (`batch_complete`, `batch_prefix`, `batch_logw_next`).
    critic = critic.to_autobatched()

    # Create an SMC sampler.
    seq_sampler = SMC(token_sampler, critic=critic)

    # Run SMC with 10 particles, 10 tokens, and an ESS threshold of 0.5.
    # If you are running this on a machine with a GPU, you should
    # see a significant performance improvement.
    with timeit("Timing sentiment-guided sampling with autobatching"):
        sequences = await seq_sampler(n_particles=10, max_tokens=10, ess_threshold=0.5)
    print("\nSentiment-guided sampling with autobatching result:")
    print(sequences.posterior)


if __name__ == "__main__":
    asyncio.run(main())
