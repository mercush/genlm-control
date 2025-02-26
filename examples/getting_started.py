from genlm_control import InferenceEngine
from genlm_control.potential import PromptedLLM, BoolFSA, Potential
from genlm_control.sampler import direct_token_sampler, eager_token_sampler

import torch
import asyncio
from arsenal import timeit
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
)  # For sentiment analysis.


async def main():
    # =============== Basic LLM Sampling =============== #

    # Load gpt2 (or any other HuggingFace model) using the HuggingFace backend
    # (Setting backend='vllm' will be much faster, but requires a GPU)
    mtl_llm = PromptedLLM.from_name("gpt2", backend="hf", temperature=0.5)

    # Set the fixed prompt prefix for the language model
    mtl_llm.set_prompt_from_str("Montreal is")

    # Load a sampler that proposes tokens by sampling directly from the LM's distribution
    sampler = direct_token_sampler(mtl_llm)

    # Create an inference engine
    engine = InferenceEngine(sampler)

    # Run SMC with 10 particles, 10 tokens, and an ESS threshold of 0.5
    sequences = await engine(n_particles=10, max_tokens=10, ess_threshold=0.5)
    print("\nBasic sampling result:")
    print(sequences.posterior)

    # Note: Sequences are lists of `bytes` objects because each token in the language model's
    # vocabulary is represented as a bytes object. In later examples, we'll see that this will make it
    # easy to combine LLM's with other potentials that operate over bytes.

    # =============== Prompt Intersection =============== #

    # Spawn a new language model (shallow copy, sharing the same underlying model)
    bos_llm = mtl_llm.spawn()
    bos_llm.set_prompt_from_str("Boston is")

    # Take the product of the two language models
    product = mtl_llm * bos_llm

    # Create a sampler that proposes tokens by sampling directly from the product's distribution
    sampler = direct_token_sampler(product)

    # Create an inference engine
    engine = InferenceEngine(sampler)

    # Run SMC with 10 particles, 10 tokens, and an ESS threshold of 0.5
    sequences = await engine(n_particles=10, max_tokens=10, ess_threshold=0.5)
    print("\nPrompt intersection result:")
    print(sequences.posterior)

    # =============== Adding Regex Constraint =============== #
    best_fsa = BoolFSA.from_regex(r"is\sthe\s(best|worst).*")

    # The following is valid but will be slow!
    # slow_sampler = direct_token_sampler(
    #    product * best_fsa.coerce(product, f=b''.join)
    # )

    # This sampler is much faster.
    sampler = eager_token_sampler(product, best_fsa)

    # Create an inference engine
    engine = InferenceEngine(sampler)

    # Run SMC with 10 particles, 10 tokens, and an ESS threshold of 0.5
    sequences = await engine(n_particles=10, max_tokens=10, ess_threshold=0.5)
    print("\nPrompt intersection with regex constraint result:")
    print(sequences.posterior)

    # =============== Custom Sentiment Analysis Potential =============== #

    # Create our own custom potential for sentiment analysis.
    # This potential has a vocabulary of bytes.
    class SentimentAnalysis(Potential):
        def __init__(self, model, tokenizer, sentiment="POSITIVE"):
            self.model = model
            self.tokenizer = tokenizer

            self.sentiment_idx = model.config.label2id.get(sentiment, None)
            if self.sentiment_idx is None:
                raise ValueError(f"Sentiment {sentiment} not found in model labels")

            super().__init__(vocabulary=list(range(256)))  # Defined over bytes

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
    print(
        await sentiment_analysis.prefix(b"so good"),
        await sentiment_analysis.prefix(b"so bad"),
    )
    # Check that it satisfies the Potential contract on a given example
    await sentiment_analysis.assert_logw_next_consistency(b"the best", top=5)
    await sentiment_analysis.assert_autoreg_fact(b"the best")

    # The following is valid but will be slow!
    # slow_sampler = eager_token_sampler(
    #    iter_potential=product, item_potential=best_fsa * sentiment_analysis
    # )

    # This setup will be much faster.
    sampler = eager_token_sampler(product, best_fsa)
    critic = sentiment_analysis.coerce(sampler.target, f=b"".join)
    engine = InferenceEngine(sampler, critic=critic)

    # Run SMC with 10 particles, 10 tokens, and an ESS threshold of 0.5
    with timeit("Sentiment-guided sampling (without autobatching)"):
        sequences = await engine(n_particles=10, max_tokens=10, ess_threshold=0.5)
    print("\nSentiment-guided sampling result:")
    print(sequences.posterior)

    # =============== Optimizing with Autobatching =============== #

    # This creates a new potential that automatically batches concurrent
    # requests to the instance methods (`prefix`, `complete`, `logw_next`)
    # and processes them using the batch methods (`batch_complete`, `batch_prefix`, `batch_logw_next`).
    critic = critic.to_autobatched()

    # Create an inference engine
    engine = InferenceEngine(sampler, critic=critic)

    # Run SMC with 10 particles, 10 tokens, and an ESS threshold of 0.5
    with timeit("Sentiment-guided sampling (with autobatching)"):
        sequences = await engine(n_particles=10, max_tokens=10, ess_threshold=0.5)
    print("\nAutobatched sampling result:")
    print(sequences.posterior)


if __name__ == "__main__":
    asyncio.run(main())
