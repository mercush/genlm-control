# Samplers

[Samplers](../reference/genlm_control/sampler/__init__) are the objects that propose new tokens during generation.

## Token Samplers

A [`TokenSampler`](../reference/genlm_control/sampler/token/#genlm_control.sampler.token.TokenSampler) generates individual tokens given a `context` sequence. Each sample $x$ is attached with a log importance weight $w$ and a log-probability $p$.[^1]

[^1]: The log-probability $p$ corresponds to the log-probability of the sample under the sampler's distribution and is returned for testing purposes.

### Direct Token Sampling

The simplest token sampler is the [`DirectTokenSampler`](../reference/genlm_control/sampler/token/#genlm_control.sampler.token.DirectTokenSampler), which samples directly from the normalized version of a potential's `logw_next` method:

```python
# Create a direct token sampler for a potential
sampler = DirectTokenSampler(potential)

# Sample a token
token, logw, logp = await sampler.sample(context)
```

`DirectTokenSampler` is efficient when the potential's `logw_next` method is efficient (e.g., for language models). However, for potentials with large vocabularies or expensive `logw_next` computations, set-based sampling strategies may be more appropriate.

### Set-based Token Sampling

A [`SetTokenSampler`](../reference/genlm_control/sampler/token/#genlm_control.sampler.token.SetTokenSampler) samples tokens by first sampling a weighted subset of tokens using a [`SetSampler`](../reference/genlm_control/sampler/set/#genlm_control.sampler.set.SetSampler), and then selects one token from the set proportional to its weight. Set-based token sampling can often be more efficient than direct token sampling at the cost of higher variance.[^2]

[^2]: "Higher variance" refers to the variance of the estimator, which is influenced by the variance of the importance weights. When a sampler has high variance, the importance weights can vary dramatically across different samples, leading to unstable estimates in downstream tasks. While high-variance samplers may generate samples efficiently, they often require more samples to achieve the same level of accuracy as lower-variance alternatives.

#### Set Samplers

SetTokenSamplers wrap a [`SetSampler`](../reference/genlm_control/sampler/set/#genlm_control.sampler.set.SetSampler), which is responsible for sampling a weighted subset of tokens. Currently, `genlm-control` provides two set samplers:

1. [`EagerSetSampler`](../reference/genlm_control/sampler/set/#genlm_control.sampler.set.EagerSetSampler) - Eagerly samples a set of tokens by sampling one "subtoken" (e.g., byte) at a time.
2. [`TopKSetSampler`](../reference/genlm_control/sampler/set/#genlm_control.sampler.set.TopKSetSampler) - Lazily enumerates the top $K$ tokens by weight and samples an additional "wildcard" token to ensure absolute continuity. This sampler is typically slower than `EagerSetSampler`.

Both of these set samplers are designed to work with two types of potentials:

1. An **iterable potential** which has a vocabulary of iterable tokens (e.g., over byte sequences)
2. An **item potential** which has a vocabulary of items which form the elements of iterable tokens (e.g., over individual bytes)

These samplers are commonly used to sample tokens from a language model's vocabulary while enforcing byte-level constraints, such as those defined by a [finite-state automaton (FSA)](../reference/genlm_control/potential/wfsa/__init__) or [context-free grammar (CFG)](../reference/genlm_control/potential/wcfg/__init__).

```python
# Create a set-based token sampler using a set sampler
set_sampler = EagerSetSampler(llm, fsa)
sampler = SetTokenSampler(set_sampler)

# Sample a token and weight
token, logw, logp = await sampler.sample(context)
```

### Factory methods

For convenience, we provide factory methods for creating token samplers from potentials.

```python
from genlm.control.sampler import direct_token_sampler, topk_token_sampler, eager_token_sampler

direct_sampler = direct_token_sampler(llm)

topk_sampler = topk_token_sampler(llm, fsa, K=10)

eager_sampler = eager_token_sampler(llm, fsa)
```

### Custom Token Samplers

It is also possible to implement custom token samplers by subclassing the [`TokenSampler`](../reference/genlm_control/sampler/token/#genlm_control.sampler.token.TokenSampler) class and implementing the [`sample`](../reference/genlm_control/sampler/token/#genlm_control.sampler.token.TokenSampler.sample) method. These implementations must satisfy the following contract.

#### Token Sampler Contract

All token samplers in `genlm-control` must generate properly weighted samples with respect to a target potential's next-token weights $\pot(\cdot \mid \bm{x})$ given a context $\xx$:

A weighted sample $(x, w) \sim q(\cdot \mid \xx)$ is properly weighted with respect to $\pot(\cdot \mid \xx)$ if, for any function $f$,

$$
\mathbb{E}_{(x,w) \sim q(\cdot \mid \xx)}[w f(x)] = \sum_{x \in \A \cup \{\eos\}} f(x)\cdot\pot(x \mid \xx)
$$

where $\mathcal{A}$ is the vocabulary of the target potenital $\pot$.


## Performance Considerations

When choosing a sampler, consider:

1. **Vocabulary size** - For large vocabularies, set-based sampling may be more efficient than direct sampling.
2. **Efficiency of `logw_next`** - If `logw_next` is expensive to compute, use a set-based sampler.
3. **Token structure** - If you are sampling from potentials whose tokens are in iterable-item relationships (e.g., bytes and individual bytes), use a set sampler that exploits this structure.
