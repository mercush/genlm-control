# Potentials

`Potential`s are the core object in `genlm-control`. A potential encodes constraints or preferences by assigning weights to sequences of tokens.

Potentials serve two main roles in controlled text generation:

1. **Building blocks of [samplers](samplers.md)** - Potentials are key components of samplers, which are used to propose new tokens during generation.
2. **Critics** - Potentials are used to guide generation by acting as *twists* which reweight partial and complete sequences during generation.

This page describes the core concepts and methods of potentials.

## Table of Contents

- [Key concepts](#key-concepts)
  - [Vocabulary](#vocabulary)
  - [Weight assignment](#weight-assignment)
  - [Next-token weights](#next-token-weights)
- [Built-in potentials](#built-in-potentials)
  - [Language models](#language-models)
  - [Finite-state automata](#finite-state-automata)
  - [Context-free grammars](#context-free-grammars)
- [Custom potentials](#custom-potentials)
  - [Defining a custom potential](#defining-a-custom-potential)
  - [Testing your custom potential](#testing-your-custom-potential)
- [Complex usage](#complex-usage)
  - [Products of potentials](#products-of-potentials)
  - [Coerced potentials](#coerced-potentials)
  - [Performance optimizations](#performance-optimizations)
- [Formalization](#formalization)
  - [Correspondance with the Potential class](#correspondance-with-the-potential-class)


## Key concepts

### Vocabulary

Each potential has a **vocabulary** which defines the set of tokens it operates on. The vocabulary is accessible via the `vocab` attribute and can contain any hashable Python objects (e.g., bytes, strings, integers). Most built-in potentials operate on vocabularies whose tokens are `bytes` or `int` objects (the latter often representing individual bytes), but can be adapted to work with other token types via coercion (see [Coerced potentials](#coerced-potentials)).

### Weight assignment

Potentials assign weights to sequences of tokens from their vocabulary. These weights are always non-negative real numbers, though they are computed in log space for numerical stability.

A potential $\Phi$ defines two core weighting functions:

1. **Complete** (`complete` method) - Assigns weights to sequences that are considered "finished" or "complete". For example, a potential enforcing grammatical correctness would assign positive weights to grammatically valid sentences and zero weights (negative infinity in log space) to invalid ones.

2. **Prefix** (`prefix` method) - Assigns weights to partial sequences that could potentially be extended into valid complete sequences. For example, a potential enforcing grammatical correctness could assign positive weights to prefixes of grammatically valid sequences.

    Given a complete method, there are many possible prefix methods that could be used, providing as much or as little information as desired. The key requirement is that if a prefix has zero weight, then all of its extensions and completions must also have zero weight - in other words, prefix cannot rule out sequences that could later become valid.

For example, consider a potential that only allows sequences of length $N$:
- The complete weight would be positive for sequences of *exactly* length $N$
- The prefix weight would be positive for sequences of length *at most* $N$

The relationship between complete and prefix weights is formalized in the [Formalization](#formalization) section.

### Next-token weights

The `logw_next` method computes weights for each possible next token (including a reserved end-of-sequence token) given a context sequence. These weights are crucial for controlled text generation as they can be used to guide the selection of the next token at each step of generation.

The `logw_next` method is implemented by default in terms of the `complete` and `prefix` methods. Potentials will often override this method to provide a more efficient implementation. The relationship between `logw_next` and `complete`/`prefix` is given in the [Formalization](#formalization) section.

## Built-in potentials

`genlm-control` comes with a number of built-in potentials that can be used in controlled text generation.

### Language models

`PromptedLLM` represents a language model conditioned on a fixed prompt prefix.

```python
# Load GPT-2 with temperature 0.5
llm = PromptedLLM.from_name("gpt2", temperature=0.5)

# Set a prompt prefix that all generations will be conditioned on
llm.set_prompt_from_str("Montreal is")
```

`PromptedLLM`s have a vocabulary of `bytes` tokens, obtained from the language model's tokenizer. See the documentation for more details.

### Finite-state automata

`genlm-control` provides two FSA implementations:

1. `WFSA` (Weighted Finite-State Automata) - For weighted constraints:
```python
# Create a WFSA from a regex pattern
# Transitions are automatically normalized to form probability distributions
wfsa = WFSA.from_regex(r"\sthe\s(best|worst).*😎")
```

2. `BoolFSA` (Boolean Finite-State Automata) - For hard constraints:
```python
# Create a boolean FSA from a regex pattern
# Transitions are binary (0 or -inf in log space)
fsa = BoolFSA.from_regex(r"\sthe\s(best|worst).*😎")
```

Both FSAs:
- Support regex patterns with standard syntax
- Operate on byte-level sequences by default
- Can be combined with other potentials via products

### Context-free grammars

Similar to FSAs, `genlm-control` provides two CFG implementations:

1. `WCFG` (Weighted Context-Free Grammar).
```python
cfg = WCFG.from_string("""
    1.0: S -> NP VP
    0.5: NP -> the N
    0.5: NP -> a N
    0.5: VP -> V NP
    0.5: N -> cat
    0.5: N -> dog
    0.5: V -> saw
    0.5: V -> chased
""")
```

2. `BoolCFG` (Boolean Context-Free Grammar).
```python
# Create a boolean CFG from a Lark grammar string
cfg = BoolCFG.from_lark("""
    start: np vp
    np: "the" n | "a" n
    vp: v np
    n: "cat" | "dog"
    v: "saw" | "chased"
""")
```

`BoolCFG`s support grammar specification via [Lark syntax](https://lark-parser.readthedocs.io/en/latest/grammar.html).

Both CFGs:
- Use Earley parsing for efficient recognition
- Can be combined with other potentials
- Operate on byte-level sequences by default

## Custom potentials

Creating custom potentials is straightforward and allows you to use arbitrary constraints.

### Creating a custom potential

To define a custom potential, you need to
1. Create a subclass of `Potential`
2. Implement the `complete` and `prefix` methods.

For improved performance, you can also override the `logw_next` method with an implementation that satisfies the properties specified in the [Formalization](#formalization) section.

Consider the following example of a potential that only allows sequences of a given length:

```python
class LengthPotential(Potential):
    """ A potential that only allows sequences of a given length. """
    def __init__(self, vocabulary, length):
        # Initialize the superclass with the potential's vocabulary.
        super().__init__(vocabulary)
        self.length = length

    async def complete(self, context):
        # Note: 0.0 = log(1.0) and float('-inf') = log(0.0)
        return 0.0 if len(context) == self.length else float('-inf')

    async def prefix(self, context):
        # Note: 0.0 = log(1.0) and float('-inf') = log(0.0)
        return 0.0 if len(context) <= self.length else float('-inf')
```

Here, the `complete` method only allows sequences of the given length. The `prefix` method, in contrast, allows sequences of any length up to the given length. This is a simple example, but it illustrates the key difference between complete and prefix weights: since a sequence with length less than the target length can always be extended to a sequence of the target length, the `prefix` method cannot rule it out. It is only when we've surpassed the target length that the `prefix` method can rule out the sequence (assigning a zero weight).

### Testing your custom potential

Potentials automatically inherit from the `PotentialTests` mixin, which provides a number of tests for validating the correctness of the potential's implementation:

* `assert_logw_next_consistency(context)`: Verifies that token-level log weights are consistent with prefix and complete scores for a given context.
* `assert_autoreg_fact(context)`: Validates that complete scores factor correctly as a sum of log token weights (with an additional correction term corresponding to the prefix weight of the empty sequence) for a given context.
* `assert_batch_consistency(contexts)`: Ensures batch operations produce identical results to their non-batch counterparts for a given batch of contexts.

```python
await potential.assert_logw_next_consistency(context)
await potential.assert_autoreg_fact(context)
await potential.assert_batch_consistency(contexts)
```

## Complex usage

### Products of potentials

The `Product` class allows you to combine two potentials. A `Product` is itself is a potential, meaning that it implements all potential methods and that it is possible to chain products to combine more than two potentials.


```python
# Example: Combining two language models
mtl_llm = PromptedLLM.from_name("gpt2")
mtl_llm.set_prompt_from_str("Montreal is")

bos_llm = mtl_llm.spawn()
bos_llm.set_prompt_from_str("Boston is")

# Create product using multiplication operator
product = mtl_llm * bos_llm
```

The product potential operates on the intersection of the two potentials' vocabularies. For a product potential:

- The vocabulary $\mathcal{A}$ is the intersection of the two potentials' vocabularies: $ \mathcal{A} = \mathcal{A}_1 \cap \mathcal{A}_2 $.
- The prefix potential $\psi$ is the product (sum in log space) of the individual prefix potentials: $ \log \psi(\bm{x}) = \log \psi_1(\bm{x}) + \log \psi_2(\bm{x}) $
- The complete potential $\phi$ is the product (sum in log space) of the individual complete potentials: $ \log \phi(\bm{x}) = \log \phi_1(\bm{x}) + \log \phi_2(\bm{x}) $
- The next-token potential $\Phi(\cdot \mid \bm{x})$ is the product (sum in log space) of the individual next-token potentials: $ \log \Phi(x \mid \bm{x}) = \log \Phi_1(x \mid \bm{x}) + \log \Phi_2(x \mid \bm{x}) $ for $x \in (\mathcal{A}_1 \cap \mathcal{A}_2) \cup \{\textsf{eos}\}$

> **Note:** Be careful when taking products of potentials with minimal vocabulary overlap, as the resulting potential will only operate on tokens present in both vocabularies. A warning will be raised if the vocabulary overlap is less than 10% of either potential's vocabulary.


### Coerced potentials

The `Coerced` class allows you to adapt a potential to work with a different vocabulary by providing a coercion function that maps between token types. This is particularly useful when combining potentials that operate on different types of tokens.

```python
# Example: Coercing a byte-level FSA to work with a language model's tokens
fsa = BoolFSA.from_regex(r"\sthe\s(best|worst).*")  # Works on bytes
llm = PromptedLLM.from_name("gpt2")  # Works on byte sequences

# Coerce the FSA to work with the LLM's tokens by joining tokens into bytes
coerced_fsa = fsa.coerce(llm, f=b''.join)

# Now we can combine them using the product operator!
product = llm * coerced_fsa
```

For a coerced potential with coercion function $f$, the prefix $\psi$ and complete $\phi$ potentials are defined as:

$$
\psi(x_1, \ldots, x_n) = \psi(f(x_1, \ldots, x_n))
$$

$$
\phi(x_1, \ldots, x_n) = \phi(f(x_1, \ldots, x_n))
$$

where $x_1, \ldots, x_n$ is a sequence of tokens in the coerced potential's vocabulary.

By default, the coerced potential's vocabulary is pruned to only include tokens that can be validly mapped to the original potential's vocabulary via the coercion function. This can be disabled by setting `prune=False`.

Common use cases for coercion include:
- Adapting byte-level constraints (like FSAs) to work with token-level language models (which have vocabularies of byte *sequences*)
- Implementing constraints that operate on processed versions of the tokens (e.g., lowercase text)
- Converting between different tokenization schemes

> **Note:** The coercion operation can impact performance, especially when mapping from a coarser token type to a finer token type (e.g., byte sequences to individual bytes). To sample tokens from a coerced product, consider using specialized samplers (e.g., `eager_token_sampler`, `topk_token_sampler`).

### Performance optimizations

`genlm-control` provides a number of performance optimizations for potentials, described in the [performance](performance.md) section.


## Formalization

This section provides a formal definition of potentials and the relationships between their complete, prefix, and next-token potentials.

**Notation** Let $\mathcal{A}$ be a vocabulary of tokens and $\textsf{eos}$ a specialized end-of-sequence token. Let $\mathcal{A}^*$ denote the set of all sequences of tokens which can be built from $\mathcal{A}$ (including the empty sequence $\epsilon$) and $\mathcal{A}^*{\textsf{eos}} = \{\bm{x}\textsf{eos} : \bm{x} \in \mathcal{A}^*\}$ the set of $\textsf{eos}$-terminated sequences. We refer to $\mathcal{A}^*$ as the set of *prefix* sequences and $\mathcal{A}^*{\textsf{eos}}$ the set of *complete* sequences.

A potential $\Phi$ is a function $\Phi: \mathcal{A}^* \cup\mathcal{A}^*{\textsf{eos}} \rightarrow \mathbb{R}_{\geq 0}$ which assigns a non-negative real number to prefix and complete sequences from its vocabulary $\mathcal{A}$:

$$
\Phi(\bm{x}) = \begin{cases}
    \psi(\bm{x}) & \text{if } \bm{x} \in \mathcal{A}^* \\
    \phi(\bm{y}) & \text{if } \bm{x} = \bm{y}\textsf{eos}, \bm{y} \in \mathcal{A}^*
\end{cases}
$$

where

* $\psi : \mathcal{A}^* \rightarrow \mathbb{R}_{\geq 0}$ is the **prefix potential**
* $\phi : \mathcal{A}^* \rightarrow \mathbb{R}_{\geq 0}$ is the **complete potential**

The complete and prefix potentials are related by the following equation:

$$
\psi(\bm{x}) > 0 \implies \phi(\bm{x}\bm{y}) > 0 \, \forall \bm{x},\bm{y} \text{ such that } \bm{x}\bm{y} \in \mathcal{A}^*
$$

Intuitively, this means that the prefix potential cannot rule out a sequence which can later on turn out to be valid according to the complete potential.

Finally, we define the **next-token weights function** $\Phi(x \mid \bm{x}) : \mathcal{A} \cup \{\textsf{eos}\} \rightarrow \mathbb{R}_{\geq 0}$, which assigns a non-negative real number to each token $x \in \mathcal{A} \cup \{\textsf{eos}\}$ given a sequence $\bm{x} \in \mathcal{A}^*$:

$$
\Phi(x \mid \bm{x}) = \frac{\Phi(\bm{x}x)}{\psi(\bm{x})} = \begin{cases}
    \frac{\psi(\bm{x}x)}{\psi(\bm{x})} & \text{if } x \in \mathcal{A} \\
    \frac{\phi(\bm{x})}{\psi(\bm{x})} & \text{if } x = \textsf{eos}
\end{cases}
$$

$\Phi(\cdot \mid \bm{x})$ is related to the complete and prefix potentials according to the following autoregressive factorization:

$$
\frac{\phi(\bm{x})}{\psi(\epsilon)} = \Phi(\textsf{eos} \mid \bm{x}) \prod_{x \in \bm{x}} \Phi(x \mid \bm{x})
$$

### Correspondance with the `Potential` class

Each of the quantities above directly corresponds to a method or attribute of the `Potential` class:

| Method/Attribute | Mathematical Quantity | Description |
|-----------------|----------------------|-------------|
| `vocab` | $\mathcal{A}$ | The vocabulary of the potential. |
| `eos` | $\textsf{eos}$ | The end-of-sequence token. |
| `vocab_eos` | $\mathcal{A} \cup \{\textsf{eos}\}$ | The vocabulary of the potential including the end-of-sequence token. |
| `complete(self, context)` | $\log \phi(\bm{x})$ | The complete potential for a given sequence. |
| `prefix(self, context)` | $\log \psi(\bm{x})$ | The prefix potential for a given sequence. |
| `logw_next(self, context)` | $\log \Phi(\cdot \mid \bm{x})$ | The next-token potential for a given prefix sequence. |
| `score(self, context)` | $\log \Phi(\bm{x})$ | The potential defined for a possibly eos-terminated sequence. |
