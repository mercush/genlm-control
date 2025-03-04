# GenLM Control

GenLM Control is a library for controlled text generation with programmable constraints. It makes use of sequential Monte Carlo (SMC) to generate text that satisfies constraints or preferences encoded by [potentials](potentials.md).

## Getting started

```python
from genlm_control import InferenceEngine
from genlm_control.potential import PromptedLLM, BoolFSA
from genlm_control.sampler import eager_token_sampler

# Create a language model potential
llm = PromptedLLM.from_name("gpt2")
llm.set_prompt_from_str("Sequential Monte Carlo is")

# Create a finite-state automaton potential using a regular expression
fsa = BoolFSA.from_regex(r"\s(good😍|bad🙁)")

# Create a token sampler that combines the language model and FSA
sampler = eager_token_sampler(llm, fsa)

# Set up the inference engine with the sampler
engine = InferenceEngine(sampler)

# Generate text using SMC. Generation is asynchronous; use `await` or `asyncio.run`.
sequences = await engine(
    n_particles=10, # Number of candidate sequences to maintain
    ess_threshold=0.5, # Threshold for resampling
    max_tokens=25, # Maximum sequence length
    verbosity=1 # Print particles at each step
)

# Show the inferred posterior distribution over sequences
sequences.posterior
```

See also the examples in `examples/getting_started.py` for more complex usage.


## Main components

### [Potentials](potentials.md)
Potentials are the core objects that guide text generation by:

* Acting as components of **samplers**, which propose new tokens at each step of the generation process.
* Serving as **critics**, which reweight sequences based on whether they satisfy the constraint encoded by the potential at each step of the generation process.

The library comes with a number of built-in potentials, including

* Language models
* Finite-state automata specified by regular expressions
* Context-free grammars specified by Lark grammars

It also supports [user-defined potentials](potentials.md#custom-potentials) and combinations of potentials using [products](potentials.md#products-of-potentials).

See the [Potentials](potentials.md) documentation for more details.

### [Samplers](samplers.md)

Samplers generate tokens by sampling from potentials or collections of potentials. This library currently supports a number of different sampling strategies which trade off quality and efficiency.

See the [Samplers](samplers.md) documentation for more details.

### Critics
Critics are used to evaluate the quality of a sequence which is in the process of being generated. Any Potential can serve as a critic. To use them in generation, pass them to the `InferenceEngine` at initialization.


## Visualization

The library includes a built-in visualization tool for inference runs, courtesy of [hfppl](https://github.com/probcomp/hfppl).

```python
from genlm_control.viz import InferenceVisualizer

# Create a visualizer (starts server on port 8000, you can specify a different port if needed)
viz = InferenceVisualizer()

# Run inference and save a record of the inference run to a JSON file
sequences = await engine(
    n_particles=10,
    ess_threshold=0.5,
    max_tokens=20,
    json_path="smc_record.json" # save the record to a JSON file
)

# Open visualization in browser
viz.visualize("smc_record.json", auto_open=True)

# Clean up when done
viz.shutdown_server()
```

Note that if you are SSH-ing onto a remote machine, you may need to set up port forwarding. Visual Studio Code automatically handles this for some ports, including the default port 8000.
