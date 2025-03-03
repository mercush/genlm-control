[![Docs](https://github.com/chisym/genlm-control/actions/workflows/docs.yml/badge.svg)](https://chisym.github.io/genlm-control/)
[![Tests](https://github.com/chisym/genlm-control/actions/workflows/pytest.yml/badge.svg)](https://chisym.github.io/genlm-control/)

# GenLM Control

GenLM Control is a controlled generation with programmable constraints. It leverages sequential Monte Carlo (SMC) methods to efficiently generate text that satisfies constraints or preferences encoded by arbitrary potential functions.

 See the [docs](https://chisym.github.io/genlm-control/) for details and [examples](https://github.com/chisym/genlm-control/tree/main/examples) for usage.

> **⚠️ Warning:** This library is currently in active development. Please report any bugs in [the issue tracker](https://github.com/chisym/genlm-control/issues).


## Quick Start

### Installation

Clone the repository:
```bash
git clone git@github.com:ChiSym/genlm-control.git
cd genlm-control
```
and install with pip:

```bash
pip install .
```

This installs the package without development dependencies. For development, install in editable mode with:

```bash
pip install -e ".[test,docs]"
```

which also installs the dependencies needed for testing (test) and documentation (docs).

## Requirements

- Python >= 3.11
- The core dependencies listed in the `pyproject.toml` file of the repository.

## Testing

When test dependencies are installed, the test suite can be run via:

```bash
pytest tests
```

## Documentation

Documentation is generated using [mkdocs](https://www.mkdocs.org/) and hosted on GitHub Pages. To build the documentation, run:

```bash
mkdocs build
```

To serve the documentation locally, run:

```bash
mkdocs serve
```
