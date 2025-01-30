# genlm-control

[![Docs](https://github.com/chi-collective/genlm-control/actions/workflows/docs.yml/badge.svg)](https://chi-collective.github.io/genlm-control/)

# GenLM Control

## Quick Start

### Installation

Clone the repository:
```bash
git clone git@github.com:chi-collective/genlm-control.git
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

> **Note**
> vLLM is not supported on macOS. On macOS systems, only CPU-based functionality (`AsyncTransformer`) will be available. GPU-accelerated features requiring vLLM (`AsyncVirtualLM`) will not work.

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
