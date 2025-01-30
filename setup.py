from setuptools import setup, find_packages

requirements = [
    'genlm_grammar @ git+https://github.com/chi-collective/genlm-grammar',
    'genlm_backend @ git+https://github.com/chi-collective/genlm-backend',
    'hfppl @ git+https://github.com/probcomp/hfppl',
    'arsenal @ git+https://github.com/timvieira/arsenal',
    'IPython',
    'numpy',
    'torch'
]

test_requirements = [
    'pytest',
    'pytest-benchmark',
    'pytest-asyncio'
]

docs_requirements = [
    'mkdocs',
    'mkdocstrings[python]',
    'mkdocs-material',
    'mkdocs-gen-files',
    'mkdocs-literate-nav',
    'mkdocs-section-index',
]

setup(
    name='genlm-control',
    version='0.0.1',
    description='',
    install_requires=requirements,
    extras_require={'test' : test_requirements, 'docs' : docs_requirements},
    python_requires='>=3.11',
    authors=['The GenLM Team'],
    readme='README.md',
    scripts=[],
    packages=find_packages(),
)