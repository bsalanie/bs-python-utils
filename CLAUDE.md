# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
make install        # create venv with uv, install editable + deps, set up pre-commit hooks
make check          # run pre-commit (ruff check + ruff format) then mypy
make test           # run pytest with doctest on all files in tests/
make build          # build wheel with uv build
make docs           # build and serve MkDocs documentation locally
```

Run a single test file:
```bash
uv run pytest tests/test_bsnputils.py
```

## Architecture

`bs-python-utils` is a pure-utilities library (no application logic) organized by domain. All modules live in `bs_python_utils/`.

**Core utilities**
- `bsutils.py` — decorators (`printargs`), error handling (`bs_error_abort`), C²-smooth extensions of log/exp (`bslog`, `bsexp`, `bsxlogx`) used throughout optimization code to avoid numerical boundary issues
- `bs_logging.py` — colored logger wrapper
- `Timer.py` — execution timing

**Numerical / scientific**
- `bsnputils.py` — NumPy array ops, bivariate polynomials (`BivariatePolynomial` class), empirical CDF, Gauss-Legendre/Hermite integration, matrix ops
- `bs_opt.py` — optimization: accelerated gradient descent, BFGS/DFP updates, L-BFGS-B wrapper, Armijo/Barzilai-Borwein step sizes
- `chebyshev.py` — Chebyshev interpolation/integration in 1D and 2D, root finding
- `bs_sparse_gaussian.py` — sparse integration E[f(X)] for X~N(0,1); precomputed grids in `GaussHermiteSparseGrids/`
- `bsstats.py` — TSLS, nonparametric estimation, random sampling
- `bivariate_quantiles.py` — 2D quantiles/ranks via optimal transport (Chernozhukov et al. 2017)
- `distance_covariances.py` — nonlinear dependence measures

**Visualization** (`bs_plots.py` aggregates):
- `bsmplutils.py` — Matplotlib; `bs_seaborn.py` — Seaborn; `bs_altair.py` — Altair

**Other**: `pandas_utils.py`, `sklearn_utils.py`, `bssputils.py`, `streamlit_utils.py`

## Tooling

- **Package manager:** `uv` (not pip/poetry)
- **Linter/formatter:** Ruff (line length 88); config in `pyproject.toml`
- **Type checker:** mypy; config in `mypy.ini`
- **Python:** >=3.12
- **Docs:** MkDocs with Material theme; docstrings use LaTeX math notation

## Type conventions

Domain-specific type aliases are defined per module (e.g., `TwoArrays`, `FloatOrArray`, `ScalarFunctionAndGradient`). Full type annotations are expected throughout; mypy is run in CI.
