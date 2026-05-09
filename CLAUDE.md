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

`bs-python-utils` is a pure-utilities library (no application logic) organized into subpackages under `bs_python_utils/`. The top-level module files (e.g. `bs_python_utils/bsnputils.py`) are thin backward-compatibility shims that re-export everything from the canonical subpackage locations.

**Dependency order** (no circular deps): `core` → `numerical` → `opt` → `stats`; `data_anal` and `viz` depend on `core` and `numerical`.

**`core/`** — bsutils, Timer, bs_logging, bs_mathstr, bs_mem
- `bsutils.py` — decorators (`printargs`), error handling (`bs_error_abort`), C²-smooth extensions of log/exp (`bslog`, `bsexp`, `bsxlogx`) used throughout optimization code to avoid numerical boundary issues
- `bs_logging.py` — colored logger wrapper
- `Timer.py` — execution timing

**`numerical/`** — bsnputils, bssputils, chebyshev, bs_sparse_gaussian
- `bsnputils.py` — NumPy array ops, bivariate polynomials (`BivariatePolynomial` class), empirical CDF, Gauss-Legendre/Hermite integration, matrix ops
- `bssputils.py` — SciPy utilities
- `chebyshev.py` — Chebyshev interpolation/integration in 1D and 2D, root finding
- `bs_sparse_gaussian.py` — sparse integration E[f(X)] for X~N(0,1); precomputed grids in `GaussHermiteSparseGrids/`

**`opt/`** — bs_opt
- `bs_opt.py` — accelerated gradient descent, BFGS/DFP updates, L-BFGS-B wrapper, Armijo/Barzilai-Borwein step sizes

**`stats/`** — bsstats, bivariate_quantiles, distance_covariances
- `bsstats.py` — TSLS, nonparametric estimation, random sampling
- `bivariate_quantiles.py` — 2D quantiles/ranks via optimal transport (Chernozhukov et al. 2017)
- `distance_covariances.py` — nonlinear dependence measures

**`data_anal/`** — pandas_utils, sklearn_utils

**`viz/`** — bs_plots, bsmplutils, bs_seaborn, bs_altair, streamlit_utils
- `bs_plots.py` — thin aggregator for the other viz modules

## Tooling

- **Package manager:** `uv` (not pip/poetry)
- **Linter/formatter:** Ruff (line length 88); config in `pyproject.toml`
- **Type checker:** mypy; config in `mypy.ini`
- **Python:** >=3.12
- **Docs:** MkDocs with Material theme; docstrings use LaTeX math notation

## Type conventions

Domain-specific type aliases are defined per module (e.g., `TwoArrays`, `FloatOrArray`, `ScalarFunctionAndGradient`). Full type annotations are expected throughout; mypy is run in CI.
