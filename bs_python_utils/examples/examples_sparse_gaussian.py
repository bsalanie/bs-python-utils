"""Integrate x -> x**4 over N(0, 1) using the sparse Gaussian grids."""

from typing import cast
import numpy as np

from bs_python_utils.bs_sparse_gaussian import setup_sparse_gaussian


def sum_x_to_the_fourth(x: np.ndarray) -> np.ndarray:
    """Vectorized integrand f(x) = sum(x**4) evaluated at the sparse grid nodes."""
    return cast(np.ndarray, np.sum(x ** 4, axis=1))


def main() -> None:
    ndims = 2
    iprec = 13

    nodes, weights = setup_sparse_gaussian(ndims, iprec)
    estimate = sum_x_to_the_fourth(nodes) @ weights
    exact = 3.0 * ndims  # E[X^4] = 3 for X ~ N(0, 1)

    print("Sparse Gaussian integration of f(x)=x^4 with X ~ N(0, 1)")
    print(f"Grid shape: nodes={nodes.shape}, weights={weights.shape}")
    print(f"Estimate: {estimate:.12f}")
    print(f"Exact value: {exact:.12f}")
    print(f"Absolute error: {abs(estimate - exact):.3e}")


if __name__ == "__main__":
    main()
