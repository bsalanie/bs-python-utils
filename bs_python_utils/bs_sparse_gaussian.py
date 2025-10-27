"""
Sets up sparse integration over a Gaussian, given text files that contain
rescaled Gauss-Hermite nodes and weights.

These files must be named `GHsparseGrid{ndims}prec{iprec}.txt`, where
`ndims` is the number of dimensions of integration
and `iprec` is a precision level that must be 9, 13, or (most precise) 17.
The file must have `(ndims+1)` columns,
with the weights in the first column.

The nodes and weights are rescaled so that `f(nodes) @ weights` approximates
`Ef(X)` for `X` an `N(0,I)` variable.
"""
from importlib import resources

import numpy as np

from bs_python_utils.bsnputils import TwoArrays
from bs_python_utils.bsutils import bs_error_abort


def read_grid_file(grid_name: str) -> np.ndarray:
    """Load a sparse Gauss-Hermite grid stored inside the package."""
    resource = (
        resources.files("bs_python_utils")
        / "GaussHermiteSparseGrids"
        / f"GHsparseGrid{grid_name}.txt"
    )
    with resources.as_file(resource) as grid_path:
        return np.loadtxt(grid_path)


def setup_sparse_gaussian(
    ndims: int,
    iprec: int,
) -> TwoArrays:
    """
    Get nodes and weights for sparse integration Ef(X) with X = N(0,1) in
    `ndims` dimensions.

    Examples:
        >>> nodes, weights = setup_sparse_gaussian(mdims, iprec)
        >>> integral_f = f(nodes) @ weights

    Args:
        ndims: number of dimensions (1 to 5)
        iprec: precision (must be 9, 13, or 17)

    Returns:
        a pair of  arrays `nodes` and `weights`;
        `nodes` has `ndims-1` columns and `weights` is a vector with the same
            number of rows.
    """
    if iprec not in [9, 13, 17]:
        bs_error_abort(
            f"We only do sparse integration with precision 9, 13, or 17, not {iprec}"
        )

    if ndims in [1, 2, 3, 4, 5]:
        grid = read_grid_file(f"{ndims}prec{iprec}").reshape((-1, ndims + 1))

        if ndims == 1:
            weights = grid[:, 0]
            nodes = grid[:, 1]
        else:
            weights = grid[:, 0]
            nodes = grid[:, 1:]
        return nodes, weights
    else:
        bs_error_abort(
            f"We only do sparse integration in one to five dimensions, not {ndims}"
        )
        return np.zeros(1), np.zeros(1)  # for mypy
