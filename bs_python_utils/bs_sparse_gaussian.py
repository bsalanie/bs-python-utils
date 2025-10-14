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

from pathlib import Path

import numpy as np

from bs_python_utils.bsnputils import TwoArrays
from bs_python_utils.bsutils import bs_error_abort


def setup_sparse_gaussian(
    ndims: int, iprec: int, GHsparsedir: str | None = None
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
        GHsparsedir: the name of a directory that contains nodes and weights

    Returns:
        a pair of  arrays `nodes` and `weights`;
        `nodes` has `ndims-1` columns and `weights` is a vector with the same
            number of rows.
    """
    GHdir = (
        Path.home() / "Dropbox" / "GHsparseGrids"
        if GHsparsedir is None
        else Path(GHsparsedir)
    )
    if iprec not in [9, 13, 17]:
        bs_error_abort(
            f"We only do sparse integration with precision 9, 13, or 17, not {iprec}"
        )

    if ndims in [1, 2, 3, 4, 5]:
        if not GHdir.exists():
            bs_error_abort("I did not find the directory with the nodes/weights files.")
        grid = np.loadtxt(GHdir / f"GHsparseGrid{ndims}prec{iprec}.txt")

        print(f"{grid.shape=}")
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
