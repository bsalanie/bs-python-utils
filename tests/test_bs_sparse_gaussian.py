from math import isclose

import numpy as np

from bs_python_utils.numerical.bs_sparse_gaussian import setup_sparse_gaussian


def test_sparse_gaussian():
    n = 5
    iprec = 13
    nodes, weights = setup_sparse_gaussian(n, iprec)

    def f(x):
        return np.sum(x**2, 1)

    intf = f(nodes) @ weights
    assert isclose(intf, n)
