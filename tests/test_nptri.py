import numpy as np

from bs_python_utils.numerical.bsnputils import set_elements_abovebelow_diagonal


def test_nptril():
    a_mat = np.array([[1, -1, 9], [2, 3, 7], [4, 5, 6]])
    a_sub = np.tril(a_mat, -1)
    a_sub_th = np.array([[0, 0, 0], [2, 0, 0], [4, 5, 0]])
    assert np.allclose(a_sub, a_sub_th)
    a_low = np.tril(a_mat)
    a_low_th = np.array([[1, 0, 0], [2, 3, 0], [4, 5, 6]])
    assert np.allclose(a_low, a_low_th)


def test_nptriu():
    a_mat = np.array([[1, -1, 9], [2, 3, 7], [4, 5, 6]])
    a_sup = np.triu(a_mat, 1)
    a_sup_th = np.array([[0, -1, 9], [0, 0, 7], [0, 0, 0]])
    assert np.allclose(a_sup, a_sup_th)
    a_high = np.triu(a_mat)
    a_high_th = np.array([[1, -1, 9], [0, 3, 7], [0, 0, 6]])
    assert np.allclose(a_high, a_high_th)


def test_set_elements_above_diagonal():
    a_mat = np.array([[1, -1, 9], [2, 3, 7], [4, 5, 6]])
    a_up = set_elements_abovebelow_diagonal(a_mat, 8, "above")
    a_up_th = np.array([[1, 8, 8], [2, 3, 8], [4, 5, 6]])
    assert np.allclose(a_up, a_up_th)


def test_set_elements_below_diagonal():
    a_mat = np.array([[1, -1, 9], [2, 3, 7], [4, 5, 6]])
    a_below = set_elements_abovebelow_diagonal(a_mat, 8, "below")
    a_below_th = np.array([[1, -1, 9], [8, 3, 7], [8, 8, 6]])
    assert np.allclose(a_below, a_below_th)


def test_set_elements_on_below_diagonal():
    a_mat = np.array([[1, -1, 9], [2, 3, 7], [4, 5, 6]])
    a_below = set_elements_abovebelow_diagonal(a_mat, 8, "on_below")
    a_below_th = np.array([[8, -1, 9], [8, 8, 7], [8, 8, 8]])
    assert np.allclose(a_below, a_below_th)


def test_set_elements_on_above_diagonal():
    a_mat = np.array([[1, -1, 9], [2, 3, 7], [4, 5, 6]])
    a_below = set_elements_abovebelow_diagonal(a_mat, 8, "on_above")
    a_below_th = np.array([[8, 8, 8], [2, 8, 8], [4, 5, 8]])
    assert np.allclose(a_below, a_below_th)
