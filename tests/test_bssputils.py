import numpy as np
import scipy.stats as sts

from bs_python_utils import bssputils


def test_describe_array_reports_std_deviation(capsys):
    data = np.array([1.0, 2.0, 4.0, 5.0])

    result = bssputils.describe_array(data, name="sample")

    captured = capsys.readouterr().out
    assert "sample has:" in captured
    assert "Std deviation:" in captured
    expected = sts.describe(data, None)
    assert result.nobs == expected.nobs
    assert result.mean == expected.mean
    assert result.minmax == expected.minmax


def test_spline_reg_interpolates_unsorted_inputs():
    x = np.array([2.0, 0.0, 1.0, 3.0])
    y = x**2
    x_new = np.linspace(0.0, 3.0, num=7)

    y_hat = bssputils.spline_reg(y, x, x_new=x_new, is_sorted=False, smooth=False)

    expected = x_new**2
    np.testing.assert_allclose(y_hat, expected, rtol=1e-9, atol=1e-9)
