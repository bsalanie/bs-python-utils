import math
from math import cos, exp, pi, sin

from bs_python_utils.numerical.bs_derivs import (
    richardson_derivative,
    richardson_second_derivative,
)


class TestRichardsonDerivative:
    """Tests for richardson_derivative function."""

    def test_linear_function(self):
        """First derivative of ax + b is a."""

        # f(x) = 2x + 3, f'(x) = 2
        def f(x, args):
            return 2 * x + 3

        x = 1.0
        h = 0.01
        result = richardson_derivative(f, x, [], h)
        assert math.isclose(result, 2.0, rel_tol=1e-10)

    def test_quadratic_function(self):
        """First derivative of x^2 is 2x."""

        # f(x) = x^2, f'(x) = 2x
        def f(x, args):
            return x**2

        x = 3.0
        h = 0.01
        result = richardson_derivative(f, x, [], h)
        expected = 2 * x
        assert math.isclose(result, expected, rel_tol=1e-8)

    def test_cubic_function(self):
        """First derivative of x^3 is 3x^2."""

        # f(x) = x^3, f'(x) = 3x^2
        def f(x, args):
            return x**3

        x = 2.0
        h = 0.01
        result = richardson_derivative(f, x, [], h)
        expected = 3 * x**2
        assert math.isclose(result, expected, rel_tol=1e-8)

    def test_quartic_function(self):
        """First derivative of x^4 is 4x^3."""

        # f(x) = x^4, f'(x) = 4x^3
        def f(x, args):
            return x**4

        x = 1.5
        h = 0.01
        result = richardson_derivative(f, x, [], h)
        expected = 4 * x**3
        assert math.isclose(result, expected, rel_tol=1e-6)

    def test_exponential_function(self):
        """First derivative of e^x is e^x."""

        # f(x) = e^x, f'(x) = e^x
        def f(x, args):
            return exp(x)

        x = 0.5
        h = 0.01
        result = richardson_derivative(f, x, [], h)
        expected = exp(x)
        assert math.isclose(result, expected, rel_tol=1e-8)

    def test_sine_function(self):
        """First derivative of sin(x) is cos(x)."""

        # f(x) = sin(x), f'(x) = cos(x)
        def f(x, args):
            return sin(x)

        x = pi / 4
        h = 0.01
        result = richardson_derivative(f, x, [], h)
        expected = cos(x)
        assert math.isclose(result, expected, rel_tol=1e-8)

    def test_cosine_function(self):
        """First derivative of cos(x) is -sin(x)."""

        # f(x) = cos(x), f'(x) = -sin(x)
        def f(x, args):
            return cos(x)

        x = pi / 3
        h = 0.01
        result = richardson_derivative(f, x, [], h)
        expected = -sin(x)
        assert math.isclose(result, expected, rel_tol=1e-8)

    def test_with_args_parameter(self):
        """Test that args are passed through correctly."""

        # f(x, args) = a*x^2 + b*x + c where args = [a, b, c]
        def f(x, args):
            a, b, c = args
            return a * x**2 + b * x + c

        x = 2.0
        h = 0.01
        args = [3.0, 2.0, 1.0]
        result = richardson_derivative(f, x, args, h)
        expected = 2 * args[0] * x + args[1]  # f'(x) = 2ax + b
        assert math.isclose(result, expected, rel_tol=1e-8)

    def test_zero_at_critical_point(self):
        """Derivative should be near zero at critical points."""

        # f(x) = x^2, critical point at x = 0
        def f(x, args):
            return x**2

        x = 0.0
        h = 0.01
        result = richardson_derivative(f, x, [], h)
        assert abs(result) < 1e-10

    def test_different_step_sizes(self):
        """Results should be accurate with different step sizes."""

        # f(x) = sin(x), f'(x) = cos(x)
        def f(x, args):
            return sin(x)

        x = pi / 6
        expected = cos(x)

        for h in [0.1, 0.01, 0.001]:
            result = richardson_derivative(f, x, [], h)
            # Smaller h should generally be more accurate
            assert math.isclose(result, expected, rel_tol=1e-5)

    def test_negative_x(self):
        """Should work correctly for negative x values."""

        # f(x) = x^3, f'(x) = 3x^2
        def f(x, args):
            return x**3

        x = -2.5
        h = 0.01
        result = richardson_derivative(f, x, [], h)
        expected = 3 * x**2
        assert math.isclose(result, expected, rel_tol=1e-8)

    def test_large_x(self):
        """Should work correctly for large x values."""

        # f(x) = x^2, f'(x) = 2x
        def f(x, args):
            return x**2

        x = 100.0
        h = 0.1
        result = richardson_derivative(f, x, [], h)
        expected = 2 * x
        assert math.isclose(result, expected, rel_tol=1e-8)

    def test_small_x(self):
        """Should work correctly for small x values close to zero."""

        # f(x) = x^2, f'(x) = 2x
        def f(x, args):
            return x**2

        x = 0.001
        h = 0.00001
        result = richardson_derivative(f, x, [], h)
        expected = 2 * x
        assert math.isclose(result, expected, rel_tol=1e-6)

    def test_fourth_order_accuracy(self):
        """Richardson extrapolation should achieve fourth-order accuracy."""

        # For a polynomial of degree 4 or less, the method should be exact
        # f(x) = x^4 - 2x^3 + x^2, f'(x) = 4x^3 - 6x^2 + 2x
        def f(x, args):
            return x**4 - 2 * x**3 + x**2

        x = 1.5
        h = 0.01
        result = richardson_derivative(f, x, [], h)
        expected = 4 * x**3 - 6 * x**2 + 2 * x
        # Fourth-order method should be very accurate for polynomials
        assert math.isclose(result, expected, rel_tol=1e-10)


class TestRichardsonSecondDerivative:
    """Tests for richardson_second_derivative function."""

    def test_quadratic_function(self):
        """Second derivative of x^2 is 2."""

        # f(x) = x^2, f''(x) = 2
        def f(x, args):
            return x**2

        x = 3.0
        h = 0.01
        result = richardson_second_derivative(f, x, [], h)
        assert math.isclose(result, 2.0, rel_tol=1e-6)

    def test_cubic_function(self):
        """Second derivative of x^3 is 6x."""

        # f(x) = x^3, f''(x) = 6x
        def f(x, args):
            return x**3

        x = 2.0
        h = 0.01
        result = richardson_second_derivative(f, x, [], h)
        expected = 6 * x
        assert math.isclose(result, expected, rel_tol=1e-6)

    def test_quartic_function(self):
        """Second derivative of x^4 is 12x^2."""

        # f(x) = x^4, f''(x) = 12x^2
        def f(x, args):
            return x**4

        x = 1.5
        h = 0.01
        result = richardson_second_derivative(f, x, [], h)
        expected = 12 * x**2
        assert math.isclose(result, expected, rel_tol=1e-5)

    def test_exponential_function(self):
        """Second derivative of e^x is e^x."""

        # f(x) = e^x, f''(x) = e^x
        def f(x, args):
            return exp(x)

        x = 0.5
        h = 0.01
        result = richardson_second_derivative(f, x, [], h)
        expected = exp(x)
        assert math.isclose(result, expected, rel_tol=1e-6)

    def test_sine_function(self):
        """Second derivative of sin(x) is -sin(x)."""

        # f(x) = sin(x), f''(x) = -sin(x)
        def f(x, args):
            return sin(x)

        x = pi / 4
        h = 0.01
        result = richardson_second_derivative(f, x, [], h)
        expected = -sin(x)
        assert math.isclose(result, expected, rel_tol=1e-5)

    def test_cosine_function(self):
        """Second derivative of cos(x) is -cos(x)."""

        # f(x) = cos(x), f''(x) = -cos(x)
        def f(x, args):
            return cos(x)

        x = pi / 3
        h = 0.01
        result = richardson_second_derivative(f, x, [], h)
        expected = -cos(x)
        assert math.isclose(result, expected, rel_tol=1e-5)

    def test_with_args_parameter(self):
        """Test that args are passed through correctly."""

        # f(x, args) = a*x^2 + b*x + c where args = [a, b, c]
        def f(x, args):
            a, b, c = args
            return a * x**2 + b * x + c

        x = 2.0
        h = 0.01
        args = [3.0, 2.0, 1.0]
        result = richardson_second_derivative(f, x, args, h)
        expected = 2 * args[0]  # f''(x) = 2a
        assert math.isclose(result, expected, rel_tol=1e-8)

    def test_inflection_point(self):
        """Second derivative should change sign at inflection point."""

        # f(x) = x^3, inflection point at x = 0
        def f(x, args):
            return x**3

        h = 0.01
        # Before inflection point
        result_neg = richardson_second_derivative(f, -1.0, [], h)
        # After inflection point
        result_pos = richardson_second_derivative(f, 1.0, [], h)

        assert result_neg < 0
        assert result_pos > 0

    def test_negative_x(self):
        """Should work correctly for negative x values."""

        # f(x) = x^4, f''(x) = 12x^2
        def f(x, args):
            return x**4

        x = -1.5
        h = 0.01
        result = richardson_second_derivative(f, x, [], h)
        expected = 12 * x**2
        assert math.isclose(result, expected, rel_tol=1e-5)

    def test_large_x(self):
        """Should work correctly for large x values."""

        # f(x) = x^2, f''(x) = 2
        def f(x, args):
            return x**2

        x = 100.0
        h = 0.1
        result = richardson_second_derivative(f, x, [], h)
        assert math.isclose(result, 2.0, rel_tol=1e-5)

    def test_composition_with_derivatives(self):
        """Verify f''(x) by comparing with numerical first derivative of f'(x)."""

        # f(x) = sin(x)
        # Compute f'' using the second derivative function
        def f(x, args):
            return sin(x)

        x = pi / 6
        h = 0.01

        second_deriv = richardson_second_derivative(f, x, [], h)
        expected = -sin(x)

        assert math.isclose(second_deriv, expected, rel_tol=1e-5)

    def test_quartic_polynomial_exact(self):
        """For degree 4 polynomial, second derivative should be very accurate."""

        # f(x) = x^4 - 2x^3 + x^2 + x - 1
        # f'(x) = 4x^3 - 6x^2 + 2x + 1
        # f''(x) = 12x^2 - 12x + 2
        def f(x, args):
            return x**4 - 2 * x**3 + x**2 + x - 1

        x = 1.5
        h = 0.01
        result = richardson_second_derivative(f, x, [], h)
        expected = 12 * x**2 - 12 * x + 2
        assert math.isclose(result, expected, rel_tol=1e-8)
