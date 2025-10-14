"""This package contains a number of functions that I have found useful in my
programming.

* `bsutils` has (inter alia) some I/O functions, error reporting, and $C^2$
  extensions of log and exp
* `bs_logging` has customized logging
* `bs_mathstr` has Unicode for math strings
* `bs_mem` reports memory usage
* `Timer` has a `Timer` class to time code execution.
* `bsnputils` has Numpy functions
* `bssputils` has Scipy functions
* `sklearn_utils` has Sklearn functions
* `pandas_utils` has Pandas functions
* `bsstats` has TSLS code; nonparametric and flexible estimation; and code to
  draw random samples
* `bs_opt` interfaces with `scipy.optimize`
* `bs_sparse_gaussian` uses sparse integration to evaluate $E f(X)$ for
  $X \\simeq N(0,1)$
* `chebyshev` has Chebyshev interpolation and integration in dimensions 1 and 2
* `distance_covariances` has measures of nonlinear dependence between random variables
* `bivariate_quantiles` computes quantiles and ranks for 2-dimensional random
  variables à la [Chernozhukov-Galichon-Hallin-Henry (*Ann. Stats.* 2017)](
  https://projecteuclid.org/journals/annals-of-statistics/volume-45/issue-1/
  MongeKantorovich-depth-quantiles-ranks-and-signs/10.1214/16-AOS1450.full).

* `bs_plots` gathers plotting routines in Matplotlib, Seaborn and Altair from
  `bsmplutils`, `bs_seaborn`, and `bs_altair`.
"""
