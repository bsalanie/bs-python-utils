"""examples using distance_covariances"""

import numpy as np

from bs_python_utils.distance_covariances import dcov_dcor, pdcov_pdcor, pvalue_pdcov

# example page 2396 of Szekely and Rizzo 2014
n = 2000
do_bootstrap = False
Z1 = np.random.normal(size=n)
Z2 = np.random.normal(size=n)
Z3 = np.random.normal(size=n)
X = Z1 + Z3
Y = Z2 + Z3
Z = Z3
print("\n\n     Test of page 2396")
dcov_XY = dcov_dcor(X, Y)
dcov_XZ = dcov_dcor(X, Z)
dcov_YZ = dcov_dcor(Y, Z)
pdcov_XYZ = pdcov_pdcor(X, Y, Z)
print(f"         dCor(X, Y)={dcov_XY.dcor}, should be 0.2062 in large samples")
print(f"         dCor(X, Z)={dcov_XZ.dcor}, should be 0.4319 in large samples")
print(f"         dCor(Y, Z)={dcov_YZ.dcor}, should be 0.4319 in large samples")
print(f"         pdCor(X, Y; Z)={pdcov_XYZ.pdcor}, should be 0.0242 in large samples")

if do_bootstrap:
    ndraws = 499
    pval = pvalue_pdcov(pdcov_XYZ)
    print(
        f"\n\n test stat={pdcov_XYZ.pdcov_stat: >.2f} has p-value {pval} for"
        f" {ndraws} draws"
    )
