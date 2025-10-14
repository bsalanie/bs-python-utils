"""examples using my sklearn code"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from bs_python_utils.sklearn_utils import skl_npreg_lasso

n_obs = 10000
X1 = -2.0 + 3.0 * np.random.uniform(size=n_obs)
X2 = np.random.normal(loc=1.0, scale=2.0, size=n_obs)
y = X1 * X2 * X2 / 100.0 - (X1 / 5.0 - X2 / 3.0) ** 3 + np.random.normal(size=n_obs)

X = np.column_stack((X1, X2))

plt.style.use("seaborn-v0_8")

degree = 10
stdsc = StandardScaler()
sfit = stdsc.fit(X)
X_scaled = sfit.transform(X)
pf = PolynomialFeatures(degree)
# Create the features and fit
X_poly = pf.fit_transform(X_scaled)

y_pred = skl_npreg_lasso(y, X, alpha=0.001)

plt.clf()

ax = plt.axes()
ax.scatter(y, y_pred)
ax.plot(y, y, "-r")
plt.show()
