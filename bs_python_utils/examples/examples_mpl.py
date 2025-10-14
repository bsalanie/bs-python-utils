"""examples using my Matplotlib functions"""

import numpy as np

from bs_python_utils.bsmplutils import bs_mpl_plot_dcm_fit

n_obs = 100
n_y = 4
rng = np.random.default_rng(12345)
y = rng.integers(0, n_y, size=n_obs)
probs = rng.uniform(size=n_obs * n_y).reshape((n_obs, n_y))

bs_mpl_plot_dcm_fit(y, probs, save_to="../Graphs/dcm_fit")
