"""example of using bs_mem"""

import tracemalloc

import numpy as np

from bs_python_utils.bs_mem import (
    memory_display_top,
    memory_display_top_diffs,
    memory_usage,
)

tracemalloc.start()
list_ex = list(range(10000))


v1 = np.ones(54546)
snapshot1 = tracemalloc.take_snapshot()
memory_display_top(snapshot1)
d = {i: li for (i, li) in enumerate(list_ex)}
m = np.random.normal(size=(3498, 12))
memory_usage(5)
snapshot2 = tracemalloc.take_snapshot()
memory_display_top_diffs(snapshot1, snapshot2)
tracemalloc.stop()
del d
memory_usage(5)
