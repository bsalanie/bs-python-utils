"""Reports on memory usage:

* `mem_usage`: prints the top `n` largest global items in memory
* `memory_display_top`: prints the top `n` largest memory allocations
  since tracing started
* `memory_display_top_diffs`: prints the top `n` largest memory
  allocations since the last snapshot.
"""

import linecache
import sys
import tracemalloc

import pandas as pd

from bs_python_utils.bsutils import print_stars


def _obj_size_fmt(num: int) -> str:
    """
    format sizes from bytes to appropriate strings depending on size

    Args:
        num: size of object in bytes

    Returns:
        its formatted size
    """
    if num < 10**3:
        return "{:.2f}{}".format(num, "B")
    elif (num >= 10**3) & (num < 10**6):
        return "{:.2f}{}".format(num / (1.024 * 10**3), "KB")
    elif (num >= 10**6) & (num < 10**9):
        return "{:.2f}{}".format(num / (1.024 * 10**6), "MB")
    else:
        return "{:.2f}{}".format(num / (1.024 * 10**9), "GB")


def memory_usage(n: int | None = 10) -> None:
    """
    prints the top `n` largest global items in memory

    Args:
        n: we report the size of the largest `n` global items

    Returns:
        nothing.
    """
    memory_usage_by_variable = pd.DataFrame(
        {k: sys.getsizeof(v) for (k, v) in globals().items()}, index=["Size"]
    )
    memory_usage_by_variable = memory_usage_by_variable.T
    total_usage = _obj_size_fmt(memory_usage_by_variable["Size"].sum())
    memory_usage_by_variable = memory_usage_by_variable.sort_values(
        by="Size", ascending=False
    ).head(n)
    memory_usage_by_variable["Size"] = memory_usage_by_variable["Size"].apply(
        lambda x: _obj_size_fmt(x)
    )
    print_stars(
        f"Currently used memory = {total_usage}\n\t\t\t\t Top {n} global objects:"
    )
    print(memory_usage_by_variable)

    return


def memory_display_top(
    snapshot: tracemalloc.Snapshot, key_type: str = "lineno", limit: int | None = 5
) -> None:
    """
    prints out the lines with the top `limit` allocations of memory since
    `tracemalloc.start()`

    Args:
        snapshot: obtained from tracemalloc.take_snapshot()
        key_type: 'lineno' gives file and line number; 'traceback' gives all
        limit: how many top allocations we want

    Returns:
        just prints.

    Examples:
       >>> tracemalloc.start()
       >>> .... execute ...
       >>> snapshot = tracemalloc.take_snapshot()
       >>> memory_display_top(snapshot)
    """

    top_stats = snapshot.statistics(key_type)

    print_stars(f"Top {limit} memory allocations")
    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]
        print(
            "#%s: %s:%s: %.1f KiB"
            % (index, frame.filename, frame.lineno, stat.size / 1024)
        )
        line = linecache.getline(frame.filename, frame.lineno).strip()
        if line:
            print("    %s" % line)

    other = top_stats[limit:]
    if other:
        size = sum(stat.size for stat in other)
        print(f"{len(other)} other: {size / 1024:.1f} KiB")
    total = sum(stat.size for stat in top_stats)
    print("Total allocated size: %.1f KiB" % (total / 1024))


def memory_display_top_diffs(
    snapshot1: tracemalloc.Snapshot,
    snapshot2: tracemalloc.Snapshot,
    key_type: str = "lineno",
    limit: int = 5,
) -> None:
    """
    prints out the lines with the top `limit` allocations between the two snapshots

    Args:
        snapshot1: previous snapshot
        snapshot2: new snapshot
        key_type: 'lineno' gives file and line number; 'traceback' gives all
        limit: how many top allocations we want

    Returns:
        just prints.

    Examples:
       >>> tracemalloc.start()
       >>> .... execute ...
       >>> snapshot1 = tracemalloc.take_snapshot()
       >>> .... execute ...
       >>> snapshot2 = tracemalloc.take_snapshot()
       >>>  memory_display_top_diffs(snapshot1, snapshot2)

    """

    top_stats = snapshot2.compare_to(snapshot1, key_type)

    print_stars(f"Top {limit} new memory allocations")
    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]
        print(
            "#%s: %s:%s: %.1f KiB"
            % (index, frame.filename, frame.lineno, stat.size / 1024)
        )
        line = linecache.getline(frame.filename, frame.lineno).strip()
        if line:
            print("    %s" % line)

    other = top_stats[limit:]
    if other:
        size = sum(stat.size for stat in other)
        print(f"{len(other)} other: {size / 1024:.1f} KiB")
    total = sum(stat.size for stat in top_stats)
    print("Total allocated size: %.1f KiB" % (total / 1024))
