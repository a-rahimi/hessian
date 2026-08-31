"""Accumulate wall-clock time per named pipeline stage.

Usage:
    with timing.record("solve/splu/factorize"):
        ...
    print(timing.report())
"""

import contextlib
import time

# Maps stage name -> (total_seconds, call_count).
_registry: dict[str, tuple[float, int]] = {}


@contextlib.contextmanager
def record(name: str):
    "Add the wall-clock duration of the enclosed block to the stage `name`."
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        total, count = _registry.get(name, (0.0, 0))
        _registry[name] = (total + elapsed, count + 1)


def reset() -> None:
    _registry.clear()


def report() -> str:
    "Format the accumulated timings as a table sorted by total time descending."
    if not _registry:
        return "no timings recorded"
    name_width = max(len("stage"), max(map(len, _registry)))
    lines = [f"{'stage':<{name_width}}  {'calls':>6}  {'total s':>9}  {'mean ms':>9}"]
    for name, (total, count) in sorted(_registry.items(), key=lambda kv: -kv[1][0]):
        lines.append(
            f"{name:<{name_width}}  {count:>6}  {total:>9.3f}  {1e3 * total / count:>9.2f}"
        )
    return "\n".join(lines)
