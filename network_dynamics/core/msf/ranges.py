"""Per-oscillator K-scan ranges from Huang et al. (2009).

Key: (dynamics, source, target) → (K_min, K_max)
"""

from __future__ import annotations

# Fallback used when (dynamics, source, target) is not in the table.
_K_RANGE_FALLBACK: tuple[float, float] = (0.0, 10.0)

K_RANGE: dict[tuple[str, int, int], tuple[float, float]] = {
    ("rossler", 0, 0): (0.,  10.),
    ("rossler", 1, 1): (0.,   5.),
    ("rossler", 2, 0): (0., 100.),
    ("lorenz",  0, 0): (0.,  30.),
    ("lorenz",  0, 1): (0.,  30.),
    ("lorenz",  1, 0): (0.,  50.),
    ("lorenz",  1, 1): (0.,  20.),
    ("lorenz",  2, 2): (0., 100.),
    ("chen",    0, 1): (0.,  30.),
    ("chen",    1, 1): (0.,  20.),
    ("chen",    2, 2): (0., 100.),
    ("chua",    0, 0): (0.,  20.),
    ("chua",    0, 1): (0.,   5.),
    ("chua",    1, 0): (0.,  30.),
    ("chua",    1, 1): (0.,  10.),
    ("chua",    1, 2): (0.,  50.),
    ("chua",    2, 0): (0.,  10.),
    ("chua",    2, 2): (0.,  10.),
    ("hr",      0, 0): (0.,   5.),
    ("hr",      0, 1): (0.,   5.),
    ("hr",      1, 0): (0.,   5.),
    ("hr",      1, 1): (0.,   3.),
}


def default_k_range(dynamics: str, source: int, target: int) -> tuple[float, float]:
    """Return (K_min, K_max) for the given oscillator and coupling scheme."""
    return K_RANGE.get((dynamics, source, target), _K_RANGE_FALLBACK)
