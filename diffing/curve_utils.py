"""
Small shared helpers for averaging per-prompt curves of possibly different
lengths into one per-position mean curve, for plotting/summary purposes.
Used by method9 (refusal probability) and method10 (free generation) alike,
so it lives here once instead of being copy-pasted into both.
"""

from collections import defaultdict

import numpy as np


def per_position_mean_dense(curves):
    """curves: list of 1D tensors/arrays, DENSE (every position present, no
    gaps), of possibly different lengths -- e.g. per-prompt JSD or
    first-token refusal-probability curves. Returns (positions, means)
    truncated to the longest curve, averaging only over the prompts that
    reached each position."""
    max_len = max((len(c) for c in curves), default=0)
    positions = list(range(max_len))
    means = [float(np.mean([c[i].item() if hasattr(c[i], "item") else c[i]
                             for c in curves if i < len(c)]))
             for i in positions]
    return positions, means


def per_position_mean_sparse(curves_with_positions):
    """curves_with_positions: list of (curve, positions) pairs whose
    positions may be strided/non-contiguous (e.g. exact-mode refusal
    probability, subsampled every `stride` steps). Returns (positions, means)
    over the union of positions seen, averaging over whichever prompts
    reached each one."""
    buckets = defaultdict(list)
    for curve, positions in curves_with_positions:
        values = curve.tolist() if hasattr(curve, "tolist") else list(curve)
        for pos, val in zip(positions, values):
            buckets[pos].append(val)
    positions = sorted(buckets)
    means = [float(np.mean(buckets[p])) for p in positions]
    return positions, means


__all__ = ["per_position_mean_dense", "per_position_mean_sparse"]
