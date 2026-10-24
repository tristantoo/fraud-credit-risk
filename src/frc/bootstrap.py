"""Seeded paired bootstrap (i.i.d. rows or contiguous time blocks) and Holm adjustment."""
import numpy as np
import pandas as pd


def resample_indices(n, B, seed, blocks=None):
    """Yield sorted index arrays (sorted so 'ties by row order' keeps meaning).
    blocks: per-row block label; whole blocks are resampled with replacement."""
    rng = np.random.default_rng(seed)
    if blocks is not None:
        labels, inv = np.unique(blocks, return_inverse=True)
        rows = [np.flatnonzero(inv == i) for i in range(len(labels))]
    for _ in range(B):
        if blocks is None:
            yield np.sort(rng.integers(0, n, n))
        else:
            yield np.sort(np.concatenate([rows[i] for i in rng.integers(0, len(rows), len(rows))]))


def summarize(point: dict, draws: pd.DataFrame) -> pd.DataFrame:
    """Percentile 95% CI and two-sided p = 2*min(P*(d<=0), P*(d>=0)), capped at 1."""
    rows = []
    for k, v in point.items():
        d = draws[k].to_numpy()
        lo, hi = np.percentile(d, [2.5, 97.5])
        p = min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))
        rows.append(dict(contrast=k, estimate=v, ci_lo=lo, ci_hi=hi, p=p))
    return pd.DataFrame(rows)


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for r, i in enumerate(order):
        running = max(running, (len(p) - r) * p[i])
        adj[i] = min(1.0, running)
    return adj
