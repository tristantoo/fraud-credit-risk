"""Shared metrics: capacity precision/recall, cost model, cutoff choice, calibration."""
import math

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def top_k_mask(score, frac=0.005):
    """Flag the ceil(frac*n) highest scores; ties broken by row order (stable sort)."""
    k = math.ceil(frac * len(score))
    order = np.argsort(-np.asarray(score), kind="stable")
    m = np.zeros(len(score), bool)
    m[order[:k]] = True
    return m


def precision_recall_at(y, score, frac=0.005):
    m = top_k_mask(score, frac)
    tp = y[m].sum()
    return tp / m.sum(), (tp / y.sum() if y.sum() else np.nan)


def cost(y, flag, amount, c_review):
    """C per flagged row + Amount of every unflagged fraud (no recovery assumed)."""
    return c_review * flag.sum() + amount[(y == 1) & ~flag].sum()


def choose_threshold(y, score, amount, c_review):
    """Score threshold minimising validation cost over all unique scores plus 'flag nothing' (inf).
    Flag rule is score >= t. Ties in cost go to the higher threshold."""
    order = np.argsort(-score, kind="stable")
    s, caught = score[order], (amount * (y == 1))[order]
    last = np.r_[s[1:] != s[:-1], True]  # last row of each unique-score group
    n_flag = np.arange(1, len(s) + 1)[last]
    c = c_review * n_flag + caught.sum() - np.cumsum(caught)[last]
    thr = np.r_[np.inf, s[last]]
    c = np.r_[caught.sum(), c]
    return float(thr[np.argmin(c)])  # argmin returns first min -> highest threshold on ties


def ece(y, p, bins=10):
    """Expected calibration error with equal-mass bins on predicted probability."""
    order = np.argsort(p, kind="stable")
    return sum(len(b) / len(p) * abs(p[b].mean() - y[b].mean()) for b in np.array_split(order, bins))


def reliability_table(y, p, bins=10, top_frac=0.005):
    order = np.argsort(p, kind="stable")
    rows = [(f"bin{i + 1}", b) for i, b in enumerate(np.array_split(order, bins))]
    rows.append(("top0.5%", np.flatnonzero(top_k_mask(p, top_frac))))
    return pd.DataFrame([dict(bin=name, n=len(b), mean_pred=p[b].mean(), obs_rate=y[b].mean(),
                              n_pos=int(y[b].sum()), p_min=p[b].min(), p_max=p[b].max())
                         for name, b in rows])


def brier(y, p):
    return float(np.mean((p - y) ** 2))


def all_metrics(y, p, amount=None, thr=None, c_review=5.0, prior=None):
    prec, rec = precision_recall_at(y, p)
    out = dict(prec_at_0p5=prec, recall_at_0p5=rec, pr_auc=average_precision_score(y, p),
               roc_auc=roc_auc_score(y, p), brier=brier(y, p), ece=ece(y, p))
    if prior is not None:
        out["brier_skill"] = 1 - out["brier"] / brier(y, np.full(len(y), prior))
    if amount is not None and thr is not None:
        flag = p >= thr
        out.update(cost=cost(y, flag, amount, c_review), flag_rate=flag.mean(),
                   cost_top0p5=cost(y, top_k_mask(p), amount, c_review),
                   cost_nothing=cost(y, np.zeros(len(y), bool), amount, c_review))
    return out
