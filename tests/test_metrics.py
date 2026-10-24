import numpy as np
from frc import metrics as M


def test_top_k_ties_by_row_order():
    m = M.top_k_mask(np.array([0.5, 0.9, 0.9, 0.1] * 50), 0.005)  # n=200 -> k=1
    assert m.sum() == 1 and m[1]


def test_precision_recall_at():
    y = np.r_[np.ones(2), np.zeros(398)].astype(int)
    s = np.r_[0.9, 0.1, np.linspace(0.8, 0.0, 398)]  # n=400 -> k=2: rows 0 and 2
    p, r = M.precision_recall_at(y, s)
    assert p == 0.5 and r == 0.5


def test_choose_threshold_brute_force():
    rng = np.random.default_rng(0)
    y = (rng.random(300) < 0.1).astype(int)
    s = np.round(rng.random(300), 2)  # ties on purpose
    a = rng.exponential(50, 300)
    for c in (1.0, 5.0, 25.0):
        t = M.choose_threshold(y, s, a, c)
        cands = np.r_[np.inf, np.unique(s)]
        costs = [M.cost(y, s >= u, a, c) for u in cands]
        best = min(costs)
        assert np.isclose(M.cost(y, s >= t, a, c), best)
        assert t == max(u for u, k in zip(cands, costs) if np.isclose(k, best))


def test_choose_threshold_flags_nothing_when_review_too_expensive():
    y, s, a = np.array([0, 1, 0]), np.array([0.2, 0.5, 0.9]), np.array([1.0, 3.0, 1.0])
    assert M.choose_threshold(y, s, a, 100.0) == np.inf


def test_ece_perfect_and_bad():
    y = np.r_[np.zeros(50), np.ones(50)]
    assert M.ece(y, y.astype(float)) == 0
    # tied p: bins follow row order -> 5 bins all-0 (|.9-0|) and 5 all-1 (|.9-1|)
    assert np.isclose(M.ece(y, np.full(100, 0.9)), 0.5)


def test_reliability_table_shape():
    rng = np.random.default_rng(1)
    p = rng.random(1000)
    t = M.reliability_table((rng.random(1000) < p).astype(int), p)
    assert len(t) == 11 and t.n.iloc[:10].sum() == 1000 and t.n.iloc[10] == 5
