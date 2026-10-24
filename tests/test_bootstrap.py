import numpy as np
import pandas as pd
from frc.bootstrap import holm, resample_indices, summarize


def test_holm_known_values():
    assert np.allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])


def test_resample_seeded_and_blocked():
    a = list(resample_indices(10, 3, 1))
    b = list(resample_indices(10, 3, 1))
    assert all((x == y).all() for x, y in zip(a, b))
    blocks = np.repeat([0, 1, 2], [2, 3, 5])
    for idx in resample_indices(10, 20, 2, blocks):
        sizes = np.bincount(blocks[idx], minlength=3)
        assert all(s % w == 0 for s, w in zip(sizes, [2, 3, 5]))  # whole blocks only


def test_summarize_p_and_ci():
    d = pd.DataFrame({"x": np.r_[np.full(97, 1.0), np.full(3, -1.0)]})
    s = summarize({"x": 1.0}, d).iloc[0]
    assert np.isclose(s.p, 0.06) and s.ci_hi == 1.0
