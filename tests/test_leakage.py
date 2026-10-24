"""Leakage audits must PASS on the real split and FAIL on deliberately leaky fixtures."""
import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from frc import data, leakage as L
from frc.models import fit_lr

FETCH = ("uv run python -c \"from sklearn.datasets import fetch_openml; "
         "fetch_openml(data_id=1597, data_home='data/raw/openml_cache')\"")
real = pytest.mark.skipif(not data.PARQUET.exists() and not data.ARFF.exists(),
                          reason=f"REAL-DATA LEAKAGE AUDIT NOT RUN: data/raw/ is empty. Download: {FETCH}")


@pytest.fixture(scope="module")
def parts():
    return data.split(data.load())


def toy(n=400, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(rng.normal(size=(n, 28)), columns=data.V)
    df["Time"] = np.arange(n, dtype=float)
    df["Amount"] = rng.exponential(50, n) + 0.01  # strictly positive
    df["Class"] = (rng.random(n) < 0.1).astype(int)
    return df


# 1. time ordering
@real
def test_time_order_real(parts):
    L.audit_time_order(*parts)


def test_time_order_bites():
    tr, va, te = data.split(toy(), 200, 300)
    L.audit_time_order(tr, va, te)
    with pytest.raises(AssertionError):  # interleaved rows
        L.audit_time_order(pd.concat([tr, te.iloc[:5]]), va, te)


# 2. preprocessing fit on train only
@real
def test_scaler_train_only_real(parts):
    Xtr = data.lr_features(parts[0])
    L.audit_scaler_train_only(fit_lr(Xtr, parts[0]["Class"])[0], Xtr)


def test_scaler_train_only_bites():
    tr, _, te = data.split(toy(), 200, 300)
    te = te.assign(V1=te.V1 + 3)  # later period drifts, so fitting on it moves the mean
    Xtr, Xte = data.lr_features(tr), data.lr_features(te)
    L.audit_scaler_train_only(StandardScaler().fit(Xtr), Xtr)
    with pytest.raises(AssertionError):
        L.audit_scaler_train_only(StandardScaler().fit(pd.concat([Xtr, Xte])), Xtr)


# 3. no label-derived features
@real
def test_no_label_features_real(parts):
    tr = parts[0]
    L.audit_no_label_features(data.lr_features(tr), tr["Class"])
    L.audit_no_label_features(data.gbm_features(tr), tr["Class"])


@pytest.mark.parametrize("bad", [
    lambda X, d: X.assign(Class=d.Class),                 # label itself
    lambda X, d: X.assign(Time=d.Time),                   # Time as a feature
    lambda X, d: X.assign(leak=d.Class.astype(float)),    # copy of Class under a new name
    lambda X, d: X.assign(V1=d.Class.astype(float)),      # copy of Class under an allowed name
    lambda X, d: X.assign(Amount=d.Class * d.Amount),     # Class*Amount under an allowed name
], ids=["Class", "Time", "class_copy_new_name", "class_copy_as_V1", "class_x_amount_as_Amount"])
def test_no_label_features_bites(bad):
    d = toy()
    X = data.gbm_features(d)
    L.audit_no_label_features(X, d.Class)
    with pytest.raises(AssertionError):
        L.audit_no_label_features(bad(X, d), d.Class)


@real
def test_class_times_amount_caught_on_real_train(parts):
    """On real train Class*Amount has AUC 0.978 (16 frauds have Amount 0): an AUC-only rule misses it."""
    tr = parts[0]
    X = data.gbm_features(tr).assign(Amount=tr.Class * tr.Amount)
    with pytest.raises(AssertionError, match="constant within Class=0"):
        L.audit_no_label_features(X, tr.Class)


# 4. no shared rows (Time ignored: it differs across slices by construction)
@real
def test_no_row_overlap_real(parts):
    tr, va, te = parts
    print("feature twins in train: test", L.audit_no_row_overlap(tr, te),
          "val", L.audit_no_row_overlap(tr, va))


def test_no_row_overlap_bites():
    tr, _, te = data.split(toy(), 200, 300)
    assert L.audit_no_row_overlap(tr, te) == 0
    fraud = te[te.Class == 1].iloc[[0]]
    legit = te[te.Class == 0].iloc[[0]]
    # planted twins differing only in Time: a non-fraud twin is counted, a fraud twin fails
    assert L.audit_no_row_overlap(pd.concat([tr, legit.assign(Time=0.5)]), te) == 1
    with pytest.raises(AssertionError, match="fraud"):
        L.audit_no_row_overlap(pd.concat([tr, fraud.assign(Time=0.5)]), te)
    with pytest.raises(AssertionError, match="fraud"):  # label flipped on the train copy
        L.audit_no_row_overlap(pd.concat([tr, legit.assign(Time=0.5, Class=1)]), te)
