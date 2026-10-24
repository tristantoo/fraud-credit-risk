"""Leakage audits (docs/PLAN.md a7). Each raises AssertionError on violation."""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ALLOWED = {f"V{i}" for i in range(1, 29)} | {"Amount", "log1p_Amount"}


def audit_time_order(train, val, test):
    assert train["Time"].max() < val["Time"].min(), "train overlaps validation in time"
    assert val["Time"].max() < test["Time"].min(), "validation overlaps test in time"


def audit_scaler_train_only(scaler, X_train):
    X = np.asarray(X_train, float)
    assert np.allclose(scaler.mean_, X.mean(0), rtol=1e-9, atol=1e-12), "scaler mean != train mean"
    assert np.allclose(scaler.scale_, X.std(0), rtol=1e-9, atol=1e-12), "scaler scale != train std"


def audit_no_label_features(X_train: pd.DataFrame, y_train, max_auc=0.999):
    """Features are whitelisted names (never Class/Time) and none is label-derived in content:
    no feature (near-)perfectly separates the label, and none is constant within a class
    (catches Class*x leaks, which have AUC < 1 whenever x ties at 0 for some frauds)."""
    cols = set(X_train.columns)
    assert not cols & {"Class", "Time"}, f"forbidden feature(s): {cols & {'Class', 'Time'}}"
    assert cols <= ALLOWED, f"non-whitelisted feature(s): {cols - ALLOWED}"
    y = np.asarray(y_train)
    for c in X_train.columns:
        x = X_train[c].to_numpy()
        auc = roc_auc_score(y, x)
        assert max(auc, 1 - auc) < max_auc, f"feature {c} separates the label (AUC {auc:.4f})"
        for k in (0, 1):
            assert np.unique(x[y == k]).size > 1, f"feature {c} is constant within Class={k}"


def audit_no_row_overlap(train, test, ignore=("Time", "Class")):
    """Twins = rows identical in every column except Time (differs across slices by construction)
    and the label. Non-fraud twins are tolerated and counted (PLAN deviation 9); a twin carrying a
    fraud label on either side fails. Returns the number of test rows with a twin in train."""
    cols = [c for c in test.columns if c not in ignore]
    htr = pd.util.hash_pandas_object(train[cols], index=False)
    hte = pd.util.hash_pandas_object(test[cols], index=False)
    shared = set(htr) & set(hte)
    in_tr, in_te = htr.isin(shared).to_numpy(), hte.isin(shared).to_numpy()
    n_fraud = int(train["Class"].to_numpy()[in_tr].sum() + test["Class"].to_numpy()[in_te].sum())
    assert n_fraud == 0, f"{n_fraud} fraud row(s) have a feature-identical twin across the split"
    return int(in_te.sum())
