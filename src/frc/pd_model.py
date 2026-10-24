"""Part b 12-month PD models and metrics (fixed settings, no tuning)."""
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from frc import metrics as M
from frc.freddie import CAT, NUM
from frc.models import fit_gbm


def _pre(scale=True):
    num = make_pipeline(SimpleImputer(strategy="median", add_indicator=True),
                        *([StandardScaler()] if scale else []))
    return ColumnTransformer([("num", num, NUM),
                              ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT)])


def fit_pd_lr(X, y):
    return make_pipeline(_pre(), LogisticRegression(C=1.0, max_iter=2000)).fit(X[NUM + CAT], y)


def fit_pd_gbm(X, y):
    pre = _pre(scale=False).fit(X[NUM + CAT])
    gbm = fit_gbm(pre.transform(X[NUM + CAT]), y)
    return make_pipeline(pre, gbm)


def audit_preproc_train_only(pipe, X_train):
    """Imputer medians and scaler means must equal train-only statistics."""
    num = pipe[0].named_transformers_["num"]
    Xn = X_train[NUM].to_numpy(float)
    med = np.nanmedian(Xn, axis=0)
    assert np.allclose(num[0].statistics_, med, equal_nan=True), "imputer not fit on train only"
    if len(num.steps) > 1:
        Z = num[0].transform(X_train[NUM])
        assert np.allclose(num[1].mean_, Z.mean(0)), "scaler not fit on train only"


def pd_metrics(y, p):
    fpr, tpr, _ = roc_curve(y, p)
    auc = roc_auc_score(y, p)
    return dict(n=len(y), default_rate=y.mean(), mean_pd=p.mean(), citl=p.mean() - y.mean(),
                auc=auc, gini=2 * auc - 1, ks=float(np.max(tpr - fpr)),
                pr_auc=average_precision_score(y, p), brier=M.brier(y, p), ece=M.ece(y, p))
