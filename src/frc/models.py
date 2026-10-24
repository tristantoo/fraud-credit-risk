"""Part a models, hyperparameters fixed a priori in docs/PLAN.md a3 (no tuning)."""
import os

os.environ.setdefault("OMP_NUM_THREADS", "4")
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402


def fit_lr(X, y):
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)).fit(X, y)


def fit_gbm(X, y):
    return HistGradientBoostingClassifier(learning_rate=0.05, max_iter=300, max_leaf_nodes=31,
                                          min_samples_leaf=20, l2_regularization=1.0,
                                          early_stopping=False, random_state=0).fit(X, y)
