"""Part a data: parse the cached OpenML 1597 ARFF (fetch_openml drops Time) and split on Time."""
import gzip
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
ARFF = ROOT / "data/raw/openml_cache/openml/openml.org/data/v1/download/1673544/creditcard.arff.gz"
PARQUET = ROOT / "data/raw/creditcard.parquet"
ARFF_MD5 = "178bcf9bb1f31a3dfe12d0e577884add"  # uncompressed ARFF, matches OpenML md5_checksum
CUT_VAL, CUT_TEST = 120_000, 145_000  # pre-registered cut points on Time (seconds)
V = [f"V{i}" for i in range(1, 29)]


def parse_arff(path=ARFF) -> pd.DataFrame:
    raw = gzip.open(path, "rb").read()
    assert hashlib.md5(raw).hexdigest() == ARFF_MD5, "ARFF md5 mismatch"
    lines = raw.decode().splitlines()
    cols = [l.split()[1] for l in lines if l.lower().startswith("@attribute")]
    start = next(i for i, l in enumerate(lines) if l.lower().startswith("@data")) + 1
    rows = [l.replace("'", "").split(",") for l in lines[start:] if l and not l.startswith("%")]
    df = pd.DataFrame(rows, columns=cols).astype(float)
    df["Class"] = df["Class"].astype(int)
    return df


def load() -> pd.DataFrame:
    if not PARQUET.exists():
        parse_arff().to_parquet(PARQUET)
    return pd.read_parquet(PARQUET)


def split(df: pd.DataFrame, cut_val=CUT_VAL, cut_test=CUT_TEST):
    """Train earlier, validate in the middle, test later. Rules fixed in docs/PLAN.md a3."""
    t = df["Time"]
    return df[t < cut_val], df[(t >= cut_val) & (t < cut_test)], df[t >= cut_test]


def lr_features(df: pd.DataFrame) -> pd.DataFrame:
    X = df[V].copy()
    X["log1p_Amount"] = np.log1p(df["Amount"])
    return X


def gbm_features(df: pd.DataFrame) -> pd.DataFrame:
    return df[V + ["Amount"]].copy()
