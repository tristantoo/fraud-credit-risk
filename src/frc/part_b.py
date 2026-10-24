"""Part b analysis: vintage curves, roll rates, out-of-time 12-month PD (docs/PLAN.md b3)."""
from pathlib import Path

import numpy as np
import pandas as pd

from frc import freddie as F, metrics as M
from frc.bootstrap import holm, resample_indices, summarize
from frc.pd_model import audit_preproc_train_only, fit_pd_gbm, fit_pd_lr, pd_metrics

SEED = 20261023


def analyze(orig_paths, svcg_paths, out_dir, B=5000, tag=""):
    """Run part b end to end and write CSVs to out_dir. `tag` prefixes file names (e.g. SYNTHETIC_)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    orig, ev, dec, rolls, lo, hi = F.build(orig_paths, svcg_paths)
    F.vintage_curves(orig, ev, hi).to_csv(out / f"{tag}part_b_vintage_curves.csv", index=False)
    F.roll_rates(rolls).to_csv(out / f"{tag}part_b_roll_rates.csv", index=False)

    train_s, s_test = F.oot_snapshots(hi, lo)
    F.audit_oot(train_s, s_test)
    tr = pd.concat([F.snapshot_frame(orig, dec, ev, s) for s in train_s], ignore_index=True)
    te = F.snapshot_frame(orig, dec, ev, s_test)
    for fr in (tr, te):
        F.audit_features_pre_snapshot(fr, F.NUM + F.CAT)
    models = {"LR": fit_pd_lr(tr, tr.label), "GBM": fit_pd_gbm(tr, tr.label)}
    audit_preproc_train_only(models["LR"], tr)
    y = te.label.to_numpy()
    p = {m: mod.predict_proba(te[F.NUM + F.CAT])[:, 1] for m, mod in models.items()}

    met = pd.DataFrame([dict(model=m, train_snapshots=f"{train_s[0]}-{train_s[-1]}", test_snapshot=s_test,
                             n_train=len(tr), train_default_rate=tr.label.mean(), **pd_metrics(y, p[m]))
                        for m in p])
    met.to_csv(out / f"{tag}part_b_pd_metrics.csv", index=False)
    pd.concat([M.reliability_table(y, p[m]).assign(model=m) for m in p]).to_csv(
        out / f"{tag}part_b_pd_reliability.csv", index=False)

    # Bootstrap over test loans: H5 (LR AUC > 0.5), H6 (LR CITL CI contains 0), GBM-LR AUC & Brier (Holm, 2)
    draws = []
    for idx in resample_indices(len(y), B, SEED):
        if y[idx].min() == y[idx].max():
            continue  # AUC undefined on a one-class resample
        r = {m: pd_metrics(y[idx], p[m][idx]) for m in p}
        draws.append(dict(LR_auc_minus_0p5=r["LR"]["auc"] - 0.5, LR_citl=r["LR"]["citl"],
                          d_auc=r["GBM"]["auc"] - r["LR"]["auc"], d_brier=r["GBM"]["brier"] - r["LR"]["brier"]))
    pt = {m: pd_metrics(y, p[m]) for m in p}
    point = dict(LR_auc_minus_0p5=pt["LR"]["auc"] - 0.5, LR_citl=pt["LR"]["citl"],
                 d_auc=pt["GBM"]["auc"] - pt["LR"]["auc"], d_brier=pt["GBM"]["brier"] - pt["LR"]["brier"])
    t = summarize(point, pd.DataFrame(draws))
    t["p_holm"] = np.nan
    sec = t.contrast.isin(["d_auc", "d_brier"])
    t.loc[sec, "p_holm"] = holm(t.loc[sec, "p"])
    t["B_used"] = len(draws)
    t.to_csv(out / f"{tag}part_b_pd_contrasts.csv", index=False)
    return met, t
