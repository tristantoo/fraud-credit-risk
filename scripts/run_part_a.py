"""Part a: run the pre-registered analysis in docs/PLAN.md end to end. Writes outputs/part_a_*."""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_curve  # noqa: E402

from frc import data, leakage as L, metrics as M  # noqa: E402
from frc.bootstrap import holm, resample_indices, summarize  # noqa: E402
from frc.models import fit_gbm, fit_lr  # noqa: E402

OUT = ROOT / "outputs"
SEED, B, C = 20261023, 5000, 5.0
PRIMARY = ["prec_at_0p5", "pr_auc", "cost"]
SECONDARY = ["recall_at_0p5", "roc_auc", "brier", "ece"]
FEATS = {"LR": data.lr_features, "GBM": data.gbm_features}
FITS = {"LR": fit_lr, "GBM": fit_gbm}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def boot(y, s, amt, thr, blocks=None, b=B):
    """Paired bootstrap of all metrics for both models on the same resampled rows."""
    rows = []
    for idx in resample_indices(len(y), b, SEED, blocks):
        r = {}
        for m in s:
            mm = M.all_metrics(y[idx], s[m][idx], amt[idx], thr[m], C)
            r.update({f"{m}_{k}": v for k, v in mm.items()})
        rows.append(r)
    d = pd.DataFrame(rows)
    pt = {}
    for m in s:
        pt.update({f"{m}_{k}": v for k, v in M.all_metrics(y, s[m], amt, thr[m], C).items()})
    return pt, d


def contrasts(pt, d):
    """GBM-LR deltas (primary Holm-adjusted; secondary unadjusted) and H4 cost contrasts."""
    dd, dp = pd.DataFrame(), {}
    for k in PRIMARY + SECONDARY:
        dd[k], dp[k] = d[f"GBM_{k}"] - d[f"LR_{k}"], pt[f"GBM_{k}"] - pt[f"LR_{k}"]
    for m in ("LR", "GBM"):
        for ref in ("nothing", "top0p5"):
            k = f"H4_{m}_cost_minus_cost_{ref}"
            dd[k], dp[k] = d[f"{m}_cost"] - d[f"{m}_cost_{ref}"], pt[f"{m}_cost"] - pt[f"{m}_cost_{ref}"]
    t = summarize(dp, dd)
    t["family"] = ["primary" if k in PRIMARY else "secondary" for k in t.contrast]
    t["p_holm"] = np.nan
    prim = t.family == "primary"
    t.loc[prim, "p_holm"] = holm(t.loc[prim, "p"])
    return t


def main():
    OUT.mkdir(exist_ok=True)
    df = data.load()
    tr, va, te = data.split(df)
    L.audit_time_order(tr, va, te)
    log("non-fraud feature twins (Time ignored, tolerated): test", L.audit_no_row_overlap(tr, te),
        "val", L.audit_no_row_overlap(tr, va))
    for f in FEATS.values():
        L.audit_no_label_features(f(tr), tr["Class"])
    split_tbl = pd.DataFrame([dict(slice=n, n=len(p), time_min=p.Time.min(), time_max=p.Time.max(),
                                   n_fraud=int(p.Class.sum()), prevalence=p.Class.mean(),
                                   fraud_amount=p.Amount[p.Class == 1].sum(),
                                   n_dup_rows=int(p.duplicated().sum()))
                              for n, p in (("train", tr), ("val", va), ("test", te))])
    split_tbl.to_csv(OUT / "part_a_split.csv", index=False)
    log("split\n", split_tbl.to_string())
    prior = tr.Class.mean()
    ytr, yva, yte = (p.Class.to_numpy() for p in (tr, va, te))
    ava, ate = va.Amount.to_numpy(), te.Amount.to_numpy()

    models, sva, ste = {}, {}, {}
    for m in FEATS:
        models[m] = FITS[m](FEATS[m](tr), ytr)
        sva[m] = models[m].predict_proba(FEATS[m](va))[:, 1]
        ste[m] = models[m].predict_proba(FEATS[m](te))[:, 1]
    L.audit_scaler_train_only(models["LR"][0], FEATS["LR"](tr))
    log("models fit; audits passed")

    # Leakage alarms (PLAN a6)
    alarms = {}
    for m in FEATS:
        alarms[f"{m}_test_minus_val_pr_auc"] = average_precision_score(yte, ste[m]) - average_precision_score(yva, sva[m])
        yperm = np.random.default_rng(SEED).permutation(ytr)
        sp = FITS[m](FEATS[m](tr), yperm).predict_proba(FEATS[m](te))[:, 1]
        alarms[f"{m}_permuted_label_test_pr_auc"] = average_precision_score(yte, sp)
    alarms["2x_test_prevalence"] = 2 * yte.mean()
    halt = any(alarms[f"{m}_test_minus_val_pr_auc"] > 0.10 or
               alarms[f"{m}_permuted_label_test_pr_auc"] > alarms["2x_test_prevalence"] for m in FEATS)
    alarms["halt"] = bool(halt)
    log("alarms", alarms)
    json.dump(alarms, open(OUT / "part_a_leakage_alarms.json", "w"), indent=2)
    if halt:
        sys.exit("HALT: leakage alarm tripped (PLAN a6); audit before believing results")

    # Cutoffs on validation, applied unchanged to test (PLAN a4); C sensitivity is descriptive
    thr = {m: M.choose_threshold(yva, sva[m], ava, C) for m in FEATS}
    cut_rows = []
    for c in (1.0, 5.0, 25.0):
        for m in FEATS:
            t = M.choose_threshold(yva, sva[m], ava, c)
            flag, bayes = ste[m] >= t, ste[m] * ate > c
            cut_rows.append(dict(C=c, model=m, threshold=t, val_cost=M.cost(yva, sva[m] >= t, ava, c),
                                 val_flag_rate=(sva[m] >= t).mean(), test_flag_rate=flag.mean(),
                                 test_cost=M.cost(yte, flag, ate, c),
                                 test_cost_top0p5=M.cost(yte, M.top_k_mask(ste[m]), ate, c),
                                 test_cost_nothing=M.cost(yte, np.zeros(len(yte), bool), ate, c),
                                 test_cost_bayes_rule=M.cost(yte, bayes, ate, c),
                                 test_bayes_flag_rate=bayes.mean()))
    cuts = pd.DataFrame(cut_rows)
    cuts["test_cost_per_1000"] = cuts.test_cost / len(yte) * 1000
    cuts.to_csv(OUT / "part_a_cutoffs_cost.csv", index=False)
    log("cutoffs\n", cuts.to_string())

    # Point metrics + calibration next to discrimination, val and test
    mt = pd.DataFrame([dict(slice=sl, model=m, **M.all_metrics(y, s[m], a, thr[m], C, prior))
                       for sl, y, s, a in (("val", yva, sva, ava), ("test", yte, ste, ate)) for m in FEATS])
    mt.to_csv(OUT / "part_a_metrics.csv", index=False)
    log("metrics\n", mt.to_string())
    pd.concat([M.reliability_table(yte, ste[m]).assign(model=m) for m in FEATS]).to_csv(
        OUT / "part_a_reliability_test.csv", index=False)
    plots(yte, ste)

    # Paired bootstraps: i.i.d. rows (primary), 15-min blocks (sensitivity), deduplicated test (sensitivity)
    res = {}
    log("iid bootstrap")
    res["iid"] = contrasts(*boot(yte, ste, ate, thr))
    log("block bootstrap")
    res["block15min"] = contrasts(*boot(yte, ste, ate, thr, blocks=(te.Time // 900).to_numpy()))
    keep = ~te.duplicated().to_numpy()
    log(f"dedup bootstrap (dropped {int((~keep).sum())} test rows)")
    res["dedup_test"] = contrasts(*boot(yte[keep], {m: ste[m][keep] for m in FEATS}, ate[keep], thr))
    allc = pd.concat([t.assign(bootstrap=k) for k, t in res.items()])
    allc.to_csv(OUT / "part_a_contrasts.csv", index=False)
    log("contrasts\n", allc.to_string())


def plots(y, s):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
    for m, p in s.items():
        fpr, tpr, _ = roc_curve(y, p)
        ax[0].plot(fpr, tpr, label=m)
        pr, rc, _ = precision_recall_curve(y, p)
        ax[1].plot(rc, pr, label=f"{m} AP={average_precision_score(y, p):.3f}")
        t = M.reliability_table(y, p, bins=10).iloc[:10]
        ax[2].plot(t.mean_pred, t.obs_rate, "o-", label=m)
    ax[0].plot([0, 1], [0, 1], "k:", lw=0.8)
    ax[0].set(xlabel="FPR", ylabel="TPR", title="ROC (test)")
    ax[1].axhline(y.mean(), color="k", ls=":", lw=0.8)
    ax[1].set(xlabel="recall", ylabel="precision", title="PR (test)")
    lim = max(ax[2].get_xlim()[1], ax[2].get_ylim()[1])
    ax[2].plot([1e-6, lim], [1e-6, lim], "k:", lw=0.8)
    ax[2].set(xscale="log", xlabel="mean predicted (equal-mass bin)",
              ylabel="observed fraud rate", title="Reliability (test, 10 equal-mass bins)")
    ax[2].set_yscale("symlog", linthresh=1e-3)
    for a in ax:
        a.legend()
    fig.tight_layout()
    fig.savefig(OUT / "part_a_curves.png", dpi=120)


if __name__ == "__main__":
    main()
