"""Part b pipeline tests on a SYNTHETIC fixture only. Real Freddie results are BLOCKED-ON-TRISTAN."""
import numpy as np
import pandas as pd
import pytest

from frc import freddie as F
from frc.part_b import analyze
from frc.pd_model import audit_preproc_train_only, fit_pd_lr
from synthetic_freddie import write_synthetic


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    return write_synthetic(tmp_path_factory.mktemp("SYNTHETIC"))


@pytest.fixture(scope="module")
def built(synth):
    return F.build([synth[0]], [synth[1]])


def test_layout_check_rejects_wrong_field_count(tmp_path):
    o, _ = write_synthetic(tmp_path, vintages=[2005], n_per=3, nfields=30)
    with pytest.raises(ValueError, match="30 fields"):
        F.read_orig(o)


def test_read_orig_sentinels_and_vintage(synth):
    o = F.read_orig(synth[0])
    assert o.credit_score.max() < 9999 and o.credit_score.isna().any()
    assert o.dti.max() < 999 and set(o.vintage) == set(range(2003, 2009))


def test_months():
    assert F.add_months(200811, 2) == 200901 and F.add_months(200901, -1) == 200812


def tiny():
    rows = [("A", 200811, "0", ""), ("A", 200812, "1", ""), ("A", 200901, "2", ""), ("A", 200902, "3", ""),
            ("A", 200903, "RA", ""), ("B", 200812, "0", ""), ("B", 200901, "0", "01")]
    d = pd.DataFrame(rows, columns=["loan_id", "period", "dlq", "zb_code"]).astype(str)
    d["cur_upb"], d["loan_age"], d["cur_rate"] = "100", "5", "5.0"
    return F.tidy_svcg(d)


def test_events_and_roll_counts():
    d = tiny()
    ev = F.loan_events(d)
    assert ev.loc["A", "first_default"] == 200902 and np.isnan(ev.loc["B", "first_default"])
    assert ev.loc["B", "prepay"] == 200901
    r = F.roll_counts(d).set_index(["from", "to"]).n
    assert r[("Current", "30")] == 1 and r[("60", "90+")] == 1 and r[("90+", "Default")] == 1
    assert r[("Current", "Prepaid")] == 1


def test_snapshot_label_and_population():
    d = tiny()
    orig = pd.DataFrame({"loan_id": ["A", "B"], "orig_upb": [200.0, 200.0]})
    for c in F.NUM + F.CAT:
        if c not in ("loan_age", "cur_rate", "upb_ratio", "dlq_s"):
            orig[c] = 1.0 if c in F.NUM else "x"
    s = F.snapshot_frame(orig, d[d.period % 100 == 12], F.loan_events(d), 200812).set_index("loan_id")
    assert s.loc["A", "label"] == 1 and s.loc["B", "label"] == 0 and s.loc["A", "dlq_s"] == 1


def test_vintage_curves_and_roll_rates(built):
    orig, ev, dec, rolls, lo, hi = built
    vc = F.vintage_curves(orig, ev, hi)
    for _, g in vc.groupby("vintage"):
        assert (np.diff(g.cum_default_rate) >= 0).all() and g.cum_default_rate.max() <= 1
        assert g.fully_observed.iloc[0] and not g.fully_observed.iloc[-1]
    rr = F.roll_rates(rolls)
    assert np.allclose(rr.groupby(["year", "from"]).rate.sum(), 1)


def test_oot_snapshots():
    assert F.oot_snapshots(201012, 200302) == ([200312, 200412, 200512, 200612, 200712, 200812], 200912)
    assert F.oot_snapshots(201011, 200302)[1] == 200812


def test_audit_oot_bites():
    F.audit_oot([200712, 200812], 200912)
    with pytest.raises(AssertionError):
        F.audit_oot([200812, 200901], 200912)  # label window reaches 201001 > S


def test_audit_features_pre_snapshot_bites(built):
    orig, ev, dec, *_ = built
    fr = F.snapshot_frame(orig, dec, ev, 200812)
    F.audit_features_pre_snapshot(fr, F.NUM + F.CAT)
    with pytest.raises(AssertionError, match="post-snapshot"):
        F.audit_features_pre_snapshot(fr, F.NUM + F.CAT + ["zb_code"])
    with pytest.raises(AssertionError, match="another month"):
        F.audit_features_pre_snapshot(fr.assign(period=200901), F.NUM + F.CAT)


def test_audit_preproc_train_only_bites(built):
    orig, ev, dec, *_ = built
    tr, te = F.snapshot_frame(orig, dec, ev, 200712), F.snapshot_frame(orig, dec, ev, 200912)
    audit_preproc_train_only(fit_pd_lr(tr, tr.label), tr)
    both = pd.concat([tr, te.assign(credit_score=te.credit_score + 40)])
    with pytest.raises(AssertionError):
        audit_preproc_train_only(fit_pd_lr(both, both.label), tr)


def test_end_to_end_SYNTHETIC(synth, tmp_path):
    met, t = analyze([synth[0]], [synth[1]], tmp_path, B=200, tag="SYNTHETIC_")
    assert {f.name for f in tmp_path.iterdir()} >= {"SYNTHETIC_part_b_pd_metrics.csv",
                                                     "SYNTHETIC_part_b_vintage_curves.csv"}
    assert met.test_snapshot.iloc[0] == 200912 and np.isfinite(met[["auc", "brier", "ece"]]).all().all()
    assert set(t.contrast) == {"LR_auc_minus_0p5", "LR_citl", "d_auc", "d_brier"}
