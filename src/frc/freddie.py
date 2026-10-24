"""Part b: Freddie Mac Single-Family Loan-Level sample pipeline (docs/PLAN.md part b).

Column positions follow the published General User Guide layout (32 pipe-delimited fields in both
the origination and the monthly performance file, no header). The layout changes between releases:
VERIFY every position below against the guide shipped with the files before trusting any output.
`check_layout` refuses files whose field count or key-field formats disagree.
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

ORIG_NFIELDS, SVCG_NFIELDS = 31, 35  # release 47 (July 2026) File Layout
ORIG = {0: "credit_score", 1: "first_pay_date", 2: "first_time_buyer", 5: "mi_pct", 7: "occupancy",
        8: "cltv", 9: "dti", 10: "orig_upb", 11: "ltv", 12: "orig_rate", 13: "channel",
        17: "property_type", 19: "loan_id", 20: "purpose", 22: "n_borrowers"}
SVCG = {0: "loan_id", 1: "period", 2: "cur_upb", 3: "dlq", 4: "loan_age", 8: "zb_code", 10: "cur_rate"}
SENTINEL = {"credit_score": 9999, "mi_pct": 999, "cltv": 999, "ltv": 999, "dti": 999, "n_borrowers": 99}
CREDIT_ZB = {"02", "03", "09", "15"}  # credit-event terminations; "01" = prepaid (competing)
NUM = ["credit_score", "ltv", "cltv", "dti", "orig_rate", "log_orig_upb", "mi_pct", "n_borrowers",
       "loan_age", "cur_rate", "upb_ratio", "dlq_s"]
CAT = ["purpose", "occupancy", "channel", "property_type", "first_time_buyer"]
# Monthly-file fields that are dated at or after termination; never features.
FORBIDDEN = {"zb_code", "zb_date", "defect_settlement_date", "mod_flag", "actual_loss", "net_sale_proceeds",
             "mi_recoveries", "non_mi_recoveries", "expenses", "first_default", "prepay", "label"}
STATES = ["Current", "30", "60", "90+", "Default", "Prepaid"]
LOAN_ID = re.compile(r"^F\d{2}Q[1-4]\w+$")


def month_index(period):
    p = np.asarray(period, int)
    return (p // 100) * 12 + p % 100 - 1


def add_months(period, k):
    m = month_index(period) + k
    return (m // 12) * 100 + m % 12 + 1


def check_layout(path, nfields, kind):
    first = Path(path).open().readline().rstrip("\n")
    n = first.count("|") + 1
    if n != nfields:
        raise ValueError(f"{path}: {n} fields, expected {nfields} for {kind}; check the User Guide")


def _read(path, cols, nfields, kind, chunksize=None):
    check_layout(path, nfields, kind)
    return pd.read_csv(path, sep="|", header=None, usecols=list(cols), dtype=str,
                       keep_default_na=False, chunksize=chunksize)


def read_orig(path) -> pd.DataFrame:
    d = _read(path, ORIG, ORIG_NFIELDS, "origination").rename(columns=ORIG)
    if not d.loan_id.str.match(LOAN_ID).mean() > 0.99:
        raise ValueError(f"{path}: loan_id column does not look like Freddie loan ids; layout drift?")
    for c in ["credit_score", "mi_pct", "cltv", "dti", "orig_upb", "ltv", "orig_rate", "n_borrowers"]:
        d[c] = pd.to_numeric(d[c].str.strip(), errors="coerce")
        if c in SENTINEL:
            d.loc[d[c] == SENTINEL[c], c] = np.nan
    d["log_orig_upb"] = np.log(d.orig_upb)
    yy = d.loan_id.str[1:3].astype(int)
    d["vintage"] = np.where(yy >= 90, 1900 + yy, 2000 + yy)
    for c in CAT:
        d[c] = d[c].str.strip().replace("", "missing")
    return d


def tidy_svcg(d: pd.DataFrame) -> pd.DataFrame:
    d = d.rename(columns=SVCG)
    if not d.period.str.fullmatch(r"\d{6}").all():
        raise ValueError("monthly reporting period is not YYYYMM; layout drift?")
    d["period"] = d.period.astype(int)
    for c in ["cur_upb", "loan_age", "cur_rate"]:
        d[c] = pd.to_numeric(d[c].str.strip(), errors="coerce")
    d["dlq"] = d.dlq.str.strip()
    d["dlq_n"] = pd.to_numeric(d.dlq, errors="coerce")
    d["zb_code"] = d.zb_code.str.strip().str.zfill(2).replace("00", "")
    return d


def read_svcg(path, chunksize=2_000_000):
    """Yield tidy chunks of the monthly file (streamed for the 8 GB RAM limit)."""
    for ch in _read(path, SVCG, SVCG_NFIELDS, "monthly performance", chunksize=chunksize):
        yield tidy_svcg(ch)


def state(d: pd.DataFrame) -> pd.Series:
    """Roll-rate state per loan-month. 90+ = delinquency >= 3 months while still active;
    Default = REO acquisition (RA) or credit-event zero balance; Prepaid = zero balance 01."""
    s = pd.Series("Other", index=d.index)
    s[d.dlq_n == 0] = "Current"
    s[d.dlq_n == 1] = "30"
    s[d.dlq_n == 2] = "60"
    s[d.dlq_n >= 3] = "90+"
    s[(d.dlq == "RA") | d.zb_code.isin(CREDIT_ZB)] = "Default"
    s[d.zb_code == "01"] = "Prepaid"
    return s


def is_default(d):
    """PD default event (PLAN b2): first month with dlq >= 3, RA, or credit-event zero balance."""
    return (d.dlq_n >= 3) | (d.dlq == "RA") | d.zb_code.isin(CREDIT_ZB)


def loan_events(d: pd.DataFrame) -> pd.DataFrame:
    """One row per loan: first default period/age, prepay period, last reported period."""
    g = d.groupby("loan_id", sort=False)
    ev = pd.DataFrame({"last_period": g.period.max()})
    dd = d[is_default(d)].sort_values("period").groupby("loan_id").first()
    ev["first_default"] = dd.period
    ev["default_age"] = dd.loan_age
    ev["prepay"] = d[d.zb_code == "01"].groupby("loan_id").period.min()
    return ev


def roll_counts(d: pd.DataFrame) -> pd.DataFrame:
    """Month-to-month transition counts (consecutive months of the same loan), by calendar year."""
    d = d.sort_values(["loan_id", "period"])
    st = state(d).to_numpy()
    same = (d.loan_id.to_numpy()[1:] == d.loan_id.to_numpy()[:-1]) & \
           (np.diff(month_index(d.period.to_numpy())) == 1)
    t = pd.DataFrame({"year": d.period.to_numpy()[:-1][same] // 100,
                      "from": st[:-1][same], "to": st[1:][same]})
    t = t[t["from"].isin(STATES) & t["to"].isin(STATES)]
    return t.groupby(["year", "from", "to"]).size().rename("n").reset_index()


def roll_rates(counts: pd.DataFrame) -> pd.DataFrame:
    c = counts.groupby(["year", "from", "to"]).n.sum().reset_index()
    c["rate"] = c.n / c.groupby(["year", "from"]).n.transform("sum")
    return c


def vintage_curves(orig: pd.DataFrame, ev: pd.DataFrame, last_period: int) -> pd.DataFrame:
    """Cumulative default rate by loan age per vintage; denominator = loans in that vintage's sample.
    `fully_observed` marks ages every loan in the vintage could have reached (right-censoring)."""
    o = orig[["loan_id", "vintage", "first_pay_date"]].merge(ev[["default_age"]], left_on="loan_id",
                                                             right_index=True, how="left")
    rows = []
    for v, g in o.groupby("vintage"):
        last_fp = int(g.first_pay_date.astype(int).max())
        max_full = int(month_index(last_period) - month_index(last_fp))
        max_age = int(month_index(last_period) - month_index(int(g.first_pay_date.astype(int).min())))
        da = g.default_age.dropna().to_numpy()
        for a in range(max_age + 1):
            rows.append(dict(vintage=v, loan_age=a, n_loans=len(g), cum_defaults=int((da <= a).sum()),
                             fully_observed=a <= max_full))
    out = pd.DataFrame(rows)
    out["cum_default_rate"] = out.cum_defaults / out.n_loans
    return out


def snapshot_frame(orig, dec_rows, ev, snapshot):
    """PD rows at December snapshot S: active, not yet defaulted, dlq 0-2, with label = default in
    (S, S+12]. Snapshot-state features come only from the S row."""
    s = dec_rows[dec_rows.period == snapshot]
    s = s[(s.dlq_n <= 2) & (s.zb_code == "")]
    s = s.merge(ev[["first_default"]], left_on="loan_id", right_index=True, how="left")
    s = s[~(s.first_default <= snapshot)]
    s = s.merge(orig, on="loan_id", how="inner")
    s["label"] = (s.first_default <= add_months(snapshot, 12)).astype(int)
    s["dlq_s"] = s.dlq_n
    s["upb_ratio"] = (s.cur_upb / s.orig_upb).where(s.cur_upb > 0)  # UPB can be masked as 0 early on
    s["snapshot"] = snapshot
    return s[["loan_id", "snapshot", "period", "label"] + NUM + CAT]


def oot_snapshots(last_period: int, first_period: int):
    """Test S = latest December with S+12 <= last period; train = Decembers <= S-12 (labels observed by S)."""
    lim = int(add_months(last_period, -12))
    s_test = (lim // 100 - (lim % 100 != 12)) * 100 + 12
    train = [y * 100 + 12 for y in range(first_period // 100, s_test // 100) if y * 100 + 12 >= first_period]
    return train, s_test


def audit_oot(train_snapshots, s_test):
    """Every training label window (S_train, S_train+12] must end by the test snapshot."""
    assert train_snapshots, "no training snapshots"
    assert max(add_months(np.array(train_snapshots), 12)) <= s_test, "training label window reaches past S"


def audit_features_pre_snapshot(frame: pd.DataFrame, features):
    bad = set(features) & FORBIDDEN
    assert not bad, f"post-snapshot field(s) used as features: {bad}"
    assert (frame.period == frame.snapshot).all(), "snapshot-state features taken from another month"


def build(orig_paths, svcg_paths):
    """Stream all files into: loans, events, December rows, roll counts, last/first reporting period."""
    orig = pd.concat([read_orig(p) for p in orig_paths], ignore_index=True)
    evs, decs, rolls, lo, hi = [], [], [], 10 ** 9, 0
    for p in svcg_paths:
        d = pd.concat(list(read_svcg(p)), ignore_index=True)  # one vintage file at a time
        evs.append(loan_events(d))
        decs.append(d[d.period % 100 == 12][["loan_id", "period", "cur_upb", "dlq_n", "loan_age",
                                             "zb_code", "cur_rate"]])
        rolls.append(roll_counts(d))
        lo, hi = min(lo, d.period.min()), max(hi, d.period.max())
    return orig, pd.concat(evs), pd.concat(decs, ignore_index=True), pd.concat(rolls), int(lo), int(hi)
