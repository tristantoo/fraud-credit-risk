"""SYNTHETIC Freddie-layout fixture generator. Numbers from it are never results."""
import numpy as np


def _line(fields: dict, n=32):
    row = [""] * n
    for i, v in fields.items():
        row[i] = str(v)
    return "|".join(row)


def write_synthetic(dirpath, vintages=range(2003, 2009), n_per=150, last=201012, seed=0, nfields=31):
    rng = np.random.default_rng(seed)
    orig_p, svcg_p = dirpath / "sample_orig_SYNTHETIC.txt", dirpath / "sample_perf_SYNTHETIC.txt"
    olines, slines = [], []
    for v in vintages:
        for i in range(n_per):
            q = rng.integers(1, 5)
            lid = f"F{v % 100:02d}Q{q}{i:06d}"
            fp = v * 100 + 3 * (q - 1) + 1 + rng.integers(0, 3)
            cs = 9999 if rng.random() < 0.02 else int(rng.integers(580, 820))
            upb, rate = int(rng.integers(100, 400)) * 1000, round(float(rng.uniform(4, 7)), 3)
            olines.append(_line({0: cs, 1: fp, 2: rng.choice(["Y", "N", "9"]), 5: rng.choice([0, 25]),
                                 7: rng.choice(["P", "I", "S"]), 8: (ltv := int(rng.integers(60, 98))),
                                 9: 999 if rng.random() < 0.02 else int(rng.integers(15, 50)), 10: upb,
                                 11: ltv, 12: rate, 13: rng.choice(["R", "B", "C"]),
                                 17: rng.choice(["SF", "CO", "PU"]), 19: lid, 20: rng.choice(["P", "C", "N"]),
                                 21: 360, 22: rng.choice([1, 2])}, nfields))
            h = min(0.2, 0.01 * np.exp((700 - (cs if cs < 9999 else 700)) / 50))
            per, age, st = fp, 0, 0
            while per <= last:
                zb = ""
                if st == 0 and rng.random() < 0.01:
                    zb = "01"
                elif st >= 4:
                    zb = "09"
                slines.append(_line({0: lid, 1: per, 2: 0 if age < 6 else int(upb * (1 - age / 400)),
                                     3: st, 4: age, 8: zb, 10: rate}, 35))
                if zb:
                    break
                st = (1 if rng.random() < h else 0) if st == 0 else (0 if rng.random() < 0.3 else st + 1)
                per = per + 1 if per % 100 < 12 else (per // 100 + 1) * 100 + 1
                age += 1
    orig_p.write_text("\n".join(olines) + "\n")
    svcg_p.write_text("\n".join(slines) + "\n")
    return orig_p, svcg_p
