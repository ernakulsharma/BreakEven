"""Preflight check on a data directory, before you spend an hour training on it.

    python -m tests.check_data --data-dir data/ieee-cis

Verifies the files are present, *labelled*, and carry the columns each phase
needs. Reads only the columns it checks, so it does not need the memory a full
load would.

The check that matters most is the label check. The Kaggle bundle ships four
CSVs and only two of them are usable here:

    train_transaction.csv  has isFraud  -> fit / calibrate / test all come from this
    train_identity.csv     device and session enrichment, joins on TransactionID
    test_transaction.csv   NO isFraud   -> labels were never released
    test_identity.csv      NO isFraud

The competition scored submissions on a hidden leaderboard, so the ground truth
for `test_*.csv` does not exist publicly. Pointing this repo at those files
gives you an unlabelled frame, and "test set" is exactly what someone reaches
for when they want to validate a model. The three splits in `train.py` are cut
*temporally out of train_transaction.csv* instead — that is what makes the test
slice a genuine future-holdout rather than a random sample of the same weeks.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from common import config

# Fields Phase 2 links entities on. Phase 1 uses far more, but these are the
# ones whose absence silently degrades ring detection rather than crashing it.
RING_FIELDS = ["card1", "card2", "card3", "card5", "addr1", "addr2",
               "P_emaildomain", "R_emaildomain", "D1"]
ID_RING_FIELDS = ["DeviceInfo", "DeviceType"]

OK, WARN, BAD = "  ok  ", " warn ", " FAIL "


class Report:
    def __init__(self) -> None:
        self.failed = False

    def line(self, status: str, msg: str, detail: str = "") -> None:
        print(f"[{status}] {msg}" + (f"  ({detail})" if detail else ""))
        if status == BAD:
            self.failed = True


def check(data_dir: Path) -> int:
    r = Report()
    print(f"checking {data_dir.resolve()}\n")

    tx_path = data_dir / "train_transaction.csv"
    id_path = data_dir / "train_identity.csv"

    # --- 1. presence ------------------------------------------------------
    if not tx_path.exists():
        r.line(BAD, "train_transaction.csv not found")
        stray = sorted(p.name for p in data_dir.glob("*.csv")) if data_dir.exists() else []
        if any(n.startswith("test_") for n in stray):
            print("\n       Found test_*.csv here. Those carry no isFraud column — Kaggle\n"
                  "       never released the labels. Use train_transaction.csv; train.py\n"
                  "       cuts its own fit/calibrate/test slices out of it temporally.")
        elif stray:
            print(f"\n       Directory holds: {', '.join(stray)}")
        else:
            print(f"\n       Create it and unzip the Kaggle bundle there:\n"
                  f"         mkdir -p {data_dir}\n"
                  f"         mv train_transaction.csv train_identity.csv {data_dir}/")
        return 2

    size_mb = tx_path.stat().st_size / 1e6
    r.line(OK, "train_transaction.csv present", f"{size_mb:,.0f} MB")

    if id_path.exists():
        r.line(OK, "train_identity.csv present", f"{id_path.stat().st_size / 1e6:,.0f} MB")
    else:
        r.line(WARN, "train_identity.csv missing",
               "device features unavailable; Phase 2 loses the device edge")

    # --- 2. labels --------------------------------------------------------
    header = pd.read_csv(tx_path, nrows=0)
    cols = set(header.columns)
    if "isFraud" not in cols:
        r.line(BAD, "no isFraud column — this file is unlabelled")
        print("\n       This looks like test_transaction.csv renamed, or the test bundle.\n"
              "       Kaggle scored the test set on a hidden leaderboard and never\n"
              "       published its labels, so it cannot be used to train, calibrate,\n"
              "       or validate anything here.")
        return 2
    r.line(OK, "isFraud present", "labelled, usable for all three splits")

    if "TransactionDT" not in cols:
        r.line(BAD, "no TransactionDT — temporal splitting is impossible")
        return 2

    # --- 3. columns each phase needs --------------------------------------
    missing_ring = [c for c in RING_FIELDS if c not in cols]
    if missing_ring:
        r.line(WARN, "Phase 2 linking fields missing", ", ".join(missing_ring))
    else:
        r.line(OK, "Phase 2 linking fields present", f"{len(RING_FIELDS)} fields")

    n_v = sum(1 for c in cols if c.startswith("V"))
    n_c = sum(1 for c in cols if c.startswith("C") and c[1:].isdigit())
    r.line(OK, "Vesta feature blocks present", f"{n_v} V-columns, {n_c} C-columns")

    # --- 4. shape, base rate, split preview -------------------------------
    print("\nreading isFraud + TransactionDT + TransactionAmt ...")
    core = pd.read_csv(tx_path, usecols=["isFraud", "TransactionDT", "TransactionAmt"])
    n = len(core)
    rate = float(core["isFraud"].mean())
    r.line(OK, "row count", f"{n:,}")

    if rate == 0 or rate == 1:
        r.line(BAD, "isFraud is constant", f"rate={rate}")
        return 2
    status = OK if 0.005 <= rate <= 0.25 else WARN
    r.line(status, "fraud base rate", f"{rate:.4%}")
    if n == 590_540 and abs(rate - 0.035) < 0.002:
        r.line(OK, "matches the published IEEE-CIS training set", "590,540 rows @ 3.50%")

    dt = core["TransactionDT"].to_numpy()
    if not np.all(np.diff(dt) >= 0):
        r.line(WARN, "rows are not sorted by TransactionDT",
               "splits still correct — they use quantiles, not row order")

    # Mirror train.three_way_split exactly, so this preview cannot drift from
    # what the trainer actually does.
    test_cut = np.quantile(dt, 1 - config.TEST_SPLIT_FRACTION)
    test_m = dt > test_cut
    cal_cut = np.quantile(dt[~test_m], 1 - config.CALIBRATION_FRACTION)
    cal_m = (~test_m) & (dt > cal_cut)
    fit_m = (~test_m) & (dt <= cal_cut)

    y = core["isFraud"].to_numpy()
    print(f"\n{'slice':<12}{'rows':>12}{'fraud':>10}{'rate':>10}   role")
    for name, m, role in (("fit", fit_m, "booster training"),
                          ("calibrate", cal_m, "early stopping + probability map"),
                          ("test", test_m, "never seen until the final report")):
        print(f"{name:<12}{m.sum():>12,}{int(y[m].sum()):>10,}"
              f"{y[m].mean():>10.3%}   {role}")

    cal_pos = int(y[cal_m].sum())
    if cal_pos >= 200:
        r.line(OK, "calibration slice supports isotonic regression",
               f"{cal_pos:,} positives, threshold is 200")
    else:
        r.line(WARN, "calibration slice is thin", 
               f"{cal_pos} positives — will fall back to Platt scaling")

    test_pos = int(y[test_m].sum())
    if test_pos < 500:
        r.line(WARN, "test slice has few positives",
               f"{test_pos} — recall and cost figures will be noisy")

    # --- 5. ring-field identifiability ------------------------------------
    # A field only links a ring if it identifies an entity. Phase 2 drops
    # fields behaving like categories, but knowing in advance which ones will
    # survive tells you what the graph is actually built from.
    present = [c for c in RING_FIELDS if c in cols]
    if present:
        print("\nreading Phase 2 linking fields ...")
        ring = pd.read_csv(tx_path, usecols=present)
        print(f"\n{'field':<18}{'distinct':>12}{'null':>9}{'avg/value':>12}   verdict")
        for c in present:
            nunique = int(ring[c].nunique(dropna=True))
            nulls = float(ring[c].isna().mean())
            avg = (n * (1 - nulls) / nunique) if nunique else float("inf")
            verdict = ("identifying" if avg < 100 else
                       "coarse — likely dropped" if avg < 2000 else
                       "category, not entity")
            print(f"{c:<18}{nunique:>12,}{nulls:>9.1%}{avg:>12,.0f}   {verdict}")

    # --- 6. memory --------------------------------------------------------
    est_gb = n * len(cols) * 4 / 1e9
    print()
    if est_gb > 2.5:
        r.line(WARN, "memory estimate is tight",
               f"~{est_gb:.1f} GB for the float32 matrix; train.py drops "
               f"high-null V-columns first")
    else:
        r.line(OK, "memory estimate", f"~{est_gb:.1f} GB for the float32 matrix")

    print()
    if r.failed:
        print("NOT usable — fix the FAIL lines above.")
        return 1
    print(f"usable. next:\n"
          f"  python -m phase1_prevention.train --data-dir {data_dir}\n"
          f"  python -m tests.smoke_test --data-dir {data_dir}\n"
          f"  python demo.py --data-dir {data_dir} --limit 100000")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=config.DATA / "ieee-cis")
    return check(ap.parse_args().data_dir)


if __name__ == "__main__":
    sys.exit(main())
