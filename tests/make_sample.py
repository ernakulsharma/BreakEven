"""Generate IEEE-CIS-shaped sample CSVs so the pipeline is runnable without Kaggle.

Column names, dtypes and null patterns mirror the real files, and abuse rings
are deliberately planted so Phase 2 has ground truth to be measured against.
This is a test fixture, NOT a substitute for the real data — model metrics from
it mean nothing. Use it to prove the code runs; use train_transaction.csv to
prove the model works.

    python -m tests.make_sample --rows 40000 --out data/sample
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

EMAILS = ["gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "anonymous.com",
          "aol.com", "comcast.net", "protonmail.com", None]
DEVICES = ["Windows", "iOS Device", "MacOS", "SAMSUNG SM-G930V", "Trident/7.0", None]
BROWSERS = ["chrome 63.0", "mobile safari 11.0", "ie 11.0 for desktop", "firefox 57.0", None]


def make_sample(n_rows: int = 40_000, n_rings: int = 25, seed: int = 42):
    rng = np.random.default_rng(seed)

    dt = np.sort(rng.integers(86_400, 86_400 * 180, n_rows)).astype("int64")
    amt = np.round(np.exp(rng.normal(4.0, 1.1, n_rows)), 2).clip(1.0, 32_000)

    df = pd.DataFrame({
        "TransactionID": np.arange(2_987_000, 2_987_000 + n_rows),
        "TransactionDT": dt,
        "TransactionAmt": amt.astype("float32"),
        "ProductCD": rng.choice(["W", "C", "R", "H", "S"], n_rows, p=[.74, .12, .06, .05, .03]),
        "card1": rng.integers(1000, 18500, n_rows),
        "card2": rng.choice(np.append(rng.integers(100, 600, 400), np.nan), n_rows),
        "card3": rng.choice([150.0, 185.0, np.nan], n_rows, p=[.88, .10, .02]),
        "card4": rng.choice(["visa", "mastercard", "american express", "discover", None],
                            n_rows, p=[.65, .30, .02, .02, .01]),
        "card5": rng.choice(np.append(rng.integers(100, 240, 90).astype(float), np.nan), n_rows),
        "card6": rng.choice(["debit", "credit", None], n_rows, p=[.74, .25, .01]),
        "addr1": rng.choice(np.append(rng.integers(100, 540, 300).astype(float), np.nan),
                            n_rows, p=[*(np.ones(300) * 0.0033), 0.01]),
        "addr2": rng.choice([87.0, 60.0, np.nan], n_rows, p=[.86, .12, .02]),
        "dist1": rng.choice(np.append(rng.integers(0, 2000, 500).astype(float), np.nan),
                            n_rows, p=[*(np.ones(500) * 0.0012), 0.4]),
        "dist2": np.where(rng.random(n_rows) < 0.07, rng.integers(0, 3000, n_rows), np.nan),
        "P_emaildomain": rng.choice(EMAILS, n_rows),
        "R_emaildomain": np.where(rng.random(n_rows) < 0.24, rng.choice(EMAILS[:-1], n_rows), None),
    })

    for i in range(1, 15):
        df[f"C{i}"] = rng.gamma(1.4, 2.0, n_rows).round(1).astype("float32")
    for i in range(1, 16):
        col = rng.gamma(2.0, 30.0, n_rows).astype("float32")
        col[rng.random(n_rows) < (0.10 + 0.04 * i)] = np.nan
        df[f"D{i}"] = col
    df["D1"] = np.abs(rng.normal(120, 110, n_rows)).round().astype("float32")
    for i in range(1, 10):
        df[f"M{i}"] = rng.choice(["T", "F", None], n_rows, p=[.42, .34, .24])
    # Vesta V-block, including the high-null columns the trainer prunes.
    vblock = {}
    for i in range(1, 140):
        col = rng.normal(0, 1, n_rows).astype("float32")
        null_rate = 0.05 if i <= 90 else rng.uniform(0.55, 0.95)
        col[rng.random(n_rows) < null_rate] = np.nan
        vblock[f"V{i}"] = col
    df = pd.concat([df, pd.DataFrame(vblock, index=df.index)], axis=1)

    # --- label: fraud correlates with amount, night hours, free email, new cards
    hour = (dt // 3600) % 24
    logit = (-5.25 + 0.30 * np.log1p(amt) + 0.55 * ((hour < 6) | (hour >= 22))
             + 0.45 * df["P_emaildomain"].isin(["anonymous.com", "protonmail.com"]).to_numpy()
             + 0.40 * (df["D1"].to_numpy() < 3) + rng.normal(0, 0.8, n_rows))
    df = df.copy()
    df["isFraud"] = (rng.random(n_rows) < 1 / (1 + np.exp(-logit))).astype("int8")
    df["_ring"] = np.nan

    # --- plant abuse rings -------------------------------------------------
    ring_rows: list[int] = []
    idx_pool = rng.choice(n_rows, size=n_rings * 12, replace=False)
    for r in range(n_rings):
        members = idx_pool[r * 12:(r + 1) * 12][:rng.integers(4, 12)]
        shared_addr = float(rng.integers(100, 540))
        shared_device = f"RingDevice-{r:03d}"
        base_dt = int(rng.integers(86_400, 86_400 * 175))
        ring_amt = float(np.round(rng.uniform(40, 400), 2))
        for j, m in enumerate(members):
            df.loc[m, "addr1"] = shared_addr
            df.loc[m, "addr2"] = 87.0
            df.loc[m, "card1"] = int(rng.integers(1000, 18500))   # many cards
            df.loc[m, "TransactionDT"] = base_dt + j * int(rng.integers(120, 900))
            df.loc[m, "TransactionAmt"] = ring_amt + rng.normal(0, 2)
            df.loc[m, "D1"] = float(rng.integers(0, 2))
            df.loc[m, "isFraud"] = 1
            df.loc[m, "_ring"] = r
        ring_rows.extend(int(x) for x in members)

    ring_truth = df[["TransactionID", "_ring"]].dropna()
    df = df.drop(columns=["_ring"]).sort_values("TransactionDT").reset_index(drop=True)

    # --- identity file: present for ~24% of rows, as in the real data ------
    has_id = rng.random(len(df)) < 0.24
    ids = df.loc[has_id, ["TransactionID"]].copy()
    m = len(ids)
    ids["DeviceType"] = rng.choice(["desktop", "mobile", None], m, p=[.52, .46, .02])
    ids["DeviceInfo"] = rng.choice(DEVICES, m)
    ids["id_30"] = rng.choice(["Windows 10", "iOS 11.1.2", "Mac OS X 10_13", None], m)
    ids["id_31"] = rng.choice(BROWSERS, m)
    ids["id_33"] = rng.choice(["1920x1080", "2208x1242", "1366x768", None], m)
    for i in range(1, 12):
        ids[f"id_{i:02d}"] = rng.normal(0, 1, m).round(2)
    # give ring members the shared device string so the graph can link them
    ring_ids = set(ring_truth["TransactionID"].astype(int))
    mask = ids["TransactionID"].isin(ring_ids)
    ids.loc[mask, "DeviceInfo"] = "RingDevice-shared"

    return df, ids.reset_index(drop=True), ring_truth


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=40_000)
    ap.add_argument("--rings", type=int, default=25)
    ap.add_argument("--out", type=Path, default=Path("data/sample"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    tx, idn, truth = make_sample(a.rows, a.rings)
    tx.to_csv(a.out / "train_transaction.csv", index=False)
    idn.to_csv(a.out / "train_identity.csv", index=False)
    truth.to_csv(a.out / "ring_truth.csv", index=False)
    print(f"wrote {len(tx):,} transactions ({tx['isFraud'].mean():.2%} fraud), "
          f"{len(idn):,} identity rows, {truth['_ring'].nunique()} planted rings -> {a.out}")


if __name__ == "__main__":
    main()
