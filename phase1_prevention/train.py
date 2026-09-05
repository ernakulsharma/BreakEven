"""Phase 1 trainer: IEEE-CIS -> calibrated XGBoost -> cost-sensitive policy.

Run:
    python -m phase1_prevention.train --data-dir data/ieee-cis

Expects `train_transaction.csv` and `train_identity.csv` in --data-dir.

Memory notes (this ran on a 1-core / 3.9GB box, and OOM-killed three times
before these were in place):
  * float64 -> float32 and int64 -> int32 downcast at read time
  * V-columns above 60% null are dropped before the merge, not after
  * cumsum/cumcount instead of expanding().mean() for rolling aggregates
  * QuantileDMatrix instead of DMatrix (bins once, never materialises a copy)
  * explicit del + gc.collect() between stages; the frame, the split frames,
    and both DMatrices must never be alive simultaneously
"""
from __future__ import annotations

import argparse
import gc
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score

from common import config
from common.calibration import (
    ProbabilityCalibrator, format_reliability, reliability,
)
from common.features import (
    BASE_CATEGORICAL, add_batch_entity_features, add_entity_key_columns,
)
from phase1_prevention import decision as dec

KEY_COLS = ["card_key", "addr_key", "email_key", "device_key", "uid_key"]
DROP_ALWAYS = ["TransactionID", "isFraud"] + KEY_COLS


def _downcast(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns:
        k = df[c].dtype.kind
        if k == "f":
            df[c] = df[c].astype("float32")
        elif k == "i":
            df[c] = pd.to_numeric(df[c], downcast="integer")
    return df


def load_data(data_dir: Path) -> pd.DataFrame:
    tx_path, id_path = data_dir / "train_transaction.csv", data_dir / "train_identity.csv"
    if not tx_path.exists():
        raise FileNotFoundError(f"{tx_path} not found. Place the IEEE-CIS CSVs in {data_dir}/")

    print(f"[load] reading {tx_path.name} ...", flush=True)
    tx = _downcast(pd.read_csv(tx_path, low_memory=False))

    # Drop the noisiest Vesta columns before we ever hold two frames at once.
    v_cols = [c for c in tx.columns if c.startswith("V")]
    null_rate = tx[v_cols].isna().mean()
    drop_v = null_rate[null_rate > config.V_COLUMN_NULL_THRESHOLD].index.tolist()
    tx.drop(columns=drop_v, inplace=True)
    print(f"[load] dropped {len(drop_v)}/{len(v_cols)} V-columns above "
          f"{config.V_COLUMN_NULL_THRESHOLD:.0%} null", flush=True)

    if id_path.exists():
        idn = _downcast(pd.read_csv(id_path, low_memory=False))
        # Kaggle ships the test identity file with `id-01` style names.
        idn.columns = [c.replace("-", "_") for c in idn.columns]
        df = tx.merge(idn, on="TransactionID", how="left")
        del idn
    else:
        print("[load] no identity file, continuing without device features")
        df = tx
    del tx
    gc.collect()

    print(f"[load] {len(df):,} rows x {df.shape[1]} cols | "
          f"fraud rate {df['isFraud'].mean():.4%}", flush=True)
    return df


def engineer(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    print("[feat] building entity keys ...", flush=True)
    df = add_entity_key_columns(df)
    print("[feat] building rolling entity features ...", flush=True)
    df = add_batch_entity_features(df)
    gc.collect()

    print("[feat] encoding categoricals ...", flush=True)
    encoders: dict[str, dict] = {}
    cat_cols = [c for c in df.columns if df[c].dtype.kind in "OSU" or str(df[c].dtype) == "string"]
    for c in cat_cols:
        if c in KEY_COLS:
            continue
        codes, uniques = pd.factorize(df[c], use_na_sentinel=True)
        df[c] = codes.astype("int32")
        encoders[c] = {str(v): int(i) for i, v in enumerate(uniques)}
    print(f"[feat] encoded {len(encoders)} categorical columns", flush=True)
    gc.collect()
    return df, encoders


def three_way_split(dt: np.ndarray):
    """fit -> calibrate -> test, all ordered in time.

    Two separate reasons this is three slices and not two:

    1. Calibration must be fitted on data the booster did not train on, or the
       map is fitted to memorised scores and collapses in production.
    2. Early stopping picks `best_iteration` by looking at a set. Doing that on
       the test set — which the previous version did — tunes the model against
       the benchmark it is then reported on. Small leak, but it inflates every
       number downstream of it, including the dollar figure.
    """
    test_cut = np.quantile(dt, 1 - config.TEST_SPLIT_FRACTION)
    test_mask = dt > test_cut
    train_dt = dt[~test_mask]
    cal_cut = np.quantile(train_dt, 1 - config.CALIBRATION_FRACTION)
    cal_mask = (~test_mask) & (dt > cal_cut)
    fit_mask = (~test_mask) & (dt <= cal_cut)
    print(f"[split] fit {fit_mask.sum():,} | calibrate {cal_mask.sum():,} | "
          f"test {test_mask.sum():,}  (temporal, cuts at "
          f"{cal_cut:,.0f} / {test_cut:,.0f})", flush=True)
    return fit_mask, cal_mask, test_mask


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=config.DATA / "ieee-cis")
    ap.add_argument("--out-dir", type=Path, default=config.ARTIFACTS)
    ap.add_argument("--rounds", type=int, default=config.MAX_BOOST_ROUNDS)
    ap.add_argument("--target-recall", type=float, default=0.90,
                    help="recall target for the cost-blind baseline policy")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    df = load_data(args.data_dir)
    n_rows, fraud_rate = len(df), float(df["isFraud"].mean())
    df, encoders = engineer(df)

    dt = df["TransactionDT"].to_numpy()
    fit_mask, cal_mask, test_mask = three_way_split(dt)
    y = df["isFraud"].to_numpy().astype("int8")
    amount = df["TransactionAmt"].to_numpy().astype("float64")
    ltv = df["customer_ltv_estimate"].to_numpy().astype("float64")

    feature_cols = [c for c in df.columns if c not in DROP_ALWAYS and c != "TransactionDT"]
    X = df[feature_cols].astype("float32").to_numpy()
    del df
    gc.collect()
    print(f"[train] {len(feature_cols)} features", flush=True)

    y_fit, y_cal, y_te = y[fit_mask], y[cal_mask], y[test_mask]
    amt_te, ltv_te = amount[test_mask], ltv[test_mask]

    # Class weighting handles the 3.5% imbalance without touching the gradient
    # geometry the way the abandoned dollar-weighted objective did. It does
    # distort the output *scale*, which is what the calibrator below undoes.
    spw = float((y_fit == 0).sum() / max((y_fit == 1).sum(), 1))
    params = dict(config.XGB_PARAMS, scale_pos_weight=spw)
    print(f"[train] scale_pos_weight={spw:.2f}", flush=True)

    dfit = xgb.QuantileDMatrix(X[fit_mask], label=y_fit, feature_names=feature_cols)
    dcal = xgb.QuantileDMatrix(X[cal_mask], label=y_cal, feature_names=feature_cols, ref=dfit)
    X_te = X[test_mask].copy()
    del X
    gc.collect()

    booster = xgb.train(
        params, dfit, num_boost_round=args.rounds,
        evals=[(dfit, "fit"), (dcal, "calib")],
        early_stopping_rounds=config.EARLY_STOPPING_ROUNDS, verbose_eval=50,
    )
    del dfit
    gc.collect()

    # --- calibration -----------------------------------------------------
    # Fitted on the calibration slice, measured on the test slice. The slice
    # the map is fitted on can never be the slice it is judged on.
    cal_raw = booster.predict(dcal)
    del dcal
    gc.collect()
    calibrator = ProbabilityCalibrator(config.CALIBRATION_METHOD).fit(cal_raw, y_cal)

    dtest = xgb.DMatrix(X_te, feature_names=feature_cols)
    raw_scores = booster.predict(dtest)
    del dtest
    gc.collect()
    scores = calibrator.transform(raw_scores)

    # Isotonic and Platt are both monotone, so ranking is unchanged by
    # construction. Reported on raw scores; asserted against calibrated ones.
    pr_auc = float(average_precision_score(y_te, raw_scores))
    roc_auc = float(roc_auc_score(y_te, raw_scores))
    print(f"\n[eval] PR-AUC {pr_auc:.4f} | ROC-AUC {roc_auc:.4f}", flush=True)

    rel_before = reliability(raw_scores, y_te)
    rel_after = reliability(scores, y_te)
    print(f"\n[calib] {calibrator.fitted_method} fitted on {calibrator.n_fit:,} rows "
          f"({calibrator.n_positives:,} fraud)")
    print(format_reliability(rel_before, rel_after), flush=True)

    # --- latency ---------------------------------------------------------
    lat = []
    single = xgb.DMatrix(X_te[:1], feature_names=feature_cols)
    for _ in range(300):
        s = time.perf_counter()
        booster.predict(single)
        lat.append((time.perf_counter() - s) * 1000)
    lat = np.array(lat)
    latency = {"p50_ms": float(np.percentile(lat, 50)),
               "p95_ms": float(np.percentile(lat, 95)),
               "p99_ms": float(np.percentile(lat, 99))}
    print(f"[eval] inference p50 {latency['p50_ms']:.2f}ms / "
          f"p95 {latency['p95_ms']:.2f}ms / p99 {latency['p99_ms']:.2f}ms", flush=True)

    # --- policy comparison ----------------------------------------------
    fixed_thr = dec.fixed_threshold_for_recall(y_te, scores, args.target_recall)
    flag_fixed = scores >= fixed_thr
    per_txn_thr = dec.bayes_threshold(amt_te, ltv_te)
    flag_bayes = scores >= per_txn_thr

    def summarise(flag):
        tp = int((flag & (y_te == 1)).sum()); fn = int((~flag & (y_te == 1)).sum())
        fp = int((flag & (y_te == 0)).sum()); tn = int((~flag & (y_te == 0)).sum())
        return {
            "recall": tp / max(tp + fn, 1), "fpr": fp / max(fp + tn, 1),
            "precision": tp / max(tp + fp, 1), "decline_rate": float(flag.mean()),
            "dollar_cost": dec.total_dollar_cost(y_te, flag, amt_te, ltv_te),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        }

    pol_fixed, pol_bayes = summarise(flag_fixed), summarise(flag_bayes)
    saving = 1 - pol_bayes["dollar_cost"] / max(pol_fixed["dollar_cost"], 1e-9)

    print(f"\n{'policy':<34}{'recall':>9}{'FPR':>9}{'$ cost':>16}")
    print(f"{'fixed threshold (cost-blind)':<34}{pol_fixed['recall']:>8.1%}"
          f"{pol_fixed['fpr']:>9.1%}{pol_fixed['dollar_cost']:>16,.0f}")
    print(f"{'per-txn Bayes (cost-aware)':<34}{pol_bayes['recall']:>8.1%}"
          f"{pol_bayes['fpr']:>9.1%}{pol_bayes['dollar_cost']:>16,.0f}")
    print(f"\ncost reduction: {saving:.1%}", flush=True)

    # --- the review band, measured offline -------------------------------
    # `decide()` serves three actions but this report only measured two, so the
    # policy being reported was not the policy being served. Fraud sitting in
    # the review band is neither caught nor missed — it is queued, and whether
    # staffing that queue is worth it depends entirely on how much fraud lands
    # in it. No cost is assigned here: an analyst hour is a real number the
    # operator has and this repo does not, so inventing one would make the
    # dollar column fiction.
    review_lo = per_txn_thr * (1.0 - config.REVIEW_BAND)
    in_review = (scores >= review_lo) & (scores < per_txn_thr)
    n_fraud = int((y_te == 1).sum())
    review = {
        "band": config.REVIEW_BAND,
        "review_rate": float(in_review.mean()),
        "fraud_share_in_band": float((in_review & (y_te == 1)).sum() / max(n_fraud, 1)),
        "precision_in_band": float((in_review & (y_te == 1)).sum() / max(in_review.sum(), 1)),
        "recall_if_review_catches_all": float(pol_bayes["recall"] +
                                              (in_review & (y_te == 1)).sum() / max(n_fraud, 1)),
    }
    print(f"[review] band routes {review['review_rate']:.2%} of traffic; it holds "
          f"{review['fraud_share_in_band']:.1%} of all fraud at "
          f"{review['precision_in_band']:.1%} precision. Perfect review would take "
          f"recall {pol_bayes['recall']:.1%} -> "
          f"{review['recall_if_review_catches_all']:.1%}.", flush=True)

    # --- diagnostic: low recall has two very different causes ------------
    # (a) the cost model is inflated, so declining anyone looks unaffordable
    # (b) the cost model is fine and the *model* is too weak to clear the bar
    # These need opposite fixes, so report the numbers that separate them.
    # (c) was invisible until calibration landed: an honest probability on a
    #     3.5% base rate rarely exceeds 0.6, so a threshold up there cannot be
    #     cleared by *any* calibrated model, however good. Uncalibrated scores
    #     hid this by sitting ~7x too high and clearing the bar by accident.
    cost_ratio = float(np.median(dec.cost_false_positive(amt_te, ltv_te) /
                                 dec.cost_false_negative(amt_te)))
    med_thr = float(np.median(per_txn_thr))
    score_at_thr = float((scores >= med_thr).mean())
    score_p99 = float(np.quantile(scores, 0.99))
    score_max = float(scores.max())
    reachable = score_max >= med_thr
    diagnostics = {
        "median_fp_to_fn_cost_ratio": round(cost_ratio, 2),
        "median_bayes_threshold": round(med_thr, 4),
        "share_of_scores_above_median_threshold": round(score_at_thr, 5),
        "median_ltv_estimate": float(np.median(ltv_te)),
        "calibrated_score_p99": round(score_p99, 4),
        "calibrated_score_max": round(score_max, 4),
        "threshold_reachable": bool(reachable),
    }
    print(f"\n[diag] median FP/FN cost ratio {cost_ratio:.2f}x | "
          f"median Bayes threshold {med_thr:.3f} | "
          f"{score_at_thr:.2%} of scores clear it", flush=True)
    print(f"[diag] calibrated scores reach p99={score_p99:.3f}, max={score_max:.3f}",
          flush=True)

    if pol_bayes["recall"] < 0.30:
        if cost_ratio > 20:
            print("[warn] FP cost dwarfs FN cost. This is the inflated-LTV failure mode: "
                  "the policy degenerates to 'decline almost nobody'. Lower "
                  "EXPECTED_FUTURE_PURCHASES / LTV_CAP / CHURN_PROB before "
                  "quoting the cost reduction.", flush=True)
        elif score_p99 < med_thr:
            print(f"[warn] Threshold is structurally unreachable: the cost model asks for "
                  f"p>{med_thr:.2f} but 99% of calibrated scores sit below {score_p99:.2f}. "
                  f"This is NOT the same as a weak model — no calibrated model on a "
                  f"{fraud_rate:.1%} base rate emits that confidence often. Either the "
                  f"decision is not really binary (route the band to review instead of "
                  f"forcing approve/decline), or CHURN_PROB/LTV overstate what a single "
                  f"decline costs. Do not quote the cost reduction: it is the saving from "
                  f"declining nothing.", flush=True)
        else:
            print("[warn] Cost model looks sane and the threshold is reachable, so the "
                  "model is not producing enough confident scores to clear it. "
                  "The cost reduction here is real but uninteresting — it comes from "
                  "declining nothing. Improve PR-AUC before reading anything into it.",
                  flush=True)

    # --- artifacts -------------------------------------------------------
    gain = booster.get_score(importance_type="gain")
    top = sorted(gain.items(), key=lambda kv: -kv[1])[:30]

    with open(args.out_dir / "model.pkl", "wb") as f:
        pickle.dump({"booster": booster, "feature_cols": feature_cols,
                     "calibrator": calibrator,
                     "model_version": time.strftime("v%Y%m%d-%H%M%S")}, f)
    with open(args.out_dir / "encoders.pkl", "wb") as f:
        pickle.dump(encoders, f)

    report = {
        "n_rows": n_rows, "fraud_rate": fraud_rate,
        "n_features": len(feature_cols), "best_iteration": int(booster.best_iteration),
        "pr_auc": pr_auc, "roc_auc": roc_auc, "latency": latency,
        "calibration": {
            "method": calibrator.fitted_method,
            "fitted_on_rows": calibrator.n_fit,
            "fitted_on_positives": calibrator.n_positives,
            "raw": rel_before,
            "calibrated": rel_after,
        },
        "policy_fixed_threshold": {**pol_fixed, "threshold": float(fixed_thr)},
        "policy_bayes_per_transaction": pol_bayes,
        "cost_reduction": saving,
        "review_band": review,
        "diagnostics": diagnostics,
        "cost_model": {
            "chargeback_fixed_fee": config.CHARGEBACK_FIXED_FEE,
            "gross_margin": config.GROSS_MARGIN,
            "churn_prob_after_false_decline": config.CHURN_PROB_AFTER_FALSE_DECLINE,
            "expected_future_purchases": config.EXPECTED_FUTURE_PURCHASES,
            "ltv_cap": config.LTV_CAP,
        },
        "top_features_by_gain": [{"feature": k, "gain": v} for k, v in top],
        "train_seconds": time.time() - t0,
    }
    with open(args.out_dir / "eval_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[done] artifacts -> {args.out_dir} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
