"""Probability calibration.

Phase 1's entire decision rule is `decline iff p > C_FP / (C_FP + C_FN)`. That
comparison is only meaningful if `p` is a *posterior probability* — the number
that, across all transactions scoring 0.30, means 30 of every 100 are fraud.

An XGBoost booster trained with `scale_pos_weight` does not produce that
number. Weighting the positive class by ~23x to fight a 3.5% base rate is good
for optimisation and good for ranking, but it shifts the output scale: the
model is fitting a reweighted distribution in which fraud is roughly half the
data. Measured on the fixture before this module existed:

    observed fraud rate 4.13%   mean predicted 29.50%   (7.1x over-predicted)
    Brier 0.1088   -- worse than always predicting the base rate (0.0396)
    ECE   0.2537

Ranking was fine (ROC-AUC 0.65); the probabilities were not. Comparing a 7x
inflated score against a cost-derived threshold silently over-declines, and the
resulting dollar figure is not the optimum it claims to be.

The fix is a monotone map from raw score to calibrated probability, fitted on a
held-out temporal slice. Because it is monotone:

  * ranking metrics (PR-AUC, ROC-AUC) are unchanged by construction
  * the recall-targeted baseline policy is unaffected -- it picks a quantile
  * only the cost-aware policy moves, which is the point: it is the only one
    that reads `p` as a number rather than as an ordering

Isotonic regression is the default because the distortion here is not a simple
squash — it varies across the range. It needs positives to fit against, so with
a thin calibration slice we fall back to Platt scaling (a 2-parameter logistic
on the log-odds), which cannot overfit the same way.
"""
from __future__ import annotations

import numpy as np

MIN_POSITIVES_FOR_ISOTONIC = 200
_EPS = 1e-6


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), _EPS, 1 - _EPS)
    return np.log(p / (1 - p))


class ProbabilityCalibrator:
    """Monotone score -> probability map. Fitted once, applied everywhere."""

    def __init__(self, method: str = "auto"):
        self.method = method
        self.fitted_method = "identity"
        self._iso = None
        self._platt = None
        self.n_fit = 0
        self.n_positives = 0

    # -- fitting ----------------------------------------------------------
    def fit(self, scores, y) -> "ProbabilityCalibrator":
        scores = np.asarray(scores, dtype=np.float64)
        y = np.asarray(y).astype(np.int8)
        self.n_fit, self.n_positives = int(y.size), int(y.sum())

        method = self.method
        if method == "auto":
            method = ("isotonic" if self.n_positives >= MIN_POSITIVES_FOR_ISOTONIC
                      else "platt")

        if self.n_positives == 0 or self.n_positives == self.n_fit:
            # Nothing to calibrate against. Stay honest rather than invent a map.
            self.fitted_method = "identity"
            return self

        if method == "isotonic":
            from sklearn.isotonic import IsotonicRegression
            self._iso = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True,
                                           out_of_bounds="clip")
            self._iso.fit(scores, y)
            self.fitted_method = "isotonic"
        else:
            from sklearn.linear_model import LogisticRegression
            self._platt = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
            self._platt.fit(_logit(scores).reshape(-1, 1), y)
            self.fitted_method = "platt"
        return self

    # -- application ------------------------------------------------------
    def transform(self, scores) -> np.ndarray:
        scores = np.asarray(scores, dtype=np.float64)
        if self.fitted_method == "isotonic":
            out = self._iso.predict(scores)
        elif self.fitted_method == "platt":
            out = self._platt.predict_proba(_logit(scores).reshape(-1, 1))[:, 1]
        else:
            out = scores
        return np.clip(out, 0.0, 1.0)

    def transform_one(self, score: float) -> float:
        """Single-transaction path. Kept separate so the online scorer never
        pays for array allocation on the hot path."""
        if self.fitted_method == "identity":
            return float(score)
        return float(self.transform(np.array([score], dtype=np.float64))[0])

    def __repr__(self) -> str:
        return (f"ProbabilityCalibrator({self.fitted_method}, n={self.n_fit:,}, "
                f"pos={self.n_positives:,})")


# ---------------------------------------------------------------------------
# Reliability measurement
# ---------------------------------------------------------------------------
def reliability(scores, y, n_bins: int = 10) -> dict:
    """Brier score, expected calibration error, and the reliability table.

    Bins are equal-*count*, not equal-width. With a 3.5% base rate almost every
    score lands in the bottom width-bins, so equal-width bins would report a
    flattering ECE built from nine near-empty buckets.
    """
    scores = np.asarray(scores, dtype=np.float64)
    y = np.asarray(y).astype(np.float64)
    if scores.size == 0:
        return {"brier": None, "ece": None, "bins": []}

    brier = float(np.mean((scores - y) ** 2))
    base = float(y.mean())
    order = np.argsort(scores)
    ece, table = 0.0, []
    for idx in np.array_split(order, min(n_bins, scores.size)):
        if idx.size == 0:
            continue
        mp, ob = float(scores[idx].mean()), float(y[idx].mean())
        ece += idx.size / scores.size * abs(mp - ob)
        table.append({"n": int(idx.size), "mean_predicted": round(mp, 5),
                      "observed_rate": round(ob, 5)})

    return {
        "brier": round(brier, 6),
        "brier_base_rate_only": round(base * (1 - base), 6),
        "ece": round(float(ece), 6),
        "mean_predicted": round(float(scores.mean()), 6),
        "observed_rate": round(base, 6),
        "over_prediction_ratio": round(float(scores.mean() / base), 3) if base > 0 else None,
        "bins": table,
    }


def format_reliability(before: dict, after: dict) -> str:
    """Side-by-side text block for the training log."""
    def row(name, b, a, fmt="{:.4f}"):
        bv = "n/a" if b is None else fmt.format(b)
        av = "n/a" if a is None else fmt.format(a)
        return f"  {name:<28}{bv:>12}{av:>12}"

    lines = [f"  {'':<28}{'raw':>12}{'calibrated':>12}",
             row("mean predicted P(fraud)", before["mean_predicted"], after["mean_predicted"]),
             row("Brier score", before["brier"], after["brier"]),
             row("expected calibration err", before["ece"], after["ece"])]
    if before.get("over_prediction_ratio") is not None:
        lines.append(row("predicted / actual", before["over_prediction_ratio"],
                         after["over_prediction_ratio"], "{:.2f}x"))
    lines.append(f"  (observed fraud rate {after['observed_rate']:.4%}, "
                 f"base-rate-only Brier {after['brier_base_rate_only']:.4f})")
    return "\n".join(lines)
