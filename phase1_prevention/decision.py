"""Cost-sensitive decisioning.

The pitch is "we don't decline good customers to catch cheap fraud". The way
that is implemented matters, and the first attempt got it wrong:

  Attempt 1 (abandoned): bake dollar costs into the XGBoost gradient as
  per-row weights. A handful of high-cost rows dominated every boosting round,
  train-rmse never converged, and PR-AUC collapsed from 0.494 to 0.08. The
  ranking was destroyed for a 2.6% dollar improvement.

  Attempt 2 (shipped): keep one well-calibrated probability model, and move
  the cost-sensitivity entirely into the *decision threshold*, computed per
  transaction. This is the textbook-correct decomposition — the model
  estimates P(fraud), the policy decides what to do with it — and it keeps the
  ranking metric intact while still optimising dollars.

Expected cost of approving:  p * C_FN
Expected cost of declining:  (1 - p) * C_FP
Decline iff p * C_FN > (1 - p) * C_FP  =>  p > C_FP / (C_FP + C_FN)
"""
from __future__ import annotations

import numpy as np

from common import config
from common.schema import CostBreakdown, Decision


def cost_false_negative(amount: np.ndarray | float) -> np.ndarray | float:
    """Missed fraud: goods gone, amount reversed, plus the acquirer dispute fee."""
    return amount * config.CHARGEBACK_LOSS_MULTIPLIER + config.CHARGEBACK_FIXED_FEE


def cost_false_positive(amount: np.ndarray | float, ltv: np.ndarray | float):
    """False decline: this sale's margin, plus churn-weighted remaining LTV."""
    return amount * config.GROSS_MARGIN + config.CHURN_PROB_AFTER_FALSE_DECLINE * ltv


def bayes_threshold(amount, ltv):
    """Per-transaction indifference point between approving and declining."""
    c_fn = cost_false_negative(amount)
    c_fp = cost_false_positive(amount, ltv)
    return c_fp / (c_fp + c_fn)


def total_dollar_cost(y_true, y_pred_flag, amount, ltv) -> float:
    """Realised cost of a decision vector on labelled data."""
    y_true = np.asarray(y_true).astype(bool)
    flag = np.asarray(y_pred_flag).astype(bool)
    amount = np.asarray(amount, dtype=np.float64)
    ltv = np.asarray(ltv, dtype=np.float64)

    fn = y_true & ~flag
    fp = ~y_true & flag
    return float(cost_false_negative(amount[fn]).sum() + cost_false_positive(amount[fp], ltv[fp]).sum())


def fixed_threshold_for_recall(y_true, scores, target_recall: float = 0.90) -> float:
    """The cost-blind baseline: one global cutoff tuned to hit a recall target.

    This is what a conventional rules/score system does, and it is the policy
    the cost-aware one is measured against.
    """
    y_true = np.asarray(y_true).astype(bool)
    scores = np.asarray(scores, dtype=np.float64)
    pos = np.sort(scores[y_true])[::-1]
    if pos.size == 0:
        return 0.5
    idx = min(int(np.ceil(target_recall * pos.size)) - 1, pos.size - 1)
    return float(pos[max(idx, 0)])


def decide(
    probability: float,
    amount: float,
    ltv: float,
    fixed_threshold: float = 0.5,
    review_band: float | None = None,
) -> tuple[Decision, CostBreakdown]:
    """Score one transaction into approve / review / decline with its rationale."""
    band = config.REVIEW_BAND if review_band is None else review_band
    c_fn = float(cost_false_negative(amount))
    c_fp = float(cost_false_positive(amount, ltv))
    thr = c_fp / (c_fp + c_fn)

    exp_approve = probability * c_fn
    exp_decline = (1.0 - probability) * c_fp

    if probability >= thr:
        decision = Decision.DECLINE
    elif probability >= thr * (1.0 - band):
        # Close to indifference: the expected costs of both actions are within
        # noise of each other, so a human review is cheap relative to the risk.
        decision = Decision.REVIEW
    else:
        decision = Decision.APPROVE

    return decision, CostBreakdown(
        cost_false_negative=round(c_fn, 2),
        cost_false_positive=round(c_fp, 2),
        customer_ltv_estimate=round(float(ltv), 2),
        bayes_threshold=round(thr, 6),
        fixed_threshold=round(float(fixed_threshold), 6),
        expected_cost_approve=round(exp_approve, 4),
        expected_cost_decline=round(exp_decline, 4),
    )


def apply_capacity_guardrail(scores, thresholds, max_review_rate: float | None = None):
    """Keep the flagged volume inside operational capacity.

    A Bayes-optimal policy is optimal against the cost model, not against the
    fraud team's headcount. If the policy wants to decline more than capacity
    allows, raise the effective floor until it fits.
    """
    cap = config.MAX_REVIEW_RATE if max_review_rate is None else max_review_rate
    scores = np.asarray(scores, dtype=np.float64)
    thresholds = np.asarray(thresholds, dtype=np.float64)
    flag = scores >= thresholds
    rate = flag.mean() if flag.size else 0.0
    if rate <= cap:
        return flag, 0.0
    # Keep only the highest-margin flags (score furthest above its own threshold).
    margin = scores - thresholds
    keep_n = int(cap * scores.size)
    cutoff = np.sort(margin[flag])[::-1][keep_n - 1] if keep_n > 0 else np.inf
    return (flag & (margin >= cutoff)), float(cutoff)
