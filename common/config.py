"""Central configuration for the risk engine.

Every tunable that affects money (cost model) or latency (feature store) lives
here so it can be audited in one place rather than scattered through the code.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = Path(os.getenv("RISK_ARTIFACTS", ROOT / "artifacts"))
DATA = Path(os.getenv("RISK_DATA", ROOT / "data"))
ARTIFACTS.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Cost model (Phase 1 decisioning)
# ---------------------------------------------------------------------------
# Cost of a MISSED fraud (false negative): the merchant eats the goods, the
# transaction amount is clawed back, and the network levies a dispute fee.
CHARGEBACK_FIXED_FEE = 25.0        # USD, typical acquirer dispute fee
CHARGEBACK_LOSS_MULTIPLIER = 1.0   # full transaction amount is reversed

# Cost of a FALSE DECLINE (false positive): lost margin on this sale, plus a
# probability-weighted slice of the customer's remaining lifetime value.
GROSS_MARGIN = 0.25                # merchant contribution margin per sale
CHURN_PROB_AFTER_FALSE_DECLINE = 0.20  # share of wrongly declined who leave

# Bounded LTV estimate. The unbounded cumulative-spend proxy used in the first
# iteration reached ~$1.6M per card1 bucket and made FP cost ~1500x FN cost,
# which degenerated the policy into "decline almost nobody" (8.9% recall).
# card1 in IEEE-CIS is a hashed bucket shared by ~44 transactions, not a
# customer id, so cumulative history is meaningless as an LTV signal.
EXPECTED_FUTURE_PURCHASES = 12     # bounded horizon, ~1 year of monthly spend
LTV_CAP = 5_000.0                  # hard ceiling, defensive

# Operating guardrails: even a cost-optimal policy must stay inside the
# review capacity and the network fraud-rate ceilings the business signed up to.
MAX_REVIEW_RATE = 0.03             # fraction of traffic routed to manual review
REVIEW_BAND = 0.35                 # relative band around threshold -> review

# ---------------------------------------------------------------------------
# Feature store
# ---------------------------------------------------------------------------
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
FEATURE_TTL_SECONDS = 60 * 60 * 24 * 90   # 90d rolling entity memory
ENTITY_NAMESPACES = ("card", "addr", "email", "device", "uid")

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
TEST_SPLIT_FRACTION = 0.20    # temporal, not random: last 20% by TransactionDT
V_COLUMN_NULL_THRESHOLD = 0.60   # drop Vesta V-columns nulled above this rate
MAX_BOOST_ROUNDS = 400
EARLY_STOPPING_ROUNDS = 30

# Calibration slice, taken from the END of the training window (never from the
# test window — the test set must stay untouched by both early stopping and
# calibration or the reported cost figure is fitted to its own benchmark).
CALIBRATION_FRACTION = 0.15   # share of the training window held out
CALIBRATION_METHOD = "auto"   # auto | isotonic | platt | identity

XGB_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": ["aucpr", "auc"],
    "tree_method": "hist",
    "max_depth": 8,
    "eta": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.6,
    "min_child_weight": 4,
    "reg_lambda": 2.0,
    "nthread": 1,
    "seed": RANDOM_SEED,
}

# ---------------------------------------------------------------------------
# Phase 3 LLM
# ---------------------------------------------------------------------------
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
OLLAMA_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "120"))
