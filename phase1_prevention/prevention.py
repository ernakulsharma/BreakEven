"""The sub-100ms scoring path.

One checkout in, one decision out. The work is:
  1. derive entity keys                        (~microseconds)
  2. one pipelined feature-store read          (~1 RTT)
  3. assemble the feature vector in model order
  4. one XGBoost predict on a 1-row DMatrix    (~0.2ms measured)
  5. per-transaction Bayes threshold           (~microseconds)
  6. fire-and-forget feature-store write

Step 3 is where train/serve skew normally creeps in, so the vector is built
from `common.features` — the same module the trainer uses — and any feature
the caller cannot supply is left as NaN, which XGBoost handles natively rather
than being silently zero-filled into a different distribution.
"""
from __future__ import annotations

import pickle
import time
from pathlib import Path

import numpy as np
import xgboost as xgb

from common import config
from common.calibration import ProbabilityCalibrator
from common.features import (
    derive_entity_features, derive_stateless_features, entity_keys,
)
from common.schema import CheckoutRequest, Decision, ScoreResponse
from phase1_prevention import decision as dec
from phase1_prevention.feature_store import FeatureStore


def request_to_row(req: CheckoutRequest) -> dict:
    """Flatten the API request into the column names the trainer used."""
    row: dict = {
        "TransactionAmt": req.amount,
        "TransactionDT": req.timestamp.timestamp() if req.timestamp else time.time(),
        "ProductCD": req.product_cd,
        "card1": req.card1, "card2": req.card2, "card3": req.card3,
        "card4": req.card4, "card5": req.card5, "card6": req.card6,
        "addr1": req.addr1, "addr2": req.addr2,
        "dist1": req.dist1, "dist2": req.dist2,
        "P_emaildomain": req.purchaser_email_domain,
        "R_emaildomain": req.recipient_email_domain,
    }
    if req.identity:
        row.update({
            "DeviceType": req.identity.device_type, "DeviceInfo": req.identity.device_info,
            "id_30": req.identity.id_30, "id_31": req.identity.id_31, "id_33": req.identity.id_33,
        })
        row.update(req.identity.extra)
    for block in (req.c_features, req.d_features, req.m_features, req.v_features):
        row.update(block)
    return row


class PreventionEngine:
    def __init__(self, artifacts_dir: Path | None = None, store: FeatureStore | None = None):
        d = Path(artifacts_dir or config.ARTIFACTS)
        self.store = store or FeatureStore()
        self.booster = None
        self.feature_cols: list[str] = []
        self.model_version = "untrained"
        self.encoders: dict[str, dict] = {}
        self.fixed_threshold = 0.5
        # Identity map until an artifact says otherwise. A missing calibrator
        # must not silently become "assume it was already calibrated".
        self.calibrator = ProbabilityCalibrator()

        model_path = d / "model.pkl"
        if model_path.exists():
            with open(model_path, "rb") as f:
                bundle = pickle.load(f)
            self.booster = bundle["booster"]
            self.feature_cols = bundle["feature_cols"]
            self.model_version = bundle.get("model_version", "unknown")
            if bundle.get("calibrator") is not None:
                self.calibrator = bundle["calibrator"]
        if (d / "encoders.pkl").exists():
            with open(d / "encoders.pkl", "rb") as f:
                self.encoders = pickle.load(f)
        report = d / "eval_report.json"
        if report.exists():
            import json
            with open(report) as f:
                self.fixed_threshold = float(
                    json.load(f).get("policy_fixed_threshold", {}).get("threshold", 0.5))

    @property
    def ready(self) -> bool:
        return self.booster is not None

    def _encode(self, col: str, value) -> float:
        """Map a categorical to the code the trainer assigned it.

        Unseen categories become NaN, not -1. -1 is a real code in the training
        distribution (pandas' NA sentinel) so reusing it would tell the model
        "this was missing" when it actually means "this is new".
        """
        table = self.encoders.get(col)
        if table is None:
            return np.nan
        return float(table.get(str(value), np.nan)) if value is not None else np.nan

    def build_vector(self, row: dict, states) -> np.ndarray:
        feats = dict(row)
        feats.update(derive_stateless_features(row))
        feats.update(derive_entity_features(states, float(row["TransactionAmt"]),
                                            row.get("TransactionDT")))
        vec = np.full(len(self.feature_cols), np.nan, dtype=np.float32)
        for i, col in enumerate(self.feature_cols):
            v = feats.get(col)
            if v is None:
                continue
            if col in self.encoders:
                vec[i] = self._encode(col, v)
            elif isinstance(v, (int, float, np.floating, np.integer, bool)):
                vec[i] = float(v)
            else:
                vec[i] = self._encode(col, v)
        return vec

    def explain(self, row: dict, states, probability: float, ltv: float) -> list[str]:
        """Short, human-readable drivers. These end up in the review queue and,
        for disputed transactions, in the Phase 3 evidence packet."""
        reasons: list[str] = []
        card = states.get("card")
        amt = float(row["TransactionAmt"])
        if card and card.count == 0:
            reasons.append("First transaction seen for this card fingerprint")
        elif card and card.mean_amount > 0 and amt > 3 * card.mean_amount:
            reasons.append(f"Amount is {amt / card.mean_amount:.1f}x this card's typical spend")
        email = states.get("email")
        if email and email.count == 0:
            reasons.append("Unseen email domain")
        dev = states.get("device")
        if dev and dev.count > 50:
            reasons.append(f"Device fingerprint shared across {dev.count} prior transactions")
        if card and card.last_seen_ts and (row.get("TransactionDT", 0) - card.last_seen_ts) < 60:
            reasons.append("Repeat attempt on this card within 60 seconds")
        if row.get("R_emaildomain") and row.get("P_emaildomain") != row.get("R_emaildomain"):
            reasons.append("Purchaser and recipient email domains differ")
        if ltv >= config.LTV_CAP:
            reasons.append("High-value customer: decline threshold raised accordingly")
        if not reasons:
            reasons.append(f"No individual red flag; model score {probability:.3f}")
        return reasons[:5]

    def score(self, req: CheckoutRequest, update_store: bool = True) -> ScoreResponse:
        t0 = time.perf_counter()
        row = request_to_row(req)
        keys = entity_keys(row)
        states = self.store.get_states(keys)
        backend_hit = self.store.backend == "redis"

        if self.ready:
            vec = self.build_vector(row, states)
            d = xgb.DMatrix(vec.reshape(1, -1), feature_names=self.feature_cols)
            raw = float(self.booster.predict(d)[0])
            # The threshold this is about to be compared against is derived
            # from dollars, so the score has to be a real probability. Applying
            # the same map the trainer fitted is what makes that true here.
            prob = self.calibrator.transform_one(raw)
        else:
            # No trained model on disk. Rather than return a fake score, fall
            # back to an explicit heuristic and flag it via model_version.
            raw = prob = _heuristic_score(row, states)

        entity_feats = derive_entity_features(states, req.amount, row.get("TransactionDT"))
        ltv = entity_feats["customer_ltv_estimate"]
        decision, cost = dec.decide(prob, req.amount, ltv, self.fixed_threshold)

        if update_store and decision is not Decision.DECLINE:
            self.store.update(keys, req.amount, row.get("TransactionDT"))

        return ScoreResponse(
            transaction_id=req.transaction_id,
            fraud_probability=round(prob, 6),
            raw_model_score=round(raw, 6),
            calibration=self.calibrator.fitted_method,
            decision=decision,
            cost=cost,
            top_reasons=self.explain(row, states, prob, ltv),
            latency_ms=round((time.perf_counter() - t0) * 1000, 3),
            model_version=self.model_version,
            feature_store_hit=backend_hit,
        )


def _heuristic_score(row: dict, states) -> float:
    """Transparent stand-in used only when no model artifact exists.

    It exists so the API, the demo, and the Phase 3 pipeline are runnable
    before training finishes. It is not a model and does not pretend to be.
    """
    s = 0.05
    card = states.get("card")
    amt = float(row.get("TransactionAmt") or 0.0)
    if card and card.count == 0:
        s += 0.15
    if card and card.mean_amount > 0 and amt > 3 * card.mean_amount:
        s += 0.25
    if row.get("R_emaildomain") and row.get("P_emaildomain") != row.get("R_emaildomain"):
        s += 0.10
    if amt > 500:
        s += 0.10
    dev = states.get("device")
    if dev and dev.count > 50:
        s += 0.15
    return float(min(s, 0.99))
