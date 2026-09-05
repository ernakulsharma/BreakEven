"""Score connected components into explainable abuse rings.

A component is only a candidate. What makes it a *ring* is the shape of the
reuse inside it, and each signal below is a specific fraud tradecraft:

  card_fanout_per_device   one device cycling many stolen cards (card testing)
  address_concentration    many cards shipping to one drop address (reshipping)
  velocity                 orders compressed into a short burst before the
                           issuer's own velocity rules can react
  amount_uniformity        near-identical amounts, the signature of scripted
                           testing rather than human shopping
  new_account_share        cards with low D1 (days since first use) clustered
                           together — freshly minted or freshly stolen

The score is a transparent weighted blend, not a second black-box model. For
this component of the system that is the right call: ring output goes to a
human analyst who has to act on it, so every point of the score needs to be
attributable to a named behaviour.
"""
from __future__ import annotations

import time
from collections import Counter

import numpy as np
import pandas as pd

from common.schema import AbuseRing, RingReport
from phase2_rings.build_graph import EntityGraph, build_graph

MIN_RING_SIZE = 3
RISK_FLAG_THRESHOLD = 0.45
# A component holding more than this share of all transactions is treated as a
# construction artifact rather than a ring.
MAX_COMPONENT_SHARE = 0.01

WEIGHTS = {
    "card_fanout_per_device": 0.28,
    "address_concentration": 0.22,
    "velocity": 0.20,
    "amount_uniformity": 0.15,
    "new_account_share": 0.15,
}


def _sat(x: float, k: float) -> float:
    """Saturating 0..1 transform. Ring badness is not linear in raw counts:
    the jump from 1 to 5 cards on a device matters far more than 50 to 55."""
    return float(x / (x + k)) if x > 0 else 0.0


def _nunique(s: pd.Series) -> int:
    return int(s.astype(str).replace({"na": np.nan}).nunique(dropna=True))


def score_component(sub: pd.DataFrame) -> tuple[float, dict, list[str]]:
    n = len(sub)
    n_cards = _nunique(sub["card_key"]) if "card_key" in sub else 0
    n_devices = _nunique(sub["device_key"]) if "device_key" in sub else 0
    n_addr = _nunique(sub["addr_key"]) if "addr_key" in sub else 0

    cards_per_device = n_cards / max(n_devices, 1)
    cards_per_addr = n_cards / max(n_addr, 1)

    if "TransactionDT" in sub and n > 1:
        span_h = max((sub["TransactionDT"].max() - sub["TransactionDT"].min()) / 3600.0, 1e-6)
        velocity = n / span_h
    else:
        span_h, velocity = 0.0, 0.0

    amt = sub["TransactionAmt"].astype(float)
    cv = float(amt.std() / amt.mean()) if n > 1 and amt.mean() > 0 else 1.0
    uniformity = float(np.clip(1.0 - cv, 0.0, 1.0))

    if "D1" in sub:
        d1 = pd.to_numeric(sub["D1"], errors="coerce")
        new_share = float((d1.fillna(999) <= 1).mean())
    else:
        new_share = 0.0

    components = {
        "card_fanout_per_device": _sat(cards_per_device - 1, 4.0),
        "address_concentration": _sat(cards_per_addr - 1, 4.0),
        "velocity": _sat(velocity, 5.0),
        "amount_uniformity": uniformity if n >= MIN_RING_SIZE else 0.0,
        "new_account_share": new_share,
    }
    score = float(sum(WEIGHTS[k] * v for k, v in components.items()))

    signals: list[str] = []
    if cards_per_device >= 3 and n_devices > 0:
        signals.append(f"{n_cards} card fingerprints on {n_devices} device(s) — card-testing pattern")
    if cards_per_addr >= 3 and n_addr > 0:
        signals.append(f"{n_cards} cards shipping to {n_addr} address(es) — reshipping/drop pattern")
    if velocity >= 5:
        signals.append(f"{velocity:.1f} transactions/hour across the ring (span {span_h:.1f}h)")
    if uniformity >= 0.85 and n >= MIN_RING_SIZE:
        signals.append(f"Near-identical amounts (CV {cv:.2f}) — scripted rather than organic")
    if new_share >= 0.5:
        signals.append(f"{new_share:.0%} of cards first seen within a day of use")
    common_amt = Counter(np.round(amt, 2)).most_common(1)
    if common_amt and common_amt[0][1] >= max(3, int(0.5 * n)):
        signals.append(f"${common_amt[0][0]:.2f} repeated {common_amt[0][1]}x")
    if not signals:
        signals.append(f"Linked cluster of {n} transactions with no strong ring signature")

    stats = {
        "n_cards": n_cards, "n_devices": n_devices, "n_addr": n_addr,
        "velocity_per_hour": velocity, "components": components,
    }
    return score, stats, signals


def rings_from_graph(
    graph: EntityGraph,
    min_size: int = MIN_RING_SIZE,
    flag_threshold: float = RISK_FLAG_THRESHOLD,
    max_rings: int = 200,
    max_members_returned: int = 50,
    max_component_share: float = MAX_COMPONENT_SHARE,
) -> list[AbuseRing]:
    df = graph.frame.copy()
    df["_component"] = graph.labels
    sizes = df["_component"].value_counts()

    # An abuse ring is a small, tight cluster. A component holding a large
    # share of all traffic is a graph-construction artifact, not a ring, and
    # scoring it produces nonsense like "29,728 cards on 354 devices". Cap it
    # and surface the fact rather than silently emitting a blob.
    size_cap = max(min_size * 10, int(max_component_share * len(df)))
    oversized = sizes[sizes > size_cap]
    candidates = sizes[(sizes >= min_size) & (sizes <= size_cap)].index

    out: list[AbuseRing] = []
    for comp in candidates:
        sub = df[df["_component"] == comp]
        score, stats, signals = score_component(sub)
        if score < flag_threshold:
            continue
        ids = (sub["TransactionID"].astype(str).tolist()
               if "TransactionID" in sub else sub.index.astype(str).tolist())
        fraud_rate = float(sub["isFraud"].mean()) if "isFraud" in sub else None
        n_email = _nunique(sub["email_key"]) if "email_key" in sub else 0
        out.append(AbuseRing(
            ring_id=f"ring-{int(comp)}",
            size=len(sub),
            n_cards=stats["n_cards"], n_addresses=stats["n_addr"],
            n_devices=stats["n_devices"], n_email_domains=n_email,
            total_amount=round(float(sub["TransactionAmt"].sum()), 2),
            fraud_rate=fraud_rate,
            velocity_per_hour=round(stats["velocity_per_hour"], 3),
            risk_score=round(min(score, 1.0), 4),
            signals=signals,
            member_transactions=ids[:max_members_returned],
        ))

    if len(oversized):
        print(f"[rings] {len(oversized)} component(s) exceeded the size cap "
              f"({size_cap:,} txns) and were excluded as graph artifacts; "
              f"largest was {int(oversized.max()):,}. If this keeps happening, a "
              f"linking field is not identifying — check "
              f"graph.dropped_namespaces and the hub limits.")

    out.sort(key=lambda r: (-r.risk_score, -r.size))
    return out[:max_rings]


def detect_rings(transactions, **kwargs) -> RingReport:
    """End-to-end: records in, scored rings out."""
    t0 = time.time()
    df = transactions if isinstance(transactions, pd.DataFrame) else pd.DataFrame(transactions)
    graph = build_graph(df, keep_edges=False)
    rings = rings_from_graph(graph, **kwargs)
    return RingReport(
        n_transactions=len(df),
        n_components=graph.n_components,
        n_rings_flagged=len(rings),
        rings=rings,
        build_seconds=round(time.time() - t0, 3),
    )


def evaluate_rings(report: RingReport, df: pd.DataFrame) -> dict:
    """How much of the labelled fraud does ring membership actually capture?

    This is the number that decides whether Phase 2 earns its place: if
    flagged rings do not concentrate fraud well above the base rate, the
    graph is decoration.
    """
    if "isFraud" not in df.columns:
        return {"note": "no labels available"}
    member_ids = {t for r in report.rings for t in r.member_transactions}
    ids = df["TransactionID"].astype(str) if "TransactionID" in df else df.index.astype(str)
    in_ring = ids.isin(member_ids).to_numpy()
    base = float(df["isFraud"].mean())
    ring_rate = float(df.loc[in_ring, "isFraud"].mean()) if in_ring.any() else 0.0
    return {
        "base_fraud_rate": base,
        "fraud_rate_in_flagged_rings": ring_rate,
        "lift": (ring_rate / base) if base > 0 else None,
        "transactions_in_flagged_rings": int(in_ring.sum()),
        "share_of_all_fraud_captured": (
            float(df.loc[in_ring, "isFraud"].sum() / max(df["isFraud"].sum(), 1))),
    }
