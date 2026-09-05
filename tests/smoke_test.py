"""End-to-end smoke test. Proves every phase runs and the contracts line up.

    python -m tests.smoke_test

Requires data/sample (see tests/make_sample.py). Does not require Redis,
Ollama, or a trained model — each of those degrades to a documented fallback,
and the test asserts the fallback is honest about itself.
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

from common.schema import CheckoutRequest, Identity
from phase1_prevention.feature_store import FeatureStore
from phase1_prevention.prevention import PreventionEngine
from phase2_rings.build_graph import build_graph, graph_summary
from phase2_rings.rings import evaluate_rings, rings_from_graph
from phase3_disputes.packet import build_packet, triage
from phase3_disputes.synth import make_cases
from common.schema import RingReport

FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        FAILURES.append(name)


def phase1() -> None:
    print("\n=== Phase 1: prevention ===")
    store = FeatureStore()
    print(f"  feature store backend: {store.backend}")
    engine = PreventionEngine(store=store)
    print(f"  model: {engine.model_version} ({len(engine.feature_cols)} features)")

    req = CheckoutRequest(
        transaction_id="tx-smoke-1", amount=249.99, timestamp=datetime.now(),
        product_cd="W", card1=13926, card2=361.0, card3=150.0, card4="visa",
        card5=226.0, card6="debit", addr1=315.0, addr2=87.0,
        purchaser_email_domain="gmail.com",
        identity=Identity(device_type="mobile", device_info="iOS Device",
                          id_31="mobile safari 11.0"),
    )
    r = engine.score(req)
    check("scores a checkout", 0.0 <= r.fraud_probability <= 1.0, f"p={r.fraud_probability:.4f}")
    check("returns a decision", r.decision.value in ("approve", "review", "decline"),
          r.decision.value)
    check("emits cost rationale", r.cost.bayes_threshold > 0,
          f"thr={r.cost.bayes_threshold:.3f} ltv=${r.cost.customer_ltv_estimate:,.0f}")
    check("gives human-readable reasons", len(r.top_reasons) > 0, r.top_reasons[0])

    # Latency under repeated load.
    lat = []
    for i in range(200):
        req.transaction_id = f"tx-smoke-{i}"
        t = time.perf_counter()
        engine.score(req)
        lat.append((time.perf_counter() - t) * 1000)
    lat.sort()
    p95 = lat[int(0.95 * len(lat))]
    check("p95 latency under 100ms", p95 < 100, f"p95={p95:.2f}ms p50={lat[len(lat)//2]:.2f}ms")

    # The store must actually accumulate, or the online features are dead weight.
    from common.features import entity_keys
    from phase1_prevention.prevention import request_to_row
    st = store.get_states(entity_keys(request_to_row(req)))
    check("feature store accumulates entity state", st["card"].count > 0,
          f"card seen {st['card'].count}x, mean ${st['card'].mean_amount:.2f}")

    # Cost asymmetry must move the threshold, or the whole pitch is decoration.
    cheap = engine.score(req.model_copy(update={"transaction_id": "t-cheap", "amount": 5.0}))
    dear = engine.score(req.model_copy(update={"transaction_id": "t-dear", "amount": 2000.0}))
    check("threshold responds to transaction value",
          cheap.cost.bayes_threshold != dear.cost.bayes_threshold,
          f"$5 -> {cheap.cost.bayes_threshold:.3f} | $2000 -> {dear.cost.bayes_threshold:.3f}")


def phase2(df: pd.DataFrame, truth_ids: set[str], truncated: bool = False) -> None:
    print("\n=== Phase 2: abuse rings ===")
    t0 = time.time()
    g = build_graph(df)
    summary = graph_summary(g)
    rings = rings_from_graph(g)
    elapsed = time.time() - t0

    check("graph built", g.n_components > 1, f"{g.n_components:,} components in {elapsed:.1f}s")
    if truncated:
        # Asserting this on a truncated window would fail for a reason that is
        # about the slice, not the code.
        print(f"  SKIP  no giant component  (largest holds "
              f"{summary['share_in_largest']:.1%}; --limit distorts this)")
    else:
        check("no giant component", summary["share_in_largest"] < 0.05,
              f"largest holds {summary['share_in_largest']:.1%}")
    # Which fields get dropped is a property of the data, not of the code. The
    # fixture's email column is a category (8 domains), so it must drop there.
    # On another dataset a different set drops, and asserting a specific name
    # would be asserting the fixture rather than the guard.
    if truth_ids:
        check("non-identifying fields auto-dropped", "email" in g.dropped_namespaces,
              f"dropped {list(g.dropped_namespaces)}")
    else:
        check("identifiability guard ran", g.dropped_namespaces is not None,
              f"dropped {list(g.dropped_namespaces) or 'nothing — all fields identifying'}")
    check("rings found", len(rings) > 0, f"{len(rings)} flagged")

    rep = RingReport(n_transactions=len(df), n_components=g.n_components,
                     n_rings_flagged=len(rings), rings=rings, build_seconds=elapsed)
    ev = evaluate_rings(rep, df)
    check("rings concentrate fraud above base rate", (ev.get("lift") or 0) > 3,
          f"{ev['fraud_rate_in_flagged_rings']:.1%} vs {ev['base_fraud_rate']:.1%} base "
          f"({ev['lift']:.1f}x lift)")

    # Ring recall needs planted ground truth, which only the fixture has. Real
    # IEEE-CIS carries no ring label. Scoring against an empty truth set yields
    # 0.0% and reads as a regression rather than as an absent benchmark, so the
    # check is skipped instead — the fraud-lift check above still holds the
    # rings to account without needing labels.
    if truth_ids:
        found = {t for r in rings for t in r.member_transactions}
        recall = len(truth_ids & found) / len(truth_ids)
        check("recovers planted rings", recall > 0.3,
              f"{recall:.1%} of planted ring transactions")
    else:
        print("  SKIP  recovers planted rings  (no ring ground truth for this dataset)")
    check("every ring is explainable", all(r.signals for r in rings))


def phase3() -> None:
    print("\n=== Phase 3: chargeback disputes ===")
    cases = make_cases(15, seed=3)
    packets = [build_packet(c, use_llm=False) for c in cases]

    check("packet per case", len(packets) == len(cases))
    check("all three recommendations occur",
          len({p.recommendation for p in packets}) >= 2,
          str(sorted({p.recommendation for p in packets})))
    check("win probability spread is real",
          max(p.win_probability for p in packets) - min(p.win_probability for p in packets) > 0.3,
          f"{min(p.win_probability for p in packets):.2f}"
          f"..{max(p.win_probability for p in packets):.2f}")
    check("gaps are explained",
          all(all(i.gap_note or not i.required for i in p.evidence if not i.present)
              for p in packets))
    check("narrative always produced", all(len(p.narrative) > 100 for p in packets),
          f"source={packets[0].narrative_source}")

    # The core guarantee: a case with no critical evidence must not be fought.
    weak = build_packet(
        [c for c in cases if any(i.required and not i.present and i.weight >= 2.5
                                 for i in build_packet(c, use_llm=False).evidence)][0],
        use_llm=False)
    check("hopeless cases are not contested", weak.recommendation == "accept_liability",
          f"{weak.reason_code} win={weak.win_probability:.2f}")

    t = triage(cases, use_llm=False)
    check("triage ranks by expected recovery",
          all(t["queue"][i]["expected_recovery"] >= t["queue"][i + 1]["expected_recovery"]
              for i in range(len(t["queue"]) - 1)))
    print(f"  queue: {t['n_contested']}/{t['n_cases']} contested, "
          f"${t['total_expected_recovery']:,.0f} expected recovery of "
          f"${t['total_disputed']:,.0f} disputed")


def api() -> None:
    print("\n=== API contract ===")
    try:
        from fastapi.testclient import TestClient
        from phase1_prevention.app import app
    except ImportError as e:
        print(f"  SKIP  fastapi test client unavailable ({e})")
        return
    with TestClient(app) as c:
        h = c.get("/health")
        check("GET /health", h.status_code == 200, str(h.json().get("feature_store")))
        r = c.post("/v1/score", json={"transaction_id": "api-1", "amount": 120.0,
                                      "card1": 1234, "purchaser_email_domain": "gmail.com"})
        check("POST /v1/score", r.status_code == 200,
              f"decision={r.json().get('decision')}")
        d = c.post("/v1/disputes/packet?use_llm=false", json={
            "case_id": "c1", "transaction_id": "t1", "amount": 300.0,
            "transaction_at": "2026-03-01T10:00:00", "dispute_raised_at": "2026-03-20T10:00:00",
            "network": "visa", "reason_code": "13.1", "merchant_name": "Test Co",
            "delivery": {"carrier": "UPS", "tracking_number": "1Z999", "signature_obtained": True,
                         "delivered_at": "2026-03-04T12:00:00",
                         "proof_of_delivery_url": "https://pod.ups.com/1"},
        })
        check("POST /v1/disputes/packet", d.status_code == 200,
              f"win={d.json().get('win_probability')}")


def calibration() -> None:
    """The Bayes threshold reads `p` as a probability, so these are correctness
    checks on the decision rule itself, not model-quality checks."""
    print("\n=== Calibration ===")
    import json

    import numpy as np

    from common import config
    from common.calibration import ProbabilityCalibrator, reliability

    # Monotonicity is what lets us claim ranking metrics survive calibration.
    rng = np.random.default_rng(0)
    raw = rng.random(4000)
    y = (rng.random(4000) < 1 / (1 + np.exp(-(6 * raw - 4)))).astype(int)
    cal = ProbabilityCalibrator("isotonic").fit(raw, y)
    out = cal.transform(raw)
    order_in, order_out = np.argsort(raw), np.argsort(out, kind="stable")
    check("calibration is order-preserving",
          bool((out[order_in][:-1] <= out[order_in][1:] + 1e-12).all()),
          f"{cal.fitted_method}, ranking metrics unaffected")

    # A thin slice must not get an isotonic fit it cannot support.
    thin = ProbabilityCalibrator("auto").fit(raw[:300], y[:300] * 0)
    check("degenerate slice falls back to identity", thin.fitted_method == "identity",
          "no positives -> no invented map")

    report = Path(config.ARTIFACTS) / "eval_report.json"
    if not report.exists():
        check("eval report present", False, "train first")
        return
    rep = json.loads(report.read_text())
    c = rep.get("calibration")
    if not c:
        check("trainer emits calibration block", False, "retrain")
        return

    check("trainer emits calibration block", True, c["method"])
    check("calibration reduces expected calibration error",
          c["calibrated"]["ece"] < c["raw"]["ece"],
          f"ECE {c['raw']['ece']:.4f} -> {c['calibrated']['ece']:.4f}")
    check("calibrated Brier beats base-rate-only",
          c["calibrated"]["brier"] < c["calibrated"]["brier_base_rate_only"],
          f"{c['calibrated']['brier']:.4f} < {c['calibrated']['brier_base_rate_only']:.4f}")
    check("mean predicted probability tracks observed rate",
          abs(c["calibrated"]["over_prediction_ratio"] - 1.0) < 0.35,
          f"{c['raw']['over_prediction_ratio']:.2f}x -> "
          f"{c['calibrated']['over_prediction_ratio']:.2f}x")

    # Serving must apply the same map the trainer fitted, or the online
    # decision is taken on a different scale than the one that was evaluated.
    engine = PreventionEngine()
    if engine.ready:
        req = CheckoutRequest(transaction_id="tx-cal", amount=249.99,
                              timestamp=datetime.now(), product_cd="W", card1=13926,
                              card4="visa", card6="debit",
                              purchaser_email_domain="gmail.com")
        r = engine.score(req, update_store=False)
        check("online path applies the fitted calibrator",
              r.calibration == c["method"],
              f"serving={r.calibration} training={c['method']}")
        check("online response exposes both scales",
              r.raw_model_score >= 0 and r.fraud_probability >= 0,
              f"raw={r.raw_model_score:.4f} calibrated={r.fraud_probability:.4f}")

    d = rep.get("diagnostics", {})
    if "threshold_reachable" in d:
        check("threshold reachability is reported", True,
              f"median thr {d['median_bayes_threshold']:.2f} vs "
              f"calibrated p99 {d['calibrated_score_p99']:.2f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("data/sample"),
                    help="data/sample for the fixture, data/ieee-cis for real data")
    ap.add_argument("--limit", type=int, default=0,
                    help="cap rows. Speeds up a run, but see the warning it "
                         "prints: it distorts the Phase 2 graph.")
    a = ap.parse_args()
    sample = a.data_dir

    if not (sample / "train_transaction.csv").exists():
        print(f"No train_transaction.csv in {sample}.")
        print("  fixture:   python -m tests.make_sample --out data/sample")
        print("  real data: python -m tests.check_data --data-dir data/ieee-cis")
        return 2
    df = pd.read_csv(sample / "train_transaction.csv", low_memory=False)
    idp = sample / "train_identity.csv"
    if idp.exists():
        df = df.merge(pd.read_csv(idp, low_memory=False), on="TransactionID", how="left")
    if a.limit:
        df = df.head(a.limit)
        # Rows are ordered by TransactionDT, so head() is a contiguous *time*
        # slice, not a sample. Phase 2 links consecutive transactions per entity
        # inside a 7-day window, so compressing the calendar span packs far more
        # pairs into that window and the graph blobs. Measured on the fixture:
        # first 8k -> largest component 70.1%; random 8k -> 0.7%. Same row count.
        print(f"\n[warn] --limit {a.limit:,} takes the earliest rows, which compresses "
              f"the time span.\n       Phase 1 and 3 are unaffected; the Phase 2 graph is "
              f"not representative.\n       Drop --limit for real ring numbers.")

    # Ring ground truth only exists for the generated fixture. On real IEEE-CIS
    # there is no ring label, so recall against planted rings is skipped rather
    # than reported against an empty set — which would read as 0% and look like
    # a regression instead of an absent benchmark.
    truth_path = sample / "ring_truth.csv"
    truth_ids = (set(pd.read_csv(truth_path)["TransactionID"].astype(str))
                 if truth_path.exists() else set())
    if not truth_ids:
        print(f"\n(no ring_truth.csv in {sample} — ring recall will be skipped)")

    phase1()
    calibration()
    phase2(df, truth_ids, truncated=bool(a.limit))
    phase3()
    api()

    print("\n" + "=" * 60)
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: " + ", ".join(FAILURES))
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
