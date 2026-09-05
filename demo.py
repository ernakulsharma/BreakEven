"""End-to-end demo: one transaction population, three pipelines.

    python demo.py --data-dir data/sample

Shows the through-line that makes this one system rather than three projects:
a transaction is scored at checkout (Phase 1), the same transaction is placed
in an entity graph where its ring shows up (Phase 2), and when it later
disputes, the ring membership and the scoring rationale become evidence in
the representment packet (Phase 3).
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from common.schema import CheckoutRequest, Identity
from phase1_prevention.feature_store import FeatureStore
from phase1_prevention.prevention import PreventionEngine
from phase2_rings.build_graph import build_graph, graph_summary
from phase2_rings.rings import evaluate_rings, rings_from_graph
from common.schema import RingReport
from phase3_disputes.packet import build_packet, triage
from phase3_disputes.synth import from_transaction_rows

RULE = "=" * 72


def row_to_request(row: dict, i: int) -> CheckoutRequest:
    def g(k, cast=None):
        v = row.get(k)
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return None
        return cast(v) if cast else v

    return CheckoutRequest(
        transaction_id=str(row.get("TransactionID", f"tx-{i}")),
        amount=float(row["TransactionAmt"]),
        timestamp=datetime(2026, 1, 1) + timedelta(seconds=float(row["TransactionDT"])),
        product_cd=g("ProductCD"), card1=g("card1", int), card2=g("card2", float),
        card3=g("card3", float), card4=g("card4"), card5=g("card5", float), card6=g("card6"),
        addr1=g("addr1", float), addr2=g("addr2", float),
        dist1=g("dist1", float), dist2=g("dist2", float),
        purchaser_email_domain=g("P_emaildomain"), recipient_email_domain=g("R_emaildomain"),
        identity=Identity(device_type=g("DeviceType"), device_info=g("DeviceInfo"),
                          id_30=g("id_30"), id_31=g("id_31"), id_33=g("id_33")),
        d_features={f"D{n}": g(f"D{n}", float) for n in range(1, 16)},
        c_features={f"C{n}": g(f"C{n}", float) for n in range(1, 15)},
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("data/sample"))
    ap.add_argument("--limit", type=int, default=0,
                    help="cap rows loaded; 0 means all. The Phase 2 graph is "
                         "O(rows), so on the full 590k set use --limit for a "
                         "walkthrough and the trainer for real numbers.")
    ap.add_argument("--n-score", type=int, default=2000)
    ap.add_argument("--n-disputes", type=int, default=12)
    ap.add_argument("--llm", action="store_true", help="use Ollama for narratives")
    a = ap.parse_args()

    df = pd.read_csv(a.data_dir / "train_transaction.csv", low_memory=False)
    idp = a.data_dir / "train_identity.csv"
    if idp.exists():
        df = df.merge(pd.read_csv(idp, low_memory=False), on="TransactionID", how="left")
    if a.limit:
        df = df.head(a.limit)
        print(f"[warn] --limit takes the earliest {a.limit:,} rows, compressing the "
              f"time span.\n       Phase 2's graph links entities inside a 7-day "
              f"window, so a compressed\n       span over-connects it. Fine for a "
              f"walkthrough, not for ring numbers.")
    print(f"loaded {len(df):,} transactions from {a.data_dir}")

    # ---------------- Phase 1 ----------------
    print(f"\n{RULE}\nPHASE 1 — real-time prevention\n{RULE}")
    engine = PreventionEngine(store=FeatureStore())
    print(f"model {engine.model_version} | {len(engine.feature_cols)} features | "
          f"feature store: {engine.store.backend}")

    sample = df.head(a.n_score).to_dict("records")
    results = [engine.score(row_to_request(r, i)) for i, r in enumerate(sample)]
    counts = pd.Series([r.decision.value for r in results]).value_counts()
    lat = sorted(r.latency_ms for r in results)
    print(f"\nscored {len(results):,} checkouts end to end "
          f"(features + model + policy, not just inference)")
    print(f"  p50 {lat[len(lat)//2]:.2f}ms | p95 {lat[int(.95*len(lat))]:.2f}ms | "
          f"p99 {lat[int(.99*len(lat))]:.2f}ms")
    for k, v in counts.items():
        print(f"  {k:<9} {v:>6,}  ({v/len(results):.1%})")

    declines = [r for r in results if r.decision.value == "decline"]
    if declines:
        d = max(declines, key=lambda r: r.cost.cost_false_negative)
        print(f"\nexample decline — {d.transaction_id}")
        print(f"  p(fraud) {d.fraud_probability:.4f} vs threshold {d.cost.bayes_threshold:.4f}")
        print(f"  approving costs ${d.cost.expected_cost_approve:,.2f}, "
              f"declining costs ${d.cost.expected_cost_decline:,.2f}")
        for r in d.top_reasons:
            print(f"    - {r}")
    print("\nthe threshold moves per transaction — a $5 order and a $2,000 order")
    print("from the same customer are held to different standards of evidence:")
    for amt in (5.0, 80.0, 800.0, 3000.0):
        s = engine.score(row_to_request({**sample[0], "TransactionAmt": amt,
                                         "TransactionID": f"illus-{amt}"}, 0),
                         update_store=False)
        print(f"  ${amt:>7,.0f}  decline threshold {s.cost.bayes_threshold:.3f}  "
              f"(FN ${s.cost.cost_false_negative:>8,.0f} vs FP ${s.cost.cost_false_positive:>8,.0f})")

    # ---------------- Phase 2 ----------------
    print(f"\n{RULE}\nPHASE 2 — abuse ring detection\n{RULE}")
    g = build_graph(df)
    summary = graph_summary(g)
    rings = rings_from_graph(g)
    report = RingReport(n_transactions=len(df), n_components=g.n_components,
                        n_rings_flagged=len(rings), rings=rings, build_seconds=0.0)

    print(f"graph: {summary['n_components']:,} components over {len(df):,} transactions; "
          f"largest holds {summary['share_in_largest']:.1%}")
    if g.dropped_namespaces:
        for ns, info in g.dropped_namespaces.items():
            print(f"  dropped '{ns}' as a linking field: {info['distinct_values']} distinct "
                  f"values, avg degree {info['avg_degree']:,.0f} — categorical, not identifying")
    print(f"  hub entities pruned: {g.pruned_hubs}")
    print(f"  link window: {(g.link_window_seconds or 0)/86400:.0f} days")

    ev = evaluate_rings(report, df)
    if "lift" in ev and ev["lift"]:
        print(f"\n{len(rings)} rings flagged | fraud rate inside them "
              f"{ev['fraud_rate_in_flagged_rings']:.1%} vs {ev['base_fraud_rate']:.1%} base "
              f"({ev['lift']:.1f}x lift)")
    for r in rings[:3]:
        print(f"\n  {r.ring_id}: {r.size} txns, {r.n_cards} cards, {r.n_devices} devices, "
              f"${r.total_amount:,.0f} | risk {r.risk_score:.2f}")
        for s in r.signals[:4]:
            print(f"    - {s}")

    # ---------------- Phase 3 ----------------
    print(f"\n{RULE}\nPHASE 3 — chargeback representment\n{RULE}")
    ring_members = {t: r.ring_id for r in rings for t in r.member_transactions}
    disputed = df[df["TransactionID"].astype(str).isin(ring_members)].head(a.n_disputes)
    if len(disputed) < a.n_disputes:
        disputed = pd.concat([disputed, df.sample(a.n_disputes - len(disputed), random_state=1)])
    rows = disputed.to_dict("records")
    for r in rows:
        r["ring_id"] = ring_members.get(str(r["TransactionID"]))
    cases = from_transaction_rows(rows)

    t = triage(cases, use_llm=a.llm)
    print(f"{t['n_cases']} disputes | {t['n_contested']} worth contesting, "
          f"{t['n_accepted']} not")
    print(f"${t['total_disputed']:,.2f} disputed -> "
          f"${t['total_expected_recovery']:,.2f} expected recovery\n")
    print(f"  {'case':<14}{'code':<9}{'amount':>10}{'win':>7}{'E[recovery]':>13}  action")
    for r in t["queue"][:8]:
        print(f"  {r['case_id']:<14}{r['reason_code']:<9}${r['amount']:>9,.2f}"
              f"{r['win_probability']:>7.2f}${r['expected_recovery']:>12,.2f}  "
              f"{r['recommendation']}")

    best = t["packets"][0]
    print(f"\n--- packet {best.case_id} ({best.reason_code} {best.reason_code_title}) ---")
    print(f"recommendation: {best.recommendation} | win probability "
          f"{best.win_probability:.0%} | respond within {best.deadline_days} days")
    print("evidence:")
    for i in best.evidence:
        mark = "+" if i.present else ("!" if i.required else "-")
        print(f"  [{mark}] {i.label}")
        if i.present and i.value:
            print(f"        {i.value}")
        elif i.gap_note:
            print(f"        gap: {i.gap_note}")
    print(f"\nnarrative ({best.narrative_source}):\n")
    print(best.narrative)

    print(f"\n{RULE}")
    print("one dataset, three pipelines: the transaction scored in Phase 1 is a node")
    print("in the Phase 2 graph, and its ring membership plus the Phase 1 rationale")
    print("become evidence in the Phase 3 packet.")


if __name__ == "__main__":
    main()
