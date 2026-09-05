"""Synthetic dispute case generator.

No public dataset combines transaction records, delivery logs, 3DS results and
dispute outcomes — that data sits inside merchant, acquirer and network systems
and is never released. So rather than substitute an unrelated corpus, we
generate the input side ourselves and keep the output side rule-driven.

The generator is built to produce a realistic *spread* of case quality, because
a dispute engine that only ever sees winnable cases learns nothing useful. Each
case is drawn with an intended difficulty:

    strong   all critical evidence present  -> should recommend represent
    partial  some required evidence missing -> partial_represent
    weak     critical evidence absent       -> accept_liability

Seeding from real IEEE-CIS rows (`from_transaction_rows`) keeps amounts, card
types, email domains and device strings on a realistic distribution, so the
Phase 3 demo runs on the same transaction population as Phases 1 and 2.
"""
from __future__ import annotations

import random
from datetime import datetime, timedelta

from common.schema import (
    AVSCVVResult, CustomerHistory, DeliveryEvent, DeliveryRecord, DisputeCase,
    ThreeDSResult,
)

CARRIERS = ["UPS", "FedEx", "USPS", "DHL Express"]
MERCHANTS = ["Northwind Outfitters", "Ridgeline Audio", "Cobalt Supply Co.",
             "Halcyon Digital", "Fernwood Home"]
FIRST = ["Priya", "Marcus", "Elena", "Tom", "Ayesha", "Diego", "Wren", "Kofi"]
LAST = ["Sandhu", "Okafor", "Rivera", "Chen", "Bergman", "Nakamura", "Osei", "Kaur"]

PHYSICAL_CODES = [("visa", "13.1"), ("mastercard", "4855"), ("visa", "13.3"),
                  ("mastercard", "4853")]
FRAUD_CODES = [("visa", "10.4"), ("mastercard", "4837")]
OTHER_CODES = [("visa", "12.6.1"), ("visa", "13.2"), ("visa", "13.6")]


def _addr(rng: random.Random) -> str:
    return (f"{rng.randint(10, 9990)} {rng.choice(['Elm','Cedar','Kingsway','Mill','Ashford'])} "
            f"{rng.choice(['St','Ave','Rd','Lane'])}, "
            f"{rng.choice(['Portland OR 97205','Austin TX 78701','Leeds LS1 4AP','Pune 411001'])}")


def _delivery(rng: random.Random, tx_at: datetime, strength: str, address: str) -> DeliveryRecord | None:
    if strength == "weak" and rng.random() < 0.6:
        return None   # nothing was ever captured — the common real-world failure
    carrier = rng.choice(CARRIERS)
    shipped = tx_at + timedelta(hours=rng.randint(4, 48))
    delivered = shipped + timedelta(days=rng.randint(1, 6)) if strength != "weak" else None
    events = [DeliveryEvent(timestamp=shipped, status="label_created", location="Origin facility")]
    if delivered:
        events += [
            DeliveryEvent(timestamp=shipped + timedelta(days=1), status="in_transit",
                          location="Regional hub"),
            DeliveryEvent(timestamp=delivered, status="delivered", location=address,
                          note="Left with recipient" if strength == "strong" else "Left at door"),
        ]
    signed = strength == "strong" and rng.random() < 0.8
    return DeliveryRecord(
        carrier=carrier,
        tracking_number=(f"1Z{rng.randint(10**11, 10**12 - 1)}" if strength != "weak" else None),
        shipped_at=shipped, delivered_at=delivered, delivery_address=address,
        signature_obtained=signed,
        signed_by=(f"{rng.choice(FIRST)} {rng.choice(LAST)}" if signed else None),
        proof_of_delivery_url=(f"https://pod.{carrier.split()[0].lower()}.com/"
                               f"{rng.randint(10**7, 10**8)}" if delivered else None),
        events=events,
    )


def _three_ds(rng: random.Random, strength: str) -> ThreeDSResult:
    if strength == "strong":
        return ThreeDSResult(attempted=True, version="2.2.0", eci="05", liability_shift=True,
                             authentication_value="CAVV-***REDACTED***", status="Y")
    if strength == "partial":
        return ThreeDSResult(attempted=True, version="2.1.0", eci="06", liability_shift=False,
                             status=rng.choice(["A", "U"]))
    return ThreeDSResult(attempted=False, liability_shift=False, status=None)


def _avs(rng: random.Random, strength: str) -> AVSCVVResult:
    if strength == "strong":
        return AVSCVVResult(avs_code="Y", avs_street_match=True, avs_zip_match=True,
                            cvv_code="M", cvv_match=True)
    if strength == "partial":
        return AVSCVVResult(avs_code="Z", avs_street_match=False, avs_zip_match=True,
                            cvv_code="M", cvv_match=True)
    return AVSCVVResult(avs_code="N", avs_street_match=False, avs_zip_match=False,
                        cvv_code="N", cvv_match=False)


def _history(rng: random.Random, strength: str, tx_at: datetime) -> CustomerHistory:
    if strength == "strong":
        n = rng.randint(6, 40)
        return CustomerHistory(
            account_created_at=tx_at - timedelta(days=rng.randint(200, 1400)),
            prior_undisputed_transactions=n, prior_disputes=0,
            same_device_prior_orders=rng.randint(3, n), same_ip_prior_orders=rng.randint(2, n),
            login_before_purchase=True)
    if strength == "partial":
        return CustomerHistory(
            account_created_at=tx_at - timedelta(days=rng.randint(10, 120)),
            prior_undisputed_transactions=rng.randint(1, 3), prior_disputes=rng.randint(0, 1),
            same_device_prior_orders=rng.randint(0, 1), same_ip_prior_orders=0,
            login_before_purchase=rng.random() < 0.5)
    return CustomerHistory(
        account_created_at=tx_at - timedelta(hours=rng.randint(1, 48)),
        prior_undisputed_transactions=0, prior_disputes=rng.randint(0, 2),
        same_device_prior_orders=0, same_ip_prior_orders=0, login_before_purchase=False)


def make_case(
    seed: int | None = None,
    strength: str | None = None,
    network: str | None = None,
    reason_code: str | None = None,
    amount: float | None = None,
    is_digital_good: bool | None = None,
    transaction_id: str | None = None,
    ring_id: str | None = None,
) -> DisputeCase:
    rng = random.Random(seed)
    strength = strength or rng.choice(["strong", "partial", "weak"])

    if reason_code is None:
        pool = FRAUD_CODES + PHYSICAL_CODES + OTHER_CODES
        network, reason_code = rng.choice(pool)
    network = network or "visa"

    digital = (rng.random() < 0.3) if is_digital_good is None else is_digital_good
    amount = float(amount if amount is not None else round(rng.uniform(18, 1250), 2))
    tx_at = datetime(2026, rng.randint(1, 7), rng.randint(1, 28),
                     rng.randint(0, 23), rng.randint(0, 59))
    raised = tx_at + timedelta(days=rng.randint(5, 70))
    billing = _addr(rng)
    # Weak cases frequently ship somewhere other than the billing address —
    # that mismatch is itself one of the strongest fraud indicators.
    ship_to = billing if strength != "weak" or rng.random() < 0.3 else _addr(rng)

    merchant = rng.choice(MERCHANTS)
    # A weak case gets a MISMATCHED descriptor, not just a missing one — an
    # unrecognisable statement line is one of the most common real causes of a
    # "I didn't authorise this" dispute against a legitimate charge.
    if strength == "weak":
        descriptor = None if rng.random() < 0.5 else f"SQ*{rng.randint(1000, 9999)}"
    else:
        descriptor = f"{merchant.split()[0].upper()}*ORDER"

    return DisputeCase(
        case_id=f"case-{rng.randint(10**6, 10**7 - 1)}",
        transaction_id=transaction_id or f"tx-{rng.randint(10**8, 10**9 - 1)}",
        amount=amount, transaction_at=tx_at, dispute_raised_at=raised,
        network=network, reason_code=reason_code,
        merchant_name=merchant,
        descriptor=descriptor,
        cardholder_name=f"{rng.choice(FIRST)} {rng.choice(LAST)}",
        is_digital_good=digital,
        ip_address=f"{rng.randint(11,223)}.{rng.randint(0,255)}.{rng.randint(0,255)}.{rng.randint(1,254)}",
        device_fingerprint=f"fp_{rng.getrandbits(48):012x}",
        billing_address=billing,
        three_ds=_three_ds(rng, strength),
        avs_cvv=_avs(rng, strength),
        delivery=(None if digital else _delivery(rng, tx_at, strength, ship_to)),
        customer_history=_history(rng, strength, tx_at),
        refund_issued=(reason_code == "13.6" and strength == "strong"),
        terms_accepted_at=(tx_at - timedelta(minutes=rng.randint(1, 20))
                           if strength != "weak" else None),
        prior_ring_id=ring_id,
    )


def make_cases(n: int = 12, seed: int = 7) -> list[DisputeCase]:
    rng = random.Random(seed)
    strengths = (["strong"] * (n // 3) + ["partial"] * (n // 3))
    strengths += ["weak"] * (n - len(strengths))
    rng.shuffle(strengths)
    return [make_case(seed=seed * 1000 + i, strength=s) for i, s in enumerate(strengths)]


def from_transaction_rows(rows, seed: int = 11, strength: str | None = None) -> list[DisputeCase]:
    """Seed cases from real transaction records (e.g. IEEE-CIS rows flagged by
    Phase 1 or clustered by Phase 2), so amounts and identifiers are consistent
    across all three pipelines."""
    rng = random.Random(seed)
    out = []
    for i, row in enumerate(rows):
        amt = float(row.get("TransactionAmt") or rng.uniform(20, 500))
        txid = str(row.get("TransactionID") or f"tx-{i}")
        # A transaction the model scored as fraud most often disputes as fraud.
        is_fraud = bool(row.get("isFraud", 0))
        net, code = rng.choice(FRAUD_CODES if is_fraud else PHYSICAL_CODES)
        out.append(make_case(seed=seed * 100 + i, strength=strength, network=net,
                             reason_code=code, amount=amt, transaction_id=txid,
                             ring_id=row.get("ring_id")))
    return out
