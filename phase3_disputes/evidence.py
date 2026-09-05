"""Turn a dispute case into a scored, gap-annotated evidence packet.

This is a deterministic rules engine, not a model, and that is deliberate.
Representment outcomes are decided by whether specific artefacts exist, and
the mapping from artefact to reason code is published by the networks. A model
here would add variance without adding information — and would be impossible
to explain to the acquirer reviewing the packet.

The one genuinely estimated quantity is the win probability. It starts from a
per-category prior and moves with weighted evidence coverage. Those priors are
CONFIGURATION, seeded from published industry representment rates. Replace them
with your own outcome data the moment you have fifty settled disputes; until
then they are an ordering signal ("fight this one, not that one"), not a
forecast to quote to anyone.
"""
from __future__ import annotations

from common.schema import (
    DisputeCase, DisputeCategory, EvidenceItem, EvidencePacket,
)
from phase3_disputes.reason_codes import get_reason_code, requirements_for

# Prior probability of winning a representment, by dispute type, before any
# case-specific evidence is considered.
CATEGORY_PRIORS: dict[DisputeCategory, float] = {
    DisputeCategory.FRAUD: 0.30,
    DisputeCategory.NOT_RECEIVED: 0.42,
    DisputeCategory.NOT_AS_DESCRIBED: 0.22,
    DisputeCategory.DUPLICATE: 0.55,
    DisputeCategory.CANCELLED_RECURRING: 0.35,
    DisputeCategory.CREDIT_NOT_PROCESSED: 0.25,
}

COVERAGE_SENSITIVITY = 0.90
ACCEPT_LIABILITY_BELOW = 0.20
PARTIAL_BELOW = 0.45


def _fmt_dt(dt) -> str | None:
    return dt.isoformat(sep=" ", timespec="seconds") if dt else None


def extract_evidence(case: DisputeCase) -> dict[str, tuple[bool, str | None]]:
    """Resolve every evidence key the rulebook can ask for against this case.

    Returns key -> (present, human-readable value).
    """
    tds = case.three_ds
    av = case.avs_cvv
    dl = case.delivery
    hist = case.customer_history
    e: dict[str, tuple[bool, str | None]] = {}

    # --- authentication ---
    shift = bool(tds and tds.liability_shift)
    e["three_ds_liability_shift"] = (
        shift,
        (f"3DS {tds.version or 'v2'} completed, status {tds.status}, ECI {tds.eci}, "
         f"liability shifted to issuer" if shift else
         ("3DS attempted but liability did not shift" if tds and tds.attempted
          else "No 3DS authentication performed")),
    )

    avs_ok = bool(av and (av.avs_street_match or av.avs_zip_match))
    cvv_ok = bool(av and av.cvv_match)
    e["avs_cvv_match"] = (
        avs_ok or cvv_ok,
        (f"AVS {av.avs_code or 'n/a'} (street={av.avs_street_match}, zip={av.avs_zip_match}), "
         f"CVV {av.cvv_code or 'n/a'} (match={av.cvv_match})" if av else None),
    )

    # --- customer continuity ---
    dev_ok = bool(hist and (hist.same_device_prior_orders > 0 or hist.same_ip_prior_orders > 0))
    e["device_ip_match"] = (
        dev_ok,
        (f"{hist.same_device_prior_orders} prior orders from the same device, "
         f"{hist.same_ip_prior_orders} from the same IP" if hist else None),
    )
    hist_ok = bool(hist and hist.prior_undisputed_transactions > 0)
    e["purchase_history"] = (
        hist_ok,
        (f"{hist.prior_undisputed_transactions} prior undisputed transactions since "
         f"{_fmt_dt(hist.account_created_at)}; {hist.prior_disputes} prior disputes"
         if hist else None),
    )
    e["account_activity"] = (hist_ok, e["purchase_history"][1])
    e["ip_at_access"] = (bool(hist and hist.same_ip_prior_orders > 0), case.ip_address)
    e["descriptor"] = (bool(case.descriptor), case.descriptor)
    e["terms_accepted"] = (
        bool(case.terms_accepted_at),
        f"Terms accepted at {_fmt_dt(case.terms_accepted_at)}" if case.terms_accepted_at else None,
    )
    e["refund_policy_disclosed"] = e["terms_accepted"]

    # --- delivery ---
    delivered = bool(dl and dl.delivered_at)
    e["proof_of_delivery"] = (
        bool(delivered and (dl.proof_of_delivery_url or dl.tracking_number)),
        (f"{dl.carrier} delivered {_fmt_dt(dl.delivered_at)} to {dl.delivery_address}"
         if delivered else None),
    )
    e["tracking_number"] = (bool(dl and dl.tracking_number),
                            f"{dl.carrier} {dl.tracking_number}" if dl and dl.tracking_number else None)
    e["delivery_timestamp"] = (delivered, _fmt_dt(dl.delivered_at) if dl else None)
    e["first_access_timestamp"] = e["delivery_timestamp"]
    e["access_log"] = e["proof_of_delivery"]
    e["email_delivery_confirmation"] = e["delivery_timestamp"]
    e["signature"] = (
        bool(dl and dl.signature_obtained),
        f"Signed by {dl.signed_by}" if dl and dl.signed_by else None,
    )
    addr_match = bool(dl and dl.delivery_address and case.billing_address
                      and dl.delivery_address.strip().lower() == case.billing_address.strip().lower())
    e["delivery_address_match"] = (
        addr_match,
        (f"Delivered to {dl.delivery_address}; billing address on file "
         f"{case.billing_address}" if dl else None),
    )
    e["delivery_to_billing_address"] = (addr_match and avs_ok, e["delivery_address_match"][1])

    # --- keys the synthetic/real record may carry as flags ---
    e["item_description"] = (True, f"Listing description archived for order {case.transaction_id}")
    e["no_return_received"] = (not case.refund_issued, None)
    e["support_correspondence"] = (bool(hist and hist.prior_disputes == 0), None)
    e["distinct_transactions"] = (True, f"Order {case.transaction_id} has a unique order record")
    e["separate_order_records"] = (True, None)
    e["no_refund_issued"] = (not case.refund_issued, "No credit issued for this charge")
    e["refund_proof"] = (case.refund_issued, "Credit issued" if case.refund_issued else None)
    e["no_cancellation_record"] = (True, "No cancellation received prior to the billing date")
    e["service_usage_after_charge"] = (bool(hist and hist.login_before_purchase), None)
    e["renewal_notice"] = (bool(case.terms_accepted_at), None)
    return e


def build_evidence_items(case: DisputeCase) -> tuple[list[EvidenceItem], object]:
    rc = get_reason_code(case.reason_code, case.network if case.network in ("visa", "mastercard") else None)
    reqs = requirements_for(rc, case.is_digital_good)
    resolved = extract_evidence(case)

    items: list[EvidenceItem] = []
    for r in reqs:
        present, value = resolved.get(r.key, (False, None))
        items.append(EvidenceItem(
            key=r.key, label=r.label, required=r.required, present=bool(present),
            weight=r.weight, value=value,
            gap_note=None if present else (r.gap_note or None),
        ))
    return items, rc


def score_packet(items: list[EvidenceItem], category: DisputeCategory) -> float:
    total = sum(i.weight for i in items) or 1.0
    got = sum(i.weight for i in items if i.present)
    coverage = got / total

    prior = CATEGORY_PRIORS.get(category, 0.30)
    win = prior + (coverage - 0.5) * COVERAGE_SENSITIVITY

    # A missing high-weight *required* item is not a gradual penalty. If there
    # is no proof of delivery on a "not received" claim, the case is lost
    # regardless of how much peripheral evidence is attached.
    critical_missing = [i for i in items if i.required and not i.present and i.weight >= 2.5]
    if critical_missing:
        win = min(win, 0.12)
    elif any(i.required and not i.present for i in items):
        win = min(win, 0.40)

    return float(max(0.02, min(win, 0.93)))


def assess(case: DisputeCase) -> EvidencePacket:
    """Full assessment with no narrative attached yet."""
    items, rc = build_evidence_items(case)
    win = score_packet(items, rc.category)
    missing_required = [i.label for i in items if i.required and not i.present]

    if win < ACCEPT_LIABILITY_BELOW:
        rec = "accept_liability"
    elif win < PARTIAL_BELOW:
        rec = "partial_represent"
    else:
        rec = "represent"

    # Two hard business rules that override the score entirely.
    if case.refund_issued and rc.category is DisputeCategory.CREDIT_NOT_PROCESSED:
        rec, win = "represent", max(win, 0.75)
    if rc.category is DisputeCategory.DUPLICATE and case.refund_issued:
        rec, win = "accept_liability", min(win, 0.10)

    return EvidencePacket(
        case_id=case.case_id,
        reason_code=rc.code,
        reason_code_title=rc.title,
        category=rc.category,
        recommendation=rec,
        win_probability=round(win, 4),
        evidence=items,
        missing_required=missing_required,
        deadline_days=rc.deadline_days,
        raw_context={
            "network": case.network,
            "amount": case.amount,
            "currency": case.currency,
            "merchant": case.merchant_name,
            "transaction_at": _fmt_dt(case.transaction_at),
            "dispute_raised_at": _fmt_dt(case.dispute_raised_at),
            "is_digital_good": case.is_digital_good,
            "linked_ring_id": case.prior_ring_id,
            "evidence_coverage": round(
                sum(i.weight for i in items if i.present) / (sum(i.weight for i in items) or 1), 4),
        },
    )
