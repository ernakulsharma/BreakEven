"""Card-network reason codes and the evidence each one actually requires.

This is the piece that replaces the CUAD plan. CUAD is 510 commercial legal
contracts annotated for clause extraction — indemnification, governing law, IP
assignment. Nothing in it resembles a transaction record, a delivery log, a
3DS authentication result, or a network reason code, so it cannot teach a
model anything about representment.

What actually governs a dispute outcome is a published, finite rulebook: each
network defines a small set of reason codes, and for each one the acquirer
requires a specific list of evidence. That is a lookup table, not a learned
function, and encoding it directly is both more accurate and more auditable
than fine-tuning against an unrelated corpus.

Deadlines and requirements below reflect the public Visa VCR / Mastercard
chargeback frameworks. They change; treat this table as configuration to be
reviewed against your acquirer's current guide, not as legal advice.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from common.schema import DisputeCategory


@dataclass(frozen=True)
class EvidenceRequirement:
    key: str
    label: str
    required: bool = True
    weight: float = 1.0
    gap_note: str = ""


@dataclass(frozen=True)
class ReasonCode:
    code: str
    network: str
    title: str
    category: DisputeCategory
    deadline_days: int
    description: str
    requirements: list[EvidenceRequirement] = field(default_factory=list)
    auto_accept_if: list[str] = field(default_factory=list)


# --- shared requirement builders ------------------------------------------

_AVS_CVV = EvidenceRequirement(
    "avs_cvv_match", "AVS and CVV match results", required=False, weight=1.0,
    gap_note="No AVS/CVV match on file; card-absent fraud claims are hard to rebut without it.")
_DEVICE_MATCH = EvidenceRequirement(
    "device_ip_match", "Device fingerprint / IP matched to prior undisputed orders",
    required=False, weight=1.5,
    gap_note="No device or IP continuity linking this order to the cardholder's history.")
_HISTORY = EvidenceRequirement(
    "purchase_history", "Prior undisputed purchases by the same cardholder",
    required=False, weight=1.2,
    gap_note="No prior undisputed order history to establish a legitimate relationship.")
_DESCRIPTOR = EvidenceRequirement(
    "descriptor", "Billing descriptor shown to the cardholder", required=False, weight=0.5,
    gap_note="Descriptor not recorded; a mismatched descriptor is a common cause of "
             "'I don't recognise this' disputes.")
_TERMS = EvidenceRequirement(
    "terms_accepted", "Timestamped acceptance of terms / refund policy",
    required=False, weight=0.8,
    gap_note="No record of the cardholder accepting the terms at checkout.")


REASON_CODES: dict[str, ReasonCode] = {
    # ---------------- Visa ----------------
    "10.4": ReasonCode(
        code="10.4", network="visa", title="Other Fraud — Card-Absent Environment",
        category=DisputeCategory.FRAUD, deadline_days=30,
        description="Cardholder claims they did not authorise a card-not-present transaction.",
        requirements=[
            EvidenceRequirement("three_ds_liability_shift",
                                "3-D Secure authentication with liability shift", True, 3.0,
                                "No 3DS liability shift. This is the single strongest defence "
                                "for 10.4 and its absence usually decides the case."),
            _AVS_CVV, _DEVICE_MATCH, _HISTORY,
            EvidenceRequirement("delivery_to_billing_address",
                                "Goods delivered to the AVS-verified billing address",
                                False, 1.5,
                                "Delivery address was not verified against the billing address."),
            _DESCRIPTOR,
        ],
        auto_accept_if=["three_ds_liability_shift_false_and_no_avs_and_no_history"],
    ),
    "13.1": ReasonCode(
        code="13.1", network="visa", title="Merchandise / Services Not Received",
        category=DisputeCategory.NOT_RECEIVED, deadline_days=30,
        description="Cardholder claims the goods or services were never delivered.",
        requirements=[
            EvidenceRequirement("proof_of_delivery", "Carrier proof of delivery", True, 3.0,
                                "No proof of delivery on file — this claim cannot be rebutted "
                                "without it."),
            EvidenceRequirement("tracking_number", "Carrier tracking number", True, 1.5,
                                "No tracking number recorded."),
            EvidenceRequirement("delivery_timestamp", "Timestamped delivery confirmation", True, 2.0,
                                "Delivery timestamp missing."),
            EvidenceRequirement("signature", "Signature or photo confirmation at delivery",
                                False, 1.5,
                                "No signature captured; for high-value orders this is often "
                                "the deciding evidence."),
            EvidenceRequirement("delivery_address_match",
                                "Delivery address matches the cardholder's billing address",
                                False, 1.2,
                                "Delivery address does not match billing address on file."),
            _HISTORY,
        ],
    ),
    "13.3": ReasonCode(
        code="13.3", network="visa", title="Not as Described or Defective Merchandise",
        category=DisputeCategory.NOT_AS_DESCRIBED, deadline_days=30,
        description="Cardholder claims the item differs materially from its description.",
        requirements=[
            EvidenceRequirement("item_description", "Product description shown at purchase",
                                True, 2.0, "The listing description at time of purchase was not captured."),
            EvidenceRequirement("proof_of_delivery", "Proof the correct item was delivered",
                                True, 2.0, "No delivery evidence for the item as shipped."),
            _TERMS,
            EvidenceRequirement("no_return_received", "Item was not returned to the merchant",
                                False, 1.5, "Return status unknown."),
            EvidenceRequirement("support_correspondence",
                                "Support correspondence showing a remedy was offered",
                                False, 1.0,
                                "No record of an attempt to resolve directly with the customer."),
        ],
    ),
    "12.6.1": ReasonCode(
        code="12.6.1", network="visa", title="Duplicate Processing",
        category=DisputeCategory.DUPLICATE, deadline_days=30,
        description="Cardholder claims they were charged twice for one purchase.",
        requirements=[
            EvidenceRequirement("distinct_transactions",
                                "Evidence the two charges are separate purchases", True, 3.0,
                                "Cannot show the charges are distinct — if they are in fact a "
                                "duplicate, refund rather than represent."),
            EvidenceRequirement("separate_order_records", "Separate order records / itemisation",
                                True, 2.0, "No separate order records available."),
            EvidenceRequirement("no_refund_issued", "No refund already issued for this charge",
                                True, 1.0, "A refund may already have been processed."),
        ],
        auto_accept_if=["duplicate_confirmed"],
    ),
    "13.2": ReasonCode(
        code="13.2", network="visa", title="Cancelled Recurring Transaction",
        category=DisputeCategory.CANCELLED_RECURRING, deadline_days=30,
        description="Cardholder claims they cancelled before the charge was taken.",
        requirements=[
            EvidenceRequirement("no_cancellation_record",
                                "No cancellation was received before the billing date", True, 2.5,
                                "A cancellation may have been received before billing."),
            _TERMS,
            EvidenceRequirement("service_usage_after_charge",
                                "Cardholder used the service after the disputed charge",
                                False, 2.0, "No usage evidence after the charge date."),
            EvidenceRequirement("renewal_notice", "Advance renewal notice sent to the cardholder",
                                False, 1.2, "No record of a renewal reminder being sent."),
        ],
    ),
    "13.6": ReasonCode(
        code="13.6", network="visa", title="Credit Not Processed",
        category=DisputeCategory.CREDIT_NOT_PROCESSED, deadline_days=30,
        description="Cardholder claims a promised refund was never issued.",
        requirements=[
            EvidenceRequirement("refund_proof", "Proof the credit was issued", True, 3.0,
                                "No credit issued — if a refund was promised, accept liability."),
            EvidenceRequirement("refund_policy_disclosed", "Refund policy disclosed at checkout",
                                True, 1.5, "Refund policy disclosure not captured."),
            EvidenceRequirement("no_return_received", "No qualifying return was received",
                                False, 1.5, "Return status unknown."),
        ],
        auto_accept_if=["refund_promised_not_issued"],
    ),
    # ------------- Mastercard -------------
    "4837": ReasonCode(
        code="4837", network="mastercard", title="No Cardholder Authorization",
        category=DisputeCategory.FRAUD, deadline_days=45,
        description="Cardholder denies authorising the transaction.",
        requirements=[
            EvidenceRequirement("three_ds_liability_shift",
                                "3-D Secure authentication with liability shift", True, 3.0,
                                "No 3DS liability shift on a fraud-coded Mastercard dispute."),
            _AVS_CVV, _DEVICE_MATCH, _HISTORY, _DESCRIPTOR,
        ],
    ),
    "4855": ReasonCode(
        code="4855", network="mastercard", title="Goods or Services Not Provided",
        category=DisputeCategory.NOT_RECEIVED, deadline_days=45,
        description="Cardholder claims goods or services were not provided.",
        requirements=[
            EvidenceRequirement("proof_of_delivery", "Carrier proof of delivery", True, 3.0,
                                "No proof of delivery on file."),
            EvidenceRequirement("tracking_number", "Carrier tracking number", True, 1.5,
                                "No tracking number recorded."),
            EvidenceRequirement("delivery_timestamp", "Timestamped delivery confirmation", True, 2.0,
                                "Delivery timestamp missing."),
            EvidenceRequirement("signature", "Signature or photo confirmation", False, 1.5,
                                "No signature captured."),
            _HISTORY,
        ],
    ),
    "4853": ReasonCode(
        code="4853", network="mastercard", title="Cardholder Dispute — Not as Described",
        category=DisputeCategory.NOT_AS_DESCRIBED, deadline_days=45,
        description="Cardholder claims the goods or services differ from the description.",
        requirements=[
            EvidenceRequirement("item_description", "Product description shown at purchase",
                                True, 2.0, "Listing description not captured."),
            EvidenceRequirement("proof_of_delivery", "Proof of delivery of the correct item",
                                True, 2.0, "No delivery evidence."),
            _TERMS,
            EvidenceRequirement("support_correspondence", "Support correspondence with the customer",
                                False, 1.0, "No support history on file."),
        ],
    ),
}

# Digital goods have no carrier, so delivery requirements are swapped for
# access logs. Applying the physical-goods template to a SaaS charge produces
# a packet that is guaranteed to fail on missing evidence that never existed.
DIGITAL_GOODS_SUBSTITUTIONS = {
    "proof_of_delivery": EvidenceRequirement(
        "access_log", "Access / download log showing the product was consumed", True, 3.0,
        "No access log showing the cardholder used the digital product."),
    "tracking_number": EvidenceRequirement(
        "account_activity", "Account activity tied to the cardholder", False, 1.5,
        "No account activity recorded."),
    "signature": EvidenceRequirement(
        "ip_at_access", "IP address at time of access matching the purchase IP", False, 1.2,
        "No IP continuity between purchase and product access."),
    "delivery_timestamp": EvidenceRequirement(
        "first_access_timestamp", "Timestamp of first access to the product", True, 2.0,
        "No first-access timestamp."),
    "delivery_address_match": EvidenceRequirement(
        "email_delivery_confirmation", "Delivery confirmation to the cardholder's email",
        False, 1.2, "No email delivery confirmation."),
}


def get_reason_code(code: str, network: str | None = None) -> ReasonCode:
    rc = REASON_CODES.get(str(code).strip())
    if rc is None:
        known = ", ".join(sorted(REASON_CODES))
        raise KeyError(f"Unknown reason code {code!r}. Known codes: {known}")
    if network and rc.network != network:
        raise ValueError(f"Reason code {code} belongs to {rc.network}, not {network}")
    return rc


def requirements_for(rc: ReasonCode, is_digital_good: bool = False) -> list[EvidenceRequirement]:
    if not is_digital_good:
        return list(rc.requirements)
    return [DIGITAL_GOODS_SUBSTITUTIONS.get(r.key, r) for r in rc.requirements]


def codes_by_category(category: DisputeCategory) -> list[ReasonCode]:
    return [rc for rc in REASON_CODES.values() if rc.category == category]
