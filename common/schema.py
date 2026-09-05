"""Pydantic contracts for all three pipelines.

Phase 1  checkout scoring request/response
Phase 2  entity graph + abuse ring output
Phase 3  chargeback case bundle + evidence packet
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

# ===========================================================================
# Phase 1 — prevention
# ===========================================================================


class Decision(str, Enum):
    APPROVE = "approve"
    REVIEW = "review"
    DECLINE = "decline"


class Identity(BaseModel):
    """Optional device/session enrichment, mirrors IEEE-CIS train_identity."""

    device_type: str | None = None          # DeviceType
    device_info: str | None = None          # DeviceInfo
    id_30: str | None = None                # OS
    id_31: str | None = None                # browser
    id_33: str | None = None                # screen resolution
    ip_address: str | None = None
    session_id: str | None = None
    extra: dict[str, float | str | None] = Field(default_factory=dict)


class CheckoutRequest(BaseModel):
    """A single checkout to be scored. Field names follow IEEE-CIS where a
    direct analogue exists so the training and serving schemas do not drift."""

    transaction_id: str
    amount: float = Field(gt=0, description="TransactionAmt, in USD")
    timestamp: datetime | None = None
    product_cd: str | None = Field(default=None, description="ProductCD")

    # Card fingerprint. card1 is the primary hashed issuer/BIN bucket.
    card1: int | None = None
    card2: float | None = None
    card3: float | None = None
    card4: str | None = None                # network: visa/mastercard/...
    card5: float | None = None
    card6: str | None = None                # credit / debit

    addr1: float | None = None
    addr2: float | None = None
    dist1: float | None = None
    dist2: float | None = None

    purchaser_email_domain: str | None = None   # P_emaildomain
    recipient_email_domain: str | None = None   # R_emaildomain

    identity: Identity | None = None

    # Vesta-style engineered blocks, passed through when the caller has them.
    c_features: dict[str, float | None] = Field(default_factory=dict)
    d_features: dict[str, float | None] = Field(default_factory=dict)
    m_features: dict[str, str | None] = Field(default_factory=dict)
    v_features: dict[str, float | None] = Field(default_factory=dict)

    @field_validator("purchaser_email_domain", "recipient_email_domain")
    @classmethod
    def _lower(cls, v: str | None) -> str | None:
        return v.lower().strip() if v else v


class CostBreakdown(BaseModel):
    """Why the threshold landed where it did — this is the auditable part."""

    cost_false_negative: float = Field(description="$ if we approve real fraud")
    cost_false_positive: float = Field(description="$ if we decline a good customer")
    customer_ltv_estimate: float
    bayes_threshold: float = Field(ge=0.0, le=1.0)
    fixed_threshold: float = Field(ge=0.0, le=1.0)
    expected_cost_approve: float
    expected_cost_decline: float


class ScoreResponse(BaseModel):
    transaction_id: str
    # Calibrated posterior. This is the number compared against the cost-derived
    # threshold, and the only one that should ever be read as "X% of these are
    # fraud". Both are surfaced because a caller debugging a decision needs to
    # see whether a surprising outcome came from the model or from the map.
    fraud_probability: float = Field(ge=0.0, le=1.0)
    raw_model_score: float = Field(ge=0.0, le=1.0)
    calibration: str = "identity"
    decision: Decision
    cost: CostBreakdown
    top_reasons: list[str] = Field(default_factory=list)
    latency_ms: float
    model_version: str
    feature_store_hit: bool = True


# ===========================================================================
# Phase 2 — abuse rings
# ===========================================================================


class EntityType(str, Enum):
    CARD = "card"
    ADDR = "addr"
    EMAIL = "email"
    DEVICE = "device"
    UID = "uid"


class RingEdge(BaseModel):
    source_transaction: str
    target_transaction: str
    shared_entity_type: EntityType
    shared_entity_value: str


class AbuseRing(BaseModel):
    ring_id: str
    size: int = Field(description="number of transactions in the component")
    n_cards: int
    n_addresses: int
    n_devices: int
    n_email_domains: int
    total_amount: float
    fraud_rate: float | None = Field(
        default=None, description="labelled fraud share, only known offline"
    )
    velocity_per_hour: float
    risk_score: float = Field(ge=0.0, le=1.0)
    signals: list[str] = Field(default_factory=list)
    member_transactions: list[str] = Field(default_factory=list)


class RingReport(BaseModel):
    n_transactions: int
    n_components: int
    n_rings_flagged: int
    rings: list[AbuseRing]
    build_seconds: float


# ===========================================================================
# Phase 3 — chargeback disputes
# ===========================================================================


class DisputeCategory(str, Enum):
    FRAUD = "fraud"
    NOT_RECEIVED = "not_received"
    NOT_AS_DESCRIBED = "not_as_described"
    DUPLICATE = "duplicate"
    CANCELLED_RECURRING = "cancelled_recurring"
    CREDIT_NOT_PROCESSED = "credit_not_processed"


class ThreeDSResult(BaseModel):
    attempted: bool
    version: str | None = None                  # "2.2.0"
    eci: str | None = None                      # electronic commerce indicator
    liability_shift: bool = False
    authentication_value: str | None = None     # CAVV present (redacted)
    status: Literal["Y", "A", "N", "U", "R", None] = None


class AVSCVVResult(BaseModel):
    avs_code: str | None = None      # "Y", "A", "Z", "N", ...
    avs_street_match: bool | None = None
    avs_zip_match: bool | None = None
    cvv_code: str | None = None      # "M" match, "N" no match
    cvv_match: bool | None = None


class DeliveryEvent(BaseModel):
    timestamp: datetime
    status: str
    location: str | None = None
    note: str | None = None


class DeliveryRecord(BaseModel):
    carrier: str | None = None
    tracking_number: str | None = None
    shipped_at: datetime | None = None
    delivered_at: datetime | None = None
    delivery_address: str | None = None
    signature_obtained: bool = False
    signed_by: str | None = None
    proof_of_delivery_url: str | None = None
    events: list[DeliveryEvent] = Field(default_factory=list)


class CustomerHistory(BaseModel):
    account_created_at: datetime | None = None
    prior_undisputed_transactions: int = 0
    prior_disputes: int = 0
    same_device_prior_orders: int = 0
    same_ip_prior_orders: int = 0
    login_before_purchase: bool = False


class DisputeCase(BaseModel):
    """Everything the engine needs to assemble a representment packet."""

    case_id: str
    transaction_id: str
    amount: float
    currency: str = "USD"
    transaction_at: datetime
    dispute_raised_at: datetime
    network: Literal["visa", "mastercard", "amex", "discover"]
    reason_code: str
    category: DisputeCategory | None = None
    merchant_name: str
    descriptor: str | None = None
    cardholder_name: str | None = None
    is_digital_good: bool = False
    ip_address: str | None = None
    device_fingerprint: str | None = None
    billing_address: str | None = None

    three_ds: ThreeDSResult | None = None
    avs_cvv: AVSCVVResult | None = None
    delivery: DeliveryRecord | None = None
    customer_history: CustomerHistory | None = None
    refund_issued: bool = False
    terms_accepted_at: datetime | None = None
    prior_ring_id: str | None = None   # link back to Phase 2


class EvidenceItem(BaseModel):
    key: str
    label: str
    required: bool
    present: bool
    weight: float = Field(ge=0.0, description="contribution to win probability")
    value: str | None = None
    gap_note: str | None = None


class EvidencePacket(BaseModel):
    case_id: str
    reason_code: str
    reason_code_title: str
    category: DisputeCategory
    recommendation: Literal["represent", "accept_liability", "partial_represent"]
    win_probability: float = Field(ge=0.0, le=1.0)
    evidence: list[EvidenceItem]
    missing_required: list[str] = Field(default_factory=list)
    deadline_days: int
    narrative: str = ""
    narrative_source: Literal["llm", "template"] = "template"
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    raw_context: dict[str, Any] = Field(default_factory=dict)
