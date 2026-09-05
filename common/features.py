"""Feature engineering shared by the offline trainer and the online API.

The single most common way a fraud model dies in production is train/serve
skew: the batch job computes a rolling average one way, the API computes it
another, and the model silently sees a different distribution than it was
fit on. So both paths derive their entity features from the *same* functions
in this module, over the same `EntityState` structure.

Offline the state comes from a leak-free cumulative groupby (only rows strictly
before the current one). Online it comes from Redis. The arithmetic is
identical in both cases.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

from common import config

# Columns that are pure passthroughs from IEEE-CIS.
BASE_NUMERIC = ["TransactionAmt", "card1", "card2", "card3", "card5", "addr1", "addr2", "dist1", "dist2"]
BASE_CATEGORICAL = [
    "ProductCD", "card4", "card6", "P_emaildomain", "R_emaildomain",
    "DeviceType", "DeviceInfo", "id_30", "id_31", "id_33",
]

# Free email providers are disproportionately represented in account-takeover
# and bulk-signup fraud; the model can learn this but the flag makes it cheap.
FREE_EMAIL_DOMAINS = {
    "gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "aol.com",
    "protonmail.com", "mail.com", "gmx.com", "yandex.ru", "live.com",
    "icloud.com", "msn.com", "comcast.net", "anonymous.com",
}


# ---------------------------------------------------------------------------
# Entity keys
# ---------------------------------------------------------------------------

def _s(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "na"
    return str(v)


def entity_keys(row: dict) -> dict[str, str]:
    """Derive the linking keys used for the feature store and the Phase 2 graph.

    These are exactly the IEEE-CIS columns that identify a shared real-world
    entity across transactions, which is what makes one dataset serve both the
    scoring model and the ring graph.
    """
    card = "|".join(_s(row.get(c)) for c in ("card1", "card2", "card3", "card5"))
    addr = "|".join(_s(row.get(c)) for c in ("addr1", "addr2"))
    email = _s(row.get("P_emaildomain"))
    device = "|".join(_s(row.get(c)) for c in ("DeviceInfo", "DeviceType", "id_30", "id_31"))
    # uid: the closest thing IEEE-CIS has to an account. card1+addr1 anchored to
    # D1 (days since first card use) approximately isolates one card+address pair.
    d1 = row.get("D1")
    d1n = "na" if d1 is None or (isinstance(d1, float) and np.isnan(d1)) else str(int(d1))
    uid = f"{_s(row.get('card1'))}|{_s(row.get('addr1'))}|{d1n}"
    return {"card": card, "addr": addr, "email": email, "device": device, "uid": uid}


# ---------------------------------------------------------------------------
# Entity state (what the feature store holds)
# ---------------------------------------------------------------------------

@dataclass
class EntityState:
    """Rolling aggregates for one entity, as of *before* the current txn."""

    count: int = 0
    amount_sum: float = 0.0
    last_seen_ts: float | None = None

    @property
    def mean_amount(self) -> float:
        return self.amount_sum / self.count if self.count else 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict | None) -> "EntityState":
        if not d:
            return cls()
        return cls(
            count=int(d.get("count", 0) or 0),
            amount_sum=float(d.get("amount_sum", 0.0) or 0.0),
            last_seen_ts=(float(d["last_seen_ts"]) if d.get("last_seen_ts") not in (None, "") else None),
        )


def bounded_ltv(card_state: EntityState, amount: float) -> float:
    """Bounded customer lifetime value estimate.

    The first iteration of this used raw cumulative card spend, which is
    unbounded (it reached ~$1.6M per card1 bucket by the end of the IEEE-CIS
    period). Because card1 is a hashed bucket shared by ~44 transactions rather
    than a customer id, that number is not a customer's value at all — it is a
    population aggregate. Feeding it into the cost model made false-positive
    cost ~1500x false-negative cost and collapsed fraud recall to 8.9%.

    The bounded form is: typical basket size x a finite future purchase horizon.
    """
    typical = card_state.mean_amount if card_state.count else amount
    typical = max(typical, 1.0)
    return float(min(typical * config.EXPECTED_FUTURE_PURCHASES, config.LTV_CAP))


def derive_entity_features(
    states: dict[str, EntityState], amount: float, now_ts: float | None
) -> dict[str, float]:
    """The shared arithmetic. Called with Redis state online, groupby state offline."""
    card = states.get("card", EntityState())
    addr = states.get("addr", EntityState())
    email = states.get("email", EntityState())
    device = states.get("device", EntityState())
    uid = states.get("uid", EntityState())

    card_mean = card.mean_amount
    since_last = np.nan
    if now_ts is not None and card.last_seen_ts is not None:
        since_last = max(0.0, now_ts - card.last_seen_ts)

    return {
        "card_cum_count": float(card.count),
        "addr_cum_count": float(addr.count),
        "email_cum_count": float(email.count),
        "device_cum_count": float(device.count),
        "uid_cum_count": float(uid.count),
        "card_mean_amt": float(card_mean),
        "amt_over_card_mean": float(amount / card_mean) if card_mean > 0 else np.nan,
        "amt_minus_card_mean": float(amount - card_mean) if card.count else np.nan,
        "uid_mean_amt": float(uid.mean_amount),
        "seconds_since_last_card_txn": float(since_last),
        # Kept as a model *feature* (the model can use raw history fine) but
        # deliberately NOT used in the cost model. See bounded_ltv above.
        "ltv_proxy": float(card.amount_sum),
        "customer_ltv_estimate": bounded_ltv(card, amount),
    }


def derive_stateless_features(row: dict) -> dict[str, float]:
    """Features computable from the single transaction alone."""
    amt = float(row.get("TransactionAmt") or 0.0)
    dt = row.get("TransactionDT")
    cents = round(amt - np.floor(amt), 4)
    p_email = (row.get("P_emaildomain") or "") or ""
    r_email = (row.get("R_emaildomain") or "") or ""

    out = {
        "amt_log": float(np.log1p(max(amt, 0.0))),
        "amt_cents": float(cents),
        "amt_is_round": float(cents == 0.0),
        "p_email_is_free": float(p_email in FREE_EMAIL_DOMAINS),
        "r_email_present": float(bool(r_email)),
        "email_domains_match": float(bool(p_email) and p_email == r_email),
    }
    if dt is not None and not (isinstance(dt, float) and np.isnan(dt)):
        # TransactionDT is seconds from an arbitrary reference point. Absolute
        # date is meaningless but hour-of-day and day-of-week are not.
        secs = float(dt)
        out["hour_of_day"] = float((secs // 3600) % 24)
        out["day_of_week"] = float((secs // 86400) % 7)
        out["is_night"] = float(out["hour_of_day"] < 6 or out["hour_of_day"] >= 22)
    else:
        out["hour_of_day"] = np.nan
        out["day_of_week"] = np.nan
        out["is_night"] = np.nan
    return out


DERIVED_FEATURE_NAMES = sorted(
    set(derive_entity_features({}, 100.0, None)) | set(derive_stateless_features({"TransactionAmt": 1.0, "TransactionDT": 0}))
)


# ---------------------------------------------------------------------------
# Offline (batch) path
# ---------------------------------------------------------------------------

def add_entity_key_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Vectorised equivalent of entity_keys() over a whole dataframe."""

    def col(name):
        return df[name].astype("string").fillna("na") if name in df.columns else pd.Series("na", index=df.index, dtype="string")

    d1 = df["D1"].fillna(-1).astype("int32").astype("string") if "D1" in df.columns else pd.Series("na", index=df.index, dtype="string")
    keys = pd.DataFrame({
        "card_key": col("card1").str.cat([col("card2"), col("card3"), col("card5")], sep="|"),
        "addr_key": col("addr1").str.cat(col("addr2"), sep="|"),
        "email_key": col("P_emaildomain"),
        "device_key": col("DeviceInfo").str.cat([col("DeviceType"), col("id_30"), col("id_31")], sep="|"),
        "uid_key": col("card1").str.cat([col("addr1"), d1], sep="|"),
    }, index=df.index)
    return pd.concat([df.drop(columns=[c for c in keys.columns if c in df.columns]), keys], axis=1)


def add_batch_entity_features(df: pd.DataFrame) -> pd.DataFrame:
    """Leak-free cumulative aggregates: every value uses only *prior* rows.

    Implemented with cumsum/cumcount rather than `expanding().mean()`, which is
    O(n^2)-ish per group and was the original OOM/slowness culprit on 590k rows
    in a 1-core, 3.9GB sandbox.
    """
    df = df.sort_values("TransactionDT", kind="mergesort").reset_index(drop=True)
    amt = df["TransactionAmt"].astype("float32")
    new: dict[str, np.ndarray] = {}

    for ns, keycol in (
        ("card", "card_key"), ("addr", "addr_key"), ("email", "email_key"),
        ("device", "device_key"), ("uid", "uid_key"),
    ):
        g = df.groupby(keycol, sort=False, observed=True)
        prior_count = g.cumcount().astype("float32")           # excludes current row
        prior_sum = (g[ "TransactionAmt" ].cumsum() - amt).astype("float32")
        new[f"{ns}_cum_count"] = prior_count.to_numpy()
        if ns in ("card", "uid"):
            with np.errstate(invalid="ignore", divide="ignore"):
                mean = np.where(prior_count > 0, prior_sum / prior_count, np.nan).astype("float32")
            new[f"{ns}_mean_amt"] = mean
        if ns == "card":
            new["ltv_proxy"] = prior_sum.to_numpy()
            last = g["TransactionDT"].shift(1)
            new["seconds_since_last_card_txn"] = (df["TransactionDT"] - last).astype("float32").to_numpy()
        del g, prior_count, prior_sum

    card_mean = new["card_mean_amt"]
    a = amt.to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        new["amt_over_card_mean"] = np.where(card_mean > 0, a / card_mean, np.nan).astype("float32")
    new["amt_minus_card_mean"] = (a - card_mean).astype("float32")

    typical = np.where(np.isnan(card_mean) | (card_mean <= 0), a, card_mean)
    new["customer_ltv_estimate"] = np.minimum(
        np.maximum(typical, 1.0) * config.EXPECTED_FUTURE_PURCHASES, config.LTV_CAP
    ).astype("float32")

    # Stateless block, vectorised.
    new["amt_log"] = np.log1p(np.maximum(a, 0.0)).astype("float32")
    cents = np.round(a - np.floor(a), 4).astype("float32")
    new["amt_cents"] = cents
    new["amt_is_round"] = (cents == 0.0).astype("float32")
    dt = df["TransactionDT"].to_numpy()
    new["hour_of_day"] = ((dt // 3600) % 24).astype("float32")
    new["day_of_week"] = ((dt // 86400) % 7).astype("float32")
    new["is_night"] = ((new["hour_of_day"] < 6) | (new["hour_of_day"] >= 22)).astype("float32")

    p = df["P_emaildomain"].astype("string").fillna("") if "P_emaildomain" in df.columns else pd.Series("", index=df.index, dtype="string")
    r = df["R_emaildomain"].astype("string").fillna("") if "R_emaildomain" in df.columns else pd.Series("", index=df.index, dtype="string")
    new["p_email_is_free"] = p.isin(FREE_EMAIL_DOMAINS).to_numpy().astype("float32")
    new["r_email_present"] = (r != "").to_numpy().astype("float32")
    new["email_domains_match"] = ((p == r) & (p != "")).to_numpy().astype("float32")

    # One concat instead of N assignments — avoids the DataFrame fragmentation
    # warning and the repeated block copies that pushed us over memory.
    return pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1)
