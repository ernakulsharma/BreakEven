"""High-speed entity feature store.

Reads five entity hashes per checkout (card, addr, email, device, uid) in a
single pipelined round trip, and writes the updated aggregates back the same
way. That keeps the store's contribution to the latency budget at roughly one
network RTT regardless of how many entities we track.

Falls back to an in-process dict when Redis is unreachable so that the API,
the tests, and a laptop demo all run without infrastructure. The fallback is
explicitly reported in the response (`feature_store_hit`) rather than silently
pretending to be Redis.
"""
from __future__ import annotations

import threading
import time

from common import config
from common.features import EntityState

try:
    import redis as _redis
except ImportError:
    _redis = None


class FeatureStore:
    def __init__(self, url: str | None = None, ttl: int | None = None):
        self.ttl = ttl or config.FEATURE_TTL_SECONDS
        self.backend = "memory"
        self._client = None
        self._mem: dict[str, dict] = {}
        self._lock = threading.Lock()

        url = url or config.REDIS_URL
        if _redis is not None:
            try:
                c = _redis.Redis.from_url(url, decode_responses=True,
                                          socket_connect_timeout=0.25, socket_timeout=0.25)
                c.ping()
                self._client, self.backend = c, "redis"
            except Exception:
                self._client = None

    # -- key layout: rf:{namespace}:{entity_value} -> hash{count,amount_sum,last_seen_ts}
    @staticmethod
    def _key(ns: str, val: str) -> str:
        return f"rf:{ns}:{val}"

    def get_states(self, keys: dict[str, str]) -> dict[str, EntityState]:
        """Fetch all entity states for one checkout in a single round trip."""
        namespaces = list(keys)
        if self.backend == "redis":
            try:
                pipe = self._client.pipeline(transaction=False)
                for ns in namespaces:
                    pipe.hgetall(self._key(ns, keys[ns]))
                raw = pipe.execute()
                return {ns: EntityState.from_dict(r) for ns, r in zip(namespaces, raw)}
            except Exception:
                # Degrade rather than fail the checkout: a payment gateway that
                # 500s because its cache blinked is worse than one that scores
                # with cold features.
                self.backend = "memory"
        with self._lock:
            return {ns: EntityState.from_dict(self._mem.get(self._key(ns, keys[ns])))
                    for ns in namespaces}

    def update(self, keys: dict[str, str], amount: float, ts: float | None = None) -> None:
        """Fold this transaction into every entity's aggregates."""
        ts = time.time() if ts is None else ts
        if self.backend == "redis":
            try:
                pipe = self._client.pipeline(transaction=False)
                for ns, val in keys.items():
                    k = self._key(ns, val)
                    pipe.hincrby(k, "count", 1)
                    pipe.hincrbyfloat(k, "amount_sum", float(amount))
                    pipe.hset(k, "last_seen_ts", ts)
                    pipe.expire(k, self.ttl)
                pipe.execute()
                return
            except Exception:
                self.backend = "memory"
        with self._lock:
            for ns, val in keys.items():
                k = self._key(ns, val)
                s = self._mem.setdefault(k, {"count": 0, "amount_sum": 0.0, "last_seen_ts": None})
                s["count"] += 1
                s["amount_sum"] += float(amount)
                s["last_seen_ts"] = ts

    def warm(self, rows) -> int:
        """Seed the store from historical rows so day-one scoring is not cold.

        In practice this is run over the same training window the model saw,
        which is what keeps the online entity features on the same scale as
        the ones the model was fit on.
        """
        from common.features import entity_keys
        n = 0
        for row in rows:
            self.update(entity_keys(row), float(row.get("TransactionAmt") or 0.0),
                        row.get("TransactionDT"))
            n += 1
        return n

    def health(self) -> dict:
        return {"backend": self.backend, "ttl_seconds": self.ttl,
                "tracked_keys": (None if self.backend == "redis" else len(self._mem))}
