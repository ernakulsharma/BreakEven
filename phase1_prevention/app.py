"""FastAPI surface for the risk engine.

    uvicorn phase1_prevention.app:app --host 0.0.0.0 --port 8000

Endpoints
    POST /v1/score            score one checkout          (Phase 1)
    POST /v1/score/batch      score many                  (Phase 1)
    POST /v1/rings/analyze    detect abuse rings          (Phase 2)
    POST /v1/disputes/packet  build an evidence packet    (Phase 3)
    GET  /health              model + feature store state
"""
from __future__ import annotations

import time
from contextlib import asynccontextmanager

from fastapi import Body, FastAPI, HTTPException

from common.schema import (
    CheckoutRequest, DisputeCase, EvidencePacket, RingReport, ScoreResponse,
)
from phase1_prevention.prevention import PreventionEngine

_engine: PreventionEngine | None = None
_started = time.time()


def engine() -> PreventionEngine:
    global _engine
    if _engine is None:
        _engine = PreventionEngine()
    return _engine


@asynccontextmanager
async def lifespan(app: FastAPI):
    e = engine()
    print(f"[api] model={e.model_version} features={len(e.feature_cols)} "
          f"store={e.store.backend}")
    yield


app = FastAPI(title="Open Risk Engine", version="0.3.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    e = engine()
    return {
        "status": "ok",
        "uptime_seconds": round(time.time() - _started, 1),
        "model_loaded": e.ready,
        "model_version": e.model_version,
        "n_features": len(e.feature_cols),
        "feature_store": e.store.health(),
    }


@app.post("/v1/score", response_model=ScoreResponse)
def score(req: CheckoutRequest) -> ScoreResponse:
    return engine().score(req)


@app.post("/v1/score/batch", response_model=list[ScoreResponse])
def score_batch(reqs: list[CheckoutRequest]) -> list[ScoreResponse]:
    if len(reqs) > 1000:
        raise HTTPException(413, "batch limit is 1000 transactions")
    e = engine()
    return [e.score(r) for r in reqs]


@app.post("/v1/rings/analyze", response_model=RingReport)
def analyze_rings(transactions: list[dict] = Body(...)) -> RingReport:
    """Offline/near-line ring detection over a window of transactions.

    Deliberately not on the checkout path: graph construction is a batch shape
    of work, and forcing it into the sub-100ms budget would compromise both.
    """
    from phase2_rings.rings import detect_rings
    if not transactions:
        raise HTTPException(400, "no transactions supplied")
    return detect_rings(transactions)


@app.post("/v1/disputes/packet", response_model=EvidencePacket)
def dispute_packet(case: DisputeCase, use_llm: bool = True) -> EvidencePacket:
    from phase3_disputes.packet import build_packet
    return build_packet(case, use_llm=use_llm)
