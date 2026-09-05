"""Assemble the final representment packet.

Division of labour, which is the important design decision here:

    RULES decide      which evidence is required, what is present, whether to
                      fight the dispute, and what the win probability is.
    THE LLM writes    the rebuttal letter from that already-decided structure.

The model never determines the outcome, never invents a fact, and never sees
a question it could answer wrongly in a way that costs money. It receives a
closed set of verified facts and turns them into the prose an acquirer will
read. If Ollama is unavailable, the deterministic template runs instead and
the packet is still complete and submittable — the narrative source is
recorded on the packet either way.

This is also why hallucination risk is low by construction: every factual
claim the letter can make already exists as a resolved evidence item.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request

from common import config
from common.schema import DisputeCase, EvidencePacket
from phase3_disputes.evidence import assess

SYSTEM_PROMPT = """You are a chargeback representment specialist writing a rebuttal letter to a card issuer.

Hard rules:
- Use ONLY the facts in the EVIDENCE list. Never invent a tracking number, timestamp, amount, name or address.
- If an evidence item is marked present=false, do not claim it exists. You may note it is unavailable, or simply omit it.
- Be factual and procedural. No emotional appeals, no speculation about the cardholder's motives, no legal threats.
- Reference the reason code and address the specific claim it represents.
- 200-320 words. Plain paragraphs, no markdown headings, no bullet lists.
- End with a single sentence requesting reversal of the chargeback."""


def _ollama_generate(prompt: str, model: str | None = None, timeout: float | None = None) -> str | None:
    """Call a local Ollama server. Returns None on any failure, never raises —
    a dispute packet must not fail to build because a side-car LLM is down."""
    body = json.dumps({
        "model": model or config.OLLAMA_MODEL,
        "prompt": prompt,
        "system": SYSTEM_PROMPT,
        "stream": False,
        "options": {"temperature": 0.2, "num_predict": 600},
    }).encode()
    req = urllib.request.Request(
        f"{config.OLLAMA_HOST.rstrip('/')}/api/generate",
        data=body, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout or config.OLLAMA_TIMEOUT) as r:
            return (json.loads(r.read().decode()).get("response") or "").strip() or None
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError, ValueError):
        return None


def build_prompt(case: DisputeCase, packet: EvidencePacket) -> str:
    present = [i for i in packet.evidence if i.present]
    absent = [i for i in packet.evidence if not i.present]
    lines = [
        f"REASON CODE: {packet.reason_code} ({packet.reason_code_title}), {case.network.upper()}",
        f"CLAIM: {packet.category.value.replace('_', ' ')}",
        f"MERCHANT: {case.merchant_name}",
        f"AMOUNT: {case.amount:.2f} {case.currency}",
        f"TRANSACTION DATE: {case.transaction_at:%Y-%m-%d %H:%M}",
        f"DISPUTE RAISED: {case.dispute_raised_at:%Y-%m-%d}",
        f"BILLING DESCRIPTOR: {case.descriptor or 'not recorded'}",
        "",
        "EVIDENCE AVAILABLE (present=true, you may cite these):",
    ]
    lines += [f"  - {i.label}: {i.value}" for i in present if i.value] or ["  (none)"]
    lines += ["", "EVIDENCE NOT AVAILABLE (present=false, do NOT claim these):"]
    lines += [f"  - {i.label}" for i in absent] or ["  (none)"]
    lines += ["", f"Write the rebuttal letter for {case.merchant_name}."]
    return "\n".join(lines)


def render_template(case: DisputeCase, packet: EvidencePacket) -> str:
    """Deterministic fallback letter. Complete and submittable on its own."""
    cited = [f"{i.label}: {i.value}" for i in packet.evidence if i.present and i.value]
    body = [
        f"Re: Chargeback representment — case {packet.case_id}, transaction "
        f"{case.transaction_id}, {case.amount:.2f} {case.currency} on "
        f"{case.transaction_at:%d %B %Y}.",
        "",
        f"{case.merchant_name} disputes the chargeback raised on "
        f"{case.dispute_raised_at:%d %B %Y} under {case.network.upper()} reason code "
        f"{packet.reason_code} ({packet.reason_code_title}). The transaction was "
        f"processed under the descriptor {case.descriptor or 'on file with the acquirer'}, "
        f"and the following records are submitted in support of the charge.",
        "",
    ]
    if cited:
        body.append("Evidence submitted:")
        body += [f"  {n}. {c}" for n, c in enumerate(cited, 1)]
        body.append("")
    if packet.missing_required:
        body.append("The following records required for this reason code are not available: "
                    + "; ".join(packet.missing_required) + ".")
        body.append("")
    if packet.recommendation == "accept_liability":
        body.append("On the evidence available, this transaction does not meet the "
                    "representment requirements for this reason code and the merchant "
                    "does not contest the chargeback.")
    else:
        body.append("The records above establish that the transaction was authorised and "
                    "fulfilled as ordered. We respectfully request that the chargeback be "
                    "reversed and the transaction amount returned to the merchant.")
    return "\n".join(body)


def build_packet(case: DisputeCase, use_llm: bool = True, model: str | None = None) -> EvidencePacket:
    packet = assess(case)

    if use_llm and packet.recommendation != "accept_liability":
        text = _ollama_generate(build_prompt(case, packet), model=model)
        if text:
            packet.narrative, packet.narrative_source = text, "llm"
            return packet

    packet.narrative = render_template(case, packet)
    packet.narrative_source = "template"
    return packet


def triage(cases: list[DisputeCase], use_llm: bool = False) -> dict:
    """Rank a queue of disputes by expected recovery.

    Representment costs staff time per case, so the ordering question is not
    "which can we win" but "which recovers the most per unit of effort".
    Expected recovery = win probability x disputed amount.
    """
    packets = [build_packet(c, use_llm=use_llm) for c in cases]
    rows = []
    for c, p in zip(cases, packets):
        rows.append({
            "case_id": p.case_id, "reason_code": p.reason_code,
            "category": p.category.value, "amount": c.amount,
            "win_probability": p.win_probability,
            "expected_recovery": round(p.win_probability * c.amount, 2),
            "recommendation": p.recommendation,
            "deadline_days": p.deadline_days,
            "missing_required": p.missing_required,
        })
    rows.sort(key=lambda r: -r["expected_recovery"])
    contested = [r for r in rows if r["recommendation"] != "accept_liability"]
    return {
        "n_cases": len(rows),
        "n_contested": len(contested),
        "n_accepted": len(rows) - len(contested),
        "total_disputed": round(sum(r["amount"] for r in rows), 2),
        "total_expected_recovery": round(sum(r["expected_recovery"] for r in contested), 2),
        "queue": rows,
        "packets": packets,
    }
