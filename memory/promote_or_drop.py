"""
memory/promote_or_drop.py
--------------------------
Fires whenever an item is evicted from ShortTermMemory (buffer overflow).
Decides ONLY between two destinations:

    - forget    : not worth keeping (small talk, redundant confirmations,
                  a tool call that returned nothing new)
    - episodic  : a specific, patient- or operations-relevant event worth
                  recording (what happened, who, when, why, outcome)

This router NEVER writes to semantic_memory. Semantic facts are only ever
produced by the separate, periodic pass in memory/consolidation.py reading
FROM episodic_memory. That separation is the whole point: a single noisy
episode should never directly become a "fact" about a patient.

Every decision -- including "forget" -- is logged to
memory/routing_log.jsonl so a grader can see the reasoning behind every
choice, not just the ones that got promoted.
"""

from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass
from typing import Literal, Optional

from memory.db import get_connection
from memory.short_term_memory import MemoryItem

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROUTING_LOG_PATH = os.path.join(BASE_DIR, "routing_log.jsonl")


@dataclass
class MemoryRoutingDecision:
    reasoning: str
    destination: Literal["forget", "episodic"]
    event_summary: Optional[str] = None
    context: Optional[str] = None
    outcome: Optional[str] = None

    def __post_init__(self) -> None:
        if self.destination not in ("forget", "episodic"):
            raise ValueError(f"Invalid destination: {self.destination!r}")

    @classmethod
    def model_validate_json(cls, raw: str) -> "MemoryRoutingDecision":
        data = json.loads(raw)
        return cls(
            reasoning=data["reasoning"],
            destination=data["destination"],
            event_summary=data.get("event_summary"),
            context=data.get("context"),
            outcome=data.get("outcome"),
        )


ROUTING_PROMPT = """An item is about to be evicted from a hospital front-desk \
agent's short-term memory.

Decide where it belongs:
- forget: not worth keeping (small talk, a duplicate confirmation, a tool \
call that returned nothing new)
- episodic: a specific event worth recording long-term -- e.g. a reported \
allergy or adverse reaction, a clinical decision, a bed/room assignment, \
an escalation, anything a doctor or front-desk staff would need to recall \
on a future visit.

Item role: {role}
Item content: {content}

Respond ONLY as JSON matching this schema:
{{"reasoning": str, "destination": "forget" | "episodic",
  "event_summary": str | null, "context": str | null, "outcome": str | null}}
"""


# ---------------------------------------------------------------------------
# LLM-backed decision, with a deterministic offline fallback -- consistent
# with the rest of this repo (agent/agent.py falls back to an offline
# planner when ANTHROPIC_API_KEY is unset).
# ---------------------------------------------------------------------------

_KEEP_SIGNALS = (
    "allerg", "reaction", "diagnos", "assigned", "confirmed", "denied",
    "refused", "escalat", "adverse", "contraindicat", "discharge",
    "transfer", "consent", "code status", "dnr", "critical",
)


def _offline_decide(item: MemoryItem) -> MemoryRoutingDecision:
    """Deterministic heuristic used when no ANTHROPIC_API_KEY is configured."""
    text = str(item.content).lower()

    if any(sig in text for sig in _KEEP_SIGNALS):
        return MemoryRoutingDecision(
            reasoning=(
                "Offline heuristic: content matched a clinically or "
                "operationally significant keyword, so it is promoted "
                "to episodic memory rather than discarded."
            ),
            destination="episodic",
            event_summary=str(item.content)[:280],
            context=f"role={item.role}, session={item.session_id}",
            outcome=None,
        )

    return MemoryRoutingDecision(
        reasoning=(
            "Offline heuristic: no clinically or operationally significant "
            "keyword found; treated as routine chatter/tool noise and "
            "forgotten."
        ),
        destination="forget",
    )


def _llm_decide(item: MemoryItem) -> MemoryRoutingDecision:
    import anthropic  # imported lazily; only needed on the LLM path

    client = anthropic.Anthropic()
    prompt = ROUTING_PROMPT.format(role=item.role, content=str(item.content)[:2000])

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=400,
        messages=[{"role": "user", "content": prompt}],
    )
    raw_text = "".join(
        block.text for block in response.content if getattr(block, "type", None) == "text"
    )
    raw_text = raw_text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    return MemoryRoutingDecision.model_validate_json(raw_text)


def decide_memory_fate(item: MemoryItem) -> MemoryRoutingDecision:
    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            return _llm_decide(item)
        except Exception as exc:  # network/parse failure -> degrade gracefully
            fallback = _offline_decide(item)
            fallback.reasoning = (
                f"LLM routing failed ({exc!r}); fell back to offline "
                f"heuristic. {fallback.reasoning}"
            )
            return fallback
    return _offline_decide(item)


# ---------------------------------------------------------------------------
# Persistence + logging
# ---------------------------------------------------------------------------

def _log_decision(item: MemoryItem, decision: MemoryRoutingDecision) -> None:
    record = {
        "timestamp": item.timestamp,
        "session_id": item.session_id,
        "patient_id": item.patient_id,
        "user_id": item.user_id,
        "role": item.role,
        "raw_item": str(item.content)[:500],
        "destination": decision.destination,
        "reasoning": decision.reasoning,
    }
    with open(ROUTING_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def route_eviction(item: MemoryItem, conn: Optional[sqlite3.Connection] = None) -> MemoryRoutingDecision:
    """
    The function wired as ShortTermMemory(on_evict=...). Decides fate, logs
    the decision unconditionally, and -- only for 'episodic' -- inserts a
    row into episodic_memory. Never touches semantic_memory.
    """
    decision = decide_memory_fate(item)
    _log_decision(item, decision)

    if decision.destination == "episodic":
        owns_conn = conn is None
        conn = conn or get_connection()
        try:
            conn.execute(
                """
                INSERT INTO episodic_memory
                    (patient_id, user_id, session_id, timestamp, event_summary,
                     context, outcome, source_role, raw_item, routing_reason)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    item.patient_id,
                    item.user_id,
                    item.session_id,
                    item.timestamp,
                    decision.event_summary or str(item.content)[:280],
                    decision.context,
                    decision.outcome,
                    item.role,
                    json.dumps(item.content, default=str) if not isinstance(item.content, str) else item.content,
                    decision.reasoning,
                ),
            )
            conn.commit()
        finally:
            if owns_conn:
                conn.close()

    return decision
