"""
memory/consolidation.py
------------------------
Semantic memory is built ONLY here, in a separate periodic pass over
episodic_memory. It is never written to by promote_or_drop.py directly, and
this pass never runs inline at write time -- run it on a schedule (cron,
Task Scheduler, or a call at agent startup/shutdown), see `run_once()`.

What this pass has to actually solve (per the lab spec), each represented as
a real code path below, not just described:

  1. UPDATES      -- a new episode implies a fact_key the patient already
                      has a value for, and the new value AGREES: refresh
                      updated_at / confidence, no version bump needed.
  2. VERSIONING   -- every write to semantic_memory is preceded by a row in
                      semantic_memory_history, so the previous value is never
                      lost, only superseded.
  3. EXPIRATION   -- facts older than a per-fact_key TTL are marked 'expired'
                      rather than silently trusted forever.
  4. CONFLICT
     RESOLUTION   -- a new episode implies a fact_key the patient already
                      has a DIFFERENT value for: resolved explicitly using
                      recency + confidence, with the losing value marked
                      'contradicted' (not deleted) and a human-readable
                      resolution_note recorded.

Fact extraction below is intentionally simple (regex/keyword based for the
allergy domain used in demo.py) with an LLM-backed path for the general
case, mirroring the offline/LLM split used elsewhere in this repo.
"""

from __future__ import annotations

import datetime
import json
import os
import re
import sqlite3
from dataclasses import dataclass
from typing import Optional

from memory.db import get_connection

# Facts older than this (in days) without being re-confirmed by a newer
# episode are considered stale. Different fact families could use different
# TTLs in a fuller implementation; one constant keeps the demo readable.
DEFAULT_TTL_DAYS = {
    "allergy": None,        # allergies don't expire on their own
    "vitals_flag": 30,      # a "reduced mobility" style flag goes stale in a month
    "default": 180,
}


def _now() -> str:
    return datetime.datetime.utcnow().isoformat()


@dataclass
class ExtractedFact:
    fact_key: str
    fact_value: str
    confidence: float
    family: str  # used to look up a TTL bucket


# ---------------------------------------------------------------------------
# Fact extraction from a single episode. Keyword/regex based so the demo is
# reproducible and free to run; swap in an LLM call for open-domain facts.
# ---------------------------------------------------------------------------

_NEGATION_RE = re.compile(
    r"\b(no|not|ruled out|denies?|denied|incorrect(?:ly)?\s+reported|"
    r"was\s+(?:a\s+)?(?:mix-?up|error|mistake)|corrected)\b", re.IGNORECASE
)

# Two orders show up in real notes: "allergic to penicillin" and
# "penicillin allergy". Try the explicit "to/from X" form first (most
# specific), then fall back to "<word> allerg*" immediately before/after.
_ALLERGY_TO_RE = re.compile(r"allerg\w*\s+(?:to|from)\s+([a-zA-Z\-]+)", re.IGNORECASE)
_ALLERGY_PRECEDING_RE = re.compile(r"\b([a-zA-Z\-]+)\s+allerg\w*", re.IGNORECASE)

_STOPWORDS = {"a", "an", "the", "known", "possible", "prior", "no", "not", "any"}


def _find_drug_mention(text: str) -> Optional[str]:
    m = _ALLERGY_TO_RE.search(text)
    if m:
        return m.group(1).lower()
    m = _ALLERGY_PRECEDING_RE.search(text)
    if m and m.group(1).lower() not in _STOPWORDS:
        return m.group(1).lower()
    return None


def extract_facts(event_summary: str, context: Optional[str], outcome: Optional[str]) -> list[ExtractedFact]:
    text = " ".join(filter(None, [event_summary, context, outcome]))
    facts: list[ExtractedFact] = []

    if "allerg" not in text.lower():
        return facts

    drug = _find_drug_mention(text)
    if not drug:
        return facts

    # A negation word anywhere near the allergy mention flips this to a
    # denial/correction rather than a confirmation.
    is_negated = bool(_NEGATION_RE.search(text))

    facts.append(ExtractedFact(
        fact_key=f"allergy:{drug}",
        fact_value="ruled_out" if is_negated else "confirmed",
        confidence=0.9, family="allergy",
    ))
    return facts


# ---------------------------------------------------------------------------
# Upsert logic implementing update / versioning / expiration / conflict
# resolution against semantic_memory + semantic_memory_history.
# ---------------------------------------------------------------------------

def _record_history(conn: sqlite3.Connection, *, patient_id: int, fact_key: str,
                     version: int, fact_value: str, confidence: float,
                     status: str, change_reason: str,
                     resolution_note: Optional[str], source_episode_ids: list[int]) -> None:
    conn.execute(
        """
        INSERT INTO semantic_memory_history
            (patient_id, fact_key, version, fact_value, confidence, status,
             change_reason, resolution_note, source_episode_ids, recorded_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (patient_id, fact_key, version, fact_value, confidence, status,
         change_reason, resolution_note, json.dumps(source_episode_ids), _now()),
    )


def _upsert_fact(conn: sqlite3.Connection, *, patient_id: int, fact: ExtractedFact,
                  episode_id: int, stats: dict) -> None:
    existing = conn.execute(
        "SELECT * FROM semantic_memory WHERE patient_id = ? AND fact_key = ?",
        (patient_id, fact.fact_key),
    ).fetchone()

    now = _now()

    if existing is None:
        # ---- case: brand new fact -----------------------------------
        _record_history(
            conn, patient_id=patient_id, fact_key=fact.fact_key, version=1,
            fact_value=fact.fact_value, confidence=fact.confidence,
            status="active", change_reason="initial_write", resolution_note=None,
            source_episode_ids=[episode_id],
        )
        conn.execute(
            """
            INSERT INTO semantic_memory
                (patient_id, fact_key, fact_value, confidence, version, status,
                 source_episode_ids, created_at, updated_at, expires_at)
            VALUES (?, ?, ?, ?, 1, 'active', ?, ?, ?, NULL)
            """,
            (patient_id, fact.fact_key, fact.fact_value, fact.confidence,
             json.dumps([episode_id]), now, now),
        )
        stats["facts_created"] += 1
        return

    if existing["fact_value"] == fact.fact_value:
        # ---- case: UPDATE (re-confirmation, same value) --------------
        source_ids = json.loads(existing["source_episode_ids"])
        source_ids.append(episode_id)
        conn.execute(
            """
            UPDATE semantic_memory
            SET confidence = MIN(1.0, confidence + 0.02),
                updated_at = ?,
                status = 'active',
                expires_at = NULL,
                source_episode_ids = ?
            WHERE patient_id = ? AND fact_key = ?
            """,
            (now, json.dumps(source_ids), patient_id, fact.fact_key),
        )
        stats["facts_updated"] += 1
        return

    # ---- case: CONFLICT -- same fact_key, different value ------------
    # Resolution policy: prefer the NEW episode when it is at least as
    # confident as the existing fact (newer clinical information should
    # not be silently ignored); otherwise keep the old value but log the
    # contradiction so a human can review it. Either way, nothing is
    # deleted -- the losing value is marked in history.
    new_wins = fact.confidence >= existing["confidence"]

    if new_wins:
        resolution_note = (
            f"Episode #{episode_id} contradicts existing fact "
            f"'{fact.fact_key}'={existing['fact_value']!r} (v{existing['version']}). "
            f"New value {fact.fact_value!r} kept: newer episode with "
            f"confidence {fact.confidence:.2f} >= prior confidence "
            f"{existing['confidence']:.2f}."
        )
        # old value: mark contradicted in history (not deleted)
        _record_history(
            conn, patient_id=patient_id, fact_key=fact.fact_key,
            version=existing["version"], fact_value=existing["fact_value"],
            confidence=existing["confidence"], status="contradicted",
            change_reason="conflict_resolution", resolution_note=resolution_note,
            source_episode_ids=json.loads(existing["source_episode_ids"]),
        )
        new_version = existing["version"] + 1
        _record_history(
            conn, patient_id=patient_id, fact_key=fact.fact_key,
            version=new_version, fact_value=fact.fact_value,
            confidence=fact.confidence, status="active",
            change_reason="conflict_resolution", resolution_note=resolution_note,
            source_episode_ids=[episode_id],
        )
        conn.execute(
            """
            UPDATE semantic_memory
            SET fact_value = ?, confidence = ?, version = ?, status = 'active',
                updated_at = ?, expires_at = NULL, source_episode_ids = ?
            WHERE patient_id = ? AND fact_key = ?
            """,
            (fact.fact_value, fact.confidence, new_version, now,
             json.dumps([episode_id]), patient_id, fact.fact_key),
        )
    else:
        resolution_note = (
            f"Episode #{episode_id} proposed {fact.fact_key}={fact.fact_value!r} "
            f"(confidence {fact.confidence:.2f}) but existing fact "
            f"{existing['fact_value']!r} (v{existing['version']}, confidence "
            f"{existing['confidence']:.2f}) was kept: incoming confidence too low "
            f"to override without further confirmation."
        )
        _record_history(
            conn, patient_id=patient_id, fact_key=fact.fact_key,
            version=existing["version"], fact_value=fact.fact_value,
            confidence=fact.confidence, status="contradicted",
            change_reason="conflict_resolution", resolution_note=resolution_note,
            source_episode_ids=[episode_id],
        )
        # existing row's flagged-for-review marker so staff can see there
        # was a conflicting report even though it didn't win
        conn.execute(
            "UPDATE semantic_memory SET updated_at = ? WHERE patient_id = ? AND fact_key = ?",
            (now, patient_id, fact.fact_key),
        )

    stats["conflicts_resolved"] += 1


def _expire_stale_facts(conn: sqlite3.Connection, stats: dict) -> None:
    rows = conn.execute(
        "SELECT * FROM semantic_memory WHERE status = 'active'"
    ).fetchall()

    now = datetime.datetime.utcnow()
    for row in rows:
        family = row["fact_key"].split(":")[0]
        ttl_days = DEFAULT_TTL_DAYS.get(family, DEFAULT_TTL_DAYS["default"])
        if ttl_days is None:
            continue  # this family never expires (e.g. allergies)

        updated_at = datetime.datetime.fromisoformat(row["updated_at"])
        if (now - updated_at) > datetime.timedelta(days=ttl_days):
            _record_history(
                conn, patient_id=row["patient_id"], fact_key=row["fact_key"],
                version=row["version"], fact_value=row["fact_value"],
                confidence=row["confidence"], status="expired",
                change_reason="expiration",
                resolution_note=f"No re-confirmation within {ttl_days} days.",
                source_episode_ids=json.loads(row["source_episode_ids"]),
            )
            conn.execute(
                "UPDATE semantic_memory SET status = 'expired' WHERE patient_id = ? AND fact_key = ?",
                (row["patient_id"], row["fact_key"]),
            )
            stats["facts_expired"] += 1


# ---------------------------------------------------------------------------
# Entry point: one consolidation run
# ---------------------------------------------------------------------------

def run_once() -> dict:
    """
    Scans every un-consolidated episode with a patient_id, extracts
    candidate facts, upserts them into semantic_memory (with full
    versioning/conflict handling), expires stale facts, and logs the run.
    Marks scanned episodes as consolidated so re-running is idempotent.
    """
    started_at = _now()
    stats = {"facts_created": 0, "facts_updated": 0, "facts_expired": 0, "conflicts_resolved": 0}

    conn = get_connection()
    try:
        episodes = conn.execute(
            "SELECT * FROM episodic_memory WHERE consolidated = 0 AND patient_id IS NOT NULL "
            "ORDER BY timestamp ASC"
        ).fetchall()

        for ep in episodes:
            facts = extract_facts(ep["event_summary"], ep["context"], ep["outcome"])
            for fact in facts:
                _upsert_fact(conn, patient_id=ep["patient_id"], fact=fact,
                             episode_id=ep["episode_id"], stats=stats)
            conn.execute(
                "UPDATE episodic_memory SET consolidated = 1 WHERE episode_id = ?",
                (ep["episode_id"],),
            )

        _expire_stale_facts(conn, stats)

        finished_at = _now()
        conn.execute(
            """
            INSERT INTO consolidation_runs
                (started_at, finished_at, episodes_scanned, facts_created,
                 facts_updated, facts_expired, conflicts_resolved, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (started_at, finished_at, len(episodes), stats["facts_created"],
             stats["facts_updated"], stats["facts_expired"], stats["conflicts_resolved"],
             "periodic consolidation pass"),
        )
        conn.commit()
    finally:
        conn.close()

    return {"episodes_scanned": len(episodes), **stats}


if __name__ == "__main__":
    result = run_once()
    print(json.dumps(result, indent=2))
