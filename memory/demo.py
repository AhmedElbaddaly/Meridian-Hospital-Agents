"""
memory/demo.py
--------------
End-to-end demo of the memory layer, standalone (no ANTHROPIC_API_KEY
required -- runs on the offline heuristic router).

Scenario (real problem, not three fake conversations):
  Patient #1 (Mohamed Adel) is seen twice.
    Visit 1 (session A): a nurse mentions a penicillin allergy while
        describing symptoms, buried among several tool calls.
    Visit 2 (session B, "later"): a doctor reviews the chart and reports
        the earlier allergy note was a mix-up -- the patient is NOT
        allergic to penicillin, confirmed via allergy panel.
  This is exactly the kind of contradiction the lab asks for: two episodes
  implying different facts, which consolidation must resolve explicitly
  (not silently overwrite).

Run:
    python -m memory.demo
"""

from __future__ import annotations

import json

from memory import consolidation
from memory.db import get_connection, init_memory_schema
from memory.promote_or_drop import route_eviction
from memory.short_term_memory import ShortTermMemory


def _print_header(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def main() -> None:
    init_memory_schema()

    patient_id = 1  # Mohamed Adel, already seeded in db/meridian_hospital.db
    user_id = 3     # Dr. Sarah Ali

    # -----------------------------------------------------------------
    # 1) Short-term memory + scratchpad in action
    # -----------------------------------------------------------------
    _print_header("1) SHORT-TERM MEMORY + SCRATCHPAD")

    stm = ShortTermMemory(max_turns=4, on_evict=route_eviction)
    stm.scratchpad.update(
        plan="Complete intake for patient #1 and confirm admission",
        subgoal="Collect allergy history before doctor sees patient",
    )
    print("Scratchpad BEFORE any pruning:", stm.get_scratchpad_header())

    # Session A: turn 1 -- the important detail, buried immediately after
    stm.add("user", "Patient reports feeling dizzy and short of breath.",
            session_id="A", patient_id=patient_id, user_id=user_id)
    stm.add("user", "Nurse note: patient has a known penicillin allergy, "
                     "had a rash last time it was prescribed.",
            session_id="A", patient_id=patient_id, user_id=user_id)

    # A wall of tool-output-shaped noise that would normally bury turn 2
    for i in range(6):
        stm.add("tool", {"tool": "get_hospital_capacity", "result": f"noise payload {i}"},
                session_id="A", patient_id=patient_id, user_id=user_id)

    stm.add("assistant", "Thanks, noted. Proceeding with intake.",
            session_id="A", patient_id=patient_id, user_id=user_id)

    print("Scratchpad AFTER buffer overflow/eviction:", stm.get_scratchpad_header())
    print("(scratchpad is untouched by eviction -- it's a separate object)")

    # -----------------------------------------------------------------
    # 2) Session B, weeks later: a doctor corrects the record
    # -----------------------------------------------------------------
    _print_header("2) SESSION B -- CORRECTION EPISODE")

    stm_b = ShortTermMemory(max_turns=2, on_evict=route_eviction)
    stm_b.add("user", "Some chit-chat, patient asked about visiting hours.",
              session_id="B", patient_id=patient_id, user_id=user_id)
    stm_b.add("assistant", "Doctor reviewed the allergy panel: the prior penicillin "
                             "allergy report was incorrect, patient is not allergic "
                             "to penicillin. Chart corrected.",
              session_id="B", patient_id=patient_id, user_id=user_id)
    stm_b.flush_all()  # force eviction of everything still buffered

    # -----------------------------------------------------------------
    # 3) Inspect what promote-or-drop routed to episodic memory
    # -----------------------------------------------------------------
    _print_header("3) EPISODIC MEMORY (after promote-or-drop routing)")

    conn = get_connection()
    rows = conn.execute(
        "SELECT episode_id, session_id, event_summary, routing_reason "
        "FROM episodic_memory WHERE patient_id = ? ORDER BY episode_id", (patient_id,)
    ).fetchall()
    for r in rows:
        print(f"  [#{r['episode_id']}] session={r['session_id']}: {r['event_summary']!r}")
        print(f"        reason: {r['routing_reason']}")
    conn.close()

    print("\n(see memory/routing_log.jsonl for the FULL routing log, "
          "including every 'forget' decision)")

    # -----------------------------------------------------------------
    # 4) Run consolidation -- this is where the conflict gets resolved
    # -----------------------------------------------------------------
    _print_header("4) CONSOLIDATION PASS (episodic -> semantic)")

    result = consolidation.run_once()
    print(json.dumps(result, indent=2))

    # -----------------------------------------------------------------
    # 5) Final semantic memory + full version history for the fact
    # -----------------------------------------------------------------
    _print_header("5) SEMANTIC MEMORY (current) + HISTORY (versioned)")

    conn = get_connection()
    print("-- current semantic_memory row --")
    for r in conn.execute(
        "SELECT * FROM semantic_memory WHERE patient_id = ?", (patient_id,)
    ).fetchall():
        print(dict(r))

    print("\n-- full semantic_memory_history for allergy:penicillin --")
    for r in conn.execute(
        "SELECT version, fact_value, status, change_reason, resolution_note "
        "FROM semantic_memory_history "
        "WHERE patient_id = ? AND fact_key = 'allergy:penicillin' "
        "ORDER BY history_id", (patient_id,)
    ).fetchall():
        print(dict(r))
    conn.close()


if __name__ == "__main__":
    main()
