# memory/ — Long-Term Memory Layer (Person 1)

## The real problem

Front-desk and clinical staff at MediCore re-ask patients (or re-read the
same tool output) for information that was already said earlier the same
call, or in an earlier visit entirely — most critically **allergy history**.
The existing MCP server (`mcp_server/`) has no notion of a conversation
persisting past a single session, and no notion of a fact ("this patient is
allergic to X") independent of the raw event that revealed it. A missed or
outdated allergy fact is not a cosmetic bug — it is the kind of gap that
leads to a wrong prescription. That is the real, costly failure mode this
folder exists to prevent.

## How each lab concern shows up here

| Concern | File | What to look for |
|---|---|---|
| Short-term memory + scratchpad | `short_term_memory.py` | `ShortTermMemory` (rolling buffer) vs `Scratchpad` (plan/sub-goal), kept as **separate objects** so pruning `messages` never touches `scratchpad`. |
| Promote-or-drop routing (forget / episodic only) | `promote_or_drop.py` | `decide_memory_fate()` returns `forget` or `episodic` with `reasoning`; **every** decision (including `forget`) is appended to `routing_log.jsonl`. Never writes to `semantic_memory`. |
| Semantic memory consolidation | `consolidation.py` | `run_once()` — a separate function you call on a schedule, reading `episodic_memory` and writing `semantic_memory` / `semantic_memory_history`. Handles: <br>• **updates** — same fact re-confirmed → confidence bump, no version churn (`_upsert_fact`, "case: UPDATE").<br>• **versioning** — every change lands in `semantic_memory_history` first (`_record_history`), old rows never deleted.<br>• **expiration** — `_expire_stale_facts()`, per-fact-family TTL (`DEFAULT_TTL_DAYS`).<br>• **conflict resolution** — "case: CONFLICT" branch in `_upsert_fact`, demoed live in `demo.py`. |
| Vector DB / RAG / Self-RAG | *(owned by Person 3, `rag/` + `retrieval_eval/`)* | not in this folder |
| Context management strategies | *(owned by Person 2, `context_eval/`)* | not in this folder — this folder only guarantees the scratchpad *survives* whatever pruning Person 2 implements |

## Schema

`schema.sql` adds four tables to the **existing** `db/meridian_hospital.db`
(no new database file, no duplicated patient data):

- `episodic_memory` — one row per promoted event, FK'd to real `Patients`/`Users`.
- `semantic_memory` — current, "live" fact per `(patient_id, fact_key)`.
- `semantic_memory_history` — append-only ledger; this is what makes versioning real instead of a fact field labelled "version" that nothing ever populates twice.
- `consolidation_runs` — proof that consolidation is a distinct, periodic pass (timestamps + counts per run), not something happening inline at write time.

## Running it

```bash
# one-time: create the memory tables inside the existing hospital db
python -m memory.db

# run the full demo: STM overflow -> routing -> episodic -> consolidation
# -> semantic memory, including a real allergy contradiction being resolved
python -m memory.demo

# run consolidation on its own (this is what a cron job / scheduler calls)
python -m memory.consolidation
```

No `ANTHROPIC_API_KEY` is required — `promote_or_drop.py` falls back to a
deterministic keyword-based router when the key is absent, consistent with
the offline stub pattern already used in `agent/agent.py`. Set
`ANTHROPIC_API_KEY` to route through Claude instead (`_llm_decide`).

## The contradiction `demo.py` resolves (real, not hypothetical)

1. **Session A**: a nurse's note mentions the patient has a *known
   penicillin allergy* — buried between a wall of tool-output noise. The
   router promotes it to `episodic_memory` (keyword match: `allerg`).
2. **Session B** (a later visit): a doctor reports the earlier allergy note
   was a mix-up — an allergy panel *ruled it out*. Also promoted to
   `episodic_memory`.
3. **Consolidation** (`run_once()`) processes both episodes in order:
   - Episode A → `semantic_memory` gets `allergy:penicillin = confirmed`, v1.
   - Episode B → contradicts v1. Confidence tie is broken in favor of the
     newer episode. v1 is marked `contradicted` in
     `semantic_memory_history` (kept, not deleted); v2 `ruled_out` becomes
     the active fact, with a human-readable `resolution_note` explaining
     exactly why.

## Integration point for Person 2 (agent loop)

`ShortTermMemory(on_evict=promote_or_drop.route_eviction)` is the only
integration surface the live agent needs: instantiate it inside the agent
loop, call `.add(role, content, session_id=..., patient_id=..., user_id=...)`
per turn, and read `.get_context()` / `.get_scratchpad_header()` when
building the next prompt. Call `memory.consolidation.run_once()` on a
schedule (or at the end of `agent.py --demo`) rather than per-turn.
