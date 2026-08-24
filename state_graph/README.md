# state_graph/ — Persistent, Recoverable State-Graph Agents

**Owner role:** State Graphs & Advanced LLM Patterns Lead  
**Company:** Meridian Hospital Network (MediCore Downtown / MediCore North)

This folder adds three **genuinely stateful** agents that sit beside the existing
memory/RAG agent and any prior decomposition/planning work. They reuse the same
`mcp_server/`, `db/meridian_hospital.db`, and do **not** re-skin scheduling or
retrieval problems from earlier labs.

---

## The three problems

| Graph | Why it cannot be a single pass | Two LLM additions | HITL trigger |
|-------|--------------------------------|-------------------|--------------|
| **post_op_recovery** | Spans days; waits on lab results that arrive on their own schedule; losing progress means re-collecting vitals and re-ordering labs | **Task decomposition** (plan the visit/check sequence) + **RAG** (pull post-op protocols) | Advance phase / escalate / discharge when confidence < 0.75 or action irreversible |
| **insurance_auth** | Waits on external insurer; denial requires a reasoned appeal; wrong resubmission wastes the claim window | **Constrained ReAct** (only whitelisted insurer-form tools) + **Tree of Thoughts** (choose appeal strategy) | Appeal filing when charge ≥ $10k or appeal confidence < 0.6 |
| **ed_surge_triage** | Irreversible actions (sedate, ICU admit, OR book) must not be taken by the agent alone; wrong order costs emergency time | **LATS** (search triage orderings against real acuity score) + **Constrained ReAct** (whitelist intake actions) | Any action in {sedate, admit_icu, book_or, transfer_out} |

None of these would produce the same result if run straight through with no pauses.

---

## Locatable concerns (grader map)

| Concern | Where to look |
|---------|----------------|
| Graph & cycle definitions | `graphs/post_op_recovery.py`, `graphs/insurance_auth.py`, `graphs/ed_surge_triage.py` — `build_*_graph()` returns `GraphDef` with `nodes` + `edges` (edges can loop, e.g. `await_lab_results → await_lab_results`) |
| Checkpointing after every meaningful transition | `checkpoint.py` → `CheckpointStore.save()`; called from `runtime.py` → `GraphRuntime._checkpoint()` after every node / HITL / failure |
| HITL node type | `runtime.request_hitl()` / `_HITLInterrupt`; nodes call it; `hitl.py` persists tasks for the platform |
| Ticket / failure-recovery path | `tickets.py`; unplanned exceptions in `GraphRuntime.step()` open a ticket (distinct code path from HITL) |
| Crash-and-resume | `GraphRuntime.resume(run_id)` loads last checkpoint; proven in `demo.py --crash-resume` |
| Two LLM additions per graph | `llm_techniques.py` + explicit comments inside each graph’s nodes |

---

## How checkpointing works

1. After every successful node, HITL pause, or failure, state is serialized to
   `graph_checkpoints` in `db/meridian_hospital.db`.
2. Kill the process mid-run → on restart call `runtime.resume(run_id)`.
3. Completed nodes are **not** re-executed; the graph continues from
   `current_node` with full `data` preserved.

---

## HITL vs Ticket (must be distinguishable)

| | HITL | Failure ticket |
|---|------|----------------|
| Nature | Expected pause — agent not allowed to decide alone | Unplanned — tool error, schema validation, unparseable model output |
| Status field | `waiting_hitl` | `failed` |
| Table | `hitl_tasks` | `failure_tickets` |
| Resume | Admin acts on platform → `resolve_hitl_task` → graph picks up decision | Admin resolves ticket → `resolve_failure_ticket` → retry from checkpoint |

---

## Running the demo

```bash
# from repo root
python -m state_graph.demo                  # all three graphs + ticket + crash-resume
python -m state_graph.demo --graph post_op_recovery
python -m state_graph.demo --crash-resume
python -m state_graph.demo --failure
```

No API key required — all LLM-shaped steps have deterministic offline fallbacks.

---

## Wiring to existing MCP / DB

- Checkpoints, HITL tasks, and tickets share `db/meridian_hospital.db`
  (same file used by `mcp_server/` and `memory/`).
- Graph nodes that perform hospital writes are designed to call MCP tools
  through the existing server; the demo uses offline executors so it runs
  without a live MCP process, but the whitelist names match MCP tool names
  (`submit_preauth`, `admit_icu`, etc.).
- Prior agents under `agent/`, `rag/`, and `memory/` are left in place and
  reused; this package only adds new state-graph agents.

---

## Prior-lab corrections touched by this role

- Decomposition / planning agent scope: any linear scheduling-style agent from
  earlier labs remains separate; these three graphs are **new** scopes with
  real waits and external branches.
- RAG: `llm_techniques.rag_retrieve` first tries the existing `rag/hybrid_search`
  index; if unavailable it falls back to embedded protocol stubs so graphs
  remain runnable while still demonstrating the RAG integration point.
