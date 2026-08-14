# Planning Lab — Person 1: Decomposition + DAG

## The real problem

Meridian's ER staff currently reshuffle the ICU/OR board **by hand** when
several critical patients arrive at once (mass-casualty, multi-vehicle
accidents, building-fire walk-ins). This is a genuine planning problem,
distinct from the Memory/RAG agent (Session 3), because:

- it needs **real branching** (several valid bed/transfer combinations exist)
- it has a **real cost to a wrong plan** (double-booking a bed a sicker
  patient needed a minute later, or leaving a critical patient unassigned
  because nobody checked another hospital's capacity)
- it has a **real difference between committing to one plan and adjusting
  mid-stream** (the real number of free beds is only known once you check;
  the right response to a shortfall only exists once you've observed one)

## What's here

| File | Concern |
|---|---|
| `models.py` | `Task` / `Plan`: DAG construction, dependency validation, cycle detection **at construction time**, topological ordering, parallel-safe batches. Forked from the reference toolkit, extended with `kind`/`mcp_tool` fields and a rule that write-kind tasks must be terminal nodes. |
| `llm_client.py` | Shared model-provider wrapper (Claude, with a deterministic offline fallback), used by every module in `planning/` — not just this one. |
| `mcp_grounding.py` | Real execution against `mcp_server/db_helpers.py` and `db/meridian_hospital.db` — no simulated tool outputs. |
| `decomposition.py` | **Decomposition-first**: the whole 6-node DAG is generated once, then executed in topological order. |
| `dynamic_decomposition.py` | **Dynamic/interleaved decomposition**: one task proposed, executed, and observed at a time — can insert a task (`escalate_overflow`) that was never in any fixed plan. |

## The DAG (decomposition-first)

```
assess_incoming   check_capacity
      \                /
       rank_by_urgency   (rank depends on assess; propose depends on both)
              \
        propose_assignment
              |
        validate_assignment
              |
        apply_and_report        <- single terminal write node
```

`assess_incoming` and `check_capacity` run in the same parallel batch
(no shared dependency). Acyclicity and the "no write mid-plan" rule are
enforced by `Plan`'s `model_validator`, so a bad DAG never reaches
execution.

## The required divergence demo

Run: `python -m planning_eval.test_cases.divergence_case`

Same request (3 critical patients, real DB currently has 2 free ICU
beds — `ICU-08`, `ICU-09`):

- **decomposition-first** commits to the 6-node plan above before
  knowing the real bed count. It has no "what if we're short" node, so
  it finishes with one patient's status left as *unresolved overflow*.
- **dynamic decomposition** observes the real shortfall right after
  `check_capacity` and inserts `escalate_overflow` — checking a partner
  hospital's real recorded ICU capacity — before continuing. That task
  never existed in the fixed plan; it was created *because of* the
  observation.

## Cost measured, not assumed

`python -m planning_eval.evaluate_decomposition` runs both methods
against 3 frozen cases against the real database and writes
`planning_eval/results/decomposition_results.json`:

| Case | Method | Task success | LLM calls | Total tokens | Latency |
|---|---|---|---|---|---|
| A — sufficient capacity | decomposition-first | ✅ | 4 | ~380 | ~0.003s |
| A — sufficient capacity | dynamic | ✅ | 6 | ~454 | ~0.003s |
| B — mass-casualty shortfall | decomposition-first | ❌ (unresolved overflow) | 5 | 431 | 0.005s |
| B — mass-casualty shortfall | dynamic | ✅ | 8 | 660 | 0.005s |
| C — mixed severity | decomposition-first | ✅ | 5 | 405 | 0.003s |
| C — mixed severity | dynamic | ✅ | 7 | 484 | 0.003s |

**Reading the table honestly:** dynamic decomposition costs ~25-50% more
tokens per run (larger context — it resends growing observation history
on every decide-next call) and is the *only* method that resolves Case
B correctly. For Case A and C, where capacity was never actually short,
decomposition-first reaches the same real outcome for less. This is why
the team ships **dynamic decomposition specifically for the top-level
surge-reshuffle request**, and keeps decomposition-first available for
sub-requests where a shortfall genuinely isn't in play — the table, not
a preference, drove that choice (see project README's final comparison
section for the merged table with Person 2/3's numbers).

## Tests

`python -m pytest planning_eval/test_decomposition.py -q` — 8 tests,
fully offline, run against the real database and clean up every write:
cycle rejection, unknown-dependency rejection, write-must-be-terminal
rejection, topological order correctness, the overflow/divergence
behavior itself, and a direct proof that a real `ICU_Beds` row changes
and is restored (not simulated).

## Running this piece standalone

```bash
cd Meridian-General-Hospital-Agents-phase2-main
pip install -r requirements.txt
pip install pydantic networkx pytest anthropic   # planning/ additions

python -m planning_eval.test_cases.divergence_case
python -m planning_eval.evaluate_decomposition
python -m pytest planning_eval/test_decomposition.py -q
```

No `ANTHROPIC_API_KEY` required — every LLM-shaped step has a
documented, deterministic, domain-aware offline fallback (see
`llm_client.py`). Set the key to route decomposition/ranking decisions
through Claude instead.
