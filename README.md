# Meridian Hospital Network — MCP Server Lab

**Company:** Meridian Hospital Network
*(fictional two-hospital regional group: MediCore Downtown, MediCore North)*

---

## Problem

Front-desk and hospital staff currently rely on manual communication to check available resources such as ICU beds and operating rooms, then update patient admission information manually.

During busy periods, staff may use general-purpose AI assistants to retrieve hospital information or perform database updates.

Allowing an AI model to directly interact with a hospital database without restrictions creates serious risks:

- Assigning an unavailable ICU bed.
- Updating patient information incorrectly.
- Modifying hospital resources without authorization.
- Performing unsafe write operations.

To prevent these failures, we implemented an **MCP server** between the AI agent and the hospital database.

The model can retrieve information through controlled read-only tools, while every write operation is performed through typed MCP tools with:

- Server-side validation.
- JSON Schema validation.
- Authorization checks.
- Business rule enforcement.

This README documents **Member 3's contribution**, including MCP client integration, MCP protocol features, testing, and demonstration.

---

## Repository Layout

```text
.
├── .gitignore
├── README.md
├── requirements.txt
│
├── agent/
│   ├── agent.py
│   ├── mcp_protocol.py
│   └── test_e2e.py
│
├── db/
│   ├── drawsql_erd.png
│   ├── erd_dbdiagram.png
│   ├── README.md
│   ├── schema.sql
│   └── seed.sql
│
└── mcp_server/
    ├── MCP.py
    ├── db_helpers.py
    ├── mock_server.py
    ├── schemas.py
    └── validation.py
```

---

## MCP Server Components

### MCP.py

Main MCP server implementation.

Responsibilities:

- Define MCP tools.
- Handle database operations.
- Apply validation rules.
- Expose MCP resources and prompts.

### schemas.py

Contains JSON Schemas used for validating MCP tool inputs.

Includes schemas for:

- Patient registration.
- Patient status updates.
- Admissions.
- ICU bed assignments.
- Operating room status updates.

### validation.py

Responsible for security and business rule validation.

Includes:

- Authorization checks.
- Patient validation.
- ICU assignment validation.
- Admission validation.
- Operating room validation.

### db_helpers.py

Database helper functions responsible for communication with the hospital database layer.

### mock_server.py

Mock MCP server used for end-to-end testing without requiring a real database connection.

---

## Running the Project

```bash
cd agent

pip install -r ../requirements.txt

python test_e2e.py

python agent.py --demo

python agent.py
```

If `ANTHROPIC_API_KEY` is not configured, the client automatically uses an offline deterministic planner and sampling stub.

If the API key is available, Claude is used for model-based tool selection and sampling.

---

## MCP Tools

| Tool | Type | Authentication | Human Confirmation | Purpose |
|---|---|---|---|---|
| `get_patient_details` | Read | None | No | Retrieve complete patient information by ID |
| `get_available_icu_beds` | Read | None | No | Retrieve available ICU beds |
| `get_hospital_capacity` | Read | None | No | Check hospital capacity information |
| `register_patient` | Write | Admin | No | Register a new patient |
| `update_patient_status` | Write | Doctor | Policy dependent | Update patient medical status |
| `create_admission` | Write | Doctor | No | Create admission record |
| `manage_icu_bed` | Write | Doctor | Yes for sensitive assignments | Assign or release ICU beds |
| `update_operating_room_status` | Write | Admin | No | Update operating room status |

---

## Capability Negotiation

During MCP initialization, the client exchanges capabilities with the server.

Supported capabilities:

- Tools
- Resources
- Prompts
- Elicitation
- Sampling
- Progress Notifications

The client stores server capabilities and checks availability before using optional MCP features.

Example:

```python
agent.supports("elicitation")
agent.supports("sampling")
```

---

## MCP Features Implemented

### Capability Negotiation

The client exchanges capabilities during the MCP initialize phase.

The server responds with supported features and the client adapts its behavior accordingly.

### Notifications

The server supports progress notifications for long-running operations.

Example:

```
notifications/progress
```

Used to provide updates while checking hospital resources.

Example:

```
Checking ICU beds...
```

### Elicitation (Human-in-the-Loop)

Sensitive operations require explicit user confirmation before execution.

Example:

```
Confirm ICU bed assignment?
```

Implemented for:

- ICU bed assignment using `manage_icu_bed`.
- Critical resource allocation operations.

This prevents unsafe automatic modifications.

### Sampling

The server can request the client model to generate content.

Example:

```
sampling/createMessage
```

Used for AI-generated admission-related text.

If no API key exists, an offline sampling stub is used.

### Resources

Hospital policies are exposed as MCP Resources instead of tools.

Available resources:

- `triage://protocols/guidelines`

  Emergency triage guidelines.

- `hospital://operating-rooms/rules`

  Operating room rules and policies.

### Prompts

Parameterized prompts are available through MCP prompts.

Available prompt:

- `triage_patient_prompt`

  Purpose:

  - Analyze patient urgency.
  - Use hospital guidelines.
  - Select suitable MCP tools.

### Defensive Tool Design

The MCP server applies multiple protection layers.

**Input Validation**

Implemented using:

- Pydantic models.
- JSON Schema validation.

**Authorization**

Write operations require proper authorization.

Examples:

- Doctors can update patient medical information.
- Admin users can register patients and manage operating room status.

**Business Rules**

The server validates:

- Patient IDs.
- Allowed patient statuses.
- ICU bed assignments.
- Admission data.
- Operating room states.

### Transport

The MCP client communicates with the mock server through:

- stdio

This follows the MCP local development workflow.

---

## Demo

Server capabilities:

```json
{
  "tools": {
    "listChanged": true
  },
  "elicitation": {},
  "sampling": {},
  "progress": {}
}
```

Available Tools:

```
get_patient_details
get_available_icu_beds
get_hospital_capacity
register_patient
update_patient_status
create_admission
manage_icu_bed
update_operating_room_status
```

**USER:**
Register a new patient

**Result:**
Patient registered successfully.

**USER:**
Get patient details

**Result:**
Patient details retrieved successfully.

**USER:**
Check available ICU beds

**Result:**
Available ICU beds returned successfully.

**USER:**
Assign ICU bed to patient

Human confirmation requested:

```
Confirm ICU bed assignment?
```

**Result:**
ICU bed assigned successfully.

**USER:**
Create admission

**Result:**
Admission created successfully.

Sampling request completed when supported.

---

## End-to-End Tests

Run:

```bash
python test_e2e.py
```

Expected output:

```
PASS: Capability Negotiation
PASS: Notifications
PASS: Human Confirmation
PASS: Resources
PASS: Prompts
PASS: Progress Tracking
PASS: Defensive Tool Design
PASS: Sampling

All tests passed
```

---

## Production Considerations

Although this prototype demonstrates MCP protocol concepts, production deployment would require:

- Secure authentication (JWT or similar).
- Complete audit logging for every write operation.
- Support for multiple simultaneous confirmation requests.
- Production-grade model integration.
- Connection with a real hospital database.
- Additional security monitoring.

---

## Session 3 Extension — Memory & RAG Lab

This section documents the memory-and-retrieval extension built on top of
the MCP Server Lab above. It reuses `mcp_server/` and `db/` as-is — nothing
in this section duplicates them.

### The problem we found

Two real gaps showed up once the MCP server above was in daily use:

1. **Nothing persists past a session.** Front-desk staff and nurses re-ask
   or re-read the same patient history every call — most critically
   **allergy history**. A missed or contradicted allergy fact isn't a UX
   annoyance, it's the kind of gap that leads to a wrong prescription.
2. **The agent can't reason over anything outside a tool call.** Doctors
   ask sedation-window, pre-op, and protocol-citation questions that live
   only in an internal clinical policy manual — too much to inject as one
   MCP Resource, and not a good fit for dozens of new single-purpose tools.

### What was built, and where

| Concern (lab rubric) | Points | Folder |
|---|---|---|
| Short-term memory + scratchpad | 5 | [`memory/`](memory/README.md) |
| Promote-or-drop routing (forget / episodic) | 6 | [`memory/`](memory/README.md) |
| Semantic memory consolidation (updates, versioning, expiration, conflict resolution) | 10 | [`memory/`](memory/README.md) |
| Context window management, all 4 strategies + comparison table | 15 | [`context_eval/`](context_eval/README.md) |
| Vector database architecture (ANN index, metadata store + index) | 8 | [`rag/`](rag/README.md) |
| Retrieval architectures — naive, hybrid, agentic + comparison table | 15 | [`rag/`](rag/README.md) + [`retrieval_eval/`](retrieval_eval/README.md) |
| Self-RAG-style verification (RAG **and** memory recall) | 8 | [`rag/self_rag.py`](rag/self_rag.py) |
| Agent/system integration | 10 | [`agent/agent.py`](agent/agent.py) (`handle_message`, `end_session`) |

Each folder's own README documents the real scenario it addresses, maps
every file to the rubric line it satisfies, and shows the actual (not
hand-typed) output of running it.

### Comparison table 1 — Context Window Management (`context_eval/`)

| Strategy | Allergy detail recalled | Avg. input tokens | Avg. output tokens | Avg. latency |
|---|---|---|---|---|
| Sliding window (last 10 turns) | 0/10 | 2814.3 | 0 | 0.0s |
| **Observation masking (keep last 3 tool outputs)** | **10/10** | 1587.7 | 0 | 0.0s |
| Recursive summarization (compact, keep last 6) | 10/10 | 1247.3 | 31 | 0.052s |
| Zone-based pruning (4 zones) | 0/10 | 1272.4 | 9 | 0.031s |

**Shipped: `observation_masking`.** It's the only strategy structurally
immune to *depth*-based data loss (sliding window and zone-based pruning
both use a turn-count cutoff and lose the buried detail; observation
masking keeps every dialogue turn regardless of age and only masks bulky
tool output, which is where MediCore's real bloat actually is). Full
methodology and analysis: [`context_eval/README.md`](context_eval/README.md).

### Comparison table 2 — Retrieval Architecture (`rag/` + `retrieval_eval/`)

| Architecture | Accuracy (12 Qs) | general | citation | multi_hop | Avg tokens/query | Avg latency/query |
|---|---|---|---|---|---|---|
| naive_rag | 11/12 | 4/4 | 4/4 | 3/4 | 566.2 | 0.0009s |
| **hybrid_search** | 11/12 | 4/4 | 4/4 | 3/4 | 569.5 | 0.001s |
| agentic_rag | **12/12** | 4/4 | 4/4 | **4/4** | 533 | 0.001s |

**Shipped: `hybrid_search` as the default**, with `agentic_rag` routed
specifically for multi-part questions (it's the only architecture that
reliably resolves questions needing two different policy sections at once).
Full methodology and analysis: [`retrieval_eval/README.md`](retrieval_eval/README.md).

### Running the full extension

```bash
pip install -r requirements.txt
pip install scikit-learn numpy   # needed by rag/embeddings.py

python -m memory.db                 # create memory tables in db/meridian_hospital.db
python -m memory.demo               # memory scenario: an allergy contradiction, resolved
python -m context_eval.evaluate     # regenerate context_eval/comparison_table.md
python -m rag.build_index           # build the policy-manual vector store
python -m rag.demo                  # all 3 RAG architectures + Self-RAG catching a failure
python -m retrieval_eval.evaluate   # regenerate retrieval_eval/comparison_table.md

python agent/agent.py --demo        # the live agent loop, memory + RAG wired in
```

No `ANTHROPIC_API_KEY` is required for any of the above — every LLM-shaped
step (promote-or-drop routing, summarization, RAG generation, Self-RAG
judging, agentic planning) has a documented, deterministic offline
fallback. Set the key to route those steps through Claude instead.

### How the live agent uses all of this (`agent/agent.py`)

`MediCoreAgent.handle_message()` is the single per-turn entry point added
to the existing MCP client loop:

1. Every message is recorded into a per-session `ShortTermMemory` (with its
   own `scratchpad`, kept physically separate from the transcript).
2. If the message reads as a policy question, it's routed to `rag/`
   (`hybrid_search` by default, `agentic_rag` for multi-part questions),
   and the answer is passed through `self_rag.verified_answer()` before
   being returned — an ungrounded answer is replaced, never shown as-is.
3. Otherwise, the existing keyword-based `decide_next_tool_call()` /
   `tools/call` path (from the MCP Server Lab) runs unchanged.
4. `MediCoreAgent.end_session()` flushes whatever is left in short-term
   memory through `promote_or_drop.route_eviction` (forget/episodic only),
   then runs **one** separate `consolidation.run_once()` pass — semantic
   memory is never written to inline, only through that periodic pass.

### Team split

This extension was divided into three pieces, each owned by one team
member (see each folder's README for the individual rationale):

1. **Memory** — `memory/` (short-term memory + scratchpad, promote-or-drop
   routing, semantic consolidation).
2. **Context evaluation + integration** — `context_eval/` and the
   `agent/agent.py` wiring above.
3. **Retrieval** — `rag/` and `retrieval_eval/` (vector store, three RAG
   architectures, Self-RAG verification).

