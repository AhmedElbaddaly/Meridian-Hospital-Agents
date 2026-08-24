"""
platform/backend/main.py

Thin FastAPI wrapper -- Person 3 (Platform lead). Routes to EXISTING code,
no new agent/business logic duplicated here:
  - agent.agent.MediCoreAgent          -> memory_rag chat
  - mcp_server.tool_registry           -> admin tool enable/disable
  - rag.vector_store.VectorStore       -> admin RAG document add/remove
  - state_graph/                       -> post_op_recovery, insurance_auth,
                                           ed_surge_triage chat + HITL/ticket admin

Run from repo root:
    pip install fastapi uvicorn
    uvicorn webapp.backend.main:app --reload
Then open http://127.0.0.1:8000/docs for interactive API testing.
"""
from __future__ import annotations

import sys
import asyncio
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

import json
import os
import sqlite3
from contextlib import asynccontextmanager
from typing import Optional
from dataclasses import asdict
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ---------------------------------------------------------------------
# Path setup so this file can import repo-root packages regardless of
# where uvicorn is launched from.
# ---------------------------------------------------------------------
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
_MCP_SERVER_DIR = os.path.join(_REPO_ROOT, "mcp_server")
_GRAPH_STATE_DB = os.path.join(_REPO_ROOT, "db", "graph_state.db")
if _MCP_SERVER_DIR not in sys.path:
    sys.path.insert(0, _MCP_SERVER_DIR)

from agent.agent import MediCoreAgent            # noqa: E402
import tool_registry                              # noqa: E402  (mcp_server/tool_registry.py)
from rag.vector_store import VectorStore, VECTOR_DB_PATH  # noqa: E402
from rag.build_index import build as build_rag_index       # noqa: E402
from rag.chunking import chunk_document, Chunk    # noqa: E402
from rag.corpus import PolicyDoc                  # noqa: E402
from rag.embeddings import get_embedder           # noqa: E402
from state_graph.hitl import list_pending_hitl, resolve_hitl_task              # noqa: E402
from state_graph.tickets import list_open_tickets, resolve_failure_ticket      # noqa: E402
from state_graph.checkpoint import CheckpointStore                             # noqa: E402
from state_graph.runtime import GraphRuntime, GraphDef                         # noqa: E402
from state_graph.graphs.insurance_auth import build_insurance_auth_graph       # noqa: E402
from state_graph.graphs.ed_surge_triage import build_ed_surge_triage_graph     # noqa: E402
from state_graph.graphs.post_op_recovery import build_post_op_recovery_graph   # noqa: E402


_GRAPH_BUILDERS = {
    "insurance_auth": build_insurance_auth_graph,
    "ed_surge_triage": build_ed_surge_triage_graph,
    "post_op_recovery": build_post_op_recovery_graph,
}


def _get_runtime_for_graph(graph_name: str, store: CheckpointStore) -> GraphRuntime:
    builder = _GRAPH_BUILDERS.get(graph_name)
    if not builder:
        raise ValueError(f"Unknown graph_name: {graph_name}")
    return GraphRuntime(builder(), store=store)

# ---------------------------------------------------------------------
# App-wide state (single process, single agent instance -- fine for a
# course-scale platform; documented here rather than hidden)
# ---------------------------------------------------------------------
class AppState:
    memory_agent: Optional[MediCoreAgent] = None
    rag_store: Optional[VectorStore] = None


state = AppState()


def _rebuild_rag_index_from_disk() -> VectorStore:
    """Every server restart loses the in-memory HNSW graph (it is never
    persisted, only rag_chunks' SQL rows + their vector_json are). Rather
    than re-embedding everything from scratch (slow, and would refit the
    embedder differently), reload each already-stored vector straight back
    into a fresh index. First-ever run (empty table) falls back to the
    real ingestion pipeline in rag/build_index.py instead."""
    store = VectorStore()

    con = sqlite3.connect(VECTOR_DB_PATH)
    con.row_factory = sqlite3.Row
    rows = con.execute("SELECT * FROM rag_chunks").fetchall()
    con.close()

    if not rows:
        print("[platform] rag_chunks empty -- running first-time ingestion (rag/build_index.py)")
        return build_rag_index(reset=True)

    print(f"[platform] Rebuilding in-memory HNSW index from {len(rows)} persisted chunks")
    import numpy as np
    for r in rows:
        vec = np.array(json.loads(r["vector_json"]))
        store.index.add(r["chunk_id"], vec)

    # The offline embedder still needs to be fit once per process so NEW
    # documents added later can be embedded -- fit it on the corpus texts
    # already on disk (their vocabulary), not the hardcoded base corpus,
    # so it reflects anything already added.
    get_embedder(texts_to_fit_on=[r["text"] for r in rows])
    return store


@asynccontextmanager
async def lifespan(app: FastAPI):
    state.memory_agent = MediCoreAgent(auto_confirm=True)
    await state.memory_agent.start()

    tool_registry.ensure_schema()
    state.rag_store = _rebuild_rag_index_from_disk()

    yield

    if state.memory_agent:
        await state.memory_agent.stop()


app = FastAPI(title="Meridian Hospital Platform API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # course-scope platform; tighten if this ever leaves localhost
    allow_methods=["*"],
    allow_headers=["*"],
)


# =======================================================================
# User-facing: agent list + chat
# =======================================================================

AGENTS = [
    {"id": "memory_rag", "name": "Front Desk & Policy Assistant",
     "description": "Patient lookups, admissions, ICU capacity, and clinical/operational policy questions."},
    {"id": "planning", "name": "Surge Planning Assistant",
     "description": "Decomposes and plans multi-step scheduling/reshuffling requests."},
    {"id": "post_op_recovery", "name": "Post-Op Recovery Coordinator",
     "description": "Multi-day recovery tracking with lab-result waits and physician sign-off."},
    {"id": "insurance_auth", "name": "Insurance Authorization Agent",
     "description": "Pre-auth submission and appeal strategy for insurer claims."},
    {"id": "ed_surge_triage", "name": "ED Surge Triage Agent",
     "description": "Triage ordering and bed/OR assignment during an ED surge, admin-approved for irreversible actions."},
]



@app.get("/api/agents")
def list_agents():
    return {"agents": AGENTS}


class ChatRequest(BaseModel):
    agent: str
    session_id: str
    message: str
    run_id: Optional[str] = None


@app.post("/api/chat")
async def chat(req: ChatRequest):
    if req.agent == "memory_rag":
        result = await state.memory_agent.handle_message(
            req.message, user_id=req.session_id
        )
        return {
            "reply": result.get("text") or json.dumps(result.get("result", result)),
            "run_id": None,
            "status": "ok",
            "hitl_task_id": None,
            "ticket_id": None,
        }

    if req.agent == "planning":
        raise HTTPException(
            status_code=501,
            detail="planning chat route pending: no planning_agent.py integration "
                   "entrypoint exists in the repo yet (see team discussion).",
        )

    if req.agent in _GRAPH_BUILDERS:
        store = CheckpointStore(db_path=_GRAPH_STATE_DB)
        try:
            runtime = _get_runtime_for_graph(req.agent, store)
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))

        if req.run_id:
            # Continuing an existing run -- picks up an admin's HITL/ticket
            # resolution automatically via runtime.resume()'s own logic.
            graph_state = runtime.resume(req.run_id)
        else:
            # New run. These graphs need structured fields (patient_id, cpt_code,
            # patients list, etc.), not free chat text -- so for a new run the
            # message body must be a JSON object with those fields.
            try:
                initial_data = json.loads(req.message)
                if not isinstance(initial_data, dict):
                    raise ValueError
            except (json.JSONDecodeError, ValueError):
                raise HTTPException(
                    status_code=422,
                    detail=(
                        f"'{req.agent}' requires structured JSON input in the message "
                        f"field to start a new run (e.g. patient_id, procedure), not "
                        f"free-text chat. Pass run_id to continue an existing run."
                    ),
                )
            graph_state = runtime.start(initial_data=initial_data)

        graph_state = runtime.run_until_pause(graph_state)

        if graph_state.status == "waiting_hitl":
            reply = f"Paused for human review: {graph_state.data.get('hitl_reason', 'awaiting admin decision')}"
        elif graph_state.status == "failed":
            reply = f"Run failed and a ticket was opened: {graph_state.error}"
        elif graph_state.status == "completed":
            reply = f"Run completed: {json.dumps(graph_state.data, default=str)}"
        else:
            reply = f"Run in progress, currently at node: {graph_state.current_node}"

        return {
            "reply": reply,
            "run_id": graph_state.run_id,
            "status": graph_state.status,
            "hitl_task_id": graph_state.hitl_task_id,
            "ticket_id": graph_state.ticket_id,
        }

    raise HTTPException(status_code=404, detail=f"Unknown agent '{req.agent}'")

# =======================================================================
# Admin: MCP tool registry (fully working today)
# =======================================================================

@app.get("/api/admin/tools")
def get_tools(agent: Optional[str] = None):
    statuses = tool_registry.list_tools(agent=agent)
    by_agent: dict[str, list] = {}
    for s in statuses:
        by_agent.setdefault(s.agent, []).append({"name": s.tool_name, "enabled": s.enabled})
    return {"agents": [{"agent": a, "tools": tools} for a, tools in by_agent.items()]}


@app.post("/api/admin/tools/{agent}/{tool_name}/enable")
def enable_tool(agent: str, tool_name: str):
    tool_registry.set_enabled(agent, tool_name, True, updated_by="admin")
    return {"agent": agent, "tool": tool_name, "enabled": True}


@app.post("/api/admin/tools/{agent}/{tool_name}/disable")
def disable_tool(agent: str, tool_name: str):
    tool_registry.set_enabled(agent, tool_name, False, updated_by="admin")
    return {"agent": agent, "tool": tool_name, "enabled": False}


# =======================================================================
# Admin: RAG documents (fully working today)
# =======================================================================

@app.get("/api/admin/rag/documents")
def list_documents():
    con = sqlite3.connect(VECTOR_DB_PATH)
    con.row_factory = sqlite3.Row
    rows = con.execute(
        """SELECT doc_id, title, department, last_reviewed, protocol_id,
                  COUNT(*) as chunk_count
           FROM rag_chunks GROUP BY doc_id"""
    ).fetchall()
    con.close()
    return {"documents": [dict(r) for r in rows]}


class AddDocumentRequest(BaseModel):
    doc_id: str
    title: str
    department: str
    protocol_id: str
    last_reviewed: str
    text: str


@app.post("/api/admin/rag/documents")
def add_document(req: AddDocumentRequest):
    doc = PolicyDoc(
        doc_id=req.doc_id, protocol_id=req.protocol_id, department=req.department,
        last_reviewed=req.last_reviewed, title=req.title, text=req.text,
    )
    chunks: list[Chunk] = chunk_document(doc)
    embedder = get_embedder()  # already fit at startup; NOT refit here
    for c in chunks:
        vector = embedder.embed(c.text)
        state.rag_store.upsert(
            c.chunk_id, c.text, vector, doc_id=c.doc_id, protocol_id=c.protocol_id,
            department=c.department, last_reviewed=c.last_reviewed, title=c.title,
        )
    return {"doc_id": req.doc_id, "chunks_added": len(chunks)}


@app.delete("/api/admin/rag/documents/{doc_id}")
def delete_document(doc_id: str):
    removed = state.rag_store.delete_by_doc_id(doc_id)
    if removed == 0:
        raise HTTPException(status_code=404, detail=f"No chunks found for doc_id={doc_id}")
    return {"doc_id": doc_id, "chunks_removed": removed}


# Admin: HITL + tickets -- fully wired to state_graph/
# =======================================================================
class HITLResolveRequest(BaseModel):
    decision: dict
    status: str = "approved"


class TicketResolveRequest(BaseModel):
    resolution_note: str
    status: str = "resolved"


@app.get("/api/admin/hitl")
def list_hitl():
    tasks = list_pending_hitl()
    return {"tasks": [asdict(t) if hasattr(t, "__dataclass_fields__") else t for t in tasks]}


@app.post("/api/admin/hitl/{task_id}/resolve")
def resolve_hitl(task_id: str, req: HITLResolveRequest):
    try:
        task = resolve_hitl_task(task_id, decision=req.decision, status=req.status)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Could not resolve HITL task: {e}")
    return {"task_id": task_id, "status": req.status, "resolved": True}


@app.get("/api/admin/tickets")
def list_tickets():
    tickets = list_open_tickets()
    return {"tickets": [asdict(t) if hasattr(t, "__dataclass_fields__") else t for t in tickets]}


@app.post("/api/admin/tickets/{ticket_id}/resolve")
def resolve_ticket(ticket_id: str, req: TicketResolveRequest):
    try:
        ticket = resolve_failure_ticket(ticket_id, resolution_note=req.resolution_note, status=req.status)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Could not resolve ticket: {e}")
    return {"ticket_id": ticket_id, "status": req.status, "resolved": True}
from fastapi.staticfiles import StaticFiles

_FRONTEND_DIR = os.path.join(_REPO_ROOT, "webapp", "frontend")
app.mount("/", StaticFiles(directory=_FRONTEND_DIR, html=True), name="frontend")
