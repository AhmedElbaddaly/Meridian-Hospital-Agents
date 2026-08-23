"""
platform/backend/main.py

Thin FastAPI wrapper -- Person 3 (Platform lead). Routes to EXISTING code,
no new agent/business logic duplicated here:
  - agent.agent.MediCoreAgent          -> memory_rag chat
  - mcp_server.tool_registry           -> admin tool enable/disable
  - rag.vector_store.VectorStore       -> admin RAG document add/remove

state-graph-backed routes (planning, post_op_recovery, insurance_auth,
ed_surge_triage, HITL queue, ticket queue) are stubbed with a clear 501 +
explanation until Person1/Person2's state_graph/ and mcp_bridge.py changes
are pushed -- NOT faked with placeholder data.

Run from repo root:
    pip install fastapi uvicorn
    uvicorn platform.backend.main:app --reload
Then open http://127.0.0.1:8000/docs for interactive API testing.
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
from contextlib import asynccontextmanager
from typing import Optional

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
if _MCP_SERVER_DIR not in sys.path:
    sys.path.insert(0, _MCP_SERVER_DIR)

from agent.agent import MediCoreAgent            # noqa: E402
import tool_registry                              # noqa: E402  (mcp_server/tool_registry.py)
from rag.vector_store import VectorStore, VECTOR_DB_PATH  # noqa: E402
from rag.build_index import build as build_rag_index       # noqa: E402
from rag.chunking import chunk_document, Chunk    # noqa: E402
from rag.corpus import PolicyDoc                  # noqa: E402
from rag.embeddings import get_embedder           # noqa: E402


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

_STATE_GRAPH_AGENTS = {"post_op_recovery", "insurance_auth", "ed_surge_triage"}


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

    if req.agent in _STATE_GRAPH_AGENTS:
        raise HTTPException(
            status_code=501,
            detail=f"'{req.agent}' pending: state_graph/ and mcp_bridge.py wiring "
                   f"not yet pushed to this branch by Person1/Person2.",
        )

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


# =======================================================================
# Admin: HITL + tickets -- STUBBED, pending state_graph/ push
# =======================================================================

@app.get("/api/admin/hitl")
def list_hitl():
    raise HTTPException(
        status_code=501,
        detail="Pending: state_graph/hitl.py not yet available on this branch.",
    )


@app.get("/api/admin/tickets")
def list_tickets():
    raise HTTPException(
        status_code=501,
        detail="Pending: state_graph/tickets.py not yet available on this branch.",
    )

from fastapi.staticfiles import StaticFiles

_FRONTEND_DIR = os.path.join(_REPO_ROOT, "webapp", "frontend")
app.mount("/", StaticFiles(directory=_FRONTEND_DIR, html=True), name="frontend")
