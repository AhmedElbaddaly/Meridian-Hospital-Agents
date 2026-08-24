"""
GraphRuntime — executes a state graph with:
  - checkpoint after every meaningful transition
  - HITL pause / resume from admin decision
  - failure tickets on unplanned errors
  - crash-and-resume from last durable checkpoint
"""

from __future__ import annotations

import copy
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from .checkpoint import Checkpoint, CheckpointStore
from .hitl import open_hitl_task, get_hitl_for_run, resolve_hitl_task
from .tickets import open_failure_ticket, get_ticket_for_run


NodeFn = Callable[["GraphState"], "GraphState"]


@dataclass
class GraphState:
    """Mutable state carried through the graph. Always checkpointed as a dict."""
    run_id: str
    graph_name: str
    data: Dict[str, Any] = field(default_factory=dict)
    current_node: str = "START"
    status: str = "running"  # running | waiting_hitl | failed | completed
    history: List[str] = field(default_factory=list)
    hitl_task_id: Optional[str] = None
    ticket_id: Optional[str] = None
    error: Optional[str] = None

    def snapshot(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "graph_name": self.graph_name,
            "data": copy.deepcopy(self.data),
            "current_node": self.current_node,
            "status": self.status,
            "history": list(self.history),
            "hitl_task_id": self.hitl_task_id,
            "ticket_id": self.ticket_id,
            "error": self.error,
        }

    @classmethod
    def from_snapshot(cls, snap: Dict[str, Any]) -> "GraphState":
        return cls(
            run_id=snap["run_id"],
            graph_name=snap["graph_name"],
            data=snap.get("data") or {},
            current_node=snap.get("current_node", "START"),
            status=snap.get("status", "running"),
            history=list(snap.get("history") or []),
            hitl_task_id=snap.get("hitl_task_id"),
            ticket_id=snap.get("ticket_id"),
            error=snap.get("error"),
        )


@dataclass
class GraphDef:
    name: str
    nodes: Dict[str, NodeFn]
    edges: Dict[str, Callable[[GraphState], str]]  # node -> next_node selector
    entry: str = "START"
    terminal: Tuple[str, ...] = ("END", "FAILED")


class GraphRuntime:
    def __init__(self, graph: GraphDef, store: Optional[CheckpointStore] = None):
        self.graph = graph
        self.store = store or CheckpointStore()

    def _checkpoint(self, state: GraphState) -> Checkpoint:
        cp = Checkpoint(
            run_id=state.run_id,
            graph_name=state.graph_name,
            node=state.current_node,
            state=state.snapshot(),
            status=state.status,
        )
        return self.store.save(cp)

    def start(self, initial_data: Optional[Dict[str, Any]] = None, run_id: Optional[str] = None) -> GraphState:
        state = GraphState(
            run_id=run_id or str(uuid.uuid4()),
            graph_name=self.graph.name,
            data=dict(initial_data or {}),
            current_node=self.graph.entry,
            status="running",
        )
        state.history.append(self.graph.entry)
        self._checkpoint(state)
        return state

    def resume(
    self,
    run_id: str,
    data_patch: Optional[Dict[str, Any]] = None
    ) -> GraphState:
        cp = self.store.latest(run_id)
        if not cp:
            raise ValueError(f"No checkpoint for run_id={run_id}")
        state = GraphState.from_snapshot(cp.state)
        # Apply admin correction before resuming
        if data_patch:
            state.data.update(data_patch)
        # If we were waiting on HITL, check whether admin has resolved it
        if state.status == "waiting_hitl":
            task = get_hitl_for_run(run_id)
            if task and task.status in ("approved", "rejected", "modified") and task.admin_decision is not None:
                state.data["hitl_decision"] = task.admin_decision
                state.data["hitl_status"] = task.status
                state.status = "running"
                # Re-enter the same HITL node so it can apply the admin decision;
                # the normal edge selector advances on the subsequent step().
                self._checkpoint(state)
        elif state.status == "failed":
            ticket = get_ticket_for_run(run_id)
            if ticket and ticket.status == "resolved":
                state.status = "running"
                state.error = None
                # Stay on the failed node so it can be retried, or advance if resolution says so
                if state.data.get("resume_next"):
                    state.current_node = state.data["resume_next"]
                    state.history.append(state.current_node)
                self._checkpoint(state)
        return state

    def step(self, state: GraphState) -> GraphState:
        """Execute exactly one node, then checkpoint. Returns updated state."""
        if state.status in ("waiting_hitl", "failed", "completed"):
            return state
        if state.current_node in self.graph.terminal:
            state.status = "completed" if state.current_node == "END" else "failed"
            self._checkpoint(state)
            return state

        node_fn = self.graph.nodes.get(state.current_node)
        if not node_fn:
            state.status = "failed"
            state.error = f"Unknown node: {state.current_node}"
            self._checkpoint(state)
            return state

        try:
            state = node_fn(state)
        except _HITLInterrupt as hitl:
            # Expected pause — open HITL task, checkpoint as waiting_hitl
            task = open_hitl_task(
                run_id=state.run_id,
                graph_name=state.graph_name,
                node=state.current_node,
                reason=hitl.reason,
                state_snapshot=state.snapshot(),
            )
            state.status = "waiting_hitl"
            state.hitl_task_id = task.task_id
            state.data["hitl_reason"] = hitl.reason
            self._checkpoint(state)
            return state
        except Exception as exc:
            # Unplanned failure — open ticket, checkpoint as failed
            ticket = open_failure_ticket(
                run_id=state.run_id,
                graph_name=state.graph_name,
                node=state.current_node,
                error_type=type(exc).__name__,
                error_message=str(exc),
                state_snapshot=state.snapshot(),
            )
            state.status = "failed"
            state.ticket_id = ticket.ticket_id
            state.error = str(exc)
            self._checkpoint(state)
            return state

        # Successful node completion → choose next edge and checkpoint
        if state.status == "waiting_hitl":
            # Node itself may have set waiting_hitl without raising
            self._checkpoint(state)
            return state

        next_selector = self.graph.edges.get(state.current_node)
        if next_selector:
            next_node = next_selector(state)
        else:
            next_node = "END"
        state.current_node = next_node
        state.history.append(next_node)
        if next_node in self.graph.terminal:
            state.status = "completed" if next_node == "END" else "failed"
        self._checkpoint(state)
        return state

    def run_until_pause(self, state: GraphState, max_steps: int = 50) -> GraphState:
        """Advance until HITL, failure, completion, or max_steps."""
        steps = 0
        while (
            state.status == "running"
            and state.current_node not in self.graph.terminal
            and steps < max_steps
        ):
            state = self.step(state)
            steps += 1
        return state


class _HITLInterrupt(Exception):
    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)


def request_hitl(reason: str) -> None:
    """Call from inside a node to trigger a genuine HITL pause."""
    raise _HITLInterrupt(reason)
