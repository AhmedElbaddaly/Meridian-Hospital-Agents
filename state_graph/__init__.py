"""
state_graph/ — Persistent, recoverable state-graph agents for Meridian Hospital.

Three genuinely stateful problems (distinct from memory/RAG and any prior
scheduling/decomposition agents):

1. post_op_recovery     — Multi-day post-operative recovery coordination
2. insurance_auth       — Elective-procedure insurance pre-authorization & appeal
3. ed_surge_triage      — Emergency-department surge triage & bed escalation

Each graph:
  - Has real cycles / waiting states / external branches
  - Persists checkpoints after every meaningful transition (SQLite)
  - Supports genuine HITL pause → admin action on platform → resume
  - Opens failure tickets (distinct from HITL) that resume from checkpoint
  - Embeds two of: task decomposition, ToT/LATS, constrained ReAct, RAG

Crash-and-resume is proven by writing durable state after each transition;
killing the process mid-run and restarting loads the last checkpoint.
"""

from .checkpoint import CheckpointStore
from .hitl import HITLTask, open_hitl_task, resolve_hitl_task
from .tickets import FailureTicket, open_failure_ticket, resolve_failure_ticket
from .runtime import GraphRuntime

from .graphs.post_op_recovery import build_post_op_recovery_graph
from .graphs.insurance_auth import build_insurance_auth_graph
from .graphs.ed_surge_triage import build_ed_surge_triage_graph

__all__ = [
    "CheckpointStore",
    "HITLTask",
    "open_hitl_task",
    "resolve_hitl_task",
    "FailureTicket",
    "open_failure_ticket",
    "resolve_failure_ticket",
    "GraphRuntime",
    "build_post_op_recovery_graph",
    "build_insurance_auth_graph",
    "build_ed_surge_triage_graph",
]
