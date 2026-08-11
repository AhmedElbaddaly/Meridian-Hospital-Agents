"""
memory/short_term_memory.py
----------------------------
Rolling short-term (working) memory buffer for the MediCore agent, PLUS a
scratchpad that is kept as a physically separate structure.

Why separate from the transcript at all?
Front-desk / triage conversations run long: a nurse describes a patient's
symptoms over many turns while the agent pulls admission history, ICU
availability, prior notes, etc. Each tool call is a large JSON blob. If the
agent's "current plan" (e.g. "step 3 of 4: confirm ICU bed before creating
admission") lived only inside the message list, any pruning strategy that
truncates or summarizes the transcript risks silently destroying the plan
mid-task. Keeping the scratchpad as its own object means every context
management strategy in context_eval/ can freely prune `messages` without
ever touching `scratchpad`.

When the buffer overflows (max_turns exceeded), the OLDEST item is evicted
and handed to the promote-or-drop router (memory/promote_or_drop.py) instead
of being silently discarded.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


def _now() -> str:
    return datetime.datetime.utcnow().isoformat()


@dataclass
class MemoryItem:
    role: str            # 'user' | 'assistant' | 'tool'
    content: Any
    timestamp: str = field(default_factory=_now)
    session_id: Optional[str] = None
    patient_id: Optional[int] = None
    user_id: Optional[int] = None


class Scratchpad:
    """
    The agent's current working state. NOT part of the message transcript,
    so it survives sliding-window truncation, summarization, masking, or
    zone-based pruning of `messages` untouched.
    """

    def __init__(self):
        self.plan: Optional[str] = None
        self.current_subgoal: Optional[str] = None
        self.pending_confirmations: list[str] = []
        self.variables: dict[str, Any] = {}

    def update(self, *, plan: Optional[str] = None,
               subgoal: Optional[str] = None,
               **variables: Any) -> None:
        if plan is not None:
            self.plan = plan
        if subgoal is not None:
            self.current_subgoal = subgoal
        self.variables.update(variables)

    def snapshot(self) -> dict:
        return {
            "plan": self.plan,
            "current_subgoal": self.current_subgoal,
            "pending_confirmations": list(self.pending_confirmations),
            "variables": dict(self.variables),
        }

    def clear(self) -> None:
        self.plan = None
        self.current_subgoal = None
        self.pending_confirmations = []
        self.variables = {}


class ShortTermMemory:
    """
    Rolling buffer of MemoryItem. On overflow, calls `on_evict(item)` --
    normally `memory.promote_or_drop.route_eviction` -- instead of dropping
    the item on the floor.
    """

    def __init__(self, max_turns: int = 20,
                 on_evict: Optional[Callable[[MemoryItem], None]] = None):
        self.max_turns = max_turns
        self.messages: list[MemoryItem] = []
        self.scratchpad = Scratchpad()
        self.on_evict = on_evict

    def add(self, role: str, content: Any, *,
            session_id: Optional[str] = None,
            patient_id: Optional[int] = None,
            user_id: Optional[int] = None) -> None:
        item = MemoryItem(
            role=role, content=content,
            session_id=session_id, patient_id=patient_id, user_id=user_id,
        )
        self.messages.append(item)
        self._enforce_budget()

    def _enforce_budget(self) -> None:
        while len(self.messages) > self.max_turns:
            oldest = self.messages.pop(0)
            if self.on_evict is not None:
                self.on_evict(oldest)

    def get_context(self) -> list[dict]:
        """What actually gets sent to the LLM: transcript + scratchpad header."""
        return [
            {"role": m.role, "content": m.content}
            for m in self.messages
        ]

    def get_scratchpad_header(self) -> str:
        sp = self.scratchpad.snapshot()
        return (
            f"Current plan: {sp['plan']}\n"
            f"Sub-goal: {sp['current_subgoal']}\n"
            f"Pending confirmations: {sp['pending_confirmations']}"
        )

    def flush_all(self) -> None:
        """Force-evict everything still in the buffer (e.g. on session end)."""
        while self.messages:
            oldest = self.messages.pop(0)
            if self.on_evict is not None:
                self.on_evict(oldest)
