"""
planning/llm_client.py
-----------------------
Single shared model-provider wrapper for the whole planning/ package.

Why this file exists (team rule): the reference toolkit
(github.com/AmrSheta22/task_decomposition_and_planning) defaults to
ChatMistralAI through LangChain. The lab requires swapping that for
"whatever you're already using elsewhere in your repo" -- this codebase
already uses Claude for the Memory/RAG agent (rag/generation.py,
memory/consolidation.py), with a documented deterministic offline
fallback when ANTHROPIC_API_KEY is unset. Every algorithm module
(decomposition, dynamic_decomposition, plan_and_solve, tree_of_thoughts,
lats, self_refine, reflexion) imports PlanningLLM from here instead of
each writing its own provider glue -- that duplication was flagged
explicitly as a risk when the team split up the work.

Two things every algorithm module needs from this wrapper, beyond just
"call the model":
1. call_count / total_tokens bookkeeping, because the lab's cost/quality
   comparison table needs real numbers, not estimates written by hand.
2. A deterministic offline mode so `planning_eval` is fully reproducible
   and gradeable without an API key -- same convention this repo already
   uses for memory/RAG (see agent/agent.py).
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable

try:
    import anthropic  # type: ignore
except ImportError:  # pragma: no cover - anthropic is optional at runtime
    anthropic = None


@dataclass
class LLMResult:
    text: str
    input_tokens: int
    output_tokens: int
    latency_s: float


class PlanningLLM:
    """Thin wrapper around Claude, with a scripted deterministic fallback.

    `structured(system, user, schema_hint, fallback_fn)` is the single
    entry point every planning/ module uses:
      - if ANTHROPIC_API_KEY is set, calls Claude and asks for JSON back
      - otherwise calls `fallback_fn(user)` -- a small, scenario-aware
        Python function that returns the same shape a real model would.
        This is NOT a random stub; each fallback encodes the actual
        hospital domain rule being tested (see decomposition.py /
        dynamic_decomposition.py), which is what makes the offline
        divergence demo meaningful instead of theater.
    """

    def __init__(self, model: str = "claude-sonnet-4-6") -> None:
        self.model = model
        self.api_key = os.environ.get("ANTHROPIC_API_KEY")
        self.online = bool(self.api_key and anthropic is not None)
        self._client = anthropic.Anthropic(api_key=self.api_key) if self.online else None

        # Bookkeeping used by planning_eval/evaluate.py for the
        # cost/quality comparison table.
        self.call_count = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.call_log: list[dict[str, Any]] = []

    def reset_counters(self) -> None:
        self.call_count = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.call_log = []

    def _record(self, label: str, result: LLMResult) -> None:
        self.call_count += 1
        self.total_input_tokens += result.input_tokens
        self.total_output_tokens += result.output_tokens
        self.call_log.append(
            {
                "label": label,
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "latency_s": round(result.latency_s, 4),
                "mode": "online" if self.online else "offline",
            }
        )

    def raw(self, system: str, user: str, label: str = "call") -> LLMResult:
        start = time.time()
        if self.online:
            resp = self._client.messages.create(
                model=self.model,
                max_tokens=1000,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
            result = LLMResult(
                text=text,
                input_tokens=resp.usage.input_tokens,
                output_tokens=resp.usage.output_tokens,
                latency_s=time.time() - start,
            )
        else:
            raise RuntimeError("raw() online-only helper called in offline mode")
        self._record(label, result)
        return result

    def structured(
        self,
        system: str,
        user: str,
        fallback_fn: Callable[[str], dict],
        label: str = "structured_call",
    ) -> dict:
        """Return a parsed dict, either from Claude (asked to emit JSON
        only) or from the deterministic domain fallback."""
        start = time.time()
        if self.online:
            resp = self._client.messages.create(
                model=self.model,
                max_tokens=1000,
                system=system + "\nRespond with ONLY valid JSON, no prose, no markdown fences.",
                messages=[{"role": "user", "content": user}],
            )
            text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
            result = LLMResult(
                text=text,
                input_tokens=resp.usage.input_tokens,
                output_tokens=resp.usage.output_tokens,
                latency_s=time.time() - start,
            )
            self._record(label, result)
            cleaned = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            return json.loads(cleaned)
        else:
            payload = fallback_fn(user)
            # approximate token cost for the comparison table even offline,
            # so the metrics stay comparable in shape to an online run.
            approx_in = max(1, len(system.split()) + len(user.split()))
            approx_out = max(1, len(json.dumps(payload).split()))
            result = LLMResult(
                text=json.dumps(payload),
                input_tokens=approx_in,
                output_tokens=approx_out,
                latency_s=time.time() - start,
            )
            self._record(label, result)
            return payload
