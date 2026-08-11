#!/usr/bin/env python3

"""
agent/agent.py
--------------
MediCore Hospital Network -- MCP Client
"""

import argparse
import asyncio
import json
import os
import sys
import time


if sys.platform == "win32":
    asyncio.set_event_loop_policy(
        asyncio.WindowsProactorEventLoopPolicy()
    )


sys.path.insert(0, os.path.dirname(__file__))

from mcp_protocol import JsonRpcEndpoint

# ---------------------------------------------------------------------------
# Memory + RAG integration (Session 3 lab extension).
# memory/ and rag/ are siblings of agent/ and mcp_server/ at the repo root,
# so the repo root needs to be importable too -- this does NOT duplicate
# anything in mcp_server/ or db/, it wires the existing agent loop below to
# the memory and retrieval layers that already reuse db/meridian_hospital.db.
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, REPO_ROOT)

from memory.short_term_memory import ShortTermMemory
from memory.promote_or_drop import route_eviction
from memory.db import init_memory_schema
from memory import consolidation

from rag.build_index import build as build_rag_index
from rag.hybrid_search import build_hybrid_index, hybrid_rag_answer
from rag.agentic_rag import agentic_rag_answer
from rag.self_rag import verified_answer


DEFAULT_SERVER_ARGS = [
    sys.executable,
    "-u",
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "mcp_server",
        "MCP.py"
    ),
]

SERVER_ARGS = (
    os.environ.get("MCP_SERVER_CMD", "").split()
    or DEFAULT_SERVER_ARGS
)
class MediCoreAgent:

    def __init__(self, auto_confirm=False):

        self.proc = None
        self.endpoint = None
        self._reader_task = None
        self.server_capabilities = {}
        self.tools = []

        self.auto_confirm = auto_confirm
        self.scripted_answers = []

        # ---- Memory + RAG integration -----------------------------------
        # Short-term buffer + scratchpad for this session; overflow/close
        # routes through promote_or_drop.route_eviction (forget/episodic
        # only -- semantic memory is only ever built by the separate
        # consolidation.run_once() pass, called in end_session() below).
        init_memory_schema()
        self.session_id = f"session-{int(time.time())}"
        self.stm = ShortTermMemory(max_turns=12, on_evict=route_eviction)

        # RAG index is built lazily on first policy question, not at
        # startup, so `--demo` runs that never ask a policy question don't
        # pay the (small) indexing cost.
        self._rag_store = None
        self._hybrid_index = None


    async def start(self):

        print("PYTHON USED:", sys.executable)

        print("Starting MCP Server...")

        self.proc = await asyncio.create_subprocess_exec(
            *SERVER_ARGS,

            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )


        if self.proc.stdin is None or self.proc.stdout is None:
            raise RuntimeError(
                "MCP server pipes were not created"
            )


        self.endpoint = JsonRpcEndpoint(
            self.proc.stdout,
            self.proc.stdin,

            request_handler=self._handle_server_request,
            notification_handler=self._handle_server_notification,

            name="medicore-client",
        )

        self._stderr_task = asyncio.create_task(
                    self._read_server_errors()
                )

        self._reader_task = asyncio.create_task(
            self.endpoint.run()
        )


        print("Initializing MCP protocol...")


        result = await self.endpoint.send_request(
            "initialize",
            {
                "protocolVersion": "2025-06-18",

                "capabilities": {
                    "elicitation": {},
                    "sampling": {}
                },

                "clientInfo": {
                    "name": "medicore-agent",
                    "version": "0.1.0"
                }
            }
        )


        self.server_capabilities = (
            result.get("capabilities", {})
        )


        await self.endpoint.send_notification(
            "initialized",
            {}
        )
        await self._refresh_tools()


        print("MCP Client Ready")



    async def _read_server_errors(self):

        if self.proc and self.proc.stderr:

            while True:

                line = await self.proc.stderr.readline()

                if not line:
                    break

                print(
                    "[SERVER]",
                    line.decode(errors="ignore").strip()
                )

    async def stop(self):

        if self._reader_task:
            self._reader_task.cancel()
            try:
                await self._reader_task
            except asyncio.CancelledError:
                pass

        if self._stderr_task:
            self._stderr_task.cancel()
            try:
                await self._stderr_task
            except asyncio.CancelledError:
                pass

        if self.proc:

            if self.proc.stdin:
                self.proc.stdin.close()

            self.proc.terminate()

            try:
                await asyncio.wait_for(self.proc.wait(), timeout=3)
            except asyncio.TimeoutError:
                self.proc.kill()
                await self.proc.wait()
            await asyncio.sleep(0.1)
    async def _refresh_tools(self):

            result = await self.endpoint.send_request(
                "tools/list",
                {}
            )

            self.tools = result.get(
                "tools",
                []
            )



    async def _handle_server_request(
            self,
            method,
            params
    ):

        if method == "elicitation/create":

            return {
                "action": "accept",
                "content": {
                    "confirm": True
                }
            }


        if method == "sampling/createMessage":

            return {
                "role": "assistant",
                "content": {
                    "type": "text",
                    "text": "offline response"
                }
            }


        raise Exception(
            f"Unsupported request {method}"
        )



    async def _handle_server_notification(
            self,
            method,
            params
    ):

        print(
            "[NOTIFICATION]",
            method,
            params
        )



    async def call_tool(
            self,
            name,
            arguments
    ):

        return await self.endpoint.send_request(
            "tools/call",
            {
                "name": name,
                "arguments": arguments
            }
        )

    # -----------------------------------------------------------------
    # Memory + RAG integration
    # -----------------------------------------------------------------

    def _ensure_rag_index(self):
        """Build the vector store + hybrid (BM25) index once per process,
        reusing rag/build_index.py -- never rebuilt from scratch here."""
        if self._rag_store is None:
            self._rag_store = build_rag_index()
            self._hybrid_index = build_hybrid_index()
        return self._rag_store, self._hybrid_index

    async def handle_message(self, msg, *, patient_id=None, user_id=None):
        """
        The single entry point run_demo()/run_interactive() now call per
        turn. Decides between three paths:

          1. A clinical/operational policy question -> rag/ (hybrid search
             by default, per retrieval_eval/'s comparison table; agentic
             RAG specifically for questions that look multi-part) ->
             Self-RAG verification before the answer is trusted.
          2. A known database action (ICU beds, patient lookup, admission,
             capacity) -> the EXISTING decide_next_tool_call() + MCP
             tools/call path, unchanged.
          3. Neither -> a plain fallback response.

        Every turn is also recorded into short-term memory, so pruning
        (context_eval/'s strategies) and promote-or-drop routing
        (memory/promote_or_drop.py) both have real data to act on.
        """
        self.stm.add("user", msg, session_id=self.session_id,
                      patient_id=patient_id, user_id=user_id)

        if is_policy_question(msg):
            store, hybrid_index = self._ensure_rag_index()

            if _looks_multi_part(msg):
                answer, chunks, trace = agentic_rag_answer(msg, store)
            else:
                answer, chunks = hybrid_rag_answer(msg, store, hybrid_index)

            verdict = verified_answer(msg, chunks, answer)
            final_text = verdict["final_answer"]

            self.stm.add("assistant", final_text, session_id=self.session_id,
                         patient_id=patient_id, user_id=user_id)

            return {
                "type": "rag_answer",
                "text": final_text,
                "grounded": verdict["issup_passed"],
                "sources": verdict["chunks_kept_after_isrel"],
            }

        call = decide_next_tool_call(msg)
        if call:
            result = await self.call_tool(call["name"], call["arguments"])
            self.stm.add("tool", {"tool": call["name"], "result": result},
                          session_id=self.session_id, patient_id=patient_id,
                          user_id=user_id)
            return {"type": "tool_result", "tool": call["name"], "result": result}

        fallback_text = "Noted, thank you."
        self.stm.add("assistant", fallback_text, session_id=self.session_id,
                      patient_id=patient_id, user_id=user_id)
        return {"type": "no_action", "text": fallback_text}

    async def end_session(self):
        """
        Call at the end of a conversation (see run_demo/run_interactive).
        Flushes whatever is still sitting in short-term memory through
        promote-or-drop (so nothing is lost silently when a session ends),
        then runs ONE consolidation pass over episodic memory. This is
        deliberately a separate, explicit step -- never something that
        happens inline per-message.
        """
        self.stm.flush_all()
        result = consolidation.run_once()
        print("[memory] consolidation run:", json.dumps(result))




def _looks_multi_part(text: str) -> bool:
    """Cheap offline heuristic: route to agentic RAG only when the question
    plausibly needs more than one policy section (matches how
    retrieval_eval/evaluate.py categorized its 'multi_hop' test questions:
    combining two distinct clinical topics in one ask)."""
    t = text.lower()
    topic_hits = sum(topic in t for topic in (
        "cardiac", "penicillin", "allerg", "pediatric", "elective", "discharge",
    ))
    return topic_hits >= 2


_POLICY_SIGNAL_WORDS = (
    "protocol", "sedation", "fasting", "allerg", "discharge", "isolation",
    "transfusion", "fall risk", "fall-risk", "pediatric", "consent",
    "pre-op", "pre operative", "screening", "policy",
)

_QUESTION_STARTERS = ("what", "how", "when", "summarize", "explain", "which", "should")


def is_policy_question(text: str) -> bool:
    """Routes a message to rag/ instead of the MCP tool-call path.

    Deliberately requires the message to actually LOOK like a question
    (contains '?' or opens with a question word), not just contain a
    clinically-flavored keyword -- otherwise a plain statement like
    "Patient reports a known penicillin allergy..." (which should just be
    recorded into short-term memory for promote-or-drop to pick up, per
    memory/) would get misrouted into the RAG path just because it
    contains 'allerg'. Kept as a keyword check (consistent with
    decide_next_tool_call() below) rather than an LLM classifier, so the
    routing itself never needs an ANTHROPIC_API_KEY.
    """
    t = text.lower()
    looks_like_question = "?" in t or t.startswith(_QUESTION_STARTERS)
    return looks_like_question and any(w in t for w in _POLICY_SIGNAL_WORDS)


def decide_next_tool_call(message):

    text = message.lower()


    if "icu beds" in text:

        return {
            "name": "get_available_icu_beds",
            "arguments": {}
        }


    if "patient details" in text:

        return {
            "name": "get_patient_details",
            "arguments": {
                "patient_id": 1
            }
        }


    if "admission" in text:

        return {
            "name": "create_admission",
            "arguments": {
                "admission": {
                    "patient_id": 1,
                    "doctor_id": 1,
                    "room_id": None,
                    "status": "Active"
                }
            }
        }


    if "capacity" in text:

        return {
            "name": "get_hospital_capacity",
            "arguments": {}
        }


    return None

DEMO_SCRIPT = [

    "Which ICU beds are available?",

    "Get patient details",

    "Patient reports a known penicillin allergy, had a rash last time it was prescribed.",

    "What does Protocol 4.2b say about cardiac-risk patients?",

    "For a 70-year-old patient with a cardiac history and a known penicillin allergy needing emergency surgery, what sedation adjustments and antibiotic handling apply?",

    "Create admission",

    "Get hospital capacity"

]



async def run_demo():

    agent = MediCoreAgent(
        auto_confirm=True
    )


    await agent.start()


    print(
        "\nCapabilities:"
    )

    print(
        json.dumps(
            agent.server_capabilities,
            indent=2
        )
    )


    print(
        "\nTools:"
    )

    print(
        [
            t["name"]
            for t in agent.tools
        ]
    )


    for msg in DEMO_SCRIPT:

        print(
            "\nUSER:",
            msg
        )

        result = await agent.handle_message(msg, patient_id=1, user_id=1)

        if result["type"] == "tool_result":
            print("Calling:", result["tool"])
            print("RESULT:", result["result"])
        elif result["type"] == "rag_answer":
            print(f"AGENT (grounded={result['grounded']}, sources={result['sources']}):")
            print(result["text"])
        else:
            print("AGENT:", result["text"])

    print("\n[memory] ending session -- flushing STM through promote-or-drop, "
          "then running one consolidation pass over episodic memory")
    await agent.end_session()

    await agent.stop()




async def run_interactive():

    agent = MediCoreAgent()

    await agent.start()


    while True:

        msg = input(
            "\nyou> "
        )


        if msg == "quit":
            break

        result = await agent.handle_message(msg, patient_id=1, user_id=1)

        if result["type"] == "tool_result":
            print("Calling:", result["tool"])
            print(result["result"])
        elif result["type"] == "rag_answer":
            print(f"[grounded={result['grounded']}, sources={result['sources']}]")
            print(result["text"])
        else:
            print(result["text"])


    await agent.end_session()
    await agent.stop()



if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--demo",
        action="store_true"
    )

    args = parser.parse_args()


    asyncio.run(
        run_demo()
        if args.demo
        else run_interactive()
    )
