"""
rag/generation.py
-------------------
Shared "generate the final answer from retrieved context" step, used
identically by naive_rag.py, hybrid_search.py, and agentic_rag.py so the
only thing being compared between architectures is retrieval quality, not
three different prompting styles.

Answers MUST be grounded only in retrieved content (per the lab's
guardrails) -- the prompt below says so explicitly, and the offline
fallback physically cannot do otherwise since it only ever echoes retrieved
chunk text back, never invents anything.
"""

from __future__ import annotations

# --- allow running this file directly (python path/to/file.py), not
# --- just as a module (python -m pkg.file) -- both now work the same.
import os as _os
import sys as _sys
_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _REPO_ROOT not in _sys.path:
    _sys.path.insert(0, _REPO_ROOT)


import os

from rag.vector_store import RetrievedChunk

GEN_PROMPT = """Answer the question using ONLY the context below. If the \
context does not contain the answer, say so explicitly -- do not guess.

Context:
{context}

Question: {question}
"""


def _offline_generate(question: str, chunks: list[RetrievedChunk]) -> str:
    """Deliberately conservative: just surfaces the retrieved chunk text
    with its citation, rather than paraphrasing -- guarantees the answer
    is grounded (it IS the retrieved text) at the cost of fluency. This is
    the fallback used when no ANTHROPIC_API_KEY is configured."""
    if not chunks:
        return "No relevant policy found in the retrieved context."
    lines = [f"[Protocol {c.protocol_id}] {c.text}" for c in chunks]
    return "Based on retrieved policy:\n" + "\n".join(lines)


def _llm_generate(question: str, chunks: list[RetrievedChunk]) -> str:
    import anthropic

    client = anthropic.Anthropic()
    context = "\n\n".join(f"[Protocol {c.protocol_id}] {c.text}" for c in chunks)
    prompt = GEN_PROMPT.format(context=context, question=question)
    response = client.messages.create(
        model="claude-sonnet-4-6", max_tokens=400,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(b.text for b in response.content if getattr(b, "type", None) == "text")


def generate_answer(question: str, chunks: list[RetrievedChunk]) -> str:
    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            return _llm_generate(question, chunks)
        except Exception:
            pass
    return _offline_generate(question, chunks)
