"""
rag/self_rag.py
------------------
Explicit, post-hoc verification -- NOT trust in whatever the nearest-
neighbor search or hybrid ranker handed back. Two checks, modeled on the
Self-RAG reflection tokens (ISREL / ISSUP from arXiv:2310.11511), applied
generically enough to cover BOTH of the lab's required targets:

    1. RAG answers            -- verify_retrieval() / verify_generation()
    2. Memory recall           -- see verify_memory_recall() at the bottom,
                                    which applies the exact same relevance
                                    check to an episodic/semantic memory
                                    item pulled back for a query, so a
                                    stale or irrelevant memory doesn't
                                    silently leak into a prompt either.

ISREL ("is this passage relevant?") runs after retrieval, before
generation. ISSUP ("is the generated answer supported by what was
retrieved?") runs after generation, before the answer reaches the user.
Failing either has a visible consequence: ISREL failure drops the chunk
before generation even sees it; ISSUP failure replaces the answer with an
explicit "not grounded" refusal rather than passing through a hallucinated
answer.

Offline fallback (no ANTHROPIC_API_KEY): lexical overlap between
question/answer and chunk text, thresholded -- coarser than an LLM judge
but deterministic and reproducible for the eval suite.
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
import re

from rag.vector_store import RetrievedChunk

_WORD_RE = re.compile(r"[a-z0-9]+")

# Lexical overlap on raw tokens is dominated by function words ("the",
# "is", "on"...) that appear in almost every sentence regardless of topic,
# which would let an off-topic question trivially "overlap" with any
# chunk. Stripping them is what makes the offline relevance/support check
# actually discriminate between topics instead of rubber-stamping
# everything -- this is the fix that made the demo's off-topic case
# (visitor parking vs. sedation policy) get caught instead of passing.
_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "on", "in", "at", "to", "for", "of", "and", "or", "with", "without",
    "this", "that", "these", "those", "it", "its", "as", "by", "from",
    "what", "when", "where", "who", "how", "does", "do", "did", "any",
    "before", "after", "so", "not", "no", "if", "we", "you", "your",
}


def _words(text: str) -> set[str]:
    return {w for w in _WORD_RE.findall(text.lower()) if w not in _STOPWORDS}


def _lexical_overlap(a: str, b: str) -> float:
    wa, wb = _words(a), _words(b)
    if not wa:
        return 0.0
    return len(wa & wb) / len(wa)


# ---------------------------------------------------------------------------
# ISREL -- is a retrieved chunk actually relevant to the question?
# ---------------------------------------------------------------------------

RELEVANCE_THRESHOLD = 0.12


def _offline_is_relevant(question: str, chunk_text: str) -> bool:
    return _lexical_overlap(question, chunk_text) >= RELEVANCE_THRESHOLD


def _llm_is_relevant(question: str, chunk_text: str) -> bool:
    import anthropic

    client = anthropic.Anthropic()
    prompt = (
        f"Question: {question}\nPassage: {chunk_text}\n\n"
        "Is this passage relevant to answering the question? "
        "Respond with exactly one word: RELEVANT or IRRELEVANT."
    )
    response = client.messages.create(
        model="claude-sonnet-4-6", max_tokens=10,
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(b.text for b in response.content if getattr(b, "type", None) == "text")
    return "IRRELEVANT" not in text.upper()


def verify_retrieval(question: str, chunks: list[RetrievedChunk]) -> tuple[list[RetrievedChunk], list[RetrievedChunk]]:
    """Returns (kept, dropped). Dropped chunks never reach generation."""
    checker = _llm_is_relevant if os.environ.get("ANTHROPIC_API_KEY") else _offline_is_relevant
    kept, dropped = [], []
    for c in chunks:
        try:
            ok = checker(question, c.text)
        except Exception:
            ok = _offline_is_relevant(question, c.text)
        (kept if ok else dropped).append(c)
    return kept, dropped


# ---------------------------------------------------------------------------
# ISSUP -- is the generated answer actually supported by the kept chunks?
# ---------------------------------------------------------------------------

SUPPORT_THRESHOLD = 0.20
NOT_GROUNDED_MESSAGE = (
    "I can't confirm this answer is supported by the retrieved policy "
    "content, so I'm not going to present it as grounded. Please rephrase "
    "the question or consult the policy manual directly."
)


def _offline_is_supported(answer: str, chunks: list[RetrievedChunk]) -> bool:
    if not chunks:
        return False
    combined = " ".join(c.text for c in chunks)
    return _lexical_overlap(answer, combined) >= SUPPORT_THRESHOLD


def _llm_is_supported(answer: str, chunks: list[RetrievedChunk]) -> bool:
    import anthropic

    client = anthropic.Anthropic()
    context = "\n\n".join(c.text for c in chunks)
    prompt = (
        f"Context:\n{context}\n\nAnswer: {answer}\n\n"
        "Is every factual claim in the answer supported by the context above? "
        "Respond with exactly one word: SUPPORTED or UNSUPPORTED."
    )
    response = client.messages.create(
        model="claude-sonnet-4-6", max_tokens=10,
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(b.text for b in response.content if getattr(b, "type", None) == "text")
    return "UNSUPPORTED" not in text.upper()


def verify_generation(answer: str, chunks: list[RetrievedChunk]) -> tuple[str, bool]:
    """Returns (final_answer, was_supported). If unsupported, the answer
    the caller should actually show the user is replaced -- the original
    (unsupported) text is never surfaced."""
    checker = _llm_is_supported if os.environ.get("ANTHROPIC_API_KEY") else _offline_is_supported
    try:
        supported = checker(answer, chunks)
    except Exception:
        supported = _offline_is_supported(answer, chunks)
    if supported:
        return answer, True
    return NOT_GROUNDED_MESSAGE, False


def verified_answer(question: str, chunks: list[RetrievedChunk], raw_answer: str) -> dict:
    """Runs the full ISREL -> generation (already done by caller) -> ISSUP
    pipeline and returns a structured verdict a grader can inspect."""
    kept, dropped = verify_retrieval(question, chunks)
    final_answer, supported = verify_generation(raw_answer, kept)
    return {
        "question": question,
        "chunks_retrieved": [c.protocol_id for c in chunks],
        "chunks_kept_after_isrel": [c.protocol_id for c in kept],
        "chunks_dropped_as_irrelevant": [c.protocol_id for c in dropped],
        "raw_answer": raw_answer,
        "final_answer": final_answer,
        "issup_passed": supported,
    }


# ---------------------------------------------------------------------------
# Same check, applied to MEMORY recall (episodic/semantic), per the lab's
# "applies to both RAG answers and to memories recalled from the episodic
# and semantic store" requirement.
# ---------------------------------------------------------------------------

def verify_memory_recall(query_context: str, recalled_fact_text: str) -> bool:
    """A recalled semantic fact or episodic memory is only injected into a
    prompt if it clears the same ISREL-style relevance bar used for RAG
    chunks -- e.g. a 3-year-old resolved fall-risk flag surfacing on an
    unrelated allergy question would fail this check and be withheld."""
    checker = _llm_is_relevant if os.environ.get("ANTHROPIC_API_KEY") else _offline_is_relevant
    try:
        return checker(query_context, recalled_fact_text)
    except Exception:
        return _offline_is_relevant(query_context, recalled_fact_text)


if __name__ == "__main__":
    from rag.build_index import build
    from rag.naive_rag import naive_rag_answer

    store = build()
    q = "What's the standard fasting window before sedation?"
    answer, chunks = naive_rag_answer(q, store)
    verdict = verified_answer(q, chunks, answer)
    import json
    print(json.dumps(verdict, indent=2)[:1200])

    print("\nMemory-recall check example:")
    print(" relevant case:", verify_memory_recall(
        "any allergy concerns before we prescribe?",
        "Patient has a documented penicillin allergy.",
    ))
    print(" irrelevant case:", verify_memory_recall(
        "any allergy concerns before we prescribe?",
        "Patient prefers a window seat during transport.",
    ))
