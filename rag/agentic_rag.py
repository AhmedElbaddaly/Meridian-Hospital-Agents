"""
rag/agentic_rag.py
--------------------
A real reasoning loop: decide what to retrieve -> retrieve -> observe ->
decide whether more retrieval is needed -> (repeat, bounded) -> answer.

This is the only architecture of the three that can handle a genuinely
multi-part question -- e.g. "a 70-year-old with a cardiac history AND a
penicillin allergy needing surgery" -- which requires two DIFFERENT policy
sections (cardiac sedation adjustments, AND antibiotic-class allergy
handling) that a single top-k retrieval tends to only half-cover.

Offline mode (no ANTHROPIC_API_KEY): the "decide what to retrieve" step
uses a keyword-based topic decomposer instead of an LLM planner --
documented and deliberately simple, consistent with the offline/LLM split
used throughout this repo. Set ANTHROPIC_API_KEY to route the planning step
through Claude instead (`_llm_plan_next_query`).
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
from dataclasses import dataclass, field

from rag.embeddings import get_embedder
from rag.generation import generate_answer
from rag.vector_store import RetrievedChunk, VectorStore

MAX_HOPS = 3

# topic -> the retrieval query that surfaces the right policy section.
# A real implementation would derive this dynamically (e.g. an LLM listing
# sub-questions); this fixed map is the offline stand-in, scoped to the
# clinical topics this corpus actually covers.
_TOPIC_QUERIES = {
    "cardiac": "sedation adjustments for cardiac-risk patients",
    "penicillin": "medication allergy administration override policy",
    "allerg": "medication allergy administration override policy",
    "pediatric": "pediatric weight-based dosing safeguard",
    "elective": "pre-operative screening for elective procedures",
    "discharge": "discharge readiness criteria",
    "transfusion": "blood transfusion consent and verification",
    "isolation": "isolation precautions for infection control",
}


@dataclass
class AgenticTrace:
    hops: list[dict] = field(default_factory=list)


def _offline_plan_next_query(question: str, already_covered: set[str]):
    q_lower = question.lower()
    for topic, query in _TOPIC_QUERIES.items():
        if topic in q_lower and topic not in already_covered:
            return topic, query
    return None, None


def _llm_plan_next_query(question: str, retrieved_so_far: list[RetrievedChunk]):
    import anthropic

    client = anthropic.Anthropic()
    context = "\n".join(f"- [{c.protocol_id}] {c.title}" for c in retrieved_so_far) or "(nothing yet)"
    prompt = f"""You are planning retrieval steps to answer a clinical policy question.

Question: {question}

Retrieved so far: {context}

If the retrieved policies already fully answer the question, respond with exactly: ANSWER
Otherwise, respond with exactly one line: RETRIEVE: <a short search query for the missing piece>
"""
    response = client.messages.create(
        model="claude-sonnet-4-6", max_tokens=60,
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(b.text for b in response.content if getattr(b, "type", None) == "text").strip()
    if text.upper().startswith("ANSWER"):
        return None
    if text.upper().startswith("RETRIEVE:"):
        return text.split(":", 1)[1].strip()
    return None


def agentic_rag_answer(question: str, store: VectorStore,
                        max_hops: int = MAX_HOPS):
    embedder = get_embedder()
    trace = AgenticTrace()
    retrieved: list[RetrievedChunk] = []
    covered_topics: set[str] = set()
    use_llm = bool(os.environ.get("ANTHROPIC_API_KEY"))

    for hop in range(max_hops):
        if use_llm:
            try:
                next_query = _llm_plan_next_query(question, retrieved)
                topic = f"llm_hop_{hop}"
            except Exception:
                use_llm = False
                topic, next_query = _offline_plan_next_query(question, covered_topics)
        else:
            topic, next_query = _offline_plan_next_query(question, covered_topics)

        if not next_query:
            trace.hops.append({"hop": hop, "action": "stop", "reason": "no further retrieval needed"})
            break

        vec = embedder.embed(next_query)
        hits = store.query(vec, top_k=2)
        retrieved.extend(h for h in hits if h.chunk_id not in {r.chunk_id for r in retrieved})
        # mark every topic keyword that maps to this same query as covered,
        # so e.g. "penicillin" and "allerg" both matching the allergy
        # section don't trigger two redundant hops
        covered_topics.update(t for t, q in _TOPIC_QUERIES.items() if q == next_query)
        covered_topics.add(topic)
        trace.hops.append({
            "hop": hop, "action": "retrieve", "query": next_query,
            "observed": [h.protocol_id for h in hits],
        })

    if not retrieved:
        # nothing topic-matched at all -- fall back to one naive retrieval
        # so agentic RAG never returns strictly worse than doing nothing
        vec = embedder.embed(question)
        retrieved = store.query(vec, top_k=3)
        trace.hops.append({"hop": 0, "action": "fallback_naive_retrieve",
                            "observed": [h.protocol_id for h in retrieved]})

    answer = generate_answer(question, retrieved)
    return answer, retrieved, trace


if __name__ == "__main__":
    from rag.build_index import build

    store = build()
    q = ("For a 70-year-old patient with a cardiac history and a known "
         "penicillin allergy who needs emergency surgery, what sedation "
         "adjustments and antibiotic handling apply?")
    answer, chunks, trace = agentic_rag_answer(q, store)
    print("Q:", q)
    for h in trace.hops:
        print(" hop:", h)
    print("Retrieved:", [c.protocol_id for c in chunks])
    print("A:", answer)
