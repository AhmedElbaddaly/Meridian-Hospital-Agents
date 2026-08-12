"""
rag/demo.py
-------------
End-to-end demo: build the vector store, ask the same question through all
three (three, no bonus Graph RAG) architectures, then show Self-RAG
verification both PASSING a well-grounded answer and CATCHING a genuine
failure (a question the corpus has no real answer for).

Run:
    python -m rag.demo
"""

from __future__ import annotations

# --- allow running this file directly (python path/to/file.py), not
# --- just as a module (python -m pkg.file) -- both now work the same.
import os as _os
import sys as _sys
_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _REPO_ROOT not in _sys.path:
    _sys.path.insert(0, _REPO_ROOT)


import json

from rag.agentic_rag import agentic_rag_answer
from rag.build_index import build
from rag.hybrid_search import build_hybrid_index, hybrid_rag_answer
from rag.naive_rag import naive_rag_answer
from rag.self_rag import verified_answer


def _print_header(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def main() -> None:
    store = build()
    hidx = build_hybrid_index()

    citation_question = "What does Protocol 4.2b say about cardiac-risk patients?"

    _print_header("SAME QUESTION, THREE ARCHITECTURES")
    print("Q:", citation_question)

    n_answer, n_chunks = naive_rag_answer(citation_question, store)
    print("\n-- naive RAG --")
    print("retrieved:", [c.protocol_id for c in n_chunks])

    h_answer, h_chunks = hybrid_rag_answer(citation_question, store, hidx)
    print("\n-- hybrid search --")
    print("retrieved:", [c.protocol_id for c in h_chunks])

    multi_hop_question = (
        "For a 70-year-old patient with a cardiac history and a known "
        "penicillin allergy who needs emergency surgery, what sedation "
        "adjustments and antibiotic handling apply?"
    )
    a_answer, a_chunks, trace = agentic_rag_answer(multi_hop_question, store)
    print("\n-- agentic RAG (multi-hop question) --")
    print("Q:", multi_hop_question)
    for h in trace.hops:
        print(" hop:", h)
    print("retrieved:", [c.protocol_id for c in a_chunks])

    # -----------------------------------------------------------------
    # Self-RAG: a well-grounded case (should pass)
    # -----------------------------------------------------------------
    _print_header("SELF-RAG VERIFICATION -- PASSING CASE")
    good_verdict = verified_answer(citation_question, n_chunks, n_answer)
    print(json.dumps({k: v for k, v in good_verdict.items() if k != "raw_answer"}, indent=2))

    # -----------------------------------------------------------------
    # Self-RAG: a genuine failure case -- the corpus has NOTHING about
    # visitor parking, so naive retrieval still returns its top-k nearest
    # chunks (ANN always returns *something*), but ISREL should reject
    # them as irrelevant, and the answer should never claim to know.
    # -----------------------------------------------------------------
    _print_header("SELF-RAG VERIFICATION -- CATCHING A REAL FAILURE")
    off_topic_question = "What is the visitor parking policy on weekends?"
    bad_answer, bad_chunks = naive_rag_answer(off_topic_question, store)
    bad_verdict = verified_answer(off_topic_question, bad_chunks, bad_answer)
    print(json.dumps({k: v for k, v in bad_verdict.items() if k != "raw_answer"}, indent=2))
    print("\nWhat the user actually sees:", repr(bad_verdict["final_answer"][:120]))


if __name__ == "__main__":
    main()
