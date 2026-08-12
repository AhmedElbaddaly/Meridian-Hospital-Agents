"""
retrieval_eval/test_questions.py
-----------------------------------
12 fixed test questions across the three categories the lab requires:
general (naive RAG should be fine), citation-heavy (hybrid should win),
and multi-part/decomposition (only agentic RAG should reliably handle).

`expected_protocol_ids`: the set of Protocol IDs a correct retrieval MUST
surface (used as ground truth for the comparison table -- deterministic,
reproducible, no LLM judge required). This file is the fixed test suite;
per the lab's guardrails it is not modified once evaluation starts.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TestQuestion:
    question: str
    category: str  # 'general' | 'citation' | 'multi_hop'
    expected_protocol_ids: set[str]


TEST_QUESTIONS: list[TestQuestion] = [
    # ---- general: naive RAG should handle these fine -------------------
    TestQuestion(
        "What's the standard fasting window before sedation?",
        "general", {"4.2a"},
    ),
    TestQuestion(
        "How is ICU bed admission approved?",
        "general", {"2.3"},
    ),
    TestQuestion(
        "What isolation precautions apply for airborne infections?",
        "general", {"6.1"},
    ),
    TestQuestion(
        "When can a patient be discharged?",
        "general", {"7.2"},
    ),

    # ---- citation-heavy: exact identifiers that embeddings tend to blur,
    #      hybrid search (BM25 side) should win these ---------------------
    TestQuestion(
        "What does Protocol 4.2b say about cardiac-risk patients?",
        "citation", {"4.2b"},
    ),
    TestQuestion(
        "What does Protocol 6.5 require for blood transfusion verification?",
        "citation", {"6.5"},
    ),
    TestQuestion(
        "What is required under Protocol 8.1 for fall-risk assessment?",
        "citation", {"8.1"},
    ),
    TestQuestion(
        "Summarize Protocol 3.4 on high-alert medication checks.",
        "citation", {"3.4"},
    ),

    # ---- multi-hop: needs two DIFFERENT policy sections combined --------
    TestQuestion(
        "For a 70-year-old patient with a cardiac history and a known "
        "penicillin allergy needing emergency surgery, what sedation "
        "adjustments and antibiotic handling apply?",
        "multi_hop", {"4.2b", "3.4"},
    ),
    TestQuestion(
        "For a pediatric patient with a documented cardiac history who "
        "needs procedural sedation, what dosing safeguard and cardiac "
        "adjustments both apply?",
        "multi_hop", {"4.8", "4.2b"},
    ),
    TestQuestion(
        "A patient is being discharged after cardiac-risk sedation -- what "
        "discharge criteria AND sedation-specific requirements apply?",
        "multi_hop", {"7.2", "4.2b"},
    ),
    TestQuestion(
        "For an elective dental cleaning on a 68-year-old with a cardiac "
        "history, what pre-operative screening and sedation adjustments "
        "are both required?",
        "multi_hop", {"4.5", "4.2b"},
    ),
]
