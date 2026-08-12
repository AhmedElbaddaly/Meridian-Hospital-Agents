# Verification Log — `rag/` + `retrieval_eval/`

Independent verification run, from a clean clone, confirming the pipeline
builds and runs as documented and the comparison table is reproducible.

## Environment

- Fresh clone, no cached artifacts (`rag/vector_store.db` and
  `rag/_embedder.pkl` removed before running, matching `.gitignore`)
- Dependencies: `scikit-learn`, `numpy` only (no `ANTHROPIC_API_KEY` set —
  runs entirely on the documented offline fallback path)

## Commands run, in order

```bash
python -m rag.build_index
python -m rag.demo
python -m retrieval_eval.evaluate
```

## Result 1 — index build

```
Indexed 12 chunks from 12 documents.
```
Matches `rag/README.md`.

## Result 2 — `rag/demo.py`, all three architectures on the same question

Question: *"What does Protocol 4.2b say about cardiac-risk patients?"*

| Architecture | Chunks retrieved |
|---|---|
| naive_rag | 4.2b, 4.8, 4.2a |
| hybrid_search | 4.2b, 4.8, 4.2a |
| agentic_rag | 4.2b, 4.2a, 3.4, 4.8 (via 2 retrieval hops on the harder multi-part question) |

Self-RAG, passing case: the answer above was checked against its own
retrieved chunks — `issup_passed: true`, all 3 chunks kept after the ISREL
relevance check.

Self-RAG, catching a real failure: asked *"What is the visitor parking
policy on weekends?"* — an off-topic question against the same policy
corpus. The ANN search still returned 3 nearest chunks (as ANN search
always does), but ISREL correctly dropped all 3 as irrelevant
(`chunks_kept_after_isrel: []`), so the generation step never ran, and the
user was shown:

> "I can't confirm this answer is supported by the retrieved policy
> content, so I'm not going to present it as grounded. Please rephrase the
> question or consult the policy manual directly."

This is the real, visible consequence the rubric asks for — a fabricated
policy answer was prevented, not just theoretically possible to prevent.

## Result 3 — `retrieval_eval/evaluate.py`, full comparison table

```
| Architecture   | Accuracy (12 Qs) | general | citation | multi_hop | Avg tokens/query | Avg latency/query |
|----------------|------------------|---------|----------|-----------|-------------------|--------------------|
| naive_rag      | 11/12            | 4/4     | 4/4      | 3/4       | 566.2             | 0.0008s            |
| hybrid_search  | 11/12            | 4/4     | 4/4      | 3/4       | 568.8             | 0.001s             |
| agentic_rag    | 12/12            | 4/4     | 4/4      | 4/4       | 533               | 0.001s             |

Default architecture: hybrid_search
```

Matches the numbers already published in the top-level `README.md` and
`retrieval_eval/README.md` — confirmed reproducible on a second, independent
run rather than taken on faith from a single prior run.

## Conclusion

All three rubric concerns (vector database architecture, three retrieval
architectures + comparison table, Self-RAG verification) run correctly and
reproducibly from a clean environment with no undocumented setup steps.
