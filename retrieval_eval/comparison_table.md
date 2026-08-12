# Retrieval Architecture Comparison

| Architecture | Accuracy (12 Qs) | general | citation | multi_hop | Avg tokens/query | Avg latency/query |
|---|---|---|---|---|---|---|
| naive_rag | 11/12 | 4/4 | 4/4 | 3/4 | 566.2 | 0.001s |
| hybrid_search | 11/12 | 4/4 | 4/4 | 3/4 | 569.5 | 0.0015s |
| agentic_rag | 12/12 | 4/4 | 4/4 | 4/4 | 533 | 0.0012s |

**Default architecture: `hybrid_search`**

'hybrid_search' selected as the default: tied for the best combined general+citation accuracy (100%) among ['naive_rag', 'hybrid_search', 'agentic_rag']. At this corpus size (12 chunks) all three architectures run in low-single-digit milliseconds, so raw latency differences are noise, not signal -- the tie is broken structurally (hybrid search's BM25 side is effectively free here and protects citation accuracy as the manual grows) rather than by over-fitting to a sub-millisecond timing measurement. Multi-part questions (category 'multi_hop') should still be routed to agentic_rag specifically, since only it reliably clears that category -- see the table.
