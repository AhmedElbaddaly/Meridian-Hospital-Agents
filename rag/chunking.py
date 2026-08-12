"""
rag/chunking.py
-----------------
Chunks each PolicyDoc into retrieval-sized pieces, carrying metadata along
with every chunk (source, department, protocol_id, last_reviewed) so it can
be written straight into the vector store's metadata payload.

Each PolicyDoc here is already a coherent policy section (as it would be
after a real 40-page PDF was parsed and its sections identified), so
chunking mostly means: split further only if a section runs long, and
always keep the protocol_id / title attached to every resulting chunk so a
citation-heavy query ("what does Protocol 4.2b say...") can be answered
even from a partial chunk.
"""

from __future__ import annotations

# --- allow running this file directly (python path/to/file.py), not
# --- just as a module (python -m pkg.file) -- both now work the same.
import os as _os
import sys as _sys
_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _REPO_ROOT not in _sys.path:
    _sys.path.insert(0, _REPO_ROOT)


from dataclasses import dataclass

from rag.corpus import CORPUS, PolicyDoc

MAX_CHUNK_WORDS = 120


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    protocol_id: str
    department: str
    last_reviewed: str
    title: str
    text: str


def _split_words(text: str, max_words: int) -> list[str]:
    words = text.split()
    return [
        " ".join(words[i:i + max_words])
        for i in range(0, len(words), max_words)
    ] or [text]


def chunk_document(doc: PolicyDoc) -> list[Chunk]:
    pieces = _split_words(doc.text, MAX_CHUNK_WORDS)
    chunks = []
    for i, piece in enumerate(pieces):
        chunks.append(Chunk(
            chunk_id=f"{doc.doc_id}::{i}",
            doc_id=doc.doc_id,
            protocol_id=doc.protocol_id,
            department=doc.department,
            last_reviewed=doc.last_reviewed,
            title=doc.title,
            # keep the protocol citation + title attached to every chunk,
            # even split ones, so hybrid/keyword search can always match it
            text=f"[Protocol {doc.protocol_id} -- {doc.title}] {piece}",
        ))
    return chunks


def chunk_corpus() -> list[Chunk]:
    all_chunks: list[Chunk] = []
    for doc in CORPUS:
        all_chunks.extend(chunk_document(doc))
    return all_chunks
