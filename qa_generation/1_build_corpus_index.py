#!/usr/bin/env python3
"""Build FAISS corpus index and (optionally) entity collision index.

Usage:
    python 1_build_corpus_index.py --input_dir ./docs --output_dir ./corpus_index
    python 1_build_corpus_index.py --input_dir ./docs --output_dir ./corpus_index --skip_ner
    python 1_build_corpus_index.py --input_dir ./docs --output_dir ./corpus_index --use_ivf

Supported input formats: .txt, .md, .pdf (via PyMuPDF)

Output files (written to --output_dir):
    corpus_chunks.json          — chunk records
    faiss.index                 — FAISS flat IP index
    faiss_id_map.json           — positional chunk-id list
    entity_collision_index.json — NER collision map (unless --skip_ner)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CHUNK_TOKEN_TARGET = 256
CHUNK_OVERLAP_TOKENS = 32
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
ENTITY_TYPES = {"PERSON", "ORG", "GPE", "PRODUCT", "WORK_OF_ART", "LOC"}
MIN_COLLISION_ENTRIES = 2  # entity must appear in >= this many chunks to be a collision

# ---------------------------------------------------------------------------
# Text utilities
# ---------------------------------------------------------------------------


def _rough_token_count(text: str) -> int:
    """Cheap whitespace tokeniser — avoids tiktoken dependency for chunking."""
    return len(text.split())


def chunk_text(text: str, doc_id: str, doc_title: str, target_tokens: int = CHUNK_TOKEN_TARGET,
               overlap: int = CHUNK_OVERLAP_TOKENS) -> List[dict]:
    """Split text into overlapping word-based chunks."""
    words = text.split()
    chunks = []
    start = 0
    chunk_index = 0
    char_pos = 0

    while start < len(words):
        end = min(start + target_tokens, len(words))
        chunk_words = words[start:end]
        chunk_text_str = " ".join(chunk_words)
        token_count = len(chunk_words)

        # Approximate start_char
        if chunk_index == 0:
            char_pos = 0
        else:
            char_pos = text.find(chunk_words[0], char_pos)

        chunks.append({
            "chunk_id": f"{doc_id}_chunk_{chunk_index:03d}",
            "doc_id": doc_id,
            "doc_title": doc_title,
            "chunk_index": chunk_index,
            "text": chunk_text_str,
            "token_count": token_count,
            "start_char": max(0, char_pos),
        })
        chunk_index += 1
        start += target_tokens - overlap

    return chunks


def _load_pdf(path: Path) -> str:
    """Extract text from a PDF using PyMuPDF (fitz). Returns full document text."""
    try:
        import fitz  # PyMuPDF
    except ImportError:
        print("PyMuPDF not installed. Run: pip install pymupdf")
        sys.exit(1)

    doc = fitz.open(str(path))
    pages = []
    for page in doc:
        text = page.get_text("text")
        if text.strip():
            pages.append(text)
    doc.close()
    # Join pages with a clear separator so sentence boundaries aren't lost
    return "\n\n".join(pages)


def load_documents(input_dir: str) -> List[dict]:
    """Load all .txt, .md, and .pdf files from input_dir recursively."""
    supported = {".txt", ".md", ".pdf"}
    docs = []
    doc_index = 0
    for path in sorted(Path(input_dir).rglob("*")):
        if path.suffix.lower() not in supported:
            continue
        if path.suffix.lower() == ".pdf":
            text = _load_pdf(path)
        else:
            text = path.read_text(encoding="utf-8", errors="replace")
        text = text.strip()
        if not text:
            continue
        doc_id = f"doc_{doc_index:04d}"
        doc_index += 1
        docs.append({"doc_id": doc_id, "doc_title": path.stem, "text": text, "path": str(path)})
    return docs


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------


def embed_chunks(chunks: List[dict], model_name: str, batch_size: int = 64) -> np.ndarray:
    """Embed all chunks using sentence-transformers. Returns (N, dim) float32 array."""
    from sentence_transformers import SentenceTransformer

    print(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)
    texts = [c["text"] for c in chunks]
    print(f"Embedding {len(texts)} chunks (batch_size={batch_size})…")
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True,
                              convert_to_numpy=True, normalize_embeddings=True)
    return embeddings.astype(np.float32)


# ---------------------------------------------------------------------------
# NER / entity collision
# ---------------------------------------------------------------------------


def _root_verb(doc, ent) -> str:
    """Return the root verb of the sentence containing the entity (signature proxy)."""
    try:
        sent = next(s for s in doc.sents if s.start <= ent.start < s.end)
        for token in sent:
            if token.dep_ == "ROOT" and token.pos_ == "VERB":
                return token.lemma_.lower()
        # Fallback: first verb in sentence
        for token in sent:
            if token.pos_ == "VERB":
                return token.lemma_.lower()
    except StopIteration:
        pass
    return ""


def build_entity_collision_index(chunks: List[dict], spacy_model: str = "en_core_web_lg") -> dict:
    """Run spaCy NER over all chunks and build the entity collision index."""
    try:
        import spacy
    except ImportError:
        print("spaCy not installed. Run: pip install spacy && python -m spacy download en_core_web_lg")
        sys.exit(1)

    print(f"Loading spaCy model: {spacy_model}")
    try:
        nlp = spacy.load(spacy_model)
    except OSError:
        print(f"Model '{spacy_model}' not found. Run: python -m spacy download {spacy_model}")
        sys.exit(1)

    entity_map: Dict[str, List[dict]] = {}

    print(f"Running NER on {len(chunks)} chunks…")
    texts = [c["text"] for c in chunks]
    for chunk, doc in zip(chunks, nlp.pipe(texts, batch_size=32)):
        for ent in doc.ents:
            if ent.label_ not in ENTITY_TYPES:
                continue
            name = ent.text.strip()
            if len(name) < 2:
                continue
            signature = _root_verb(doc, ent)
            context_start = max(0, ent.start_char - 60)
            context_end = min(len(chunk["text"]), ent.end_char + 60)
            entry = {
                "chunk_id": chunk["chunk_id"],
                "label": ent.label_,
                "signature": signature,
                "context_snippet": chunk["text"][context_start:context_end],
            }
            entity_map.setdefault(name, []).append(entry)

    # Keep only entities appearing in >= 2 chunks with differing signatures
    collisions: Dict[str, dict] = {}
    for name, entries in entity_map.items():
        if len(entries) < MIN_COLLISION_ENTRIES:
            continue
        signatures = {e["signature"] for e in entries}
        if len(signatures) < 2:
            continue  # same verb root — probably same referent
        labels = list({e["label"] for e in entries})
        entity_type = "/".join(sorted(labels))
        collisions[name] = {"entity_type": entity_type, "chunks": entries}

    return {
        "built_at": datetime.now(timezone.utc).isoformat(),
        "total_collisions": len(collisions),
        "collisions": collisions,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Build FAISS corpus index for QA generation")
    parser.add_argument("--input_dir", required=True, help="Directory with raw .txt/.md documents")
    parser.add_argument("--output_dir", default="corpus_index", help="Output directory")
    parser.add_argument("--embed_model", default=EMBED_MODEL, help="Sentence-transformers model name")
    parser.add_argument("--chunk_tokens", type=int, default=CHUNK_TOKEN_TARGET,
                        help="Target token count per chunk")
    parser.add_argument("--overlap_tokens", type=int, default=CHUNK_OVERLAP_TOKENS,
                        help="Overlap tokens between adjacent chunks")
    parser.add_argument("--batch_size", type=int, default=64, help="Embedding batch size")
    parser.add_argument("--skip_ner", action="store_true", help="Skip entity collision index build")
    parser.add_argument("--spacy_model", default="en_core_web_lg",
                        help="spaCy model for NER (e.g. en_core_web_trf for best accuracy)")
    parser.add_argument("--use_ivf", action="store_true",
                        help="Use IVF index instead of flat (future upgrade path)")
    parser.add_argument("--export_text", action="store_true",
                        help="Export full document text as .txt files for OpenCode grep/read")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load documents
    print(f"Loading documents from: {args.input_dir}")
    docs = load_documents(args.input_dir)
    if not docs:
        print("No .txt or .md files found. Exiting.")
        sys.exit(1)
    print(f"Loaded {len(docs)} documents.")

    # 2. Chunk
    all_chunks: List[dict] = []
    for doc in docs:
        all_chunks.extend(chunk_text(
            doc["text"], doc["doc_id"], doc["doc_title"],
            target_tokens=args.chunk_tokens, overlap=args.overlap_tokens,
        ))
    print(f"Created {len(all_chunks)} chunks.")

    chunks_path = os.path.join(args.output_dir, "corpus_chunks.json")
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)
    print(f"Saved chunks → {chunks_path}")

    # 3. Embed
    embeddings = embed_chunks(all_chunks, args.embed_model, batch_size=args.batch_size)
    dim = embeddings.shape[1]
    print(f"Embedding dim: {dim}")

    # 4. Build FAISS index
    from utils.faiss_store import FAISSStore

    store = FAISSStore(dim=dim, use_ivf=args.use_ivf)
    if args.use_ivf:
        store.train(embeddings)
    chunk_ids = [c["chunk_id"] for c in all_chunks]
    store.add_batch(chunk_ids, embeddings)

    index_path = os.path.join(args.output_dir, "faiss.index")
    id_map_path = os.path.join(args.output_dir, "faiss_id_map.json")
    store.save(index_path, id_map_path)
    print(f"Saved FAISS index → {index_path} ({len(store)} vectors)")

    # 5. Export plain-text files for OpenCode grep/read
    if args.export_text:
        text_dir = os.path.join(args.output_dir, "..", "corpus_text")
        text_dir = os.path.normpath(text_dir)
        os.makedirs(text_dir, exist_ok=True)
        for doc in docs:
            txt_path = os.path.join(text_dir, f"{doc['doc_id']}_{doc['doc_title']}.txt")
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(doc["text"])
        print(f"Exported {len(docs)} text file(s) → {text_dir}")

    # 6. Entity collision index
    if not args.skip_ner:
        collision_index = build_entity_collision_index(all_chunks, spacy_model=args.spacy_model)
        collision_path = os.path.join(args.output_dir, "entity_collision_index.json")
        with open(collision_path, "w", encoding="utf-8") as f:
            json.dump(collision_index, f, indent=2, ensure_ascii=False)
        print(f"Saved entity collision index → {collision_path} "
              f"({collision_index['total_collisions']} collisions)")
    else:
        print("Skipping NER (--skip_ner set).")

    print("Done.")


if __name__ == "__main__":
    main()
