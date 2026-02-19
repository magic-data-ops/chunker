#!/usr/bin/env python3
"""Build corpus text directory from raw documents.

Usage:
    python build_corpus_index.py --input_dir ./docs --output_dir ./corpus_index

Supported input formats: .txt, .md, .pdf (via PyMuPDF)

Output:
    ../corpus_text/*.txt — plain-text exports for Claude Code Grep/Read
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List


# ---------------------------------------------------------------------------
# Text utilities
# ---------------------------------------------------------------------------


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
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Build corpus text directory for QA generation")
    parser.add_argument("--input_dir", required=True, help="Directory with raw .txt/.md/.pdf documents")
    parser.add_argument("--output_dir", default="corpus_index", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load documents
    print(f"Loading documents from: {args.input_dir}")
    docs = load_documents(args.input_dir)
    if not docs:
        print("No .txt, .md, or .pdf files found. Exiting.")
        sys.exit(1)
    print(f"Loaded {len(docs)} documents.")

    # 2. Export plain-text files for Claude Code Grep/Read
    text_dir = os.path.join(args.output_dir, "..", "corpus_text")
    text_dir = os.path.normpath(text_dir)
    os.makedirs(text_dir, exist_ok=True)
    for doc in docs:
        txt_path = os.path.join(text_dir, f"{doc['doc_id']}_{doc['doc_title']}.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(doc["text"])
    print(f"Exported {len(docs)} text file(s) → {text_dir}")

    print("Done.")


if __name__ == "__main__":
    main()
