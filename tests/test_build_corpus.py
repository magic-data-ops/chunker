"""Tests for qa_generation/build_corpus_index.py — document loading."""
from __future__ import annotations

from qa_generation import build_corpus_index as build


class TestLoadDocuments:
    def test_loads_txt_files(self, tmp_path):
        (tmp_path / "doc1.txt").write_text("Hello world.")
        (tmp_path / "doc2.md").write_text("# Title\nContent here.")
        docs = build.load_documents(str(tmp_path))
        assert len(docs) == 2
        assert docs[0]["doc_id"] == "doc_0000"
        assert docs[1]["doc_id"] == "doc_0001"

    def test_skips_empty_files(self, tmp_path):
        (tmp_path / "empty.txt").write_text("")
        (tmp_path / "good.txt").write_text("Some content.")
        docs = build.load_documents(str(tmp_path))
        assert len(docs) == 1
        assert docs[0]["doc_title"] == "good"

    def test_skips_unsupported_extensions(self, tmp_path):
        (tmp_path / "data.csv").write_text("a,b,c")
        (tmp_path / "doc.txt").write_text("Real doc.")
        docs = build.load_documents(str(tmp_path))
        assert len(docs) == 1

    def test_empty_dir_returns_empty(self, tmp_path):
        docs = build.load_documents(str(tmp_path))
        assert docs == []

    def test_recursive_loading(self, tmp_path):
        sub = tmp_path / "subdir"
        sub.mkdir()
        (sub / "nested.txt").write_text("Nested content.")
        (tmp_path / "top.txt").write_text("Top level.")
        docs = build.load_documents(str(tmp_path))
        assert len(docs) == 2
