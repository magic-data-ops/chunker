"""Tests for qa_generation/generate_qa_chains.py — JSON extraction, pair validation, chain conversion,
stream-json parsing, and provenance extraction."""
from __future__ import annotations

import json
import os

import pytest
from qa_generation import generate_qa_chains as gen


# ---------------------------------------------------------------------------
# _extract_json_array
# ---------------------------------------------------------------------------


class TestExtractJsonArray:
    def test_plain_json(self):
        text = '[{"question": "What?", "golden_answer": "Yes."}]'
        result = gen._extract_json_array(text)
        assert result is not None
        assert len(result) == 1
        assert result[0]["question"] == "What?"

    def test_fenced_json(self):
        text = '```json\n[{"question": "What?", "golden_answer": "Yes."}]\n```'
        result = gen._extract_json_array(text)
        assert result is not None
        assert len(result) == 1

    def test_garbage_returns_none(self):
        text = "This is not JSON at all, just random text."
        result = gen._extract_json_array(text)
        assert result is None


# ---------------------------------------------------------------------------
# _validate_pair
# ---------------------------------------------------------------------------


class TestValidatePair:
    def test_good_pair(self, sample_pair, sample_category):
        ok, reason = gen._validate_pair(sample_pair, sample_category)
        assert ok, f"Expected valid pair, got: {reason}"

    def test_short_answer(self, sample_pair, sample_category):
        pair = {**sample_pair, "golden_answer": "Yes"}
        ok, reason = gen._validate_pair(pair, sample_category)
        assert not ok
        assert "too short" in reason

    def test_short_evidence(self, sample_pair, sample_category):
        pair = {**sample_pair, "evidence_snippets": ["sit"]}
        ok, reason = gen._validate_pair(pair, sample_category)
        assert not ok
        assert "too short" in reason

    def test_no_question_mark(self, sample_pair, sample_category):
        pair = {**sample_pair, "question": "Tell me about the company"}
        ok, reason = gen._validate_pair(pair, sample_category)
        assert not ok
        assert "?" in reason

    def test_no_evidence(self, sample_pair, sample_category):
        pair = {**sample_pair, "evidence_snippets": []}
        ok, reason = gen._validate_pair(pair, sample_category)
        assert not ok
        assert "no evidence" in reason

    def test_hop_constraints_too_many(self, sample_pair, sample_category):
        """single_chunk_factoid allows max_hops=1, so 2 snippets should fail."""
        pair = {
            **sample_pair,
            "evidence_snippets": [
                "Founded in 1927, the company initially manufactured agricultural equipment.",
                "The company later pivoted to electronics production in the mid-1960s.",
            ],
        }
        ok, reason = gen._validate_pair(pair, sample_category)
        assert not ok
        assert "too many" in reason

    def test_hop_constraints_too_few(self, multi_hop_category):
        """multi_hop_reasoning requires min_hops=2, so 1 snippet should fail."""
        pair = {
            "question": "What connects the two events?",
            "golden_answer": "Both events occurred during the same decade and influenced policy changes.",
            "evidence_snippets": [
                "Founded in 1927, the company initially manufactured agricultural equipment.",
            ],
            "source_files": ["doc_0000_history.txt"],
        }
        ok, reason = gen._validate_pair(pair, multi_hop_category)
        assert not ok
        assert "too few" in reason or "multi_hop" in reason

    def test_multi_hop_requires_two_snippets(self, multi_hop_category):
        pair = {
            "question": "What connects the two events?",
            "golden_answer": "Both events occurred during the same decade and influenced policy changes.",
            "evidence_snippets": [
                "Founded in 1927, the company initially manufactured agricultural equipment.",
            ],
            "source_files": ["doc_0000_history.txt"],
        }
        ok, reason = gen._validate_pair(pair, multi_hop_category)
        assert not ok

    def test_entity_evidence_fallback(self):
        """When evidence_snippets is empty, fall back to entities[].evidence_snippet."""
        entity_cat = {"name": "entity_disambiguation", "description": "test", "min_hops": 1, "max_hops": 3}
        pair = {
            "question": "What are the two meanings of BERT?",
            "golden_answer": "BERT refers to both a robot and a language model.",
            "evidence_snippets": [],
            "entities": [
                {"label": "BERT — Robot", "evidence_snippet": "BERT2 is a physical humanoid robot assistant used in an HRI experiment."},
                {"label": "BERT — NLP", "evidence_snippet": "BERT stands for Bidirectional Encoder Representations from Transformers."},
            ],
            "source_files": ["corpus.txt"],
        }
        ok, reason = gen._validate_pair(pair, entity_cat)
        assert ok
        # evidence_snippets should now be populated from entities
        assert len(pair["evidence_snippets"]) == 2


# ---------------------------------------------------------------------------
# _pairs_to_chains
# ---------------------------------------------------------------------------


class TestPairsToChains:
    def test_empty_sources_no_crash(self, sample_category):
        """Bug 2: empty source_files should not crash."""
        pair = {
            "question": "What year?",
            "golden_answer": "The company was founded in the year 1927.",
            "evidence_snippets": ["Founded in 1927, the company initially manufactured agricultural equipment."],
            "source_files": [],
        }
        chains = gen._pairs_to_chains([pair], sample_category, "fallback.txt")
        assert len(chains) == 1
        assert chains[0]["source_file"] == "fallback.txt"
        assert chains[0]["prompt_seed_file"] == "fallback.txt"

    def test_schema_fields(self, sample_pair, sample_category):
        chains = gen._pairs_to_chains([sample_pair], sample_category, "seed.txt")
        assert len(chains) == 1
        chain = chains[0]
        required_fields = {
            "chain_id", "category", "source_file", "prompt_seed_file", "question",
            "final_answer", "hop_path", "hop_count", "termination_reason",
            "single_answer_heuristic", "generated_at",
        }
        assert required_fields.issubset(chain.keys())
        assert chain["category"] == "single_chunk_factoid"
        assert chain["hop_count"] >= 1

    def test_skips_empty_question_or_answer(self, sample_category):
        pairs = [
            {"question": "", "golden_answer": "some answer here that is long enough"},
            {"question": "What?", "golden_answer": ""},
        ]
        chains = gen._pairs_to_chains(pairs, sample_category, "seed.txt")
        assert len(chains) == 0

    def test_provenance_attached(self, sample_pair, sample_category):
        """When provenance is provided, it should be attached to chains."""
        provenance = gen.ProvenanceReport(
            tool_provenances=[],
            files_accessed=[{"file": "doc_0000.txt", "line_range": (1, 50), "content_length": 1000}],
            grep_queries=[{"pattern": "test", "files_hit": ["doc_0000.txt"]}],
            unique_files=["doc_0000.txt"],
            total_content_read_chars=1000,
        )
        chains = gen._pairs_to_chains([sample_pair], sample_category, "seed.txt",
                                      provenance=provenance)
        assert len(chains) == 1
        assert "provenance_report" in chains[0]
        assert chains[0]["provenance_report"]["unique_files"] == ["doc_0000.txt"]
        assert chains[0]["provenance_report"]["tool_call_count"] == 0
        assert "context_span_lines" in chains[0]

    def test_evidence_locations_attached(self, sample_category):
        """evidence_locations from agent output should pass through to chain."""
        pair = {
            "question": "What year was it founded?",
            "golden_answer": "The company was founded in 1927.",
            "evidence_snippets": ["Founded in 1927, the company initially manufactured agricultural equipment."],
            "source_files": ["doc_0000.txt"],
            "evidence_locations": [{"file": "doc_0000.txt", "start_line": 120, "end_line": 125}],
        }
        chains = gen._pairs_to_chains([pair], sample_category, "seed.txt")
        assert len(chains) == 1
        assert chains[0]["evidence_locations"] == [{"file": "doc_0000.txt", "start_line": 120, "end_line": 125}]

    def test_source_file_vs_prompt_seed_file(self, sample_category):
        """source_file comes from pair's source_files, prompt_seed_file from the seed."""
        pair = {
            "question": "What year was it founded?",
            "golden_answer": "The company was founded in 1927.",
            "evidence_snippets": ["Founded in 1927, the company initially manufactured agricultural equipment."],
            "source_files": ["doc_0001_other.txt"],
        }
        chains = gen._pairs_to_chains([pair], sample_category, "doc_0000_seed.txt")
        assert len(chains) == 1
        assert chains[0]["source_file"] == "doc_0001_other.txt"
        assert chains[0]["prompt_seed_file"] == "doc_0000_seed.txt"


# ---------------------------------------------------------------------------
# _parse_stream_json (golden file test)
# ---------------------------------------------------------------------------


FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")


class TestParseStreamJson:
    @pytest.fixture
    def golden_stdout(self):
        path = os.path.join(FIXTURES_DIR, "sample_claude_stream.jsonl")
        with open(path, "r") as f:
            return f.read()

    def test_extracts_tool_events(self, golden_stdout):
        reply_text, reply_json, tool_events, meta, _reasoning = gen._parse_stream_json(golden_stdout)
        # Should have 3 tool events: Grep, Read, Glob
        assert len(tool_events) == 3
        tools_used = [e["tool"] for e in tool_events]
        assert "Grep" in tools_used
        assert "Read" in tools_used
        assert "Glob" in tools_used

    def test_extracts_reply_text(self, golden_stdout):
        reply_text, reply_json, tool_events, meta, _reasoning = gen._parse_stream_json(golden_stdout)
        assert "Brown v. Board of Education" in reply_text

    def test_extracts_reply_json(self, golden_stdout):
        reply_text, reply_json, tool_events, meta, _reasoning = gen._parse_stream_json(golden_stdout)
        assert reply_json is not None
        assert len(reply_json) == 1
        assert reply_json[0]["question"] == "What was the ruling in Brown v. Board of Education?"

    def test_extracts_meta(self, golden_stdout):
        reply_text, reply_json, tool_events, meta, _reasoning = gen._parse_stream_json(golden_stdout)
        assert meta.get("session_id") == "test-session-001"
        assert meta.get("cli_version") == "2.1.47"
        assert meta.get("duration_ms") == 8500
        assert meta.get("cost_usd") == 0.0085
        assert meta.get("is_error") is False

    def test_tool_results_matched(self, golden_stdout):
        """Each tool_use should have a corresponding result."""
        reply_text, reply_json, tool_events, meta, _reasoning = gen._parse_stream_json(golden_stdout)
        for event in tool_events:
            assert event["result"] is not None, f"Tool {event['tool']} has no result"

    def test_survives_extra_fields(self):
        """Parser should not crash on unknown event types."""
        stdout = '{"type":"unknown_event","data":"foo"}\n{"type":"result","subtype":"success","result":"test","is_error":false,"duration_ms":100,"total_cost_usd":0.001,"num_turns":1,"session_id":"s1"}\n'
        reply_text, reply_json, tool_events, meta, _reasoning = gen._parse_stream_json(stdout)
        assert reply_text == "test"
        assert meta["duration_ms"] == 100

    def test_incomplete_text_block_no_crash(self):
        """A text block missing 'text' key should be skipped, not crash."""
        stdout = json.dumps({"type": "assistant", "message": {"content": [{"type": "text"}]}}) + "\n"
        stdout += json.dumps({"type": "result", "result": "final", "is_error": False, "duration_ms": 50, "total_cost_usd": 0.0, "num_turns": 1}) + "\n"
        reply_text, reply_json, tool_events, meta, _reasoning = gen._parse_stream_json(stdout)
        assert reply_text == "final"

    def test_incomplete_tool_use_block_no_crash(self):
        """A tool_use block missing 'id' or 'name' should be skipped."""
        stdout = json.dumps({"type": "assistant", "message": {"content": [{"type": "tool_use", "id": "t1"}]}}) + "\n"  # missing 'name'
        stdout += json.dumps({"type": "assistant", "message": {"content": [{"type": "tool_use", "name": "Grep"}]}}) + "\n"  # missing 'id'
        stdout += json.dumps({"type": "result", "result": "ok", "is_error": False, "duration_ms": 1, "total_cost_usd": 0.0, "num_turns": 1}) + "\n"
        reply_text, reply_json, tool_events, meta, _reasoning = gen._parse_stream_json(stdout)
        assert len(tool_events) == 0  # Both should be skipped

    def test_result_event_preferred_over_intermediate_text(self):
        """The result event should be used as reply even when intermediate text exists."""
        stdout = json.dumps({"type": "assistant", "message": {"content": [{"type": "text", "text": "Let me search..."}]}}) + "\n"
        stdout += json.dumps({"type": "result", "result": '[{"question": "Q?", "golden_answer": "A long enough answer here."}]', "is_error": False, "duration_ms": 1, "total_cost_usd": 0.0, "num_turns": 1}) + "\n"
        reply_text, reply_json, tool_events, meta, _reasoning = gen._parse_stream_json(stdout)
        # Should use the result event, not the intermediate "Let me search..."
        assert reply_json is not None
        assert reply_json[0]["question"] == "Q?"

    def test_reasoning_blocks_captured(self):
        """Intermediate assistant text blocks should be captured in reasoning_blocks."""
        stdout = json.dumps({"type": "assistant", "message": {"content": [{"type": "text", "text": "Let me search for BERT..."}]}}) + "\n"
        stdout += json.dumps({"type": "assistant", "message": {"content": [{"type": "text", "text": "Found two papers about BERT."}]}}) + "\n"
        stdout += json.dumps({"type": "result", "result": "final", "is_error": False, "duration_ms": 1, "total_cost_usd": 0.0, "num_turns": 1}) + "\n"
        reply_text, reply_json, tool_events, meta, reasoning = gen._parse_stream_json(stdout)
        assert len(reasoning) == 2
        assert "Let me search for BERT..." in reasoning
        assert "Found two papers about BERT." in reasoning


# ---------------------------------------------------------------------------
# Provenance extraction
# ---------------------------------------------------------------------------


class TestParseReadLineRange:
    def test_arrow_format(self):
        text = "     1→First line\n     2→Second line\n     3→Third line"
        result = gen._parse_read_line_range(text)
        assert result == (1, 3)

    def test_tab_format(self):
        text = "  1520\tContext before.\n  1521\tThe case was argued.\n  1522\tChief Justice."
        result = gen._parse_read_line_range(text)
        assert result == (1520, 1522)

    def test_no_line_numbers(self):
        text = "Just plain text without line numbers"
        result = gen._parse_read_line_range(text)
        assert result is None


class TestParseGrepHits:
    def test_standard_format(self):
        text = "doc_0000.txt:1523: The Supreme Court ruled.\ndoc_0001.txt:847: Another ruling."
        hits = gen._parse_grep_hits(text)
        assert len(hits) == 2
        assert hits[0]["file"] == "doc_0000.txt"
        assert hits[0]["line_no"] == 1523
        assert hits[1]["file"] == "doc_0001.txt"

    def test_no_hits(self):
        text = "No matches found"
        hits = gen._parse_grep_hits(text)
        assert len(hits) == 0


class TestExtractProvenance:
    def test_read_provenance(self):
        events = [{
            "tool": "Read",
            "tool_use_id": "toolu_001",
            "input": {"file_path": "/tmp/corpus/doc_0000.txt", "offset": 1520, "limit": 10},
            "result": "  1520\tLine one.\n  1521\tLine two.\n  1522\tLine three.",
            "result_meta": {"file": {"filePath": "/tmp/corpus/doc_0000.txt", "startLine": 1520, "numLines": 3}},
        }]
        report = gen._extract_provenance(events)
        assert len(report.tool_provenances) == 1
        assert report.tool_provenances[0].file_path == "/tmp/corpus/doc_0000.txt"
        assert report.tool_provenances[0].line_range == (1520, 1522)
        assert "doc_0000.txt" in report.unique_files
        assert len(report.files_accessed) == 1

    def test_grep_provenance(self):
        events = [{
            "tool": "Grep",
            "tool_use_id": "toolu_002",
            "input": {"pattern": "Supreme Court"},
            "result": "doc_0000.txt:1523: The Supreme Court ruled.\ndoc_0001.txt:847: Another ruling.",
        }]
        report = gen._extract_provenance(events)
        assert len(report.grep_queries) == 1
        assert report.grep_queries[0]["pattern"] == "Supreme Court"
        assert len(report.grep_queries[0]["files_hit"]) == 2

    def test_empty_events(self):
        report = gen._extract_provenance([])
        assert len(report.tool_provenances) == 0
        assert report.total_content_read_chars == 0

    def test_non_string_result_no_crash(self):
        """tool_result.content can be a list/dict — should not raise AttributeError."""
        events = [{
            "tool": "Read",
            "tool_use_id": "toolu_003",
            "input": {"file_path": "/tmp/doc.txt"},
            "result": [{"type": "text", "text": "line 1"}, {"type": "text", "text": "line 2"}],
        }]
        report = gen._extract_provenance(events)
        assert len(report.tool_provenances) == 1
        assert report.tool_provenances[0].content_length > 0

    def test_none_result_no_crash(self):
        """result=None (incomplete tool call) should not crash."""
        events = [{
            "tool": "Grep",
            "tool_use_id": "toolu_004",
            "input": {"pattern": "test"},
            "result": None,
        }]
        report = gen._extract_provenance(events)
        assert len(report.tool_provenances) == 1
        assert report.tool_provenances[0].content_length == 0


# ---------------------------------------------------------------------------
# _coerce_str
# ---------------------------------------------------------------------------


class TestCoerceStr:
    def test_string_passthrough(self):
        assert gen._coerce_str("hello") == "hello"

    def test_list_to_json(self):
        result = gen._coerce_str([{"type": "text", "text": "hi"}])
        assert '"text"' in result
        assert '"hi"' in result

    def test_dict_to_json(self):
        result = gen._coerce_str({"key": "value"})
        assert '"key"' in result

    def test_none_to_empty(self):
        assert gen._coerce_str(None) == ""

    def test_int_to_str(self):
        assert gen._coerce_str(42) == "42"


# ---------------------------------------------------------------------------
# Context span (provenance-based)
# ---------------------------------------------------------------------------


class TestRunSpan:
    def test_single_read(self):
        report = gen.ProvenanceReport(
            files_accessed=[{"file": "doc_0000.txt", "line_range": (100, 120), "content_length": 500}],
        )
        span = gen._compute_run_span(report)
        assert span == 20  # 120 - 100

    def test_two_reads_same_file(self):
        report = gen.ProvenanceReport(
            files_accessed=[
                {"file": "doc_0000.txt", "line_range": (100, 120), "content_length": 500},
                {"file": "doc_0000.txt", "line_range": (5000, 5020), "content_length": 500},
            ],
        )
        span = gen._compute_run_span(report)
        assert span == 4920  # 5020 - 100

    def test_two_reads_different_files(self):
        report = gen.ProvenanceReport(
            files_accessed=[
                {"file": "doc_0000.txt", "line_range": (100, 200), "content_length": 500},
                {"file": "doc_0001.txt", "line_range": (1, 50), "content_length": 500},
            ],
        )
        span = gen._compute_run_span(report)
        assert span == 100  # max within-file span: 200-100

    def test_no_reads_zero_span(self):
        report = gen.ProvenanceReport()
        span = gen._compute_run_span(report)
        assert span == 0


class TestPairSpan:
    def test_pair_with_evidence_locations(self):
        """Pair-level span uses evidence_locations when available."""
        pair = {
            "evidence_locations": [
                {"file": "doc_0000.txt", "start_line": 100, "end_line": 120},
                {"file": "doc_0000.txt", "start_line": 500, "end_line": 520},
            ],
        }
        span = gen._compute_pair_span(pair)
        assert span == 420  # 520 - 100

    def test_pair_falls_back_to_run_span(self):
        """Without evidence_locations, falls back to run-level provenance."""
        pair = {"question": "test?"}
        provenance = gen.ProvenanceReport(
            files_accessed=[{"file": "doc_0000.txt", "line_range": (100, 200), "content_length": 500}],
        )
        span = gen._compute_pair_span(pair, provenance)
        assert span == 100

    def test_pair_no_data_returns_zero(self):
        pair = {"question": "test?"}
        span = gen._compute_pair_span(pair)
        assert span == 0

    def test_pair_with_partial_evidence_locations(self):
        """evidence_locations with missing start/end should be skipped."""
        pair = {
            "evidence_locations": [
                {"file": "doc_0000.txt", "start_line": 100},  # missing end_line
                {"file": "doc_0000.txt", "start_line": 200, "end_line": 220},
            ],
        }
        span = gen._compute_pair_span(pair)
        assert span == 20  # Only uses the complete location: 220 - 200

    def test_pair_cross_file_evidence(self):
        """Cross-file evidence gives upper-bound span."""
        pair = {
            "evidence_locations": [
                {"file": "doc_0000.txt", "start_line": 100, "end_line": 120},
                {"file": "doc_0001.txt", "start_line": 5000, "end_line": 5020},
            ],
        }
        span = gen._compute_pair_span(pair)
        assert span == 4920  # 5020 - 100


# ---------------------------------------------------------------------------
# Seed context sampling
# ---------------------------------------------------------------------------


class TestSampleSeedContext:
    def test_returns_text_and_filename(self, tmp_path):
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        (corpus / "doc_0000.txt").write_text("A" * 1000)
        rng = gen.random.Random(42)
        ctx, fname = gen._sample_seed_context(str(corpus), rng, max_chars=200)
        assert len(ctx) <= 200
        assert len(ctx) > 0
        assert fname == "doc_0000.txt"

    def test_empty_dir_returns_empty(self, tmp_path):
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        rng = gen.random.Random(42)
        ctx, fname = gen._sample_seed_context(str(corpus), rng)
        assert ctx == ""
        assert fname == ""

    def test_short_file_returns_full_text(self, tmp_path):
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        (corpus / "doc_0000.txt").write_text("Short text.")
        rng = gen.random.Random(42)
        ctx, fname = gen._sample_seed_context(str(corpus), rng, max_chars=500)
        assert ctx == "Short text."
        assert fname == "doc_0000.txt"


# ---------------------------------------------------------------------------
# ClaudeRunResult
# ---------------------------------------------------------------------------


class TestClaudeRunResult:
    def test_default_values(self):
        result = gen.ClaudeRunResult()
        assert result.reply_text == ""
        assert result.reply_json is None
        assert result.tool_events == []
        assert result.errors == []

    def test_fields_settable(self):
        result = gen.ClaudeRunResult(
            reply_text="test",
            reply_json=[{"q": "a"}],
            meta={"model": "sonnet"},
        )
        assert result.reply_text == "test"
        assert result.meta["model"] == "sonnet"
