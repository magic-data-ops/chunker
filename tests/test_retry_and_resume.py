"""Tests for retry logic and resume behavior in generate_qa_chains.py."""
from __future__ import annotations

import asyncio
import json
import random
from collections import Counter
from unittest.mock import AsyncMock, patch

import pytest
from qa_generation import generate_qa_chains as gen


def _make_valid_pair(index: int = 0) -> dict:
    return {
        "question": f"What happened in event {index}?",
        "golden_answer": f"Event {index} occurred during the industrial revolution and had lasting impacts on policy.",
        "evidence_snippets": [
            f"Event {index} occurred during the industrial revolution and had lasting impacts on policy and trade."
        ],
        "source_files": ["doc_0000_history.txt"],
    }


def _make_invalid_pair() -> dict:
    return {
        "question": "What?",
        "golden_answer": "Yes",  # too short
        "evidence_snippets": ["x"],  # too short
        "source_files": ["doc_0000_history.txt"],
    }


def _make_claude_run_result(reply_text: str = "", reply_json=None, errors=None):
    """Helper to create a ClaudeRunResult for mocking."""
    return gen.ClaudeRunResult(
        reply_text=reply_text,
        reply_json=reply_json,
        tool_events=[],
        raw_stdout="",
        meta={},
        errors=errors or [],
    )


@pytest.fixture
def factoid_category():
    return {
        "name": "single_chunk_factoid",
        "description": "Simple factual retrieval.",
        "min_hops": 1,
        "max_hops": 1,
    }


@pytest.fixture
def tmp_corpus(tmp_path):
    """Create a minimal corpus_text directory with one .txt file."""
    corpus_dir = tmp_path / "corpus_text"
    corpus_dir.mkdir()
    (corpus_dir / "doc_0000_history.txt").write_text("Some corpus text content here.")
    return str(corpus_dir)


class TestNoProgressGuard:
    def test_breaks_after_three_consecutive_zero_progress(self, factoid_category, tmp_corpus):
        """consecutive_zero_progress = 3 should break the loop."""
        # Agent always returns parseable but invalid pairs
        invalid_response = json.dumps([_make_invalid_pair()])

        call_count = 0

        async def mock_run(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return _make_claude_run_result(
                reply_text=invalid_response,
                reply_json=[_make_invalid_pair()],
            )

        with patch.object(gen, "_run_claude_code", side_effect=mock_run):
            chains = asyncio.get_event_loop().run_until_complete(
                gen.generate_for_category(
                    category=factoid_category,
                    corpus_text_dir=tmp_corpus,
                    n_chains=10,
                    claude_bin="claude",
                    model="sonnet",
                    max_budget_usd=0.50,
                    batch_size=2,
                    gen_template="{{CATEGORY_NAME}} {{N_PAIRS}} {{FILE_LIST}} {{SEED_CONTEXT}} {{SEED_ENTITY}} {{CATEGORY_DESCRIPTION}}",
                    max_retries=2,
                )
            )
        assert len(chains) == 0
        # Should have stopped after 3 consecutive zero-progress batches
        assert call_count == 3


class TestRetryOnParseFailure:
    def test_retries_on_parse_failure_then_succeeds(self, factoid_category, tmp_corpus):
        """Parse failure should retry, then succeed on a valid response."""
        valid_pairs = [_make_valid_pair()]
        valid_response = json.dumps(valid_pairs)
        responses = [
            _make_claude_run_result(reply_text="not json garbage"),
            _make_claude_run_result(reply_text="still not json"),
            _make_claude_run_result(reply_text=valid_response, reply_json=valid_pairs),
        ]
        call_idx = 0

        async def mock_run(*args, **kwargs):
            nonlocal call_idx
            resp = responses[min(call_idx, len(responses) - 1)]
            call_idx += 1
            return resp

        with patch.object(gen, "_run_claude_code", side_effect=mock_run):
            chains = asyncio.get_event_loop().run_until_complete(
                gen.generate_for_category(
                    category=factoid_category,
                    corpus_text_dir=tmp_corpus,
                    n_chains=1,
                    claude_bin="claude",
                    model="sonnet",
                    max_budget_usd=0.50,
                    batch_size=1,
                    gen_template="{{CATEGORY_NAME}} {{N_PAIRS}} {{FILE_LIST}} {{SEED_CONTEXT}} {{SEED_ENTITY}} {{CATEGORY_DESCRIPTION}}",
                    max_retries=3,
                )
            )
        assert len(chains) == 1


class TestRemainingDecrementedByValidCount:
    def test_remaining_decremented_by_valid_not_batch(self, factoid_category, tmp_corpus):
        """Only valid pairs should decrement remaining, not total parsed pairs."""
        # Return 3 pairs: 1 valid + 2 invalid
        mixed_pairs = [
            _make_valid_pair(0),
            _make_invalid_pair(),
            _make_invalid_pair(),
        ]
        mixed_response = json.dumps(mixed_pairs)

        async def mock_run(*args, **kwargs):
            return _make_claude_run_result(
                reply_text=mixed_response,
                reply_json=mixed_pairs,
            )

        with patch.object(gen, "_run_claude_code", side_effect=mock_run):
            chains = asyncio.get_event_loop().run_until_complete(
                gen.generate_for_category(
                    category=factoid_category,
                    corpus_text_dir=tmp_corpus,
                    n_chains=1,
                    claude_bin="claude",
                    model="sonnet",
                    max_budget_usd=0.50,
                    batch_size=3,
                    gen_template="{{CATEGORY_NAME}} {{N_PAIRS}} {{FILE_LIST}} {{SEED_CONTEXT}} {{SEED_ENTITY}} {{CATEGORY_DESCRIPTION}}",
                    max_retries=2,
                )
            )
        assert len(chains) == 1


class TestResumePerCategoryCounts:
    def test_partial_category_not_skipped(self, tmp_path):
        """Bug 5: a category with some chains should not be skipped entirely."""
        # Simulate existing output with 2 chains for a category that needs 5
        existing = [
            {
                "chain_id": f"existing-{i}",
                "category": "single_chunk_factoid",
                "question": f"Q{i}?",
                "final_answer": f"A{i}",
                "hop_path": [],
                "hop_count": 1,
            }
            for i in range(2)
        ]
        per_category = 5

        cat_counts = Counter(c["category"] for c in existing)
        categories = [
            {"name": "single_chunk_factoid", "description": "test", "min_hops": 1, "max_hops": 1},
            {"name": "multi_hop_reasoning", "description": "test", "min_hops": 2, "max_hops": 4},
        ]

        # Filter as the real code does
        rank_categories = [c for c in categories
                           if cat_counts.get(c["name"], 0) < per_category]

        # single_chunk_factoid has 2 < 5, so it should NOT be filtered out
        names = [c["name"] for c in rank_categories]
        assert "single_chunk_factoid" in names
        assert "multi_hop_reasoning" in names

        # Remaining for single_chunk_factoid should be 3
        remaining = per_category - cat_counts.get("single_chunk_factoid", 0)
        assert remaining == 3


class TestTimeoutHandling:
    def test_timeout_error_adds_to_errors(self, factoid_category, tmp_corpus):
        """Timeout should result in error and move to next batch."""
        call_count = 0

        async def mock_run(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return _make_claude_run_result(errors=["timeout"])

        with patch.object(gen, "_run_claude_code", side_effect=mock_run):
            chains = asyncio.get_event_loop().run_until_complete(
                gen.generate_for_category(
                    category=factoid_category,
                    corpus_text_dir=tmp_corpus,
                    n_chains=2,
                    claude_bin="claude",
                    model="sonnet",
                    max_budget_usd=0.50,
                    batch_size=1,
                    gen_template="{{CATEGORY_NAME}} {{N_PAIRS}} {{FILE_LIST}} {{SEED_CONTEXT}} {{SEED_ENTITY}} {{CATEGORY_DESCRIPTION}}",
                    max_retries=1,
                )
            )
        assert len(chains) == 0
