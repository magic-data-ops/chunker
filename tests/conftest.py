"""Shared fixtures and helpers for tests."""
from __future__ import annotations

import json
import os
import tempfile

import pytest


# ---------------------------------------------------------------------------
# Sample data fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_category():
    return {
        "name": "long_context_citation",
        "display_name": "Retrieves information in a long context source and correctly cites its source",
        "description": "Simple factual retrieval with citation.",
        "min_hops": 1,
        "max_hops": 5,
        "domain_scope": "test_corpus",
    }


@pytest.fixture
def multi_hop_category():
    return {
        "name": "multi_hop_reasoning",
        "display_name": "Performs multi-hop reasoning across documents that are widely separated in context",
        "description": "Multi-hop reasoning questions.",
        "min_hops": 2,
        "max_hops": 10,
        "domain_scope": "test_corpus",
    }


@pytest.fixture
def sample_pair():
    return {
        "question": "What year was the company founded?",
        "golden_answer": "The company was founded in 1927 and initially manufactured agricultural equipment.",
        "evidence_snippets": [
            "Founded in 1927, the company initially manufactured agricultural equipment before pivoting to electronics in 1965."
        ],
        "source_files": ["doc_0000_history.txt"],
    }


@pytest.fixture
def sample_chain():
    return {
        "chain_id": "test-chain-001",
        "category": "long_context_citation",
        "source_file": "doc_0000_history.txt",
        "prompt_seed_file": "doc_0000_history.txt",
        "question": "What year was the company founded?",
        "final_answer": "The company was founded in 1927.",
        "hop_path": [
            {
                "hop_index": 0,
                "chunk_id": "doc_0000_chunk_003",
                "chunk_text": "Founded in 1927, the company initially manufactured agricultural equipment.",
                "partial_answer": "Founded in 1927",
                "retrieval_score": 0.85,
            }
        ],
        "hop_count": 1,
        "termination_reason": "agent_complete",
        "approved": True,
        "category_suitability_score": 0.9,
        "answer_completeness_score": 0.85,
    }
