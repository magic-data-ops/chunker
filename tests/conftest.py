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


@pytest.fixture
def sample_multiturn_chain():
    return {
        "chain_id": "test-chain-mt-001",
        "category": "entity_disambiguation",
        "source_file": "corpus_doc.txt",
        "prompt_seed_file": "corpus_doc.txt",
        "question": "How do the two Smith entities differ in this corpus?",
        "final_answer": "John Smith was a plaintiff in a property case, while Robert Smith was a defendant in a criminal case.",
        "hop_path": [
            {
                "hop_index": 0,
                "chunk_id": "corpus_doc.txt:evidence_0",
                "chunk_text": "John Smith filed a property dispute claim against the county in 2015.",
                "partial_answer": "John Smith was a plaintiff in a property case.",
                "retrieval_score": None,
            },
            {
                "hop_index": 1,
                "chunk_id": "corpus_doc.txt:evidence_1",
                "chunk_text": "Robert Smith was charged with grand theft in People v. Smith, 2018.",
                "partial_answer": "Robert Smith was a defendant in a criminal case.",
                "retrieval_score": None,
            },
        ],
        "hop_count": 2,
        "termination_reason": "agent_complete",
        "evidence_locations": [
            {"file": "corpus_doc.txt", "start_line": 100, "end_line": 120},
            {"file": "corpus_doc.txt", "start_line": 500, "end_line": 520},
        ],
        "difficulty": "medium",
        "entities": [
            {
                "label": "John Smith — Property Plaintiff",
                "description": "Plaintiff in a property dispute against the county",
                "evidence_snippet": "John Smith filed a property dispute claim against the county in 2015.",
                "evidence_location": {"file": "corpus_doc.txt", "start_line": 100, "end_line": 120},
            },
            {
                "label": "Robert Smith — Criminal Defendant",
                "description": "Defendant charged with grand theft",
                "evidence_snippet": "Robert Smith was charged with grand theft in People v. Smith, 2018.",
                "evidence_location": {"file": "corpus_doc.txt", "start_line": 500, "end_line": 520},
            },
        ],
        "disambiguation_statement": "Two different individuals named Smith in different legal contexts.",
        "num_turns": 3,
        "conversation_history": [
            {
                "turn_index": 1,
                "user": "I see references to someone named Smith in the corpus. Can you tell me about the property case?",
                "assistant": "Yes, John Smith filed a property dispute claim against the county in 2015. He was the plaintiff seeking to resolve a boundary issue.",
            },
            {
                "turn_index": 2,
                "user": "Interesting. I also noticed a Smith in a criminal case. What can you tell me about that?",
                "assistant": "That would be a different person — Robert Smith was charged with grand theft in People v. Smith in 2018. He was the defendant in that criminal proceeding.",
            },
        ],
    }
