"""Tests for multi-turn conversation generation (Step 5)."""
from __future__ import annotations

import json
import os
import sys

import pytest
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "qa_generation"))

from generate_multiturn import (
    _format_entities,
    _format_evidence,
    _parse_conversation_history,
    _validate_conversation_history,
)

# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------

QA_DIR = os.path.join(os.path.dirname(__file__), "..", "qa_generation")
CATEGORIES_CFG = os.path.join(QA_DIR, "qa_config", "categories.yaml")
CATEGORIES_ENRON_CFG = os.path.join(QA_DIR, "qa_config", "categories_enron.yaml")


# ---------------------------------------------------------------------------
# Test parse_conversation_history
# ---------------------------------------------------------------------------


class TestParseConversationHistory:
    def test_plain_json_array(self):
        raw = json.dumps([
            {"turn_index": 1, "user": "Hello", "assistant": "Hi there, how can I help?"},
        ])
        result = _parse_conversation_history(raw)
        assert result is not None
        assert len(result) == 1
        assert result[0]["user"] == "Hello"

    def test_fenced_json(self):
        raw = '```json\n[{"turn_index": 1, "user": "Hello", "assistant": "Hi there, how can I help?"}]\n```'
        result = _parse_conversation_history(raw)
        assert result is not None
        assert len(result) == 1

    def test_wrapper_object(self):
        raw = json.dumps({
            "conversation_history": [
                {"turn_index": 1, "user": "Hello", "assistant": "Hi there, how can I help?"},
            ]
        })
        result = _parse_conversation_history(raw)
        assert result is not None
        assert len(result) == 1

    def test_garbage_returns_none(self):
        result = _parse_conversation_history("This is not JSON at all.")
        assert result is None

    def test_empty_array(self):
        result = _parse_conversation_history("[]")
        assert result is not None
        assert result == []

    def test_json_with_preamble(self):
        raw = 'Here is the conversation:\n[{"turn_index": 1, "user": "Question?", "assistant": "Answer about the topic."}]'
        result = _parse_conversation_history(raw)
        assert result is not None
        assert len(result) == 1


# ---------------------------------------------------------------------------
# Test validate_conversation_history
# ---------------------------------------------------------------------------


class TestValidateConversationHistory:
    def _make_turns(self, n):
        return [
            {
                "turn_index": i + 1,
                "user": f"This is user message number {i + 1}?",
                "assistant": f"This is a helpful assistant response for turn {i + 1}.",
            }
            for i in range(n)
        ]

    def test_valid_history(self):
        ok, reason = _validate_conversation_history(self._make_turns(4), 4)
        assert ok, reason

    def test_wrong_turn_count(self):
        ok, reason = _validate_conversation_history(self._make_turns(3), 4)
        assert not ok
        assert "expected 4" in reason

    def test_missing_user_field(self):
        turns = [{"assistant": "Response about the topic here."}]
        ok, reason = _validate_conversation_history(turns, 1)
        assert not ok
        assert "missing" in reason

    def test_missing_assistant_field(self):
        turns = [{"user": "This is a user question about the topic?"}]
        ok, reason = _validate_conversation_history(turns, 1)
        assert not ok
        assert "missing" in reason

    def test_empty_user_message(self):
        turns = [{"turn_index": 1, "user": "", "assistant": "This is a response about the topic."}]
        ok, reason = _validate_conversation_history(turns, 1)
        assert not ok
        assert "empty" in reason

    def test_empty_assistant_message(self):
        turns = [{"turn_index": 1, "user": "This is a question about the topic?", "assistant": ""}]
        ok, reason = _validate_conversation_history(turns, 1)
        assert not ok
        assert "empty" in reason

    def test_short_user_message(self):
        turns = [{"turn_index": 1, "user": "Hi?", "assistant": "This is a helpful response about the topic."}]
        ok, reason = _validate_conversation_history(turns, 1)
        assert not ok
        assert "too short" in reason

    def test_short_assistant_message(self):
        turns = [{"turn_index": 1, "user": "Tell me about the Smith case?", "assistant": "Short."}]
        ok, reason = _validate_conversation_history(turns, 1)
        assert not ok
        assert "too short" in reason

    def test_zero_expected_passes(self):
        ok, reason = _validate_conversation_history([], 0)
        assert ok

    def test_not_a_list(self):
        ok, reason = _validate_conversation_history("not a list", 1)
        assert not ok
        assert "not a list" in reason

    def test_turn_not_a_dict(self):
        ok, reason = _validate_conversation_history(["not a dict"], 1)
        assert not ok
        assert "not a dict" in reason


# ---------------------------------------------------------------------------
# Test format helpers
# ---------------------------------------------------------------------------


class TestFormatEvidence:
    def test_formats_hop_path(self):
        chain = {
            "hop_path": [
                {"hop_index": 0, "chunk_text": "First evidence snippet."},
                {"hop_index": 1, "chunk_text": "Second evidence snippet."},
            ]
        }
        result = _format_evidence(chain)
        assert "[Evidence 0]" in result
        assert "[Evidence 1]" in result
        assert "First evidence snippet." in result

    def test_empty_hop_path(self):
        result = _format_evidence({"hop_path": []})
        assert "no evidence" in result

    def test_fallback_to_evidence_snippets(self):
        chain = {
            "hop_path": [],
            "evidence_snippets": ["Fallback snippet here."],
        }
        result = _format_evidence(chain)
        assert "Fallback snippet" in result


class TestFormatEntities:
    def test_formats_entity_list(self):
        chain = {
            "entities": [
                {"label": "Entity A", "description": "Description of A"},
                {"label": "Entity B", "description": "Description of B"},
            ]
        }
        result = _format_entities(chain)
        assert "Entity A" in result
        assert "Description of B" in result

    def test_empty_entities(self):
        result = _format_entities({"entities": []})
        assert "no entities" in result


# ---------------------------------------------------------------------------
# Test category config num_turns
# ---------------------------------------------------------------------------


class TestCategoryNumTurns:
    def test_categories_yaml_has_num_turns(self):
        with open(CATEGORIES_CFG) as f:
            cfg = yaml.safe_load(f)
        for cat in cfg["categories"]:
            assert "num_turns" in cat, f"Missing num_turns in {cat['name']}"
            assert isinstance(cat["num_turns"], int)
            assert cat["num_turns"] >= 1

    def test_categories_enron_yaml_has_num_turns(self):
        with open(CATEGORIES_ENRON_CFG) as f:
            cfg = yaml.safe_load(f)
        for cat in cfg["categories"]:
            assert "num_turns" in cat, f"Missing num_turns in {cat['name']}"
            assert isinstance(cat["num_turns"], int)
            assert cat["num_turns"] >= 1

    def test_multiturn_scenario_present_when_needed(self):
        for cfg_path in (CATEGORIES_CFG, CATEGORIES_ENRON_CFG):
            with open(cfg_path) as f:
                cfg = yaml.safe_load(f)
            for cat in cfg["categories"]:
                if cat.get("num_turns", 1) > 1:
                    assert "multiturn_scenario" in cat, (
                        f"Missing multiturn_scenario in {cat['name']} "
                        f"(num_turns={cat['num_turns']}) in {cfg_path}"
                    )
                    assert len(cat["multiturn_scenario"].strip()) > 20

    def test_num_turns_range(self):
        """All num_turns should be between 1 and 10."""
        for cfg_path in (CATEGORIES_CFG, CATEGORIES_ENRON_CFG):
            with open(cfg_path) as f:
                cfg = yaml.safe_load(f)
            for cat in cfg["categories"]:
                assert 1 <= cat["num_turns"] <= 10, (
                    f"num_turns={cat['num_turns']} out of range in {cat['name']}"
                )
