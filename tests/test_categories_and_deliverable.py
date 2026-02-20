"""Tests for category config, deliverable export, CSV export, and grouped schema."""
from __future__ import annotations

import csv
import json
import os
import uuid

import pytest
import yaml

from qa_generation import generate_qa_chains as gen

CATEGORIES_CFG = os.path.join(
    os.path.dirname(__file__), "..", "qa_generation", "qa_config", "categories.yaml"
)
CATEGORIES_ENRON_CFG = os.path.join(
    os.path.dirname(__file__), "..", "qa_generation", "qa_config", "categories_enron.yaml"
)


# ---------------------------------------------------------------------------
# Category config tests
# ---------------------------------------------------------------------------


class TestCategoryConfig:
    @pytest.fixture
    def categories(self):
        with open(CATEGORIES_CFG) as f:
            cfg = yaml.safe_load(f)
        return cfg["categories"]

    def test_twelve_categories_exist(self, categories):
        assert len(categories) == 12

    def test_every_category_has_required_fields(self, categories):
        for cat in categories:
            assert "name" in cat, f"Missing 'name' in {cat}"
            assert "display_name" in cat, f"Missing 'display_name' in {cat.get('name')}"
            assert "description" in cat, f"Missing 'description' in {cat.get('name')}"
            assert "domain_scope" in cat, f"Missing 'domain_scope' in {cat.get('name')}"
            assert "min_hops" in cat, f"Missing 'min_hops' in {cat.get('name')}"
            assert "max_hops" in cat, f"Missing 'max_hops' in {cat.get('name')}"

    def test_all_domain_scopes_consistent(self, categories):
        scopes = {cat["domain_scope"] for cat in categories}
        assert len(scopes) == 1, f"Multiple domain_scopes found: {scopes}"

    def test_slugs_are_unique(self, categories):
        names = [c["name"] for c in categories]
        assert len(names) == len(set(names)), f"Duplicate slugs: {names}"

    def test_slugs_are_valid_identifiers(self, categories):
        for cat in categories:
            assert cat["name"].replace("_", "").isalnum(), \
                f"Slug '{cat['name']}' is not a valid identifier"

    def test_hop_ranges_valid(self, categories):
        for cat in categories:
            assert cat["min_hops"] >= 1
            assert cat["max_hops"] >= cat["min_hops"]


# ---------------------------------------------------------------------------
# Deliverable sample conversion
# ---------------------------------------------------------------------------


def _make_chain(category="entity_disambiguation", n_hops=2) -> dict:
    """Build a minimal internal chain for testing."""
    return {
        "chain_id": str(uuid.uuid4()),
        "category": category,
        "source_file": "corpus_doc.txt",
        "prompt_seed_file": "corpus_doc.txt",
        "question": "What is the difference between the two uses of Smith?",
        "final_answer": "Smith in case A is a plaintiff; Smith in case B is a defendant.",
        "hop_path": [
            {
                "hop_index": i,
                "chunk_id": f"corpus_doc.txt:evidence_{i}",
                "chunk_text": f"Evidence snippet {i} with sufficient length for validation here.",
                "partial_answer": f"Partial answer for hop {i} that is long enough.",
                "retrieval_score": None,
            }
            for i in range(n_hops)
        ],
        "hop_count": n_hops,
        "evidence_locations": [
            {"file": "corpus_doc.txt", "start_line": 100 + i * 1000, "end_line": 150 + i * 1000}
            for i in range(n_hops)
        ],
        "entities": [
            {
                "label": f"Smith — Case {'AB'[i]}",
                "description": f"Smith in case {'AB'[i]}",
                "evidence_snippet": f"Evidence snippet {i} with sufficient length for validation here.",
                "evidence_location": {"file": "corpus_doc.txt", "start_line": 100 + i * 1000, "end_line": 150 + i * 1000},
            }
            for i in range(n_hops)
        ],
    }


class TestChainToDeliverableSample:
    def test_maps_required_fields(self):
        chain = _make_chain()
        sample = gen._chain_to_deliverable_sample(chain)
        assert "relevant_context" in sample
        assert "context_location_in_file" in sample
        assert "suggested_prompt" in sample
        assert "golden_response" in sample

    def test_relevant_context_from_hop_path(self):
        chain = _make_chain(n_hops=2)
        sample = gen._chain_to_deliverable_sample(chain)
        assert "Evidence snippet 0" in sample["relevant_context"]
        assert "Evidence snippet 1" in sample["relevant_context"]

    def test_suggested_prompt_is_question(self):
        chain = _make_chain()
        sample = gen._chain_to_deliverable_sample(chain)
        assert sample["suggested_prompt"] == chain["question"]

    def test_golden_response_is_final_answer(self):
        chain = _make_chain()
        sample = gen._chain_to_deliverable_sample(chain)
        assert sample["golden_response"] == chain["final_answer"]

    def test_locations_from_evidence_locations(self):
        chain = _make_chain(n_hops=2)
        sample = gen._chain_to_deliverable_sample(chain)
        assert len(sample["context_location_in_file"]) == 2
        assert sample["context_location_in_file"][0]["file"] == "corpus_doc.txt"

    def test_locations_fallback_to_entities(self):
        chain = _make_chain(n_hops=2)
        del chain["evidence_locations"]
        sample = gen._chain_to_deliverable_sample(chain)
        assert len(sample["context_location_in_file"]) == 2


# ---------------------------------------------------------------------------
# Deliverable sample validation
# ---------------------------------------------------------------------------


class TestValidateDeliverableSample:
    def test_valid_sample_passes(self):
        sample = {
            "relevant_context": "Some context text here.",
            "context_location_in_file": [{"file": "test.txt", "start_line": 1, "end_line": 10}],
            "suggested_prompt": "What happened?",
            "golden_response": "The court ruled in favor of the plaintiff.",
        }
        ok, reason = gen._validate_deliverable_sample(sample)
        assert ok

    def test_empty_context_fails(self):
        sample = {
            "relevant_context": "",
            "context_location_in_file": [{"file": "test.txt", "start_line": 1, "end_line": 10}],
            "suggested_prompt": "What?",
            "golden_response": "Answer.",
        }
        ok, reason = gen._validate_deliverable_sample(sample)
        assert not ok
        assert "relevant_context" in reason

    def test_no_location_fails(self):
        sample = {
            "relevant_context": "Some text.",
            "context_location_in_file": [],
            "suggested_prompt": "What?",
            "golden_response": "Answer.",
        }
        ok, reason = gen._validate_deliverable_sample(sample)
        assert not ok
        assert "location" in reason

    def test_empty_prompt_fails(self):
        sample = {
            "relevant_context": "Some text.",
            "context_location_in_file": [{"file": "test.txt", "start_line": 1, "end_line": 10}],
            "suggested_prompt": "",
            "golden_response": "Answer.",
        }
        ok, reason = gen._validate_deliverable_sample(sample)
        assert not ok
        assert "suggested_prompt" in reason

    def test_empty_golden_response_fails(self):
        sample = {
            "relevant_context": "Some text.",
            "context_location_in_file": [{"file": "test.txt", "start_line": 1, "end_line": 10}],
            "suggested_prompt": "What?",
            "golden_response": "",
        }
        ok, reason = gen._validate_deliverable_sample(sample)
        assert not ok
        assert "golden_response" in reason


# ---------------------------------------------------------------------------
# Grouped deliverable builder
# ---------------------------------------------------------------------------


class TestBuildGroupedDeliverable:
    @pytest.fixture
    def two_categories(self):
        return [
            {
                "name": "entity_disambiguation",
                "display_name": "Disambiguates distinct entities",
                "description": "test",
                "min_hops": 1,
                "max_hops": 10,
                "domain_scope": "test_corpus",
            },
            {
                "name": "multi_hop_reasoning",
                "display_name": "Performs multi-hop reasoning",
                "description": "test",
                "min_hops": 2,
                "max_hops": 10,
                "domain_scope": "test_corpus",
            },
        ]

    def test_groups_by_category(self, two_categories):
        chains = [
            _make_chain("entity_disambiguation") for _ in range(3)
        ] + [
            _make_chain("multi_hop_reasoning") for _ in range(3)
        ]
        deliverable, errors = gen._build_grouped_deliverable(chains, two_categories, 3)
        assert len(errors) == 0
        assert len(deliverable["categories"]) == 2
        for cat_block in deliverable["categories"]:
            assert len(cat_block["samples"]) == 3

    def test_has_metadata(self, two_categories):
        chains = [_make_chain("entity_disambiguation") for _ in range(3)] + \
                 [_make_chain("multi_hop_reasoning") for _ in range(3)]
        deliverable, _ = gen._build_grouped_deliverable(chains, two_categories, 3)
        assert "generated_at" in deliverable
        assert "domain_scope" in deliverable

    def test_category_display_name_preserved(self, two_categories):
        chains = [_make_chain("entity_disambiguation") for _ in range(3)] + \
                 [_make_chain("multi_hop_reasoning") for _ in range(3)]
        deliverable, _ = gen._build_grouped_deliverable(chains, two_categories, 3)
        display_names = {c["category_display_name"] for c in deliverable["categories"]}
        assert "Disambiguates distinct entities" in display_names
        assert "Performs multi-hop reasoning" in display_names

    def test_insufficient_samples_errors(self, two_categories):
        # Only 2 chains for entity_disambiguation, need 3
        chains = [_make_chain("entity_disambiguation") for _ in range(2)] + \
                 [_make_chain("multi_hop_reasoning") for _ in range(3)]
        deliverable, errors = gen._build_grouped_deliverable(chains, two_categories, 3)
        assert len(errors) == 1
        assert "entity_disambiguation" in errors[0]
        # Only multi_hop should appear
        assert len(deliverable["categories"]) == 1

    def test_zero_chains_all_errors(self, two_categories):
        deliverable, errors = gen._build_grouped_deliverable([], two_categories, 3)
        assert len(errors) == 2
        assert len(deliverable["categories"]) == 0

    def test_samples_contain_only_required_fields(self, two_categories):
        chains = [_make_chain("entity_disambiguation") for _ in range(3)] + \
                 [_make_chain("multi_hop_reasoning") for _ in range(3)]
        deliverable, _ = gen._build_grouped_deliverable(chains, two_categories, 3)
        required_keys = {"relevant_context", "context_location_in_file", "suggested_prompt", "golden_response"}
        for cat_block in deliverable["categories"]:
            for sample in cat_block["samples"]:
                assert set(sample.keys()) == required_keys


# ---------------------------------------------------------------------------
# CSV deliverable export
# ---------------------------------------------------------------------------


class TestExportDeliverableCsv:
    @pytest.fixture
    def deliverable(self):
        """A minimal deliverable dict with 2 categories, 2 samples each."""
        return {
            "generated_at": "2026-01-01T00:00:00+00:00",
            "domain_scope": "test_corpus",
            "categories": [
                {
                    "category_id": "entity_disambiguation",
                    "category_display_name": "Disambiguates distinct entities",
                    "samples": [
                        {
                            "relevant_context": "Context A1.",
                            "context_location_in_file": [
                                {"file": "cases.txt", "start_line": 10, "end_line": 20},
                                {"file": "cases.txt", "start_line": 30, "end_line": 40},
                            ],
                            "suggested_prompt": "Question A1?",
                            "golden_response": "Answer A1.",
                        },
                        {
                            "relevant_context": "Context A2.",
                            "context_location_in_file": [
                                {"file": "cases.txt", "start_line": 50, "end_line": 60},
                            ],
                            "suggested_prompt": "Question A2?",
                            "golden_response": "Answer A2.",
                        },
                    ],
                },
                {
                    "category_id": "multi_hop_reasoning",
                    "category_display_name": "Performs multi-hop reasoning",
                    "samples": [
                        {
                            "relevant_context": "Context B1.",
                            "context_location_in_file": [
                                {"file": "cases.txt", "start_line": 100, "end_line": 200},
                            ],
                            "suggested_prompt": "Question B1?",
                            "golden_response": "Answer B1.",
                        },
                    ],
                },
            ],
        }

    def test_csv_has_correct_header(self, deliverable, tmp_path):
        csv_path = str(tmp_path / "test.csv")
        gen._export_deliverable_csv(deliverable, csv_path)
        with open(csv_path, encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
        assert header == [
            "description",
            "relevant_context",
            "context_location_in_file",
            "template_question",
            "golden_response",
        ]

    def test_csv_row_count(self, deliverable, tmp_path):
        csv_path = str(tmp_path / "test.csv")
        gen._export_deliverable_csv(deliverable, csv_path)
        with open(csv_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 3  # 2 + 1 samples

    def test_description_is_display_name(self, deliverable, tmp_path):
        csv_path = str(tmp_path / "test.csv")
        gen._export_deliverable_csv(deliverable, csv_path)
        with open(csv_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert rows[0]["description"] == "Disambiguates distinct entities"
        assert rows[1]["description"] == "Disambiguates distinct entities"
        assert rows[2]["description"] == "Performs multi-hop reasoning"

    def test_context_location_is_valid_json(self, deliverable, tmp_path):
        csv_path = str(tmp_path / "test.csv")
        gen._export_deliverable_csv(deliverable, csv_path)
        with open(csv_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                locs = json.loads(row["context_location_in_file"])
                assert isinstance(locs, list)
                for loc in locs:
                    assert "file" in loc
                    assert "start_line" in loc
                    assert "end_line" in loc

    def test_fields_mapped_correctly(self, deliverable, tmp_path):
        csv_path = str(tmp_path / "test.csv")
        gen._export_deliverable_csv(deliverable, csv_path)
        with open(csv_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            row = next(reader)
        assert row["relevant_context"] == "Context A1."
        assert row["template_question"] == "Question A1?"
        assert row["golden_response"] == "Answer A1."

    def test_atomic_write(self, deliverable, tmp_path):
        """No .tmp file left behind after successful write."""
        csv_path = str(tmp_path / "out.csv")
        gen._export_deliverable_csv(deliverable, csv_path)
        assert os.path.exists(csv_path)
        assert not os.path.exists(csv_path + ".tmp")


# ---------------------------------------------------------------------------
# Enron category config tests
# ---------------------------------------------------------------------------


class TestEnronCategoryConfig:
    @pytest.fixture
    def categories(self):
        with open(CATEGORIES_ENRON_CFG) as f:
            cfg = yaml.safe_load(f)
        return cfg["categories"]

    def test_eleven_categories_exist(self, categories):
        assert len(categories) == 11

    def test_no_cross_context_synthesis(self, categories):
        names = [c["name"] for c in categories]
        assert "cross_context_synthesis" not in names

    def test_every_category_has_required_fields(self, categories):
        for cat in categories:
            assert "name" in cat
            assert "display_name" in cat
            assert "description" in cat
            assert "domain_scope" in cat
            assert "min_hops" in cat
            assert "max_hops" in cat

    def test_all_domain_scopes_enron(self, categories):
        for cat in categories:
            assert cat["domain_scope"] == "enron_email_corpus"

    def test_slugs_are_unique(self, categories):
        names = [c["name"] for c in categories]
        assert len(names) == len(set(names))


# ---------------------------------------------------------------------------
# Corpus stem derivation
# ---------------------------------------------------------------------------


class TestCorpusStem:
    def test_single_file_uses_stem(self, tmp_path):
        (tmp_path / "enron_complete.txt").write_text("data")
        assert gen._corpus_stem(str(tmp_path)) == "enron_complete"

    def test_multiple_files_uses_dir_name(self, tmp_path):
        (tmp_path / "part1.txt").write_text("a")
        (tmp_path / "part2.txt").write_text("b")
        assert gen._corpus_stem(str(tmp_path)) == tmp_path.name

    def test_no_files_uses_dir_name(self, tmp_path):
        assert gen._corpus_stem(str(tmp_path)) == tmp_path.name
