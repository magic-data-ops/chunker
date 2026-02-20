# chunker

Automated pipeline that uses Claude Code CLI to generate scored Q&A evaluation metadata from document corpora. No vector DB -- the agent searches plain-text files directly.

## Workflow

**1. Index (PDF only)** -- Extract text from PDF documents into `corpus_text/` via PyMuPDF. Skip this step if your data is already `.txt` or `.md` -- place those files directly in `corpus_text/`.
```bash
python qa_generation/build_corpus_index.py --input_dir ./docs --output_dir ./corpus_index
```

**2. Generate** -- For each evaluation category, spawn a Claude Code agent that autonomously searches the corpus (Grep/Read/Glob + parallel sub-agents), then returns structured Q&A chains with questions, golden answers, verbatim evidence snippets with file+line locations, entity metadata, and difficulty ratings. Each pair is validated (format, length, hop-count bounds) and saved atomically with full provenance logs. Runs in parallel across categories; supports multi-machine sharding.
```bash
python qa_generation/generate_qa_chains.py \
  --corpus_text_dir ./corpus_text \
  --output qa_chains_raw.json \
  --samples-per-category 3 \
  --model sonnet \
  --max-budget-usd 1.00
```

**3. Validate** -- A second Claude agent scores every chain on two dimensions: **category suitability** and **answer completeness** (both 0.0--1.0). Chains with both scores >= 0.7 are approved; rejected chains get a written reason. The validator also produces a polished answer. Runs up to 32 validations in parallel.
```bash
python qa_generation/contractor_polish.py \
  --input qa_chains_raw.json \
  --output qa_chains_validated.json
```

**4. Report** -- Aggregate scores and check four dataset-level quality targets (all must be >= 80%): overall approval rate, mean category suitability, mean answer completeness, and threshold pass rate (both scores >= 0.8). Broken down per category with hop-count distribution.
```bash
python qa_generation/validate_qa_dataset.py \
  --input qa_chains_validated.json \
  --output validation_report.json
```

## Corpora

- **California state case law** -- 12 categories
- **Enron email corpus** -- 11 categories

Categories include entity disambiguation, multi-hop reasoning, temporal ordering, numerical aggregation, hierarchy comprehension, semantic deduplication, and others.

## Outputs

- `qa_chains_validated.json` -- scored chains with evidence, entities, and approval status
- `qa_deliverable_grouped.json` + `.csv` -- grouped deliverable with context, prompts, and golden responses
- `validation_report.json` -- per-category metrics and target pass/fail

## Usage

```bash
# Interactive TUI -- walks through configuration and runs selected steps
./run.sh
```

The TUI offers five modes:

| Mode | Steps run |
|------|-----------|
| Full pipeline | Index -> Generate -> Validate -> Report |
| Generate + validate + report | Skip indexing (corpus already built) |
| Just generate | Only produce raw Q&A chains |
| Just validate | Only score existing chains |
| Just report | Only aggregate metrics |

It prompts for corpus path, samples per category, model (sonnet/opus/haiku), budget per call, and optional advanced settings (batch size, concurrency, custom categories config, custom prompt template). All outputs are auto-named with the corpus stem and date (e.g. `enron_complete_20260220_qa_chains_raw.json`).

## Properties

Resumable, concurrent, distributable (sharding), fault-tolerant (atomic saves, retries, zero-progress guards).

Full technical architecture: [architecture.md](architecture.md)
