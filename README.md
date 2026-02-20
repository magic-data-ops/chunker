# chunker

Automated pipeline that uses Claude Code CLI to generate scored Q&A evaluation metadata from document corpora. A Claude agent autonomously searches plain-text files (no vector DB needed), produces multi-hop question-answer chains with source evidence, and a second agent validates each chain for quality.

## Pipeline

```
Documents (.pdf/.txt/.md)
  1. Index       -- extract plain text from source documents
  2. Generate    -- Claude agent searches corpus, produces Q&A chains with evidence
  3. Validate    -- second Claude agent scores each chain (LLM-as-judge)
  4. Report      -- aggregate metrics against >=80% quality targets
```

The generation agent uses Grep/Read/Glob tools to find evidence across the corpus and can spawn parallel sub-agents for broad searches. Each chain includes the question, golden answer, hop-by-hop evidence trail with file locations, entity metadata, and difficulty rating.

## Validation (two-pass quality gate)

**Step 3 -- LLM-as-judge.** A second Claude agent independently reviews every generated chain. For each chain it receives the category definition, the question, and the full hop-by-hop evidence path, then scores two dimensions:

- **Category suitability** (0.0--1.0) -- does the question genuinely require the declared reasoning skill (e.g., multi-hop, disambiguation)?
- **Answer completeness** (0.0--1.0) -- do the collected evidence hops fully support the golden answer?

Chains with both scores >= 0.7 are approved; rejected chains include a written reason. The validator also produces a polished answer that synthesizes the evidence without referencing internal hop structure.

**Step 4 -- Quality report.** The final step aggregates all scores and checks four dataset-level targets (all must be >= 80%):

| Target | Metric |
|--------|--------|
| Overall approval rate | % of chains approved |
| Mean category suitability | Average suitability score across approved chains |
| Mean answer completeness | Average completeness score across approved chains |
| Threshold pass rate | % of chains where *both* scores >= 0.8 |

The report breaks these down per category, includes a hop-count distribution, and flags any category that falls below target.

## Corpora

- **California state case law** -- 12 evaluation categories
- **Enron email corpus** -- 11 evaluation categories

Categories: entity disambiguation, multi-hop reasoning, temporal ordering, numerical aggregation, cross-context synthesis, hierarchy comprehension, semantic deduplication, and others.

## Outputs

- `qa_chains_validated.json` -- scored chains with evidence, entity metadata, and approval status
- `qa_deliverable_grouped.json` + `.csv` -- grouped deliverable with context, prompts, and golden responses
- `validation_report.json` -- per-category approval rates, mean scores, and target pass/fail

## Key properties

- **Resumable** -- restarts skip already-processed items
- **Concurrent** -- parallel generation across categories, parallel validation across chains
- **Distributable** -- sharding support for multi-machine runs
- **Fault-tolerant** -- atomic saves, retries, and zero-progress guards

Full technical architecture with Mermaid diagrams: [architecture.md](architecture.md)
