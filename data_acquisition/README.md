# Legal Casebook Corpus Generation

Generates a ~200M token evaluation corpus by downloading real US case law from the Harvard Caselaw Access Project and augmenting it with LLM-generated companion documents (briefs, statutes, law reviews, etc.). The assembled corpus feeds into the existing QA generation pipeline (`generate_qa_chains.py`).

## Why generate synthetic documents from real ones?

The core challenge in building long-context evaluation datasets is creating a corpus where questions *require* reasoning across many distant parts of a large document collection. Real case law alone gives us authentic legal text, but the document types are homogeneous -- all court opinions.

This pipeline solves that by using real cases as **seed material** to generate realistic companion documents that reference the same parties, citations, statutes, and facts. The result is a multi-document-type corpus where:

- A **plaintiff brief** argues one side of a real case, citing other real cases from the corpus
- A **defendant brief** argues the opposite side of the same case
- A **statute** is referenced by multiple real opinions and interpreted differently in a **committee report** vs. a **regulatory guidance** document
- A **law review article** analyzes a circuit split across multiple real holdings
- A **treatise section** tracks how a legal standard evolved across decades of real cases

Because the synthetic documents are grounded in real case metadata (exact party names, citation strings, dollar amounts, judge names from the entity graph), they create natural cross-reference networks. A question like *"What damages did the court award in Stevens v. Department of Treasury, and how does the plaintiff brief's demand compare?"* requires finding information across a real opinion and a synthetic brief -- potentially separated by millions of tokens in the assembled corpus.

## How it works

```
Real case law (static.case.law)         LLM (OpenCode Zen API)
        |                                        |
        v                                        |
  Phase 1: Download & filter                     |
  28K+ court opinions                            |
        |                                        |
        v                                        |
  Phase 2: Extract knowledge graph               |
  judges, parties, citations,                    |
  statutes, numerical facts                      |
        |                                        |
        +-----> Phase 3: Generate augmentation <--+
        |       Uses real case data as context
        |       to ground LLM-generated documents
        |                |
        v                v
  Phase 4: Assemble into single corpus
  Interleave real + synthetic by topic
  (deliberately non-chronological)
        |
        v
  Phase 5: Verify quality
  Cross-references, entity consistency,
  distribution checks
        |
        v
  legal_casebook_complete.txt (~200M tokens)
```

### Phase 1: Download real case law

Downloads court opinions from the Harvard Caselaw Access Project ([static.case.law](https://static.case.law)), a free research archive of 6.7M+ US court decisions.

```bash
python data_acquisition/download_cap.py \
  --reporters us f3d f2d ny3d ny-2d sw3d sw2d \
  --topics "contract law" "employment law" "environmental regulation" "intellectual property" \
  --date-start 1950 --date-end 2025 \
  --target-tokens 150000000 \
  --circuit-filter 9th-circuit 5th-circuit
```

Each case is saved as a plain text file with a structured header:

```
CASE_ID: 6215164
CASE_NAME: Stevens v. Department of Treasury
COURT: Supreme Court of the United States
DATE: 1991-04-24
CITATIONS: 500 U.S. 1; 111 S. Ct. 1562
JUDGES: Stevens, J.
TOPIC: employment law
JURISDICTION: United States
---
{full opinion text}
```

A master index (`metadata/case_index.json`) tracks all cases with their metadata, citation links, and token counts.

### Phase 2: Extract entity & citation graph

Parses the downloaded cases to build a structured knowledge graph -- the backbone that keeps synthetic documents consistent with real ones.

```bash
python data_acquisition/extract_graph.py \
  --corpus-dir corpus_text/legal_casebook
```

Extracts (all rule-based, no LLM cost):

| Entity type | Method | Example |
|---|---|---|
| Parties | Regex on "X v. Y" patterns | Stevens (plaintiff), Department of Treasury (defendant) |
| Citation edges | Match extracted cites against case index | Case A cites Case B via "500 U.S. 1" |
| Statutes | 4 regex families (USC, named acts, state codes, CFR) | 42 U.S.C. § 1983, Title VII, Clean Air Act |
| Numerical facts | Dollar amounts, vote counts, time periods | $1,500,000 damages, 5-to-4 decision |
| Holdings | Optional LLM extraction (`--extract-holdings`) | "Court held that..." |

Output: `metadata/entity_graph.json` -- used by Phase 3 to ground synthetic documents in real facts.

### Phase 3: Generate augmented documents

This is where real data becomes a multi-document-type corpus. For each real case, the pipeline generates companion documents by prompting an LLM with the real case text + entity graph context.

```bash
OPENCODE_API_KEY=sk-... python data_acquisition/generate_augmentation.py \
  --corpus-dir corpus_text/legal_casebook \
  --layer A --layer B --layer C \
  --concurrency 8 \
  --model claude-sonnet-4-5
```

Three layers of synthetic documents:

**Layer A -- Litigation documents** (grounded in individual cases):
| Type | What the LLM receives | What it generates |
|---|---|---|
| Plaintiff brief | Real opinion + parties + related cases | Legal argument for the plaintiff, citing real cases |
| Defendant brief | Same context, opposite side | Counterargument citing different real cases |
| Motion for summary judgment | Case facts + statutes | Procedural motion referencing real statutory provisions |
| Discovery motion | Case timeline + parties | Discovery dispute with real party names |
| Settlement agreement | Numerical facts from graph | Agreement with dollar amounts consistent with the real case |
| Court order | Case metadata + procedural history | Procedural ruling referencing the real case timeline |

**Layer B -- Legislative materials** (grounded in statutes referenced by multiple cases):
| Type | What it generates |
|---|---|
| Statutory text | Full statute sections with subsections and definitions |
| Committee report | Legislative intent analysis referencing real court interpretations |
| Floor debate | Multiple speakers disagreeing about the statute's scope |
| Regulatory guidance | Agency interpretation that may conflict with court holdings |

**Layer C -- Secondary sources** (grounded in topic clusters of cases):
| Type | What it generates |
|---|---|
| Law review article | Analyzes circuit splits across real holdings |
| Practice guide | Jurisdiction-by-jurisdiction comparison with checklists |
| Treatise section | Deep hierarchical analysis (§7.03[1][a][i]) of legal standards |

The grounding mechanism works like this:

1. Select a real case from the corpus
2. Pull its metadata from `case_index.json` (parties, court, date, citations)
3. Pull related cases from the citation graph in `entity_graph.json`
4. Pull entities: judges, statutes, numerical facts, holdings
5. Render a prompt template with all this real context
6. LLM generates a document that references real names, citations, and facts
7. Validate that generated content references entities that exist in the graph

This ensures that a question requiring cross-document reasoning (e.g., *"Compare the plaintiff's damages claim in the brief with what the court actually awarded"*) has grounded, consistent answers across the real opinion and the synthetic brief.

**API**: Direct HTTP to `https://opencode.ai/zen/v1/chat/completions` (OpenAI-compatible) via async `aiohttp`. Supports concurrent generation, atomic saves every 25 docs, and full resume.

### Phase 4: Assemble final corpus

Merges all real cases and synthetic documents into a single `legal_casebook_complete.txt`.

```bash
python data_acquisition/assemble_corpus.py \
  --corpus-dir corpus_text/legal_casebook \
  --seed 42
```

Key design choices:

- **Topic-based interleaving**: Documents are grouped by legal topic, then round-robin interleaved across topics. This means related documents (an opinion and its briefs) are *not* adjacent -- they're separated by documents from other topics.
- **Non-chronological order**: Within each topic group, documents are shuffled. A 1991 opinion might appear after a 2015 opinion.
- **Document type markers**: Each document gets a header identifying its type, enabling evaluation questions about source prioritization.

```
=== DOCUMENT 1/5000 | TYPE: COURT_OPINION ===
CASE_ID: 6215164
...
=== DOCUMENT 2/5000 | TYPE: PLAINTIFF_BRIEF ===
DOC_ID: layerA_6215164_plaintiff_brief
...
=== DOCUMENT 3/5000 | TYPE: STATUTORY_TEXT ===
DOC_ID: layerB_42_U_S_C___1983_statutory_text
...
```

These design choices create natural evaluation challenges:

| Challenge | How the corpus creates it |
|---|---|
| Temporal ordering | Reconstructing chronology from shuffled documents |
| Cross-context synthesis | Combining info from opinion + brief + statute |
| Source prioritization | Court opinion vs. committee report vs. agency guidance |
| Entity tracking | Same party appears across opinion, brief, settlement |
| Numerical aggregation | Damages scattered across opinion, brief, settlement |
| Conflicting information | Plaintiff brief vs. defendant brief on the same facts |

### Phase 5: Verify quality

Runs 9 automated checks on the assembled corpus.

```bash
python data_acquisition/verify_corpus.py \
  --corpus-dir corpus_text/legal_casebook
```

Checks: token count, structural integrity, cross-reference resolution, entity consistency, numerical consistency, document type diversity, topic diversity, jurisdiction diversity, and non-chronological ordering.

## File structure

```
data_acquisition/
  download_cap.py              # Phase 1: download from static.case.law
  extract_graph.py             # Phase 2: build entity/citation graph
  generate_augmentation.py     # Phase 3: LLM generation via OpenCode Zen API
  assemble_corpus.py           # Phase 4: merge into final corpus
  verify_corpus.py             # Phase 5: quality checks
  check_overlap.py             # N-gram overlap check (synthetic vs source)
  prompts/
    brief_generator.txt        # Layer A prompt template
    legislative_generator.txt  # Layer B prompt template
    secondary_source_generator.txt  # Layer C prompt template

corpus_text/legal_casebook/
  cases/{reporter}/{id}.txt    # Individual case opinions
  augmented/
    layer_a/*.txt              # Generated litigation documents
    layer_b/*.txt              # Generated legislative materials
    layer_c/*.txt              # Generated secondary sources
  metadata/
    case_index.json            # Master case index
    entity_graph.json          # Knowledge graph
    assembly_stats.json        # Corpus statistics
    verification_report.json   # Quality check results
    overlap_report.json        # N-gram overlap report
  legal_casebook_complete.txt  # Final assembled corpus

qa_generation/
  qa_config/categories_legal_casebook.yaml  # 12 eval categories
  prompts/qa_gen_agent_legal_casebook.txt   # QA agent prompt

run_legal_casebook.sh          # Orchestration script (phases 1-5)
```

## Running the full pipeline

```bash
# Full pipeline (all phases)
./run_legal_casebook.sh

# Single phase
./run_legal_casebook.sh --phase 3

# Phase range
./run_legal_casebook.sh --phase 1-3

# Resume interrupted run
./run_legal_casebook.sh --resume

# Dry run (print commands only)
./run_legal_casebook.sh --dry-run
```

Or run phases individually:

```bash
# Phase 1: Download cases
python data_acquisition/download_cap.py \
  --output-dir corpus_text/legal_casebook \
  --reporters us f3d f2d ny3d ny-2d sw3d sw2d \
  --target-tokens 150000000

# Phase 2: Extract graph
python data_acquisition/extract_graph.py \
  --corpus-dir corpus_text/legal_casebook

# Phase 3: Generate augmented documents (requires OPENCODE_API_KEY)
OPENCODE_API_KEY=sk-... python data_acquisition/generate_augmentation.py \
  --corpus-dir corpus_text/legal_casebook \
  --concurrency 8 --model claude-sonnet-4-5

# Phase 4: Assemble corpus
python data_acquisition/assemble_corpus.py \
  --corpus-dir corpus_text/legal_casebook

# Phase 5: Verify
python data_acquisition/verify_corpus.py \
  --corpus-dir corpus_text/legal_casebook
```

## Prerequisites

```bash
uv pip install requests tqdm aiohttp
```

Phase 3 requires an [OpenCode](https://opencode.ai) API key set as `OPENCODE_API_KEY`.

## Downstream: QA generation

The assembled corpus plugs directly into the existing QA pipeline:

```bash
python qa_generation/generate_qa_chains.py \
  --corpus_text_dir corpus_text/legal_casebook \
  --categories_cfg qa_generation/qa_config/categories_legal_casebook.yaml \
  --prompt-template qa_generation/prompts/qa_gen_agent_legal_casebook.txt
```

This uses Claude Code (not OpenCode) to generate evaluation Q&A pairs across 12 categories: cross-context synthesis, long-context citation, hierarchy comprehension, entity state tracking, entity disambiguation, multi-hop reasoning, semantic deduplication, temporal ordering, domain scoping, source prioritization, numerical aggregation, and conflicting information synthesis.
