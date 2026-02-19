# Long-Context QA Generation Pipeline — Architecture

## End-to-End Pipeline

```mermaid
flowchart TB
    subgraph INPUT["Document Corpus"]
        PDF[".pdf (PyMuPDF)"]
        TXT[".txt / .md"]
    end

    subgraph STEP1["Step 1 — Build Corpus Index"]
        direction TB
        LOAD["Load documents<br/>.txt · .md · .pdf (PyMuPDF)"]
        EXPORT["Export plain-text files<br/>for agent grep / read tools"]

        LOAD --> EXPORT
    end

    subgraph ARTIFACTS1["Index Artifacts"]
        A_TXT["corpus_text/*.txt"]
    end

    subgraph STEP2["Step 2 — Generate Q&A Chains"]
        direction TB

        subgraph CATS["Category Config (categories.yaml)"]
            CAT_ED["entity_disambiguation<br/>hops 1–3"]
            CAT_MH["multi_hop_reasoning<br/>hops 2–4"]
            CAT_SC["single_chunk_factoid<br/>hops 1"]
        end

        subgraph SEED_RUNTIME["Current Runtime Seeding (generate_qa_chains.py)"]
            S_CTX["Pick random passage from corpus_text/*.txt<br/>for SEED_CONTEXT"]
            S_ENT["SEED_ENTITY set to empty string"]
        end

        RENDER["Prompt Renderer<br/>qa_gen_agent.txt<br/>injects: FILE_LIST · CATEGORY_NAME · CATEGORY_DESCRIPTION · N_PAIRS · SEED_CONTEXT · SEED_ENTITY"]

        subgraph HARNESS["Claude Code CLI Subprocess"]
            OC_CMD["claude -p --output-format stream-json<br/>--dir corpus_text/"]

            subgraph AGENT_GEN["Claude Agent"]
                direction LR
                A_MODEL["sonnet/opus/haiku · configurable"]
                A_GREP["Grep (multi-word phrases)"]
                A_READ["Read (±50 lines context)"]
                A_OUT["JSON array of Q&A pairs"]
            end
        end

        subgraph POSTPROC["Post-Processing"]
            EXTRACT["JSON Extractor<br/>strip fences → json.loads → regex fallback"]
            VALIDATE["Pair Validator<br/>question ends ? · answer ≥20ch<br/>evidence ≥20ch · hop count in range"]
            CONVERT["Chain Converter<br/>UUID chain_id · hop_path construction<br/>chunk_id resolution via ChunkStore"]
        end

        CATS --> SEED_RUNTIME
        SEED_RUNTIME --> RENDER
        RENDER --> HARNESS
        HARNESS --> EXTRACT
        EXTRACT --> VALIDATE --> CONVERT
    end

    subgraph STEP3["Step 3 — Contractor Polish"]
        direction TB
        FMT["Format hop path<br/>human-readable text block"]
        RENDER3["Prompt Renderer<br/>contractor_validator.txt<br/>injects: CHAIN_ID · CATEGORY · HOP_PATH_TEXT · QUESTION"]

        subgraph SESSION["Claude Code CLI Subprocess"]
            S_CMD["claude -p --output-format json"]

            subgraph AGENT_VAL["contractor_validator Agent"]
                direction LR
                V_MODEL["sonnet · text-only evaluation"]
                V_SCORE["Score: category_suitability 0.0–1.0<br/>Score: answer_completeness 0.0–1.0"]
                V_APPROVE["Prompt instructs: approve if both ≥ 0.7<br/>script trusts returned approved flag"]
            end

            S_CMD --> AGENT_VAL
        end

        PARSE3["Parse validation JSON<br/>fallback: approved=false · scores=0.0"]
        FMT --> RENDER3 --> SESSION --> PARSE3
    end

    subgraph STEP4["Step 4 — Validation Report"]
        direction LR
        AGG["Per-category breakdown<br/>approval rate · mean scores<br/>hop distribution · termination reasons"]
        TARGETS["Target assessment<br/>≥80% approval · ≥80% suitability<br/>≥80% completeness · ≥80% pass rate"]
        AGG --> TARGETS
    end

    subgraph OUTPUTS["Pipeline Outputs"]
        O_RAW["qa_chains_raw.json"]
        O_VAL["qa_chains_validated.json"]
        O_RPT["validation_report.json"]
    end

    INPUT --> STEP1
    STEP1 --> ARTIFACTS1
    ARTIFACTS1 --> STEP2
    STEP2 --> O_RAW
    O_RAW --> STEP3
    STEP3 --> O_VAL
    O_VAL --> STEP4
    STEP4 --> O_RPT

    style STEP1 fill:#1a1a2e,stroke:#16213e,color:#e0e0e0
    style STEP2 fill:#1a1a2e,stroke:#16213e,color:#e0e0e0
    style STEP3 fill:#1a1a2e,stroke:#16213e,color:#e0e0e0
    style STEP4 fill:#1a1a2e,stroke:#16213e,color:#e0e0e0
    style HARNESS fill:#2d1b69,stroke:#8e44ad,color:#e0e0e0
    style SESSION fill:#2d1b69,stroke:#8e44ad,color:#e0e0e0
    style AGENT_GEN fill:#0f3460,stroke:#533483,color:#e0e0e0
    style AGENT_VAL fill:#0f3460,stroke:#533483,color:#e0e0e0
```

## Agent Configuration & Tool Permissions

```mermaid
flowchart LR
    subgraph AGENTS["Claude Code CLI Agents"]
        direction TB

        subgraph GEN["Step 2: qa_generator"]
            GEN_M["claude -p (stream-json)"]
            GEN_T["model: configurable (sonnet/opus/haiku)"]
            GEN_P["Allowed: Read · Grep · Glob · Bash(ls/wc)"]
        end

        subgraph CONT["Step 3: contractor_validator"]
            CONT_M["claude -p (json output)"]
            CONT_T["model: sonnet · text-only evaluation"]
        end
    end

    subgraph INVOCATION["Invocation Mechanism"]
        I_SUB["Both steps: CLI subprocess<br/>claude -p --output-format ..."]
    end

    GEN --> I_SUB
    CONT --> I_SUB

    style GEN fill:#1e3a5f,stroke:#2980b9,color:#e0e0e0
    style CONT fill:#1e3a5f,stroke:#2980b9,color:#e0e0e0
```

## Step 2 Detail — Agentic grep/read Loop

```mermaid
sequenceDiagram
    participant PY as generate_qa_chains.py
    participant CC as claude -p (subprocess)
    participant AGENT as Claude (sonnet/opus/haiku)
    participant FS as corpus_text/*.txt

    PY->>CC: claude -p --output-format stream-json<br/>"Generate 5 entity_disambiguation pairs..."

    loop up to 50 agent steps
        AGENT->>FS: grep "multi-word phrase"
        FS-->>AGENT: matched lines + positions
        AGENT->>FS: read file.txt (lines 100–200)
        FS-->>AGENT: passage context
        Note over AGENT: Evaluate evidence sufficiency.<br/>Chain to related concepts if multi-hop.
    end

    AGENT-->>CC: JSON array [{question, golden_answer,<br/>evidence_snippets, source_files}]
    CC-->>PY: stdout stream-json events<br/>{type: "result", result: "[...]"}

    Note over PY: _extract_json_array() → _validate_pair() → _pairs_to_chains()
```

## Step 3 Detail — Contractor Validation

```mermaid
sequenceDiagram
    participant PY as contractor_polish.py
    participant QA as QAAgent
    participant CC as claude -p (subprocess)

    PY->>QA: agent.generate(rendered_prompt)
    QA->>CC: claude -p --output-format json<br/>--model sonnet

    CC-->>QA: {result: "{approved, scores, polished_answer}"}
    QA-->>PY: parsed validation result

    Note over PY: Merge scores into chain → atomic save every 50
```

## Data Schema — Chain Record Lifecycle

```mermaid
flowchart LR
    subgraph S2["Step 2 produces"]
        direction TB
        F1["chain_id: UUID"]
        F2["category: str"]
        F3["question: str"]
        F4["final_answer: str"]
        F5["hop_count: int"]
        F6["hop_path: &#91;{hop_index, chunk_id,<br/>chunk_text, partial_answer,<br/>retrieval_score, provenance}&#93;"]
        F7["source_file · prompt_seed_file · termination_reason<br/>single_answer_heuristic · generated_at"]
    end

    subgraph S3["Step 3 enriches with"]
        direction TB
        V1["approved: bool"]
        V2["category_suitability_score: 0.0–1.0"]
        V3["answer_completeness_score: 0.0–1.0"]
        V4["polished_answer: str"]
        V5["rejection_reason: str | null"]
        V6["validated_at: ISO-8601"]
    end

    subgraph S4["Step 4 aggregates"]
        direction TB
        R1["total_chains / total_approved"]
        R2["per_category: count · rate · mean scores"]
        R3["hop_distribution histogram"]
        R4["target assessment (≥80% thresholds)"]
    end

    S2 -->|"+ validation"| S3
    S3 -->|"aggregate"| S4
```

## Resilience & Concurrency Patterns

```mermaid
flowchart TB
    subgraph RESILIENCE["Resilience"]
        AT["Atomic Saves<br/>write .tmp → os.rename()"]
        RE["Resume<br/>Step 2: per-category counts from existing output<br/>Step 3: done_ids set · skip processed"]
        RT["Retry<br/>max_retries=2 default on parse failure<br/>abort after 3× zero-progress"]
        CK["Checkpoints<br/>Step 3 saves every 50 chains"]
        JF["JSON Fallback<br/>strip fences → json.loads → regex"]
    end

    subgraph CONCURRENCY["Concurrency"]
        S2C["Step 2: asyncio.Semaphore(3)<br/>category-level parallelism<br/>async subprocess · 600s timeout"]
        S3C["Step 3: asyncio.Semaphore(32)<br/>chain-level parallelism<br/>HTTP session API · tqdm progress"]
        SH["Sharding<br/>--rank / --world_size<br/>category-level distribution"]
    end
```

## Module Dependency Graph

```mermaid
flowchart TB
    LAUNCH["launch_qa_gen.py"]
    S1["build_corpus_index.py (run directly)"]

    LAUNCH --> S2["generate_qa_chains.py"]
    LAUNCH --> S3["contractor_polish.py"]
    LAUNCH --> S4["validate_qa_dataset.py"]

    S2 --> YAML["qa_config/categories.yaml"]

    S3 --> LLM["utils/llm_client.py<br/>QAAgent"]

    style LAUNCH fill:#2d1b69,stroke:#8e44ad,color:#e0e0e0
    style S1 fill:#1a1a2e,stroke:#16213e,color:#e0e0e0
    style S2 fill:#1a1a2e,stroke:#16213e,color:#e0e0e0
    style S3 fill:#1a1a2e,stroke:#16213e,color:#e0e0e0
    style S4 fill:#1a1a2e,stroke:#16213e,color:#e0e0e0
```

## TL;DR — High-Level Flow

```mermaid
flowchart LR
    DOCS["📄 Documents"]
    INDEX["1 · Index<br/>load + export text"]
    GEN["2 · Generate<br/>agentic grep/read<br/>over corpus"]
    VAL["3 · Validate<br/>LLM-as-judge<br/>scoring"]
    RPT["4 · Report<br/>aggregate<br/>metrics"]
    OUT["✅ Scored<br/>QA Dataset"]

    DOCS --> INDEX
    INDEX --> GEN
    GEN --> VAL
    VAL --> RPT
    RPT --> OUT

    style DOCS fill:#0f3460,stroke:#533483,color:#e0e0e0
    style INDEX fill:#1a1a2e,stroke:#16213e,color:#e0e0e0
    style GEN fill:#2d1b69,stroke:#8e44ad,color:#e0e0e0
    style VAL fill:#2d1b69,stroke:#8e44ad,color:#e0e0e0
    style RPT fill:#1a1a2e,stroke:#16213e,color:#e0e0e0
    style OUT fill:#0f3460,stroke:#533483,color:#e0e0e0
```
