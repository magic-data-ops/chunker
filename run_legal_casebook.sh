#!/usr/bin/env bash
set -euo pipefail

# Legal Casebook Corpus Generation Pipeline
#
# Builds ~200M token legal casebook corpus from real public-domain case law
# (Harvard CAP) + LLM-augmented companion documents via OpenCode CLI.
#
# Usage:
#   ./run_legal_casebook.sh              # Full pipeline (phases 1-5)
#   ./run_legal_casebook.sh --phase 1    # Single phase
#   ./run_legal_casebook.sh --phase 1-3  # Phase range
#   ./run_legal_casebook.sh --resume     # Resume interrupted runs
#   ./run_legal_casebook.sh --dry-run    # Print commands only
#
# Prerequisites:
#   pip install requests tqdm
#   opencode CLI (https://opencode.ai/download) for phases 2-3
#   ANTHROPIC_API_KEY env var for OpenCode to access Anthropic models

CORPUS_DIR="corpus_text/legal_casebook"

REPORTERS="us f3d f2d ny3d ny-2d sw3d sw2d"
CIRCUIT_FILTER="9th-circuit 5th-circuit"
DATE_START=1950
DATE_END=2025
TARGET_TOKENS=150000000
MAX_PER_REPORTER=5000

AUGMENT_CONCURRENCY=8
AUGMENT_MODEL="claude-haiku-4-5"
AUGMENT_MAX_CASES=1000

DRY_RUN=false
PHASE_START=1
PHASE_END=5
RESUME=false

while [ $# -gt 0 ]; do
    case "$1" in
        --phase)
            shift
            case "$1" in
                *-*)
                    PHASE_START="${1%-*}"
                    PHASE_END="${1#*-}"
                    ;;
                *)
                    PHASE_START="$1"
                    PHASE_END="$1"
                    ;;
            esac
            shift
            ;;
        --dry-run) DRY_RUN=true; shift ;;
        --resume)  RESUME=true; shift ;;
        --help|-h) head -20 "$0" | tail -15; exit 0 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

run_cmd() {
    if "$DRY_RUN"; then
        echo "  -> $*"
    else
        "$@"
    fi
}

echo "Legal Casebook Corpus Pipeline -- Phases ${PHASE_START}-${PHASE_END}"
if "$RESUME"; then echo "  Resume mode enabled"; fi
if "$DRY_RUN"; then echo "  DRY RUN -- commands printed only"; fi
echo ""

python3 -c "import requests, tqdm" 2>/dev/null || {
    echo "ERROR: Missing Python deps. Run: pip install requests tqdm"
    exit 1
}

# Phase 1: Download and filter case law from CAP
if [ "$PHASE_START" -le 1 ] && [ "$PHASE_END" -ge 1 ]; then
    echo "== PHASE 1: Download and Filter Case Law from static.case.law =="
    echo "   Target: ~${TARGET_TOKENS} tokens, reporters: ${REPORTERS}"
    echo ""

    CMD=(
        python3 data_acquisition/download_cap.py
        --output-dir "$CORPUS_DIR"
        --reporters $REPORTERS
        --topics "contract law" "employment law" "environmental regulation" "intellectual property"
        --date-start "$DATE_START"
        --date-end "$DATE_END"
        --target-tokens "$TARGET_TOKENS"
        --max-per-reporter "$MAX_PER_REPORTER"
        --circuit-filter $CIRCUIT_FILTER
    )
    if "$RESUME"; then CMD+=(--resume); fi

    run_cmd "${CMD[@]}"
    echo "[DONE] Phase 1"
    echo ""
fi

# Phase 2: Extract entity and citation graph
if [ "$PHASE_START" -le 2 ] && [ "$PHASE_END" -ge 2 ]; then
    echo "== PHASE 2: Extract Entity and Citation Graph =="
    echo ""

    CMD=(
        python3 data_acquisition/extract_graph.py
        --corpus-dir "$CORPUS_DIR"
        --concurrency "$AUGMENT_CONCURRENCY"
        --model "$AUGMENT_MODEL"
    )
    # Uncomment to enable LLM holding extraction (costs money):
    # CMD+=(--extract-holdings)

    run_cmd "${CMD[@]}"
    echo "[DONE] Phase 2"
    echo ""
fi

# Phase 3: Generate LLM-augmented companion documents via OpenCode
if [ "$PHASE_START" -le 3 ] && [ "$PHASE_END" -ge 3 ]; then
    echo "== PHASE 3: Generate Augmented Companion Documents =="
    echo "   Layer A: briefs, motions, settlements, orders"
    echo "   Layer B: statutes, committee reports, debates, guidance"
    echo "   Layer C: law reviews, practice guides, treatises"
    echo "   Model: ${AUGMENT_MODEL}, Concurrency: ${AUGMENT_CONCURRENCY}"
    echo ""

    CMD=(
        python3 data_acquisition/generate_augmentation.py
        --corpus-dir "$CORPUS_DIR"
        --concurrency "$AUGMENT_CONCURRENCY"
        --model "$AUGMENT_MODEL"
        --max-cases "$AUGMENT_MAX_CASES"
        --layer A --layer B --layer C
    )
    if "$RESUME"; then CMD+=(--resume); fi

    run_cmd "${CMD[@]}"
    echo "[DONE] Phase 3"
    echo ""
fi

# Phase 4: Assemble final corpus
if [ "$PHASE_START" -le 4 ] && [ "$PHASE_END" -ge 4 ]; then
    echo "== PHASE 4: Assemble Final Corpus =="
    echo ""

    run_cmd python3 data_acquisition/assemble_corpus.py \
        --corpus-dir "$CORPUS_DIR" \
        --output "${CORPUS_DIR}/legal_casebook_complete.txt" \
        --seed 42

    echo "[DONE] Phase 4"
    echo ""
fi

# Phase 5: Verify corpus quality
if [ "$PHASE_START" -le 5 ] && [ "$PHASE_END" -ge 5 ]; then
    echo "== PHASE 5: Verify Corpus Quality =="
    echo ""

    run_cmd python3 data_acquisition/verify_corpus.py \
        --corpus-dir "$CORPUS_DIR" \
        --corpus-file "${CORPUS_DIR}/legal_casebook_complete.txt"

    # N-gram overlap check (synthetic vs source documents)
    run_cmd python3 data_acquisition/check_overlap.py \
        --corpus-dir "$CORPUS_DIR" \
        --ngram-size 5 \
        --threshold 15.0

    echo "[DONE] Phase 5"
    echo ""
fi

echo "========================================"
echo "CORPUS GENERATION COMPLETE"
echo "========================================"
echo ""
echo "Corpus:   ${CORPUS_DIR}/legal_casebook_complete.txt"
echo "Metadata: ${CORPUS_DIR}/metadata/"
echo ""
echo "To run the QA pipeline on this corpus:"
echo ""
echo "  python qa_generation/generate_qa_chains.py \\"
echo "    --corpus_text_dir ${CORPUS_DIR} \\"
echo "    --categories_cfg qa_generation/qa_config/categories_legal_casebook.yaml \\"
echo "    --prompt-template qa_generation/prompts/qa_gen_agent_legal_casebook.txt"
echo ""
