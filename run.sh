#!/usr/bin/env bash
# Interactive Q&A Pipeline Runner
# Run with: ./run.sh

set -euo pipefail

# ─── Colors ───────────────────────────────────────────────────────────────────

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
DIM='\033[2m'
NC='\033[0m'

# ─── Print helpers ────────────────────────────────────────────────────────────

print_header() {
  printf "\n${BOLD}${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"
  printf "${BOLD}${CYAN}  %s${NC}\n" "$1"
  printf "${BOLD}${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n\n"
}

print_step() {
  printf "${BOLD}${BLUE}>> %s${NC}\n" "$1"
}

print_success() {
  printf "${GREEN}  OK: %s${NC}\n" "$1"
}

print_error() {
  printf "${RED}  ERROR: %s${NC}\n" "$1"
}

print_warn() {
  printf "${YELLOW}  NOTE: %s${NC}\n" "$1"
}

print_item() {
  printf "  ${BOLD}%-22s${NC} %s\n" "$1" "$2"
}

# ─── Input helpers ────────────────────────────────────────────────────────────

prompt_text() {
  local label="$1"
  local default="$2"
  local result

  printf "${BOLD}  %s${NC} ${DIM}[%s]${NC}: " "$label" "$default" >&2
  read -r result
  result="${result:-$default}"
  printf "%s" "$result"
}

prompt_number() {
  local label="$1"
  local default="$2"
  local min="${3:-1}"
  local max="${4:-99999}"

  while true; do
    printf "${BOLD}  %s${NC} ${DIM}[%s]${NC}: " "$label" "$default" >&2
    read -r input
    input="${input:-$default}"

    if [[ "$input" =~ ^[0-9]+$ ]] && [ "$input" -ge "$min" ] && [ "$input" -le "$max" ]; then
      printf "%s" "$input"
      return
    fi
    print_error "Enter a number between $min and $max" >&2
  done
}

prompt_decimal() {
  local label="$1"
  local default="$2"

  while true; do
    printf "${BOLD}  %s${NC} ${DIM}[%s]${NC}: " "$label" "$default" >&2
    read -r input
    input="${input:-$default}"

    if [[ "$input" =~ ^[0-9]+\.?[0-9]*$ ]]; then
      printf "%s" "$input"
      return
    fi
    print_error "Enter a valid number (e.g. 0.50)" >&2
  done
}

prompt_yes_no() {
  local label="$1"
  local default="$2"  # "y" or "n"

  local hint="y/N"
  [ "$default" = "y" ] && hint="Y/n"

  while true; do
    printf "${BOLD}  %s${NC} ${DIM}[%s]${NC}: " "$label" "$hint" >&2
    read -r input
    input="${input:-$default}"

    case "$input" in
      [yY]|[yY][eE][sS]) return 0 ;;
      [nN]|[nN][oO])     return 1 ;;
      *)                  print_error "Please enter y or n" >&2 ;;
    esac
  done
}

prompt_choice() {
  local label="$1"
  shift
  local options=("$@")
  local n=${#options[@]}

  printf "\n${BOLD}  %s${NC}\n\n" "$label" >&2
  for i in "${!options[@]}"; do
    printf "    ${CYAN}%d)${NC}  %s\n" "$((i + 1))" "${options[$i]}" >&2
  done
  printf "\n" >&2

  while true; do
    printf "${BOLD}  Choose${NC} ${DIM}[1]${NC}: " >&2
    read -r input
    input="${input:-1}"

    if [[ "$input" =~ ^[0-9]+$ ]] && [ "$input" -ge 1 ] && [ "$input" -le "$n" ]; then
      printf "%s" "$input"
      return
    fi
    print_error "Enter a number between 1 and $n" >&2
  done
}

# ─── Dependency checks ───────────────────────────────────────────────────────

check_deps() {
  print_step "Checking dependencies"

  local ok=true

  if command -v python3 &>/dev/null; then
    print_success "python3 found ($(python3 --version 2>&1))"
  else
    print_error "python3 not found. Please install Python 3.12+"
    ok=false
  fi

  if command -v claude &>/dev/null; then
    print_success "claude CLI found"
  else
    print_error "claude CLI not found. Install from: https://docs.anthropic.com/en/docs/claude-code"
    ok=false
  fi

  if [ -d "qa_generation" ]; then
    print_success "qa_generation/ directory found"
  else
    print_error "qa_generation/ not found. Run this script from the project root."
    ok=false
  fi

  if [ "$ok" = false ]; then
    printf "\n"
    print_error "Missing dependencies. Fix the above issues and try again."
    exit 1
  fi

  printf "\n"
}

# ─── Run a pipeline step ─────────────────────────────────────────────────────

run_step() {
  local step_name="$1"
  shift
  local cmd=("$@")

  printf "\n${BOLD}${CYAN}── Running: %s ──${NC}\n\n" "$step_name"
  printf "${DIM}  %s${NC}\n\n" "${cmd[*]}"

  if "${cmd[@]}"; then
    print_success "$step_name completed"
    return 0
  else
    local rc=$?
    print_error "$step_name failed (exit code $rc)"
    return $rc
  fi
}

# ─── Main ─────────────────────────────────────────────────────────────────────

main() {
  print_header "Q&A Dataset Pipeline"

  check_deps

  # ── Step 1: What to run ──

  local mode
  mode=$(prompt_choice "What would you like to do?" \
    "Full pipeline  (build corpus -> generate -> validate -> report)" \
    "Generate + validate + report  (corpus already built)" \
    "Just generate Q&A chains" \
    "Just validate existing chains" \
    "Just generate report")
  printf "\n"

  # ── Step 2: Paths ──

  print_step "Configure paths"

  local docs_dir="./docs"
  local output_dir="./corpus_index"
  local corpus_dir="./corpus_text"
  local outputs_dir="./outputs"
  # Dynamic output naming derived after corpus_dir is known
  local raw_output=""
  local validated_output=""
  local report_output=""

  # Build corpus — need docs directory
  if [ "$mode" = "1" ]; then
    docs_dir=$(prompt_text "Documents directory" "$docs_dir")
    printf "\n"

    if [ ! -d "$docs_dir" ]; then
      print_error "Directory '$docs_dir' does not exist."
      exit 1
    fi

    local file_count
    file_count=$(find "$docs_dir" -type f \( -name "*.txt" -o -name "*.md" -o -name "*.pdf" \) | wc -l | tr -d ' ')
    print_success "Found $file_count document(s) in $docs_dir"
  fi

  # Generation — need corpus directory
  if [ "$mode" = "1" ] || [ "$mode" = "2" ] || [ "$mode" = "3" ]; then
    if [ "$mode" != "1" ]; then
      corpus_dir=$(prompt_text "Corpus text directory" "$corpus_dir")
      printf "\n"

      if [ ! -d "$corpus_dir" ]; then
        print_error "Directory '$corpus_dir' does not exist. Run the full pipeline first, or point to your corpus."
        exit 1
      fi

      local txt_count
      txt_count=$(find "$corpus_dir" -maxdepth 1 -name "*.txt" | wc -l | tr -d ' ')
      print_success "Found $txt_count text file(s) in $corpus_dir"
    fi

    # Derive dynamic output names from corpus stem + date
    local corpus_stem
    local txt_file_count
    txt_file_count=$(find "$corpus_dir" -maxdepth 1 -name "*.txt" 2>/dev/null | wc -l | tr -d ' ')
    if [ "$txt_file_count" -eq 1 ]; then
      corpus_stem=$(basename "$(ls "$corpus_dir"/*.txt | head -1)" .txt)
    else
      corpus_stem=$(basename "$corpus_dir")
    fi
    local ts
    ts=$(date +%Y%m%d)

    raw_output="$outputs_dir/${corpus_stem}_${ts}_qa_chains_raw.json"
    validated_output="$outputs_dir/${corpus_stem}_${ts}_qa_chains_validated.json"
    report_output="$outputs_dir/${corpus_stem}_${ts}_validation_report.json"

    raw_output=$(prompt_text "Output file for raw chains" "$raw_output")
    printf "\n"
  fi

  # Validation — need raw chains
  if [ "$mode" = "4" ]; then
    raw_output=$(prompt_text "Raw chains file to validate" "$raw_output")
    printf "\n"

    if [ ! -f "$raw_output" ]; then
      print_error "File '$raw_output' not found. Generate chains first."
      exit 1
    fi
  fi

  # Report — need validated chains
  if [ "$mode" = "5" ]; then
    validated_output=$(prompt_text "Validated chains file" "$validated_output")
    printf "\n"

    if [ ! -f "$validated_output" ]; then
      print_error "File '$validated_output' not found. Run validation first."
      exit 1
    fi
  fi

  # ── Step 3: Generation settings ──

  local samples_per_cat=3
  local model="sonnet"
  local budget="2.00"
  local batch_size=5
  local concurrency=3
  local log_dir="$outputs_dir/logs"
  local categories_cfg=""
  local prompt_template=""
  local save_raw="n"

  if [ "$mode" = "1" ] || [ "$mode" = "2" ] || [ "$mode" = "3" ]; then
    printf "\n"
    print_step "Generation settings"

    samples_per_cat=$(prompt_number "Samples per category" "3" 1 100)
    printf "\n"

    local model_choice
    model_choice=$(prompt_choice "Claude model" \
      "Sonnet  — good balance of speed and quality (recommended)" \
      "Opus    — highest quality, slower, costs more" \
      "Haiku   — fastest and cheapest, lower quality")

    case "$model_choice" in
      1) model="sonnet" ;;
      2) model="opus" ;;
      3) model="haiku" ;;
    esac

    budget=$(prompt_decimal "Max cost per Claude call (USD)" "2.00")
    printf "\n"

    if prompt_yes_no "Configure advanced settings?" "n"; then
      printf "\n"
      batch_size=$(prompt_number "Batch size (pairs per call)" "5" 1 20)
      printf "\n"
      concurrency=$(prompt_number "Concurrency (parallel calls)" "3" 1 10)
      printf "\n"
      log_dir=$(prompt_text "Log directory" "$log_dir")
      printf "\n"
      categories_cfg=$(prompt_text "Categories config (empty for default)" "")
      printf "\n"
      prompt_template=$(prompt_text "Prompt template (empty for default)" "")
      printf "\n"
      if prompt_yes_no "Save raw Claude output for debugging?" "n"; then
        save_raw="y"
      fi
    fi
  fi

  # ── Step 4: Summary ──

  printf "\n"
  print_header "Review Settings"

  local steps_label=""
  case "$mode" in
    1) steps_label="Build -> Generate -> Validate -> Report" ;;
    2) steps_label="Generate -> Validate -> Report" ;;
    3) steps_label="Generate" ;;
    4) steps_label="Validate" ;;
    5) steps_label="Report" ;;
  esac

  print_item "Pipeline:" "$steps_label"

  if [ "$mode" = "1" ]; then
    print_item "Documents:" "$docs_dir"
  fi

  if [ "$mode" = "1" ] || [ "$mode" = "2" ] || [ "$mode" = "3" ]; then
    print_item "Corpus:" "$corpus_dir"
    print_item "Samples/category:" "$samples_per_cat"
    print_item "Model:" "$model"
    print_item "Budget/call:" "\$$budget"
    print_item "Batch size:" "$batch_size"
    print_item "Concurrency:" "$concurrency"
    print_item "Log directory:" "$log_dir"
    print_item "Raw output:" "$raw_output"

    if [ "$save_raw" = "y" ]; then
      print_item "Debug output:" "enabled"
    fi
  fi

  if [ "$mode" != "3" ]; then
    print_item "Validated output:" "$validated_output"
  fi
  if [ "$mode" != "3" ] && [ "$mode" != "4" ]; then
    print_item "Report:" "$report_output"
  fi

  printf "\n"

  if ! prompt_yes_no "Start pipeline?" "y"; then
    printf "\n"
    print_warn "Cancelled."
    exit 0
  fi

  # ── Step 5: Execute ──

  printf "\n"
  print_header "Running Pipeline"

  mkdir -p "$outputs_dir" "$log_dir"

  local failed=false

  # Build corpus
  if [ "$mode" = "1" ]; then
    if ! run_step "Build Corpus" python3 qa_generation/build_corpus_index.py \
        --input_dir "$docs_dir" \
        --output_dir "$output_dir"; then
      failed=true
    fi
  fi

  # Generate chains
  if [ "$failed" = false ] && { [ "$mode" = "1" ] || [ "$mode" = "2" ] || [ "$mode" = "3" ]; }; then
    local gen_cmd=(python3 qa_generation/generate_qa_chains.py
      --corpus_text_dir "$corpus_dir"
      --output "$raw_output"
      --samples-per-category "$samples_per_cat"
      --batch_size "$batch_size"
      --concurrency "$concurrency"
      --model "$model"
      --max-budget-usd "$budget"
      --log-dir "$log_dir"
    )
    [ -n "$categories_cfg" ] && gen_cmd+=(--categories_cfg "$categories_cfg")
    [ -n "$prompt_template" ] && gen_cmd+=(--prompt-template "$prompt_template")
    [ "$save_raw" = "y" ] && gen_cmd+=(--save-raw-runs)

    if ! run_step "Generate Q&A Chains" "${gen_cmd[@]}"; then
      failed=true
    fi
  fi

  # Validate chains
  if [ "$failed" = false ] && { [ "$mode" = "1" ] || [ "$mode" = "2" ] || [ "$mode" = "4" ]; }; then
    if ! run_step "Validate Chains" python3 qa_generation/contractor_polish.py \
        --input "$raw_output" \
        --output "$validated_output" \
        --concurrency "$concurrency"; then
      failed=true
    fi
  fi

  # Generate report
  if [ "$failed" = false ] && { [ "$mode" = "1" ] || [ "$mode" = "2" ] || [ "$mode" = "5" ]; }; then
    if ! run_step "Generate Report" python3 qa_generation/validate_qa_dataset.py \
        --input "$validated_output" \
        --output "$report_output"; then
      failed=true
    fi
  fi

  # ── Results ──

  printf "\n"
  if [ "$failed" = true ]; then
    print_header "Pipeline finished with errors"
    print_error "One or more steps failed. Check the output above for details."
    exit 1
  fi

  print_header "Pipeline Complete"

  if [ "$mode" = "1" ] || [ "$mode" = "2" ] || [ "$mode" = "3" ]; then
    print_item "Raw chains:" "$raw_output"
    print_item "Run logs:" "$log_dir"
    print_item "Deliverable + CSV:" "(auto-named in output directory)"
  fi
  if [ "$mode" != "3" ] && [ "$mode" != "5" ]; then
    print_item "Validated chains:" "$validated_output"
  fi
  if [ "$mode" != "3" ] && [ "$mode" != "4" ]; then
    print_item "Report:" "$report_output"
  fi

  printf "\n"
  print_success "Done!"
  printf "\n"
}

main "$@"
