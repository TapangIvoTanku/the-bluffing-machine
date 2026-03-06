#!/usr/bin/env bash
# =============================================================================
# THE BLUFFING MACHINE — MASTER REPLICATION SCRIPT
# Runs all experiments, analysis, and figure generation in one command.
# Usage: bash run_all.sh
# Estimated runtime: 3–5 hours (API rate limits are the bottleneck)
# =============================================================================

set -e  # Exit on any error

echo "============================================================"
echo "  THE BLUFFING MACHINE — FULL REPLICATION PIPELINE"
echo "  $(date)"
echo "============================================================"

# Check for API key
if [ -z "$OPENAI_API_KEY" ]; then
    echo "ERROR: OPENAI_API_KEY environment variable not set."
    echo "Please run: export OPENAI_API_KEY='your-key-here'"
    exit 1
fi

# Create output directories
mkdir -p data/raw data/processed figures

echo ""
echo "STEP 1/4: Running LLM signaling game simulations..."
echo "    (900 games × 3 models × 2 treatments)"
python3 code/simulation_engine.py
echo "    ✓ Simulations complete"

echo ""
echo "STEP 2/4: Analyzing results and computing metrics..."
python3 code/analyze_results.py
echo "    ✓ Analysis complete"

echo ""
echo "STEP 3/4: Generating publication-quality figures..."
python3 code/generate_figures.py
echo "    ✓ Figures generated"

echo ""
echo "STEP 4/4: Compiling LaTeX paper..."
cd paper
pdflatex -interaction=nonstopmode bluffing_machine.tex > /dev/null
bibtex bluffing_machine > /dev/null
pdflatex -interaction=nonstopmode bluffing_machine.tex > /dev/null
pdflatex -interaction=nonstopmode bluffing_machine.tex > /dev/null
cd ..
echo "    ✓ PDF compiled"

echo ""
echo "============================================================"
echo "  REPLICATION COMPLETE"
echo "  Results:  data/processed/"
echo "  Figures:  figures/"
echo "  Paper:    paper/bluffing_machine.pdf"
echo "  $(date)"
echo "============================================================"
