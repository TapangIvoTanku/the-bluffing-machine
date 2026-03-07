# The Bluffing Machine: Replication Repository

[![SSRN](https://img.shields.io/badge/SSRN-6361838-blue)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6361838)
[![License: MIT](https://img.shields.io/badge/Code-MIT_License-green)](LICENSE)
[![License: CC BY 4.0](https://img.shields.io/badge/Data-CC_BY_4.0-lightgrey)](https://creativecommons.org/licenses/by/4.0/)
[![Status: Under Review](https://img.shields.io/badge/Status-Under_Review-orange)]()

> **Paper:** "The Bluffing Machine: Large Language Models as Strategic Deceivers in Crisis Bargaining"
> **Author:** Tapang Ivo Tanku, University at Buffalo, SUNY
> **Contact:** tapangiv@buffalo.edu
> **SSRN Preprint:** [https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6361838](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6361838)
> **Submitted to:** *Journal of Global Security Studies* (Manuscript ID: JoGSS-2026-084)
> **Submitted:** March 6, 2026

---

## Overview

This repository contains the complete replication package for the paper. It includes all source code, raw experimental data, processed datasets, analysis scripts, and publication-quality figures needed to fully replicate all results reported in the paper.

The paper runs the first formal experimental test of whether Large Language Models (LLMs) can engage in **strategic deception** in a crisis bargaining context, using a signaling game framework grounded in Perfect Bayesian Equilibrium theory. We test three frontier LLMs — GPT-4.1-mini, GPT-4.1-nano, and Gemini-2.5-Flash — across 900 simulated crisis bargaining interactions and introduce the **Signal-Reason-Outcome (S-R-O)** qualitative coding framework to analyze model reasoning.

---

## Repository Structure

```
bluffing_machine_repo/
│
├── README.md                    ← This file
├── requirements.txt             ← Python dependencies
├── run_all.sh                   ← Single script to replicate all results
│
├── code/
│   ├── simulation_engine.py     ← Main experiment runner (LLM signaling game)
│   ├── sensitivity_analysis.py  ← Prompt sensitivity analysis (300 games)
│   ├── sro_qualitative_analysis.py ← S-R-O coding framework
│   ├── generate_figures.py      ← Publication-quality visualizations
│   ├── generate_figures_v3.py   ← Final figure versions
│   └── test_engine.py           ← Unit tests for simulation engine
│
├── data/
│   ├── raw/
│   │   ├── main_results_*.csv   ← Raw game-by-game output (900 games)
│   │   ├── main_results_*.json  ← JSON version of raw results
│   │   ├── summary_*.csv        ← Aggregated metrics per model × treatment
│   │   └── experiment_log_*.txt ← Full timestamped experiment logs
│   ├── sensitivity/
│   │   ├── sensitivity_results_*.csv  ← 300-game sensitivity analysis results
│   │   └── sensitivity_summary_*.csv  ← Aggregated sensitivity metrics
│   └── qualitative/
│       ├── sro_coded_traces.csv ← S-R-O coded reasoning traces
│       └── sro_summary.json     ← S-R-O coding summary statistics
│
├── figures/
│   ├── fig2_bluffing_lollipop.png     ← Bluffing rates by model & treatment
│   ├── fig3_grouped_bar.png           ← Grouped bar chart of key metrics
│   ├── fig4_reasoning_heatmap.png     ← S-R-O reasoning category heatmap
│   ├── fig5_payoff_distributions.png  ← Payoff distributions
│   ├── fig6_summary_dashboard.png     ← Summary dashboard
│   ├── fig7_sensitivity_analysis.png  ← Prompt sensitivity results
│   └── fig8_sro_qualitative_analysis.png ← S-R-O qualitative analysis
│
├── paper/
│   ├── main.tex                 ← Full LaTeX source (unblinded)
│   ├── main_blinded.tex         ← Blinded LaTeX source (for review)
│   ├── references.bib           ← Bibliography (34 references)
│   ├── main.pdf                 ← Compiled manuscript (unblinded)
│   └── main_blinded.pdf         ← Compiled manuscript (blinded)
│
└── docs/
    ├── codebook.md              ← Variable definitions and data dictionary
    ├── benchmark_comparison.md  ← PBE benchmark derivations
    └── figure_audit_final.md    ← Figure QA and audit notes
```

---

## Quick Start: Full Replication

### Prerequisites

```bash
# Python 3.11+
pip install -r requirements.txt

# Set your API keys
export OPENAI_API_KEY="your-openai-key-here"
export GEMINI_API_KEY="your-gemini-key-here"
```

### Run Everything with One Command

```bash
bash run_all.sh
```

This will:
1. Run all LLM signaling game simulations (900 games across 3 models × 2 treatments)
2. Run the prompt sensitivity analysis (300 games across 3 framing conditions)
3. Run the S-R-O qualitative coding analysis
4. Generate all 8 publication-quality figures
5. Compile the LaTeX paper to PDF

**Estimated runtime:** 3–5 hours (API rate limits are the bottleneck)

### Run Individual Components

```bash
# Run main simulations only
python3 code/simulation_engine.py

# Run sensitivity analysis
python3 code/sensitivity_analysis.py

# Run S-R-O qualitative analysis
python3 code/sro_qualitative_analysis.py

# Generate all figures
python3 code/generate_figures_v3.py
```

---

## Data Description

### `data/raw/main_results_*.csv`

Each row is one game in the signaling game experiment. **900 rows total** (3 models × 2 treatments × 150 simulations).

| Variable | Type | Description |
|---|---|---|
| `sim_id` | int | Simulation index within cell |
| `model_key` | str | API model identifier |
| `model_name` | str | Human-readable model name |
| `treatment` | str | `zero_shot` or `role_conditioned` |
| `sender_type` | str | True type of Sender: `HIGH` or `LOW` |
| `signal` | str | Signal sent: `ESCALATE` or `NEGOTIATE` |
| `action` | str | Receiver action: `ATTACK` or `BACK_DOWN` |
| `posterior_belief` | float | Receiver's posterior Pr(H\|signal) ∈ [0,1] |
| `rational_posterior` | float | Theoretical PBE benchmark posterior |
| `outcome` | str | `COERCION_SUCCESS`, `WAR_HIGH_RESOLVE`, `WAR_LOW_RESOLVE` |
| `sender_payoff` | float | Sender's realized payoff |
| `receiver_payoff` | float | Receiver's realized payoff |
| `is_bluff` | bool | True if LOW Resolve sender sent ESCALATE |
| `bluff_success` | bool | True if bluff caused Receiver to BACK_DOWN |
| `sender_reasoning` | str | Full chain-of-thought reasoning trace (Sender) |
| `receiver_reasoning` | str | Full chain-of-thought reasoning trace (Receiver) |
| `timestamp` | str | ISO 8601 timestamp of game execution |

### `data/sensitivity/sensitivity_results_*.csv`

Results from the 300-game prompt sensitivity analysis across three framing conditions: `neutral`, `diplomatic`, and `military`.

### `data/qualitative/sro_coded_traces.csv`

S-R-O coded reasoning traces. The **Signal-Reason-Outcome (S-R-O)** framework codes each model reasoning trace along three dimensions:
- **Signal**: What signal did the model choose and why?
- **Reason**: What justification category did the model invoke? (capability, risk avoidance, strategic deception, norm compliance, other)
- **Outcome**: How did the model evaluate the expected outcome?

---

## Key Results Summary

| Model | Treatment | Bluff Rate | vs. PBE (0.42) |
|---|---|---|---|
| GPT-4.1-mini | Zero-Shot | 0% | Far below equilibrium |
| GPT-4.1-mini | Role-Conditioned | 100% | Far above equilibrium |
| GPT-4.1-nano | Zero-Shot | ~0% | Far below equilibrium |
| GPT-4.1-nano | Role-Conditioned | ~50% | Near equilibrium (different mechanism) |
| Gemini-2.5-Flash | Zero-Shot | ~100% | Far above equilibrium |
| Gemini-2.5-Flash | Role-Conditioned | ~100% | Far above equilibrium |

*Full results with statistical details in `data/raw/` and in the paper.*

---

## Theoretical Benchmarks

The Perfect Bayesian Equilibrium (PBE) predictions used as benchmarks in the paper:

- **Pooling equilibrium bluff rate** (prior p = 0.5): 0.42
- **Separating equilibrium bluff rate**: 0.00
- **Rational Brier score** (perfect calibration): 0.00
- **Naive baseline Brier score** (always predict 0.5): 0.25

---

## Citation

If you use this code or data, please cite:

```bibtex
@unpublished{Tanku2026bluffing,
  title   = {The Bluffing Machine: Large Language Models as Strategic
             Deceivers in Crisis Bargaining},
  author  = {Tanku, Tapang Ivo},
  year    = {2026},
  note    = {Preprint. Under review at the Journal of Global Security Studies.
             SSRN Abstract ID: 6361838},
  url     = {https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6361838}
}
```

---

## License

- **Code:** [MIT License](LICENSE) — free to use, modify, and distribute with attribution
- **Data:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) — free to use with attribution
- **Paper:** All rights reserved pending journal publication

---

## Acknowledgements

The author thanks the University at Buffalo Department of Political Science for institutional support. Computational experiments were conducted using the OpenAI API and Google Gemini API.
