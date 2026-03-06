# The Bluffing Machine: Replication Repository

> **Paper:** "The Bluffing Machine: Generative AI, Strategic Deception, and the Limits of Deterrence Theory"
> **Author:** Tapang Ivo Tanku, University at Buffalo, SUNY
> **Contact:** tapangiv@buffalo.edu
> **SSRN Preprint:** *(forthcoming)*
> **Status:** Under review

---

## Overview

This repository contains the complete replication package for the paper. It includes all source code, raw experimental data, processed datasets, analysis scripts, and publication-quality figures needed to fully replicate all results reported in the paper.

The paper runs the first formal experimental test of whether Large Language Models (LLMs) can engage in **strategic deception** in a crisis bargaining context, using a signaling game framework grounded in Perfect Bayesian Equilibrium theory.

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
│   ├── analyze_results.py       ← Metric computation and statistical analysis
│   ├── generate_figures.py      ← All publication-quality visualizations
│   └── utils.py                 ← Shared utilities
│
├── data/
│   ├── raw/
│   │   ├── main_results.csv     ← Raw game-by-game simulation output (900 games)
│   │   ├── reputation_results.json ← Repeated game sequences (120 sequences × 10 rounds)
│   │   └── experiment_log.txt   ← Full timestamped experiment log
│   └── processed/
│       ├── summary_by_model_treatment.csv  ← Aggregated metrics per cell
│       ├── calibration_data.csv            ← Posterior belief data for calibration curves
│       └── reputation_decay_data.csv       ← Round-by-round bluff success rates
│
├── figures/
│   ├── fig1_signaling_game_tree.png
│   ├── fig2_bluffing_frequency.png
│   ├── fig3_calibration_curves.png
│   ├── fig4_reputation_decay.png
│   ├── fig5_reasoning_heatmap.png
│   ├── fig6_payoff_distributions.png
│   └── fig7_latency_tokens_dashboard.png
│
├── paper/
│   ├── bluffing_machine.tex     ← Full LaTeX source
│   ├── references.bib           ← Bibliography
│   └── bluffing_machine.pdf     ← Compiled manuscript
│
└── docs/
    ├── codebook.md              ← Variable definitions and data dictionary
    ├── experimental_design.md   ← Detailed methodology documentation
    └── CHANGELOG.md             ← Version history
```

---

## Quick Start: Full Replication

### Prerequisites

```bash
# Python 3.11+
pip install -r requirements.txt

# Set your OpenAI API key
export OPENAI_API_KEY="your-key-here"
```

### Run Everything with One Command

```bash
bash run_all.sh
```

This will:
1. Run all LLM signaling game simulations (900 games across 3 models × 2 treatments)
2. Run the reputation decay experiment (120 sequences × 10 rounds)
3. Compute all metrics and generate processed datasets
4. Generate all 7 publication-quality figures
5. Compile the LaTeX paper to PDF

**Estimated runtime:** 3–5 hours (API rate limits are the bottleneck)

### Run Individual Components

```bash
# Run simulations only
python3 code/simulation_engine.py

# Analyze existing results
python3 code/analyze_results.py --input data/raw/main_results.csv

# Generate figures only
python3 code/generate_figures.py --input data/processed/
```

---

## Data Description

### `data/raw/main_results.csv`

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
| `sender_confidence` | float | Sender's self-reported confidence ∈ [0,1] |
| `receiver_confidence` | float | Receiver's self-reported confidence ∈ [0,1] |
| `sender_latency_ms` | int | API response time for Sender (milliseconds) |
| `receiver_latency_ms` | int | API response time for Receiver (milliseconds) |
| `sender_tokens` | int | Total tokens used by Sender call |
| `receiver_tokens` | int | Total tokens used by Receiver call |
| `game_duration_ms` | int | Total wall-clock time for game (milliseconds) |
| `timestamp` | str | ISO 8601 timestamp of game execution |

### `data/raw/reputation_results.json`

JSON array of repeated game sequences. Each sequence has:
- `model_key`, `model_name`, `seq_id`
- `rounds`: array of 11 rounds (round 0 = forced failed bluff; rounds 1–10 = free play)

### `data/processed/summary_by_model_treatment.csv`

Aggregated metrics per model × treatment cell:

| Variable | Description |
|---|---|
| `bluff_rate` | Proportion of LOW Resolve senders sending ESCALATE |
| `bluff_success_rate` | Proportion of bluffs that caused BACK_DOWN |
| `brier_score` | Brier score of Receiver posterior beliefs (lower = better calibrated) |
| `edi` | Equilibrium Deviation Index (mean \|posterior − rational_posterior\|) |
| `avg_sender_payoff` | Mean sender payoff across all games |
| `avg_receiver_payoff` | Mean receiver payoff across all games |
| `avg_latency_ms` | Mean game duration in milliseconds |
| `total_tokens` | Total API tokens consumed in cell |
| `cell_duration_s` | Wall-clock time to complete cell (seconds) |

---

## Key Results Summary

| Model | Treatment | Bluff Rate | PBE Benchmark | Brier Score | EDI |
|---|---|---|---|---|---|
| GPT-4.1-mini | Zero-Shot | *see paper* | 0.42 | *see paper* | *see paper* |
| GPT-4.1-mini | Role-Conditioned | *see paper* | 0.42 | *see paper* | *see paper* |
| GPT-4.1-nano | Zero-Shot | *see paper* | 0.42 | *see paper* | *see paper* |
| GPT-4.1-nano | Role-Conditioned | *see paper* | 0.42 | *see paper* | *see paper* |
| Gemini-2.5-Flash | Zero-Shot | *see paper* | 0.42 | *see paper* | *see paper* |
| Gemini-2.5-Flash | Role-Conditioned | *see paper* | 0.42 | *see paper* | *see paper* |

*Full results in `data/processed/summary_by_model_treatment.csv` and in the paper.*

---

## Theoretical Benchmarks

The Perfect Bayesian Equilibrium (PBE) predictions used as benchmarks in the paper:

- **Pooling equilibrium bluff rate** (prior p = 0.5): 0.42
- **Separating equilibrium bluff rate**: 0.00
- **Rational Brier score** (perfect calibration): 0.00
- **Naive baseline Brier score** (always predict 0.5): 0.25
- **Rational reputation decay**: exponential, ~31 pp decline over 10 rounds

---

## Citation

If you use this code or data, please cite:

```bibtex
@article{Tanku2026bluffing,
  title   = {The Bluffing Machine: Generative AI, Strategic Deception,
             and the Limits of Deterrence Theory},
  author  = {Tanku, Tapang Ivo},
  journal = {Working Paper, University at Buffalo},
  year    = {2026},
  url     = {https://github.com/TapangIvoTanku/bluffing-machine}
}
```

---

## License

- **Code:** MIT License — free to use, modify, and distribute with attribution
- **Data:** CC BY 4.0 — free to use with attribution
- **Paper:** All rights reserved pending journal publication

---

## Acknowledgements

The author thanks the University at Buffalo Department of Political Science for institutional support. Computational experiments were conducted using the OpenAI API.
