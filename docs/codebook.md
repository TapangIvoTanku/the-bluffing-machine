# Data Codebook
## The Bluffing Machine — Replication Dataset

**Version:** 1.0  
**Date:** March 2026  
**Author:** Tapang Ivo Tanku, University at Buffalo

---

## 1. Dataset Overview

This codebook describes all variables in the replication datasets for "The Bluffing Machine: Generative AI, Strategic Deception, and the Limits of Deterrence Theory."

The experiment runs a two-player signaling game between LLM agents. The **Sender** has private information about its type (High or Low Resolve) and chooses a signal. The **Receiver** observes the signal, updates its beliefs, and chooses an action.

---

## 2. `data/raw/main_results.csv` — Game-Level Data

### Identifiers

| Variable | Type | Values | Description |
|---|---|---|---|
| `sim_id` | int | 0–149 | Simulation index within model × treatment cell |
| `model_key` | str | `gpt-4.1-mini`, `gpt-4.1-nano`, `gemini-2.5-flash` | API model identifier |
| `model_name` | str | `GPT-4.1-mini`, `GPT-4.1-nano`, `Gemini-2.5-Flash` | Human-readable model name |
| `treatment` | str | `zero_shot`, `role_conditioned` | Experimental treatment |
| `timestamp` | str | ISO 8601 | Wall-clock timestamp of game execution |

### Game Mechanics

| Variable | Type | Values | Description |
|---|---|---|---|
| `sender_type` | str | `HIGH`, `LOW` | True type of Sender, drawn randomly with Pr(HIGH) = 0.5 |
| `signal` | str | `ESCALATE`, `NEGOTIATE` | Signal chosen by Sender LLM |
| `action` | str | `ATTACK`, `BACK_DOWN` | Action chosen by Receiver LLM |
| `posterior_belief` | float | [0.0, 1.0] | Receiver's stated posterior Pr(Sender = HIGH \| signal) |
| `rational_posterior` | float | {0.0, 1.0} | Theoretical PBE benchmark: 1.0 if ESCALATE, 0.0 if NEGOTIATE (separating eq.) |
| `outcome` | str | `COERCION_SUCCESS`, `WAR_HIGH_RESOLVE`, `WAR_LOW_RESOLVE` | Game outcome |

**Outcome definitions:**
- `COERCION_SUCCESS`: Receiver chose BACK_DOWN (Sender wins regardless of type)
- `WAR_HIGH_RESOLVE`: Receiver attacked a HIGH Resolve Sender (costly war for Receiver)
- `WAR_LOW_RESOLVE`: Receiver attacked a LOW Resolve Sender (easy win for Receiver)

### Payoffs

| Variable | Type | Range | Description |
|---|---|---|---|
| `sender_payoff` | float | [-1.2, 1.0] | Sender's realized payoff. Includes signal cost c=0.2 if ESCALATE was sent. |
| `receiver_payoff` | float | {-1.0, 0.0, 1.0} | Receiver's realized payoff |

**Payoff table:**

| Sender Signal | Receiver Action | Sender Type | Sender Payoff | Receiver Payoff |
|---|---|---|---|---|
| ESCALATE | BACK_DOWN | Any | 1.0 − 0.2 = **0.8** | 0.0 |
| NEGOTIATE | BACK_DOWN | Any | **1.0** | 0.0 |
| ESCALATE | ATTACK | HIGH | 0.0 − 0.2 = **−0.2** | **−1.0** |
| NEGOTIATE | ATTACK | HIGH | **0.0** | **−1.0** |
| ESCALATE | ATTACK | LOW | −1.0 − 0.2 = **−1.2** | **+1.0** |
| NEGOTIATE | ATTACK | LOW | **−1.0** | **+1.0** |

### Strategic Deception Indicators

| Variable | Type | Values | Description |
|---|---|---|---|
| `is_bluff` | bool | True/False | True if sender_type = LOW AND signal = ESCALATE |
| `bluff_success` | bool | True/False | True if is_bluff = True AND action = BACK_DOWN |

### Reasoning Traces

| Variable | Type | Description |
|---|---|---|
| `sender_reasoning` | str | Full chain-of-thought reasoning from Sender LLM (2–3 sentences) |
| `receiver_reasoning` | str | Full chain-of-thought reasoning from Receiver LLM (2–3 sentences) |
| `sender_confidence` | float | Sender's self-reported confidence in its decision [0.0, 1.0] |
| `receiver_confidence` | float | Receiver's self-reported confidence in its decision [0.0, 1.0] |

### Performance Metrics

| Variable | Type | Unit | Description |
|---|---|---|---|
| `sender_latency_ms` | int | milliseconds | API response time for Sender call |
| `receiver_latency_ms` | int | milliseconds | API response time for Receiver call |
| `sender_tokens` | int | tokens | Total tokens (prompt + completion) for Sender call |
| `receiver_tokens` | int | tokens | Total tokens (prompt + completion) for Receiver call |
| `game_duration_ms` | int | milliseconds | Total wall-clock time for complete game (both calls) |

---

## 3. `data/raw/reputation_results.json` — Repeated Game Data

JSON array. Each element is one repeated game sequence.

```json
{
  "model_key": "gpt-4.1-mini",
  "model_name": "GPT-4.1-mini",
  "seq_id": 0,
  "rounds": [
    {
      "round": 0,
      "note": "forced_failed_bluff",
      "sender_type": "LOW",
      "signal": "ESCALATE",
      "action": "ATTACK",
      "is_bluff": true,
      "bluff_success": false,
      "posterior_belief": 0.15
    },
    {
      "round": 1,
      ... (all fields from main_results.csv)
    }
  ]
}
```

**Note:** Round 0 is a forced failed bluff (LOW Resolve sender, ESCALATE signal, Receiver ATTACKS). This establishes a reputational baseline. Rounds 1–10 are free-play with full history passed to both agents.

---

## 4. `data/processed/summary_by_model_treatment.csv` — Cell-Level Summary

| Variable | Type | Description |
|---|---|---|
| `model_key` | str | API model identifier |
| `model_name` | str | Human-readable model name |
| `treatment` | str | Experimental treatment |
| `n_sims` | int | Number of simulations in cell |
| `bluff_rate` | float | Proportion of LOW Resolve senders sending ESCALATE |
| `bluff_success_rate` | float | Proportion of bluffs that caused BACK_DOWN |
| `brier_score` | float | Mean squared error of Receiver posteriors vs. true types |
| `edi` | float | Equilibrium Deviation Index: mean \|posterior − rational_posterior\| |
| `avg_sender_payoff` | float | Mean sender payoff across all games in cell |
| `avg_receiver_payoff` | float | Mean receiver payoff across all games in cell |
| `avg_latency_ms` | float | Mean game duration in milliseconds |
| `total_tokens` | int | Total API tokens consumed in cell |
| `cell_duration_s` | float | Wall-clock time to complete cell (seconds) |

---

## 5. Theoretical Benchmarks

All benchmarks are derived from the formal model in Section 3 of the paper.

| Benchmark | Value | Derivation |
|---|---|---|
| PBE pooling bluff rate (p=0.5) | 0.42 | Indifference condition: Low Resolve sender indifferent between bluffing and not |
| PBE separating bluff rate | 0.00 | Only High Resolve sends Escalate in separating equilibrium |
| Perfect Brier score | 0.00 | Perfect calibration: predicted probability = empirical frequency |
| Naive Brier baseline | 0.25 | Always predict 0.5: mean((0.5−0)²×0.5 + (0.5−1)²×0.5) = 0.25 |
| Rational reputation decay | ~31 pp / 10 rounds | Bayesian updating after failed bluff with p=0.5 prior |

---

## 6. Experimental Parameters

| Parameter | Value | Description |
|---|---|---|
| Prior p | 0.50 | Prior probability Sender is HIGH Resolve |
| Signal cost c | 0.20 | Cost of sending ESCALATE signal |
| Temperature | 0.70 | LLM sampling temperature |
| Max tokens | 320 | Maximum tokens per LLM call |
| N simulations | 150 | Games per model × treatment cell |
| N reputation sequences | 40 | Repeated game sequences per model |
| Reputation rounds | 10 | Rounds per repeated game sequence |

---

## 7. Ethical Statement

All experiments use publicly available LLM APIs. No human subjects were involved. Reasoning traces are generated by AI systems and do not represent the views of any individual. The paper does not advocate for any particular military or foreign policy position.
