# Benchmarking Notes — Top Papers vs. The Bluffing Machine

## Paper 1: Rivera et al. (2024) — "Escalation Risks from LLMs in Military and Diplomatic Decision-Making"
Published: ACM FAccT 2024 (top CS/AI fairness venue). 124 citations in ~2 years.
Authors: Georgia Tech + Stanford (Jacquelyn Schneider — Hoover Wargaming Initiative)

### Their Research Design:
- 8 autonomous LLM nation agents per simulation (GPT-4, GPT-3.5, Claude 2, Llama-2)
- 3 distinct geopolitical scenarios (different starting conditions)
- 14 turns per simulation, 27 discrete action choices per turn
- Dynamic variables: military capacity, GDP, nuclear capabilities
- Escalation Scoring (ES) framework — custom metric based on political science literature
- JSON-structured outputs with chain-of-thought reasoning
- Temperature = 1.0 for GPT models, 0.9 for Llama-2
- Appendix: 57 pages, 46 figures, 11 tables
- Sensitivity analysis on prompt variations
- Qualitative + quantitative results

### What They Did Well:
1. Very detailed appendix with all prompts, nation descriptions, action lists
2. Escalation scoring framework grounded in IR literature (Kahn, Schelling, Patches)
3. Multiple scenarios with different starting conditions
4. Sensitivity analysis on prompt wording
5. Clear limitations section
6. Policy recommendations section
7. Released code and data publicly

### What They Did NOT Do:
1. No formal game-theoretic model — purely descriptive/empirical
2. No equilibrium benchmark — no way to say if behavior is "rational" or not
3. No fine-tuning experiment
4. No cross-model statistical comparison with significance tests
5. No Bayesian belief calibration analysis
6. No reputation/repeated game dynamics

## What Our Paper Has That They Don't:
1. Formal signaling game with Perfect Bayesian Equilibrium — gives a RATIONAL BENCHMARK
2. Bayesian belief calibration (Brier scores) — measures epistemic quality
3. Equilibrium Deviation Index — quantifies how far from rational play
4. Bluffing as the specific strategic behavior (not just escalation generally)
5. Explicit type asymmetry (High vs. Low Resolve) — the core of deterrence theory
6. Statistical significance testing (binomial tests, Wilson CIs)

## What We Are Missing Compared to Them:
1. SENSITIVITY ANALYSIS — they varied prompts and showed robustness; we did not
2. APPENDIX DEPTH — they have 57 pages of appendix with all prompts, all nation descriptions
3. MULTIPLE SCENARIOS — they ran 3 different geopolitical scenarios; we ran 1 signaling game
4. QUALITATIVE ANALYSIS — they coded the reasoning traces qualitatively; we only did keyword counts
5. POLICY RECOMMENDATIONS SECTION — they have a dedicated section; ours is in discussion
6. LIMITATIONS SECTION — needs to be more detailed and self-critical
7. SAMPLE SIZE — they ran many more simulations; our 900 is adequate but they had more conditions

## Gaps to Fix in Our Paper:
- Add prompt sensitivity analysis (vary the framing, show results are robust)
- Expand appendix with all prompts verbatim
- Add a second scenario (e.g., nuclear crisis variant)
- Strengthen limitations section
- Add explicit policy recommendations section
- Add qualitative coding of reasoning traces (not just keyword counts)

## Paper 2: Payne (2026) — "AI Arms and Influence: Frontier Models Exhibit Sophisticated Reasoning in Simulated Nuclear Crises"
Published: arXiv:2602.14740, February 2026. King's College London. 1 citation (brand new).

### Their Research Design:
- 3 frontier models: GPT-5.2, Claude Sonnet 4, Gemini 3 Flash
- 21 simulated nuclear crisis scenarios (tournament structure)
- 329 turns of play total
- ~780,000 words of structured reasoning generated
- Three-phase architecture per turn: Reflection → Forecast → Decision
- Analyzed: deception, credibility management, prediction accuracy, self-awareness
- Key finding: 95% nuclear signaling, models never chose accommodation or surrender
- "Deadline effect" — temporal framing dramatically changed behavior
- Validated against: Schelling (commitment), Kahn (escalation), Jervis (misperception)
- GitHub repo: https://github.com/kennethpayne01/project_kahn_public

### What They Did Well:
1. Reflection-Forecast-Decision architecture makes AI reasoning transparent
2. Tournament structure (21 games) provides statistical power
3. Explicitly tested against canonical IR theories (Schelling, Kahn, Jervis)
4. "Deadline effect" is a novel and policy-relevant finding
5. 780,000 words of reasoning = massive qualitative corpus
6. Public GitHub repo with all code and data

### What They Did NOT Do:
1. No formal game-theoretic model — no rational equilibrium benchmark
2. No statistical significance tests (no p-values, no confidence intervals)
3. No Bayesian belief calibration
4. No cross-treatment experimental design (no zero-shot vs. role-conditioned comparison)
5. No fine-tuning experiment
6. Only 3 models, no comparison across model sizes
7. No quantitative measure of "how far from rational" — purely descriptive

## OVERALL BENCHMARK ASSESSMENT

### Where Our Paper Beats Both:
1. FORMAL MODEL: We have a signaling game with PBE — neither paper has this. This is our biggest advantage.
2. RATIONAL BENCHMARK: We can say "GPT-4.1-mini deviates X% from the PBE" — they cannot.
3. BAYESIAN CALIBRATION: Brier scores — unique to our paper.
4. EXPERIMENTAL DESIGN: Zero-shot vs. role-conditioned — a clean causal comparison.
5. STATISTICAL RIGOR: Binomial tests, Wilson CIs, p-values — neither paper has this.

### Where We Fall Short:
1. SCALE: Payne has 329 turns, 780K words. Rivera has 57-page appendix, 46 figures. We have 900 games — adequate but not overwhelming.
2. SCENARIO VARIETY: Both papers have multiple scenarios. We have one signaling game structure.
3. QUALITATIVE DEPTH: Payne's reflection-forecast-decision architecture gives rich qualitative data. Our reasoning trace analysis is keyword-based only.
4. PROMPT SENSITIVITY: Rivera varies prompts. We do not test robustness to prompt wording.
5. POLICY SECTION: Both papers have strong policy recommendation sections. Ours needs strengthening.
6. APPENDIX: Both papers have extensive appendices with all prompts verbatim. Ours needs this.

### Priority Upgrades Needed:
CRITICAL:
- Add prompt sensitivity analysis (3 prompt variants, show results hold)
- Add full appendix with all prompts verbatim
- Strengthen policy recommendations section
- Strengthen limitations section with specific threats to validity

IMPORTANT:
- Add a second scenario variant (e.g., nuclear crisis framing)
- Add qualitative coding of reasoning traces (not just keywords)
- Increase sample size if possible (target 1,500-2,000 games)

NICE TO HAVE:
- Reflection-Forecast-Decision architecture (inspired by Payne)
- More models (if budget allows)
