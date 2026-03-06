# Benchmark Comparison: The Bluffing Machine vs. Top Published Papers

This table provides a rigorous, section-by-section comparison of our paper against the two most relevant and high-quality published papers in the AI + security space: Rivera et al. (2024) and Payne (2026).

| Section | Our Paper ("The Bluffing Machine") | Rivera et al. (2024) | Payne (2026) | Verdict |
|---|---|---|---|---|
| **Theoretical Framework** | **Signaling Game with Perfect Bayesian Equilibrium (PBE)** | Descriptive, based on IR literature (Kahn, Schelling) | Descriptive, based on IR literature (Schelling, Jervis) | **We are stronger.** Our formal model provides a rational benchmark that neither paper has. |
| **Research Question** | Can LLMs learn to bluff and manage credibility? | Do LLMs escalate conflict in wargames? | How do LLMs reason in nuclear crises? | **We are more focused.** Our question is more specific and theoretically grounded. |
| **Experimental Design** | **2x3 factorial design (Zero-Shot vs. Role-Conditioned) x 3 models** | 3 scenarios x 5 models | 21 scenarios x 3 models (tournament) | **They are stronger on scenario variety.** We have a cleaner causal comparison, but they have more diverse contexts. |
| **Methodology** | Signaling game simulation, Bayesian belief calibration, EDI | Wargame simulation, custom escalation score (ES) | Crisis simulation, Reflection-Forecast-Decision architecture | **Mixed.** Our methodology is more quantitatively rigorous (PBE, Brier scores). Payne's is more qualitatively rich. |
| **Data** | 900 real LLM simulation runs | Not specified, but likely thousands of runs | 329 turns, 780,000 words of reasoning | **They are stronger on scale.** Payne's qualitative corpus is massive. |
| **Statistical Analysis** | **Binomial tests, Wilson CIs, p-values** | Descriptive statistics, no significance tests | Descriptive statistics, no significance tests | **We are much stronger.** We are the only paper with formal statistical tests. |
| **Results** | Quantitative findings on bluffing rates, calibration, EDI | Qualitative and quantitative findings on escalation patterns | Qualitative and quantitative findings on nuclear signaling | **We are more precise.** Our results are tied directly to the formal model and statistical tests. |
| **Visualizations** | 6 premium figures (lollipop, heatmap, violin, etc.) | 46 figures (mostly bar charts and line graphs) | Not specified, but likely many figures | **We are stronger on quality.** Our figures are more modern and interpretable. |
| **Qualitative Analysis** | Keyword-based reasoning trace analysis | Qualitative coding of reasoning traces | Deep qualitative analysis via R-F-D architecture | **They are much stronger.** Payne's R-F-D architecture is a major innovation we lack. |
| **Sensitivity Analysis** | **None** | **Yes (prompt variations)** | **Yes (deadline effect)** | **We are weaker.** This is a critical gap we must fix. |
| **Appendix** | Minimal | **57 pages, all prompts verbatim** | Not specified, but likely extensive | **We are much weaker.** Our appendix needs to be dramatically expanded. |
| **Replication** | Full public GitHub repo with code, data, figures | Public GitHub repo | Public GitHub repo | **All are strong.** This is now the standard. |
| **Policy Relevance** | Strong implications for AI in diplomacy and warfare | Strong policy recommendations section | Strong policy recommendations section | **We are comparable, but need a dedicated section.** |

## Overall Assessment and Path Forward

Our paper has a **stronger theoretical core and more rigorous quantitative methodology** than either of the benchmark papers. The use of a formal game model and statistical testing is a major advantage that makes our paper stand out.

However, we are **weaker on qualitative depth, scenario variety, and robustness checks (sensitivity analysis).** The lack of a detailed appendix and a dedicated policy recommendations section also needs to be addressed.

To elevate our paper to a level that is not just competitive but *superior* to these published works, we need to address these gaps directly. The next step is to upgrade the paper based on this benchmark analysis.
