# Figure Audit v3 — Overlap and Visual Issues

## Fig 2 (Lollipop) — CLEAN ✅
- No overlaps. All labels clearly separated.
- "0%" labels for zero-value GPT models are visible at the left edge.
- p-value and percentage on separate lines — no collision.
- n= labels below dots — clear.
- VERDICT: No fixes needed.

## Fig 3 (Grouped Bar) — MOSTLY CLEAN, 2 ISSUES ⚠️
- ISSUE 1: "100%" label for GPT-mini role-conditioned overlaps with the annotation arrow box at the top. The "100%" text and the yellow annotation box are very close.
- ISSUE 2: The "Bluff Success Rate" title in the middle panel is partially overlapping with "100%100%" value labels at the top — the title text runs into the bar value labels.
- VERDICT: Fix title position and annotation box placement.

## Fig 4 (Heatmap) — CRITICAL OVERLAP ❌
- CRITICAL: The subtitle "What Concepts Do LLMs Invoke When Making Strategic Signaling Decisions?" is overlapping with the model group labels "GPT-mini", "GPT-nano", "Gemini" that appear at the top of the heatmap columns. The colored model labels (GPT-mini in blue, GPT-nano in green, Gemini in red) are printed directly on top of the subtitle text.
- The subtitle text and the column group labels are colliding badly at the top of the figure.
- VERDICT: Remove the top column group labels OR increase top margin significantly.

## Fig 5 (Violin) — 2 ISSUES ⚠️
- ISSUE 1: In the Receiver Payoff panel, the x-axis label "Role-Conditioned" for GPT-mini is cut off — shows "Role-Conditioned" but the last part is truncated.
- ISSUE 2: The legend box is partially overlapping the bottom of the GPT-nano violin in the Receiver panel.
- ISSUE 3: All Md=0.00 labels are crowded together at the zero line — they overlap each other in the Receiver panel.
- VERDICT: Increase bottom margin, move legend below figure, space out Md= labels.

## Fig 6 (Dashboard) — 3 ISSUES ⚠️
- ISSUE 1: "Bluff Success Rate" panel title overlaps with "100%100%100%" value labels at the top of the bars — the title text runs directly into the percentage labels.
- ISSUE 2: "Brier Score (↓ better)" panel — the "0.31" label is clipped at the top edge of the panel.
- ISSUE 3: "Avg. Sender Payoff" panel — "0.79" and "0.80" labels are very close together and nearly touching.
- VERDICT: Increase ylim on affected panels, add padding above bars, fix title positions.

## Fig 8 (S-R-O) — 1 REMAINING ISSUE ⚠️
- ISSUE: Panel C annotation box (yellow "Key finding" box) is still partially overlapping with the bar for "Bayesian Update". The box is inside the bar area.
- Panel D heatmap is now working correctly — no issues.
- VERDICT: Move annotation box outside the plot area or to a corner that doesn't overlap bars.
