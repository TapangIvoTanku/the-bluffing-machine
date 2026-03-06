# Figure QA Notes — Premium v2

## Fig 1 (Game Tree): PASS
- Numeric payoffs clearly shown: (-0.2,-1), (+0.8,0), etc.
- Legend box in bottom-left explains (Sender, Receiver) format
- Nodes clearly labeled: Nature (red), S(H)/S(L) (blue), R (grey)
- BLUFF annotation on the Escalate(E) branch from S(L) — clear
- No overlapping text

## Fig 2 (Lollipop): PASS
- Zero-value dots are large and clearly visible
- Labels (0%, 78%, 89%, 93%, 100%) are all well-positioned with no overlap
- PBE dashed line clearly marked at 42%
- Footer note explains CI, PBE, and N

## Fig 3 (Calibration): PASS
- Shaded overconfident/underconfident regions added
- n= counts per bin shown above each dot
- Brier score box in top-left of each panel
- Subtitle explains interpretation of diagonal

## Fig 4 (Heatmap): PASS
- Model group labels (GPT-4.1-mini, GPT-4.1-nano, Gemini) shown below in brand colors
- White separators between model groups
- Cell text is white on dark, dark on light — good contrast
- Colorbar with % labels on right

## Fig 5 (Payoffs): MOSTLY PASS — minor issue
- Horizontal layout is much more readable
- Median annotations (Md=0.80, Md=0.00) on right side — no overlap
- IQR boxes visible
- ISSUE: All sender medians = 0.80 and all receiver medians = 0.00 — correct data
  but the violin shapes for GPT-4.1-nano look very flat/elongated (wide IQR)
  This is genuine data, not a rendering issue.

## Fig 6 (Dashboard): PASS
- Consistent x-labels: GPT-mini, GPT-nano, Gemini — no hyphenation issues
- Zero reference line in payoff panels
- PBE=42% annotation on Bluff Rate panel
- Value labels properly positioned above bars
- Negative bars (Gemini receiver payoff = -0.07) correctly shown below zero
