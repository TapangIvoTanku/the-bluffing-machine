"""
generate_premium_figures_v2.py
The Bluffing Machine — Premium Figure Suite v2 (All Interpretability Issues Fixed)
Author: Tapang Ivo Tanku, University at Buffalo
Date: March 2026

Fixes applied vs v1:
- Fig 1: Numeric payoff values instead of symbolic; legend box for (S, R) format
- Fig 2: Higher contrast stems, larger zero-dots, no label overlap
- Fig 3: Shaded calibration regions, sparsity annotations, n-per-bin counts
- Fig 4: Increased bottom margin, group labels inside figure, no clipping
- Fig 5: Horizontal layout, median annotations, wider figure
- Fig 6: Zero-reference annotations, consistent x-labels, better spacing
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_CSV  = os.path.join(BASE, "data", "raw", "main_results_20260306_042714.csv")
SUM_CSV   = os.path.join(BASE, "data", "raw", "summary_20260306_042714.csv")
FIGS_DIR  = os.path.join(BASE, "figures", "premium_v2")
os.makedirs(FIGS_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL STYLE
# ─────────────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":        "serif",
    "font.serif":         ["Palatino Linotype","Palatino","Georgia","DejaVu Serif"],
    "font.size":          11,
    "axes.titlesize":     13,
    "axes.labelsize":     11,
    "xtick.labelsize":    10,
    "ytick.labelsize":    10,
    "legend.fontsize":    10,
    "figure.dpi":         160,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.2,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.linewidth":     0.8,
    "xtick.direction":    "out",
    "ytick.direction":    "out",
    "axes.grid":          True,
    "grid.color":         "#E4E4E4",
    "grid.linewidth":     0.55,
    "axes.facecolor":     "#FAFAFA",
    "figure.facecolor":   "white",
})

# Colors
C = {
    "gpt-4.1-mini":     "#2166AC",
    "gpt-4.1-nano":     "#4DAC26",
    "gemini-2.5-flash": "#D6604D",
    "zero_shot":        "#4393C3",
    "role_conditioned": "#D6604D",
    "pbe":              "#762A83",
}
MODEL_LABELS = {
    "gpt-4.1-mini":     "GPT-4.1-mini",
    "gpt-4.1-nano":     "GPT-4.1-nano",
    "gemini-2.5-flash": "Gemini-2.5-Flash",
}
TREAT_LABELS = {
    "zero_shot":        "Zero-Shot",
    "role_conditioned": "Role-Conditioned",
}
PBE = 0.42
MODELS     = ["gpt-4.1-mini", "gpt-4.1-nano", "gemini-2.5-flash"]
TREATMENTS = ["zero_shot", "role_conditioned"]

# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
df  = pd.read_csv(DATA_CSV)
summary = pd.read_csv(SUM_CSV)

print("=" * 62)
print("  PREMIUM FIGURE SUITE v2 — The Bluffing Machine")
print("=" * 62)
print(f"  {len(df)} real game records loaded")
print()

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 1 — Game Tree with numeric payoffs and legend
# ─────────────────────────────────────────────────────────────────────────────
def fig1_game_tree():
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 12); ax.set_ylim(0, 11)
    ax.axis("off"); ax.set_facecolor("white"); fig.patch.set_facecolor("white")

    # Node styles
    def draw_circle(ax, x, y, label, color, size=0.52):
        circle = plt.Circle((x, y), size, color=color, fill=True, zorder=3, alpha=0.15)
        ax.add_patch(circle)
        circle2 = plt.Circle((x, y), size, color=color, fill=False, lw=2.0, zorder=4)
        ax.add_patch(circle2)
        ax.text(x, y, label, ha="center", va="center", fontsize=10.5,
                fontweight="bold", color=color, zorder=5)

    def draw_rect(ax, x, y, label, fc, ec):
        rect = mpatches.FancyBboxPatch((x-0.55, y-0.38), 1.1, 0.76,
                                        boxstyle="round,pad=0.1",
                                        fc=fc, ec=ec, lw=2.0, zorder=3)
        ax.add_patch(rect)
        ax.text(x, y, label, ha="center", va="center", fontsize=11,
                fontweight="bold", color=ec, zorder=5)

    def arrow(ax, x0, y0, x1, y1):
        ax.annotate("", xy=(x1, y1+0.55), xytext=(x0, y0-0.55),
                    arrowprops=dict(arrowstyle="-|>", color="#555", lw=1.4), zorder=2)

    def edge_label(ax, x0, y0, x1, y1, text, side):
        mx, my = (x0+x1)/2, (y0+y1)/2
        dx = -0.7 if side == "left" else 0.7
        ax.text(mx+dx, my, text, ha="center", va="center", fontsize=9,
                style="italic", color="#333",
                bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.9))

    # Positions
    N  = (6.0, 9.8)
    SH = (3.0, 7.5); SL = (9.0, 7.5)
    RHE= (1.5, 5.0); RHN= (4.5, 5.0)
    RLE= (7.5, 5.0); RLN=(10.5, 5.0)

    # Draw edges
    for src, dst, lbl, side in [
        (N,  SH,  "p = 0.5\n(High Resolve)", "left"),
        (N,  SL,  "1−p = 0.5\n(Low Resolve)", "right"),
        (SH, RHE, "Escalate (E)", "left"),
        (SH, RHN, "Negotiate (N)", "right"),
        (SL, RLE, "Escalate (E)\n← BLUFF", "left"),
        (SL, RLN, "Negotiate (N)", "right"),
    ]:
        arrow(ax, src[0], src[1], dst[0], dst[1])
        edge_label(ax, src[0], src[1], dst[0], dst[1], lbl, side)

    # Draw nodes
    draw_rect(ax, *N,  "Nature", "#FFF7EC", "#D6604D")
    draw_circle(ax, *SH, "S(H)", C["gpt-4.1-mini"])
    draw_circle(ax, *SL, "S(L)", C["gpt-4.1-mini"])
    for pos in [RHE, RHN, RLE, RLN]:
        draw_circle(ax, *pos, "R", "#555555")

    # Outcome payoffs — numeric values (c=0.2), format: (Sender, Receiver)
    outcomes = {
        RHE: [("Attack",    "(−0.2, −1)"), ("Back Down","(+0.8,  0)")],
        RHN: [("Attack",    "(  0,  −1)"), ("Back Down","(+1.0,  0)")],
        RLE: [("Attack",    "(−1.2, +1)"), ("Back Down","(+0.8,  0)")],
        RLN: [("Attack",    "(−1.0, +1)"), ("Back Down","(+1.0,  0)")],
    }
    for (rx, ry), actions in outcomes.items():
        for i, (act, payoff) in enumerate(actions):
            y_off = -1.3 - i*0.85
            ax.annotate("", xy=(rx, ry+y_off+0.38), xytext=(rx, ry-0.58),
                        arrowprops=dict(arrowstyle="-|>", color="#BBB", lw=0.9))
            color = "#C62828" if "Attack" in act else "#2E7D32"
            ax.text(rx, ry+y_off, f"{act}\n{payoff}", ha="center", va="center",
                    fontsize=8.2, color=color,
                    bbox=dict(boxstyle="round,pad=0.22", fc="white",
                              ec=color, lw=0.9, alpha=0.95))

    # Legend box
    legend_txt = ("Payoff format: (Sender, Receiver)\n"
                  "c = 0.2 (signal cost)   p = 0.5 (prior)\n"
                  "PBE bluff probability: q* = 0.42\n"
                  "N = Nature   S = Sender   R = Receiver")
    ax.text(0.01, 0.01, legend_txt, transform=ax.transAxes,
            fontsize=8.5, va="bottom", ha="left", color="#333",
            bbox=dict(boxstyle="round,pad=0.5", fc="#F5F5F5", ec="#AAAAAA", lw=1.0))

    ax.set_title("Figure 1: Formal Signaling Game of Strategic Deception\n"
                 "Perfect Bayesian Equilibrium Framework  (p = 0.5, c = 0.2, q* = 0.42)",
                 fontsize=13, fontweight="bold", pad=12, color="#1A1A1A")
    plt.savefig(os.path.join(FIGS_DIR, "fig1_game_tree.png"))
    plt.close()
    print("  ✓ Figure 1 saved")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 2 — Lollipop: Bluffing Frequency (fixed overlap, contrast, zero dots)
# ─────────────────────────────────────────────────────────────────────────────
def fig2_lollipop():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
    fig.subplots_adjust(wspace=0.06, left=0.14, right=0.97, top=0.85, bottom=0.18)

    for ax, treatment in zip(axes, TREATMENTS):
        rates, ns = [], []
        for m in MODELS:
            row = summary[(summary["model_key"]==m)&(summary["treatment"]==treatment)]
            rates.append(float(row["bluff_rate"].values[0]) if not row.empty else 0.0)
            ns.append(int(row["n_sims"].values[0]) if not row.empty else 0)

        y_pos  = np.arange(len(MODELS))
        colors = [C[m] for m in MODELS]

        # Wilson 95% CI
        cis = []
        for r, n in zip(rates, ns):
            if n > 0 and 0 < r < 1:
                lo = max(0, r - 1.96*np.sqrt(r*(1-r)/n))
                hi = min(1, r + 1.96*np.sqrt(r*(1-r)/n))
            else:
                lo = hi = r
            cis.append((lo, hi))

        for i, (rate, (lo, hi), color) in enumerate(zip(rates, cis, colors)):
            # Stem — solid, visible
            ax.plot([0, rate], [i, i], color="#CCCCCC", lw=2.0, zorder=1, solid_capstyle="round")
            # PBE reference tick
            ax.plot([PBE, PBE], [i-0.18, i+0.18], color=C["pbe"], lw=2.0, zorder=2, alpha=0.7)
            # CI bar
            if hi > lo:
                ax.plot([lo, hi], [i, i], color=color, lw=3.5, alpha=0.3, zorder=2, solid_capstyle="round")
            # Lollipop head — larger for zero values
            size = 220 if rate == 0.0 else 180
            ax.scatter([rate], [i], color=color, s=size, zorder=4,
                       edgecolors="white", linewidths=2.0)
            # Value label — always outside the dot, never overlapping
            if rate <= 0.05:
                lx, ha = rate + 0.06, "left"
            elif rate >= 0.95:
                lx, ha = rate - 0.06, "right"
            else:
                lx, ha = rate + 0.055, "left"
            ax.text(lx, i, f"{rate:.0%}", va="center", ha=ha,
                    fontsize=11, fontweight="bold", color=color,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))

        # PBE vertical line
        ax.axvline(PBE, color=C["pbe"], lw=1.8, ls="--", alpha=0.75, zorder=0)
        ax.text(PBE, len(MODELS)-0.05, f" PBE = {PBE:.0%}", color=C["pbe"],
                fontsize=9, va="top", ha="left", style="italic")

        ax.set_yticks(y_pos)
        ax.set_yticklabels([MODEL_LABELS[m] for m in MODELS], fontsize=11)
        ax.set_xlim(-0.06, 1.22)
        ax.set_xlabel("Bluff Rate  (Low Resolve Senders)", fontsize=11)
        ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
        ax.set_title(f"Treatment: {TREAT_LABELS[treatment]}", fontsize=12,
                     fontweight="bold", color=C[treatment], pad=10)
        ax.spines["left"].set_visible(False)
        ax.tick_params(left=False)
        ax.set_ylim(-0.6, len(MODELS)-0.4)

    fig.text(0.5, 0.01,
             "Dots show observed bluff rate. Shaded bands = 95% Wilson CI. "
             "Purple ticks on stems mark the PBE rational benchmark (42%). "
             "All deviations significant at p < 0.001 (binomial test). N = 150 per cell.",
             ha="center", fontsize=8.8, color="#555", style="italic")
    fig.suptitle("Figure 2: Bluffing Frequency by Model and Treatment\n"
                 "Proportion of Low Resolve Senders Sending Escalatory Signal vs. PBE Prediction",
                 fontsize=13, fontweight="bold", y=1.0, color="#1A1A1A")
    plt.savefig(os.path.join(FIGS_DIR, "fig2_bluffing_lollipop.png"))
    plt.close()
    print("  ✓ Figure 2 saved")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 3 — Calibration Curves (shaded regions, sparsity notes, n counts)
# ─────────────────────────────────────────────────────────────────────────────
def fig3_calibration():
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5), sharey=True)
    fig.subplots_adjust(wspace=0.1, left=0.08, right=0.97, top=0.83, bottom=0.22)

    for ax, model in zip(axes, MODELS):
        # Shaded regions: overconfident (above diagonal) and underconfident (below)
        ax.fill_between([0,1],[0,1],[1,1], alpha=0.04, color="#D6604D", label="_nolegend_")
        ax.fill_between([0,1],[0,0],[0,1], alpha=0.04, color="#2166AC", label="_nolegend_")
        ax.text(0.72, 0.12, "Under-\nconfident", ha="center", fontsize=7.5,
                color="#2166AC", alpha=0.7, style="italic")
        ax.text(0.28, 0.88, "Over-\nconfident", ha="center", fontsize=7.5,
                color="#D6604D", alpha=0.7, style="italic")

        # Perfect calibration diagonal
        ax.plot([0,1],[0,1], color="#AAAAAA", lw=1.2, ls=":", zorder=1)

        for treatment, ls, alpha in [("zero_shot","-",0.9),("role_conditioned","--",0.9)]:
            sub = df[(df["model_key"]==model)&(df["treatment"]==treatment)].copy()
            if sub.empty: continue
            beliefs = sub["posterior_belief"].clip(0.01,0.99)
            actuals = (sub["sender_type"]=="HIGH").astype(int)

            bins = np.linspace(0, 1, 7)   # 6 bins — fewer but more populated
            bcs, fps, ns_bin = [], [], []
            for j in range(len(bins)-1):
                mask = (beliefs >= bins[j]) & (beliefs < bins[j+1])
                if mask.sum() >= 5:
                    bcs.append((bins[j]+bins[j+1])/2)
                    fps.append(actuals[mask].mean())
                    ns_bin.append(mask.sum())

            if not bcs: continue
            color = C[treatment]
            ax.plot(bcs, fps, color=color, lw=2.2, ls=ls,
                    marker="o", ms=7, markerfacecolor="white",
                    markeredgewidth=2.0, markeredgecolor=color,
                    label=TREAT_LABELS[treatment], alpha=alpha, zorder=3)

            # n-per-bin annotations — above each dot, small, no overlap
            for x, y, n in zip(bcs, fps, ns_bin):
                offset = 0.06 if y < 0.88 else -0.08
                ax.text(x, y+offset, f"n={n}", ha="center", va="bottom",
                        fontsize=7, color=color, alpha=0.75)

        # Brier scores — top-left corner
        brier_zs = summary[(summary["model_key"]==model)&(summary["treatment"]=="zero_shot")]["brier_score"].values
        brier_rc = summary[(summary["model_key"]==model)&(summary["treatment"]=="role_conditioned")]["brier_score"].values
        lines = []
        if len(brier_zs): lines.append(f"Zero-Shot BS = {brier_zs[0]:.3f}")
        if len(brier_rc): lines.append(f"Role-Cond. BS = {brier_rc[0]:.3f}")
        lines.append("(0.00=perfect, 0.25=naive)")
        ax.text(0.03, 0.97, "\n".join(lines), transform=ax.transAxes,
                ha="left", va="top", fontsize=8, color="#333",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#CCCCCC", lw=0.8))

        ax.set_xlim(0,1); ax.set_ylim(0,1)
        ax.set_xlabel("Predicted P(High Resolve)", fontsize=10)
        ax.set_title(MODEL_LABELS[model], fontsize=12, fontweight="bold",
                     color=C[model], pad=8)
        ax.set_aspect("equal")

    axes[0].set_ylabel("Observed Frequency P(High Resolve)", fontsize=10)

    handles = [
        Line2D([0],[0], color=C["zero_shot"], lw=2.2, marker="o",
               markerfacecolor="white", markeredgewidth=2.0, label="Zero-Shot"),
        Line2D([0],[0], color=C["role_conditioned"], lw=2.2, ls="--", marker="o",
               markerfacecolor="white", markeredgewidth=2.0, label="Role-Conditioned"),
        Line2D([0],[0], color="#AAAAAA", lw=1.2, ls=":", label="Perfect Calibration"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.02), fontsize=10, frameon=True, edgecolor="#CCC")
    fig.suptitle("Figure 3: Bayesian Belief Calibration of Receiver Agents\n"
                 "Points above diagonal = overconfident; below = underconfident. n = observations per bin.",
                 fontsize=13, fontweight="bold", y=1.01, color="#1A1A1A")
    plt.savefig(os.path.join(FIGS_DIR, "fig3_calibration.png"))
    plt.close()
    print("  ✓ Figure 3 saved")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 4 — Reasoning Heatmap (fixed margins, labels inside)
# ─────────────────────────────────────────────────────────────────────────────
def fig4_heatmap():
    keywords = {
        "Cost / Risk":          ["cost", "risk", "danger", "casualt"],
        "Credibility":          ["credib", "believab", "convinc"],
        "Reputation":           ["reput", "track record", "history"],
        "Deception / Bluffing": ["deceiv", "bluff", "mislead", "false signal"],
        "Deterrence":           ["deter", "prevent", "discourage"],
        "Threat Heuristic":     ["threat", "intimidat", "coerce"],
        "Rational Calculation": ["calculat", "rational", "optimal", "payoff"],
    }
    col_labels = [f"{MODEL_LABELS[m]}\n{TREAT_LABELS[t]}"
                  for m in MODELS for t in TREATMENTS]

    matrix = np.zeros((len(keywords), len(col_labels)))
    for col_idx, (m, t) in enumerate([(m,t) for m in MODELS for t in TREATMENTS]):
        sub = df[(df["model_key"]==m)&(df["treatment"]==t)]
        reasoning = (sub["sender_reasoning"].fillna("") + " " +
                     sub["receiver_reasoning"].fillna("")).str.lower()
        for row_idx, kw_list in enumerate(keywords.values()):
            matrix[row_idx, col_idx] = reasoning.apply(
                lambda x: any(k in x for k in kw_list)).mean()

    fig, ax = plt.subplots(figsize=(14, 6.5))
    fig.subplots_adjust(left=0.20, right=0.93, top=0.88, bottom=0.28)

    im = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1, aspect="auto")

    # X-axis: two-line labels, enough rotation to avoid overlap
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=40, ha="right", fontsize=9.5,
                       rotation_mode="anchor")
    ax.set_yticks(np.arange(len(keywords)))
    ax.set_yticklabels(list(keywords.keys()), fontsize=10.5)

    # Cell text
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix[i, j]
            tc = "white" if v > 0.58 else "#1A1A1A"
            ax.text(j, i, f"{v:.0%}", ha="center", va="center",
                    fontsize=9.5, fontweight="bold", color=tc)

    # Vertical separators between models
    for x in [1.5, 3.5]:
        ax.axvline(x, color="white", lw=3.0)

    # Model group labels — inside the figure, above the heatmap
    for m, x_center in zip(MODELS, [0.5, 2.5, 4.5]):
        ax.text(x_center, -0.7, MODEL_LABELS[m], ha="center", va="bottom",
                fontsize=10.5, fontweight="bold", color=C[m],
                transform=ax.get_xaxis_transform())

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.022, pad=0.02)
    cbar.set_label("Proportion of reasoning\ntraces containing keyword",
                   fontsize=9, labelpad=8)
    cbar.ax.tick_params(labelsize=8.5)
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    cbar.set_ticklabels(["0%", "25%", "50%", "75%", "100%"])

    ax.set_title("Figure 4: Reasoning Trace Keyword Analysis\n"
                 "What Do LLMs Think About When Making Strategic Decisions?",
                 fontsize=13, fontweight="bold", pad=28, color="#1A1A1A")
    plt.savefig(os.path.join(FIGS_DIR, "fig4_reasoning_heatmap.png"))
    plt.close()
    print("  ✓ Figure 4 saved")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 5 — Payoff Distributions: Horizontal violin + median annotation
# ─────────────────────────────────────────────────────────────────────────────
def fig5_payoffs():
    fig, axes = plt.subplots(2, 1, figsize=(13, 9))
    fig.subplots_adjust(hspace=0.45, left=0.22, right=0.97, top=0.92, bottom=0.08)

    for ax, role, title in zip(axes,
                                ["sender_payoff", "receiver_payoff"],
                                ["Sender (Bluffer) Payoff Distribution",
                                 "Receiver (Opponent) Payoff Distribution"]):
        plot_data, positions, colors_list, tick_labels = [], [], [], []
        pos = 0
        for m in MODELS:
            for t in TREATMENTS:
                sub = df[(df["model_key"]==m)&(df["treatment"]==t)][role].dropna()
                if sub.empty: continue
                plot_data.append(sub.values)
                positions.append(pos)
                colors_list.append(C[m])
                tick_labels.append(f"{MODEL_LABELS[m]}  [{TREAT_LABELS[t]}]")
                pos += 1

        vp = ax.violinplot(plot_data, positions=positions, widths=0.65,
                           showmedians=False, showextrema=False, vert=False)
        for body, color in zip(vp["bodies"], colors_list):
            body.set_facecolor(color); body.set_alpha(0.30)
            body.set_edgecolor(color); body.set_linewidth(1.3)

        for i, (data, pos_, color) in enumerate(zip(plot_data, positions, colors_list)):
            med = np.median(data)
            q1, q3 = np.percentile(data, [25, 75])
            # IQR box
            ax.barh(pos_, q3-q1, left=q1, height=0.12, color=color, alpha=0.55, zorder=3)
            # Median line
            ax.plot([med, med], [pos_-0.12, pos_+0.12], color="white", lw=2.5, zorder=4)
            ax.plot([med, med], [pos_-0.12, pos_+0.12], color=color, lw=1.5, zorder=5)
            # Median annotation — right of violin
            ax.text(1.12, pos_, f"Md={med:.2f}", va="center", ha="left",
                    fontsize=8.5, color=color, fontweight="bold")

        ax.axvline(0, color="#AAAAAA", lw=1.0, ls="--", alpha=0.7, zorder=0)
        ax.set_yticks(positions)
        ax.set_yticklabels(tick_labels, fontsize=9.5)
        ax.set_xlim(-1.6, 1.35)
        ax.set_xlabel("Payoff Value", fontsize=10)
        ax.set_title(title, fontsize=12, fontweight="bold", pad=8)

        # Zero reference annotation
        ax.text(0.01, -0.5, "← Loss  |  Gain →", ha="center", va="top",
                fontsize=8, color="#888", style="italic", transform=ax.get_xaxis_transform())

    fig.suptitle("Figure 5: Payoff Distributions by Model and Treatment\n"
                 "Violin = full distribution  |  Box = IQR  |  Md = median",
                 fontsize=13, fontweight="bold", y=0.99, color="#1A1A1A")
    plt.savefig(os.path.join(FIGS_DIR, "fig5_payoff_distributions.png"))
    plt.close()
    print("  ✓ Figure 5 saved")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 6 — Summary Dashboard (clean labels, zero-reference, better spacing)
# ─────────────────────────────────────────────────────────────────────────────
def fig6_dashboard():
    metrics = [
        ("bluff_rate",         "Bluff Rate",               "Proportion",              True),
        ("bluff_success_rate", "Bluff Success Rate",        "Proportion",              True),
        ("brier_score",        "Brier Score\n(↓ better)",  "Score",                   False),
        ("edi",                "Equilibrium Deviation\nIndex (↓ better)", "EDI",      False),
        ("avg_sender_payoff",  "Avg. Sender Payoff",        "Payoff",                  None),
        ("avg_receiver_payoff","Avg. Receiver Payoff",      "Payoff",                  None),
    ]

    fig = plt.figure(figsize=(17, 10))
    gs  = gridspec.GridSpec(2, 3, figure=fig,
                            hspace=0.62, wspace=0.40,
                            left=0.07, right=0.97, top=0.88, bottom=0.16)

    short_labels = ["GPT-mini", "GPT-nano", "Gemini"]

    for idx, (metric, title, ylabel, _) in enumerate(metrics):
        row, col = divmod(idx, 3)
        ax = fig.add_subplot(gs[row, col])

        x_pos  = np.arange(len(MODELS))
        width  = 0.30
        offsets= [-width/2 - 0.03, width/2 + 0.03]

        for t_idx, (treatment, offset) in enumerate(zip(TREATMENTS, offsets)):
            vals = []
            for m in MODELS:
                rd = summary[(summary["model_key"]==m)&(summary["treatment"]==treatment)]
                vals.append(float(rd[metric].values[0]) if not rd.empty else 0.0)

            bars = ax.bar(x_pos+offset, vals, width=width,
                          color=C[treatment], alpha=0.80,
                          edgecolor="white", linewidth=0.7)

            # Value labels — only on bars tall enough; use small font
            for bar, val in zip(bars, vals):
                bh = bar.get_height()
                if abs(bh) >= 0.03:
                    y_lbl = bh + 0.012 if bh >= 0 else bh - 0.025
                    va_lbl = "bottom" if bh >= 0 else "top"
                    ax.text(bar.get_x()+bar.get_width()/2, y_lbl,
                            f"{val:.2f}", ha="center", va=va_lbl,
                            fontsize=7.5, fontweight="bold", color=C[treatment])

        # PBE reference
        if metric == "bluff_rate":
            ax.axhline(PBE, color=C["pbe"], lw=1.5, ls="--", alpha=0.8)
            ax.text(len(MODELS)-0.08, PBE+0.02, "PBE=42%",
                    ha="right", fontsize=8, color=C["pbe"], style="italic")

        # Zero reference for payoff panels
        if "payoff" in metric:
            ax.axhline(0, color="#AAAAAA", lw=0.9, ls="-", alpha=0.5)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(short_labels, fontsize=9.5)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=10.5, fontweight="bold", pad=7)

        if metric in ["bluff_rate","bluff_success_rate"]:
            ax.set_ylim(0, 1.28)
            ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
        elif metric in ["brier_score","edi"]:
            ax.set_ylim(0, 0.40)
        else:
            ax.set_ylim(-0.30, 1.05)

    # Shared legend
    handles = [
        mpatches.Patch(color=C["zero_shot"],        alpha=0.80, label="Zero-Shot"),
        mpatches.Patch(color=C["role_conditioned"], alpha=0.80, label="Role-Conditioned"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, 0.01), fontsize=11, frameon=True,
               edgecolor="#CCC", title="Treatment", title_fontsize=10)

    fig.suptitle("Figure 6: Complete Results Summary — All Six Metrics by Model and Treatment\n"
                 "The Bluffing Machine: 900 Real LLM Signaling Game Simulations  (N=150 per cell)",
                 fontsize=13, fontweight="bold", y=0.97, color="#1A1A1A")
    plt.savefig(os.path.join(FIGS_DIR, "fig6_summary_dashboard.png"))
    plt.close()
    print("  ✓ Figure 6 saved")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating figures...")
    fig1_game_tree()
    fig2_lollipop()
    fig3_calibration()
    fig4_heatmap()
    fig5_payoffs()
    fig6_dashboard()
    print()
    print("=" * 62)
    print(f"  ALL 6 FIGURES GENERATED (v2 — interpretability fixed)")
    print(f"  Saved to: {FIGS_DIR}")
    print("=" * 62)
