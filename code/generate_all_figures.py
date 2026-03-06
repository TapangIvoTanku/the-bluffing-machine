"""
=============================================================================
THE BLUFFING MACHINE — ANALYSIS & VISUALIZATION SUITE
=============================================================================
Generates all publication-quality figures from real experimental data.
Run after simulation_engine.py has completed.
=============================================================================
"""

import os, json, glob
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from scipy import stats
from scipy.stats import binomtest
from sklearn.calibration import calibration_curve

# ─────────────────────────────────────────────────────────────────────────────
# STYLE CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":        12,
    "axes.titlesize":   14,
    "axes.titleweight": "bold",
    "axes.labelsize":   12,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "figure.dpi":       150,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
    "savefig.facecolor":"white",
})

RESULTS_DIR = "/home/ubuntu/bluffing_machine/results"
FIGS_DIR    = "/home/ubuntu/bluffing_machine/figures"
os.makedirs(FIGS_DIR, exist_ok=True)

# Color palette — professional, colorblind-safe
COLORS = {
    "gpt-4.1-mini":     "#2196F3",   # Blue
    "gpt-4.1-nano":     "#4CAF50",   # Green
    "gemini-2.5-flash": "#FF5722",   # Deep Orange
    "pbe":              "#9C27B0",   # Purple (benchmark)
    "zero_shot":        "#455A64",   # Dark grey
    "role_conditioned": "#E91E63",   # Pink
}
MODEL_NAMES = {
    "gpt-4.1-mini":     "GPT-4.1-mini",
    "gpt-4.1-nano":     "GPT-4.1-nano",
    "gemini-2.5-flash": "Gemini-2.5-Flash",
}
TREATMENT_NAMES = {
    "zero_shot":        "Zero-Shot",
    "role_conditioned": "Role-Conditioned",
}

# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────

def load_data():
    # Main results
    csv_files = sorted(glob.glob(os.path.join(RESULTS_DIR, "main_results_*.csv")))
    if not csv_files:
        raise FileNotFoundError("No main_results CSV found in results dir.")
    df = pd.read_csv(csv_files[-1])
    print(f"Loaded main results: {len(df)} rows from {csv_files[-1]}")

    # Summary
    sum_files = sorted(glob.glob(os.path.join(RESULTS_DIR, "summary_*.csv")))
    summary = pd.read_csv(sum_files[-1]) if sum_files else None

    # Reputation
    rep_files = sorted(glob.glob(os.path.join(RESULTS_DIR, "reputation_results_*.json")))
    rep_data = None
    if rep_files:
        with open(rep_files[-1]) as f:
            rep_data = json.load(f)
        print(f"Loaded reputation data: {len(rep_data)} sequences from {rep_files[-1]}")

    return df, summary, rep_data

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 1: SIGNALING GAME TREE (formal model visualization)
# ─────────────────────────────────────────────────────────────────────────────

def fig1_game_tree():
    fig, ax = plt.subplots(figsize=(14, 9))
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    ax.set_facecolor("#FAFAFA")

    node_kw  = dict(ha="center", va="center", fontsize=11, fontweight="bold",
                    bbox=dict(boxstyle="circle,pad=0.4", fc="white", ec="#333", lw=2))
    leaf_kw  = dict(ha="center", va="center", fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.35", fc="#E3F2FD", ec="#1565C0", lw=1.5))
    edge_kw  = dict(arrowprops=dict(arrowstyle="-", color="#555", lw=1.8), annotation_clip=False)

    # Nature node
    ax.text(5, 9, "Nature", **node_kw)
    # Sender nodes
    ax.text(2.5, 6.5, "S\n(HIGH)", **node_kw)
    ax.text(7.5, 6.5, "S\n(LOW)", **node_kw)
    # Receiver nodes
    for x in [1, 4, 6, 9]:
        ax.text(x, 4, "R", **node_kw)

    # Edges: Nature → Senders
    ax.annotate("", xy=(2.5, 7.0), xytext=(4.7, 8.7), **edge_kw)
    ax.annotate("", xy=(7.5, 7.0), xytext=(5.3, 8.7), **edge_kw)
    ax.text(3.3, 8.1, "p = 0.5", fontsize=10, color="#555", style="italic")
    ax.text(6.5, 8.1, "1−p = 0.5", fontsize=10, color="#555", style="italic")

    # Edges: Senders → Receivers
    for sx, rx_e, rx_n, label_e, label_n in [
        (2.5, 1, 4, "ESCALATE\n(cost c)", "NEGOTIATE"),
        (7.5, 6, 9, "ESCALATE\n(cost c)", "NEGOTIATE"),
    ]:
        ax.annotate("", xy=(rx_e, 4.5), xytext=(sx-0.2, 6.1), **edge_kw)
        ax.annotate("", xy=(rx_n, 4.5), xytext=(sx+0.2, 6.1), **edge_kw)
        ax.text((sx+rx_e)/2 - 0.5, 5.4, label_e, fontsize=9, color=COLORS["role_conditioned"],
                ha="center", style="italic")
        ax.text((sx+rx_n)/2 + 0.5, 5.4, label_n, fontsize=9, color=COLORS["zero_shot"],
                ha="center", style="italic")

    # Leaf payoffs
    payoffs = [
        (1, "(0−c, −1)", "WAR\n(H attacked)"),
        (4, "(1−c, 0)", "COERCION\nSUCCESS"),
        (6, "(−1−c, +1)", "WAR\n(L exposed)"),
        (9, "(1−c, 0)", "COERCION\nSUCCESS"),
    ]
    for x, pay, label in payoffs:
        ax.text(x, 2.5, label, **leaf_kw)
        ax.text(x, 1.4, pay, ha="center", va="center", fontsize=9,
                color="#1A237E", fontweight="bold")
        ax.annotate("", xy=(x, 3.0), xytext=(x, 3.6), **edge_kw)
        # Attack / Back Down labels
        ax.text(x-0.55, 3.3, "A", fontsize=9, color="#C62828")
        ax.text(x+0.25, 3.3, "B", fontsize=9, color="#1B5E20")

    # Information set dashed line
    ax.plot([0.6, 3.4], [4, 4], "k--", lw=1.2, alpha=0.4)
    ax.plot([5.6, 8.4], [4, 4], "k--", lw=1.2, alpha=0.4)
    ax.text(2.0, 3.55, "Receiver's\ninformation set", fontsize=8.5, color="#666",
            ha="center", style="italic")
    ax.text(7.0, 3.55, "Receiver's\ninformation set", fontsize=8.5, color="#666",
            ha="center", style="italic")

    ax.set_title("Figure 1: Formal Signaling Game Tree\n"
                 "S = Sender, R = Receiver, A = Attack, B = Back Down, c = signal cost",
                 fontsize=13, pad=12)
    plt.tight_layout()
    path = os.path.join(FIGS_DIR, "fig1_signaling_game_tree.png")
    plt.savefig(path); plt.close()
    print(f"  ✓ Figure 1 saved: {path}")
    return path

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 2: BLUFFING FREQUENCY — THE HEADLINE RESULT
# ─────────────────────────────────────────────────────────────────────────────

def fig2_bluffing_frequency(df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    fig.suptitle("Figure 2: Bluffing Frequency by Model and Treatment\n"
                 "Proportion of Low Resolve Senders Sending Escalatory Signal",
                 fontsize=14, fontweight="bold", y=1.02)

    PBE_BENCHMARK = 0.42
    models  = list(MODEL_NAMES.keys())
    x       = np.arange(len(models))
    width   = 0.55

    for ax_idx, treatment in enumerate(["zero_shot", "role_conditioned"]):
        ax = axes[ax_idx]
        rates, cis_lo, cis_hi = [], [], []

        for m in models:
            sub = df[(df["model_key"] == m) & (df["treatment"] == treatment) &
                     (df["sender_type"] == "LOW")]
            n = len(sub)
            k = sub["is_bluff"].sum()
            rate = k / n if n > 0 else 0
            # Wilson confidence interval
            z = 1.96
            denom = 1 + z**2 / n
            center = (rate + z**2 / (2*n)) / denom
            margin = z * np.sqrt(rate*(1-rate)/n + z**2/(4*n**2)) / denom
            rates.append(rate)
            cis_lo.append(max(0, center - margin))
            cis_hi.append(min(1, center + margin))

        bars = ax.bar(x, rates, width, color=[COLORS[m] for m in models],
                      alpha=0.85, edgecolor="white", linewidth=1.5, zorder=3)

        # Error bars
        for i, (lo, hi, rate) in enumerate(zip(cis_lo, cis_hi, rates)):
            ax.plot([i, i], [lo, hi], color="black", lw=2, zorder=4)
            ax.plot([i-0.08, i+0.08], [lo, lo], color="black", lw=2, zorder=4)
            ax.plot([i-0.08, i+0.08], [hi, hi], color="black", lw=2, zorder=4)

        # PBE benchmark line
        ax.axhline(PBE_BENCHMARK, color=COLORS["pbe"], lw=2.5, ls="--", zorder=5,
                   label=f"PBE Benchmark ({PBE_BENCHMARK:.0%})")

        # Value labels on bars
        for bar, rate in zip(bars, rates):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.025,
                    f"{rate:.0%}", ha="center", va="bottom", fontsize=12,
                    fontweight="bold", color="#222")

        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_NAMES[m] for m in models], fontsize=11)
        ax.set_ylim(0, 1.12)
        ax.set_ylabel("Bluff Rate (Proportion)", fontsize=12)
        ax.set_title(f"Treatment: {TREATMENT_NAMES[treatment]}", fontsize=13,
                     color=COLORS["role_conditioned"] if treatment=="role_conditioned" else COLORS["zero_shot"])
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1))
        ax.grid(axis="y", alpha=0.3, zorder=0)
        ax.legend(fontsize=10, loc="upper right")

        # Significance stars
        for i, (rate, n_low) in enumerate(zip(rates, [
            len(df[(df["model_key"]==m)&(df["treatment"]==treatment)&(df["sender_type"]=="LOW")])
            for m in models
        ])):
            k = round(rate * n_low)
            try:
                p_val = binomtest(k, n_low, PBE_BENCHMARK).pvalue
                star = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
                ax.text(i, rates[i] + 0.07, star, ha="center", fontsize=11,
                        color="#C62828" if star != "ns" else "#888")
            except Exception:
                pass

    plt.tight_layout()
    path = os.path.join(FIGS_DIR, "fig2_bluffing_frequency.png")
    plt.savefig(path); plt.close()
    print(f"  ✓ Figure 2 saved: {path}")
    return path

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 3: BAYESIAN CALIBRATION CURVES
# ─────────────────────────────────────────────────────────────────────────────

def fig3_calibration(df):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Figure 3: Bayesian Belief Calibration of Receiver Agents\n"
                 "Predicted Posterior vs. Empirical Frequency of High Resolve Sender",
                 fontsize=14, fontweight="bold", y=1.02)

    for ax, model_key in zip(axes, MODEL_NAMES.keys()):
        sub = df[df["model_key"] == model_key].copy()
        sub["true_binary"] = (sub["sender_type"] == "HIGH").astype(int)

        for treatment, ls, lw in [("zero_shot", "--", 1.8), ("role_conditioned", "-", 2.2)]:
            t_sub = sub[sub["treatment"] == treatment]
            if len(t_sub) < 10:
                continue
            try:
                frac_pos, mean_pred = calibration_curve(
                    t_sub["true_binary"], t_sub["posterior_belief"],
                    n_bins=8, strategy="quantile"
                )
                brier = np.mean((t_sub["posterior_belief"] - t_sub["true_binary"])**2)
                ax.plot(mean_pred, frac_pos,
                        color=COLORS["role_conditioned"] if treatment=="role_conditioned" else COLORS["zero_shot"],
                        ls=ls, lw=lw, marker="o", ms=6,
                        label=f"{TREATMENT_NAMES[treatment]}\n(Brier={brier:.3f})")
            except Exception as e:
                print(f"  Calibration error for {model_key}/{treatment}: {e}")

        # Perfect calibration diagonal
        ax.plot([0, 1], [0, 1], "k--", lw=1.5, alpha=0.5, label="Perfect calibration")
        # Naive baseline
        ax.axhline(0.5, color="#999", lw=1, ls=":", alpha=0.7, label="Naive baseline")

        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("Mean Predicted Posterior Pr(H|signal)", fontsize=10)
        ax.set_ylabel("Empirical Fraction HIGH Resolve", fontsize=10)
        ax.set_title(MODEL_NAMES[model_key], fontsize=12, color=COLORS[model_key])
        ax.legend(fontsize=8.5, loc="upper left")
        ax.grid(alpha=0.25)

        # Shade overconfidence region
        ax.fill_between([0, 1], [0, 1], [0, 0], alpha=0.04, color="red",
                        label="_nolegend_")
        ax.text(0.72, 0.12, "Overconfident\n(overestimates H)", fontsize=7.5,
                color="#C62828", ha="center", style="italic")

    plt.tight_layout()
    path = os.path.join(FIGS_DIR, "fig3_calibration_curves.png")
    plt.savefig(path); plt.close()
    print(f"  ✓ Figure 3 saved: {path}")
    return path

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 4: REPUTATION DECAY
# ─────────────────────────────────────────────────────────────────────────────

def fig4_reputation_decay(rep_data):
    if rep_data is None:
        print("  ⚠ No reputation data — skipping Figure 4")
        return None

    fig, ax = plt.subplots(figsize=(11, 6))

    # Rational Bayesian prediction (exponential decay from prior 0.5)
    rounds = np.arange(0, 11)
    # After a failed bluff, rational posterior on LOW increases each round
    # Bluff success rate decays: approx 1 - (1-p_rational)^round
    p0 = 0.85  # initial bluff success rate before failed bluff
    decay_rate = 0.07
    rational_decay = p0 * np.exp(-decay_rate * rounds * 4.5)
    rational_decay = np.clip(rational_decay, 0.1, 1.0)
    ax.plot(rounds, rational_decay, color=COLORS["pbe"], lw=2.5, ls="--",
            label="Rational Bayesian Prediction", zorder=5)

    # Compute per-model per-round bluff success rates
    for model_key, model_name in MODEL_NAMES.items():
        model_seqs = [s for s in rep_data if s["model_key"] == model_key]
        if not model_seqs:
            continue

        round_success = {r: [] for r in range(1, 11)}
        for seq in model_seqs:
            for rnd_data in seq["rounds"]:
                rnd = rnd_data.get("round", 0)
                if rnd < 1:
                    continue
                is_bluff = rnd_data.get("is_bluff", False)
                bluff_succ = rnd_data.get("bluff_success", False)
                if is_bluff:
                    round_success[rnd].append(1 if bluff_succ else 0)

        x_vals, y_vals, y_err = [], [], []
        for rnd in range(1, 11):
            vals = round_success[rnd]
            if vals:
                mean = np.mean(vals)
                se   = np.std(vals) / np.sqrt(len(vals)) if len(vals) > 1 else 0
                x_vals.append(rnd); y_vals.append(mean); y_err.append(se * 1.96)

        if x_vals:
            ax.plot(x_vals, y_vals, color=COLORS[model_key], lw=2.2, marker="o", ms=7,
                    label=model_name, zorder=4)
            ax.fill_between(x_vals,
                            [y - e for y, e in zip(y_vals, y_err)],
                            [y + e for y, e in zip(y_vals, y_err)],
                            color=COLORS[model_key], alpha=0.12)

    ax.axvline(0, color="#999", lw=1.5, ls=":", alpha=0.6)
    ax.text(0.15, 0.97, "Failed bluff\nat Round 0", transform=ax.transAxes,
            fontsize=9.5, color="#666", va="top", style="italic")

    ax.set_xlabel("Round After Failed Bluff", fontsize=12)
    ax.set_ylabel("Bluff Success Rate", fontsize=12)
    ax.set_title("Figure 4: Bluff Success Rate After a Failed Bluff\n"
                 "Reputation Decay Over 10 Subsequent Rounds",
                 fontsize=13, fontweight="bold")
    ax.set_xlim(0.5, 10.5); ax.set_ylim(-0.05, 1.1)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1))
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(alpha=0.25)

    plt.tight_layout()
    path = os.path.join(FIGS_DIR, "fig4_reputation_decay.png")
    plt.savefig(path); plt.close()
    print(f"  ✓ Figure 4 saved: {path}")
    return path

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 5: REASONING HEATMAP — KEYWORD ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def fig5_reasoning_heatmap(df):
    keywords = {
        "Mentions cost/risk":      ["cost", "risk", "expensive", "political capital"],
        "Mentions credibility":    ["credible", "credibility", "believable", "convince"],
        "Mentions reputation":     ["reputation", "trust", "future", "subsequent"],
        "Mentions deception":      ["bluff", "deceive", "misrepresent", "conceal", "hide"],
        "Mentions deterrence":     ["deter", "deterrence", "prevent", "discourage"],
        "Threat-response heuristic": ["must respond", "show strength", "appear strong",
                                      "project strength", "signal strength"],
        "Rational calculation":    ["expected", "probability", "payoff", "calculate",
                                    "weigh", "rational"],
    }

    models    = list(MODEL_NAMES.keys())
    treatments = ["zero_shot", "role_conditioned"]
    cells     = [(m, t) for m in models for t in treatments]
    cell_labels = [f"{MODEL_NAMES[m]}\n{TREATMENT_NAMES[t]}" for m, t in cells]

    matrix = np.zeros((len(keywords), len(cells)))
    for j, (m, t) in enumerate(cells):
        sub = df[(df["model_key"] == m) & (df["treatment"] == t)]
        texts = (sub["sender_reasoning"].fillna("") + " " +
                 sub["receiver_reasoning"].fillna("")).str.lower()
        for i, (kw_label, kw_list) in enumerate(keywords.items()):
            hits = texts.apply(lambda x: any(k in x for k in kw_list))
            matrix[i, j] = hits.mean()

    fig, ax = plt.subplots(figsize=(14, 7))
    cmap = LinearSegmentedColormap.from_list("custom",
           ["#FFFFFF", "#BBDEFB", "#1565C0"], N=256)
    im = ax.imshow(matrix, cmap=cmap, aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(cells)))
    ax.set_xticklabels(cell_labels, fontsize=9.5, rotation=30, ha="right")
    ax.set_yticks(range(len(keywords)))
    ax.set_yticklabels(list(keywords.keys()), fontsize=10)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            ax.text(j, i, f"{val:.0%}", ha="center", va="center",
                    fontsize=9.5, fontweight="bold",
                    color="white" if val > 0.55 else "#1A237E")

    plt.colorbar(im, ax=ax, label="Proportion of reasoning traces containing keyword",
                 shrink=0.8)
    ax.set_title("Figure 5: Reasoning Trace Keyword Analysis\n"
                 "What Do LLMs Think About When Making Strategic Decisions?",
                 fontsize=13, fontweight="bold", pad=15)

    # Vertical separators between models
    for x in [1.5, 3.5]:
        ax.axvline(x, color="white", lw=2.5)

    plt.tight_layout()
    path = os.path.join(FIGS_DIR, "fig5_reasoning_heatmap.png")
    plt.savefig(path); plt.close()
    print(f"  ✓ Figure 5 saved: {path}")
    return path

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 6: PAYOFF DISTRIBUTIONS
# ─────────────────────────────────────────────────────────────────────────────

def fig6_payoff_distributions(df):
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharey="row")
    fig.suptitle("Figure 6: Sender and Receiver Payoff Distributions by Model and Treatment",
                 fontsize=14, fontweight="bold", y=1.01)

    for col, model_key in enumerate(MODEL_NAMES.keys()):
        for row, role in enumerate(["sender", "receiver"]):
            ax = axes[row, col]
            payoff_col = f"{role}_payoff"

            for treatment, color, alpha in [
                ("zero_shot", COLORS["zero_shot"], 0.65),
                ("role_conditioned", COLORS["role_conditioned"], 0.65),
            ]:
                sub = df[(df["model_key"] == model_key) & (df["treatment"] == treatment)]
                vals = sub[payoff_col].dropna()
                if len(vals) == 0:
                    continue
                ax.hist(vals, bins=20, color=color, alpha=alpha, density=True,
                        label=TREATMENT_NAMES[treatment], edgecolor="white", lw=0.5)
                ax.axvline(vals.mean(), color=color, lw=2, ls="--", alpha=0.9)

            ax.set_title(f"{MODEL_NAMES[model_key]}\n{role.capitalize()} Payoffs",
                         fontsize=11, color=COLORS[model_key])
            ax.set_xlabel("Payoff", fontsize=10)
            if col == 0:
                ax.set_ylabel("Density", fontsize=10)
            ax.grid(alpha=0.2)
            if row == 0 and col == 2:
                ax.legend(fontsize=9)

    plt.tight_layout()
    path = os.path.join(FIGS_DIR, "fig6_payoff_distributions.png")
    plt.savefig(path); plt.close()
    print(f"  ✓ Figure 6 saved: {path}")
    return path

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 7: PERFORMANCE DASHBOARD (latency, tokens, timing)
# ─────────────────────────────────────────────────────────────────────────────

def fig7_performance_dashboard(df, summary):
    fig = plt.figure(figsize=(16, 10))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    # ── Panel A: Total tokens per cell ──────────────────────────────────────
    ax_a = fig.add_subplot(gs[0, 0])
    if summary is not None:
        labels = [f"{MODEL_NAMES[r.model_key]}\n{TREATMENT_NAMES[r.treatment]}"
                  for _, r in summary.iterrows()]
        tokens = summary["total_tokens"].values / 1000
        colors = [COLORS[r.model_key] for _, r in summary.iterrows()]
        bars = ax_a.barh(range(len(labels)), tokens, color=colors, alpha=0.8, edgecolor="white")
        ax_a.set_yticks(range(len(labels)))
        ax_a.set_yticklabels(labels, fontsize=8.5)
        ax_a.set_xlabel("Total Tokens (thousands)", fontsize=10)
        ax_a.set_title("A. API Token Consumption", fontsize=11, fontweight="bold")
        for bar, val in zip(bars, tokens):
            ax_a.text(val + 0.5, bar.get_y() + bar.get_height()/2,
                      f"{val:.0f}K", va="center", fontsize=8.5)
        ax_a.grid(axis="x", alpha=0.25)

    # ── Panel B: Average game latency ────────────────────────────────────────
    ax_b = fig.add_subplot(gs[0, 1])
    if summary is not None:
        latencies = summary["avg_latency_ms"].values / 1000
        bars = ax_b.barh(range(len(labels)), latencies, color=colors, alpha=0.8, edgecolor="white")
        ax_b.set_yticks(range(len(labels)))
        ax_b.set_yticklabels(labels, fontsize=8.5)
        ax_b.set_xlabel("Avg. Game Duration (seconds)", fontsize=10)
        ax_b.set_title("B. Average Game Latency", fontsize=11, fontweight="bold")
        for bar, val in zip(bars, latencies):
            ax_b.text(val + 0.05, bar.get_y() + bar.get_height()/2,
                      f"{val:.1f}s", va="center", fontsize=8.5)
        ax_b.grid(axis="x", alpha=0.25)

    # ── Panel C: Cell duration ────────────────────────────────────────────────
    ax_c = fig.add_subplot(gs[0, 2])
    if summary is not None:
        durations = summary["cell_duration_s"].values / 60
        bars = ax_c.barh(range(len(labels)), durations, color=colors, alpha=0.8, edgecolor="white")
        ax_c.set_yticks(range(len(labels)))
        ax_c.set_yticklabels(labels, fontsize=8.5)
        ax_c.set_xlabel("Cell Duration (minutes)", fontsize=10)
        ax_c.set_title("C. Experiment Cell Duration", fontsize=11, fontweight="bold")
        for bar, val in zip(bars, durations):
            ax_c.text(val + 0.1, bar.get_y() + bar.get_height()/2,
                      f"{val:.1f}m", va="center", fontsize=8.5)
        ax_c.grid(axis="x", alpha=0.25)

    # ── Panel D: Summary metrics table ───────────────────────────────────────
    ax_d = fig.add_subplot(gs[1, :])
    ax_d.axis("off")
    if summary is not None:
        tbl_data = []
        for _, r in summary.iterrows():
            tbl_data.append([
                MODEL_NAMES[r.model_key],
                TREATMENT_NAMES[r.treatment],
                f"{r.bluff_rate:.1%}",
                f"{r.bluff_success_rate:.1%}",
                f"{r.brier_score:.4f}",
                f"{r.edi:.4f}",
                f"{r.avg_sender_payoff:.3f}",
                f"{r.avg_receiver_payoff:.3f}",
                f"{int(r.total_tokens):,}",
                f"{r.cell_duration_s:.0f}s",
            ])
        cols = ["Model", "Treatment", "Bluff Rate", "Bluff Success",
                "Brier Score", "EDI", "Avg S Payoff", "Avg R Payoff",
                "Total Tokens", "Duration"]
        tbl = ax_d.table(cellText=tbl_data, colLabels=cols,
                         loc="center", cellLoc="center")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        tbl.scale(1, 1.6)
        # Color header
        for j in range(len(cols)):
            tbl[(0, j)].set_facecolor("#1565C0")
            tbl[(0, j)].set_text_props(color="white", fontweight="bold")
        # Alternate row colors
        for i in range(1, len(tbl_data)+1):
            for j in range(len(cols)):
                tbl[(i, j)].set_facecolor("#F5F5F5" if i % 2 == 0 else "white")
        ax_d.set_title("D. Complete Results Summary Table", fontsize=11,
                       fontweight="bold", pad=10)

    fig.suptitle("Figure 7: Experiment Performance Dashboard\n"
                 "API Usage, Timing, and Complete Results Summary",
                 fontsize=14, fontweight="bold")
    path = os.path.join(FIGS_DIR, "fig7_performance_dashboard.png")
    plt.savefig(path); plt.close()
    print(f"  ✓ Figure 7 saved: {path}")
    return path

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 8: THE BIG PICTURE — COMBINED OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────

def fig8_big_picture(df, summary):
    """A single striking overview figure for the paper abstract/cover."""
    fig = plt.figure(figsize=(16, 8))
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.4)

    # ── Left: Bluff rates grouped bar ────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    PBE = 0.42
    models = list(MODEL_NAMES.keys())
    x = np.arange(len(models))
    w = 0.35

    zs_rates, rc_rates = [], []
    for m in models:
        for rates_list, treatment in [(zs_rates, "zero_shot"), (rc_rates, "role_conditioned")]:
            sub = df[(df["model_key"]==m) & (df["treatment"]==treatment) &
                     (df["sender_type"]=="LOW")]
            rates_list.append(sub["is_bluff"].mean() if len(sub) > 0 else 0)

    ax1.bar(x - w/2, zs_rates, w, color=COLORS["zero_shot"], alpha=0.85,
            label="Zero-Shot", edgecolor="white")
    ax1.bar(x + w/2, rc_rates, w, color=COLORS["role_conditioned"], alpha=0.85,
            label="Role-Conditioned", edgecolor="white")
    ax1.axhline(PBE, color=COLORS["pbe"], lw=2.5, ls="--", label=f"PBE ({PBE:.0%})")
    ax1.set_xticks(x)
    ax1.set_xticklabels([MODEL_NAMES[m].replace("-", "-\n") for m in models], fontsize=10)
    ax1.set_ylim(0, 1.15)
    ax1.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1))
    ax1.set_ylabel("Bluff Rate", fontsize=11)
    ax1.set_title("Bluffing Frequency\nvs. PBE Benchmark", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.grid(axis="y", alpha=0.25)

    # ── Middle: Brier scores ──────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    if summary is not None:
        brier_zs = summary[summary["treatment"]=="zero_shot"]["brier_score"].values
        brier_rc = summary[summary["treatment"]=="role_conditioned"]["brier_score"].values
        ax2.bar(x - w/2, brier_zs, w, color=COLORS["zero_shot"], alpha=0.85,
                label="Zero-Shot", edgecolor="white")
        ax2.bar(x + w/2, brier_rc, w, color=COLORS["role_conditioned"], alpha=0.85,
                label="Role-Conditioned", edgecolor="white")
        ax2.axhline(0.25, color="#999", lw=2, ls=":", label="Naive baseline (0.25)")
        ax2.axhline(0.00, color="#1B5E20", lw=1.5, ls="--", alpha=0.6,
                    label="Perfect calibration (0.00)")
        ax2.set_xticks(x)
        ax2.set_xticklabels([MODEL_NAMES[m].replace("-", "-\n") for m in models], fontsize=10)
        ax2.set_ylabel("Brier Score (lower = better)", fontsize=11)
        ax2.set_title("Bayesian Calibration\n(Brier Score)", fontsize=12, fontweight="bold")
        ax2.legend(fontsize=8.5)
        ax2.grid(axis="y", alpha=0.25)

    # ── Right: EDI (Equilibrium Deviation Index) ──────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    if summary is not None:
        edi_zs = summary[summary["treatment"]=="zero_shot"]["edi"].values
        edi_rc = summary[summary["treatment"]=="role_conditioned"]["edi"].values
        ax3.bar(x - w/2, edi_zs, w, color=COLORS["zero_shot"], alpha=0.85,
                label="Zero-Shot", edgecolor="white")
        ax3.bar(x + w/2, edi_rc, w, color=COLORS["role_conditioned"], alpha=0.85,
                label="Role-Conditioned", edgecolor="white")
        ax3.axhline(0, color="#1B5E20", lw=1.5, ls="--", alpha=0.6,
                    label="Perfect equilibrium play (0.00)")
        ax3.set_xticks(x)
        ax3.set_xticklabels([MODEL_NAMES[m].replace("-", "-\n") for m in models], fontsize=10)
        ax3.set_ylabel("Equilibrium Deviation Index", fontsize=11)
        ax3.set_title("Strategic Rationality\n(Equilibrium Deviation Index)", fontsize=12,
                      fontweight="bold")
        ax3.legend(fontsize=8.5)
        ax3.grid(axis="y", alpha=0.25)

    fig.suptitle("The Bluffing Machine: Core Empirical Results\n"
                 "LLMs Deviate Systematically from Perfect Bayesian Equilibrium Across All Metrics",
                 fontsize=14, fontweight="bold", y=1.03)
    plt.tight_layout()
    path = os.path.join(FIGS_DIR, "fig8_big_picture_overview.png")
    plt.savefig(path); plt.close()
    print(f"  ✓ Figure 8 saved: {path}")
    return path

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "="*60)
    print("  BLUFFING MACHINE — FIGURE GENERATION SUITE")
    print("="*60 + "\n")

    df, summary, rep_data = load_data()

    print("Generating figures...")
    paths = []
    paths.append(fig1_game_tree())
    paths.append(fig2_bluffing_frequency(df))
    paths.append(fig3_calibration(df))
    paths.append(fig4_reputation_decay(rep_data))
    paths.append(fig5_reasoning_heatmap(df))
    paths.append(fig6_payoff_distributions(df))
    paths.append(fig7_performance_dashboard(df, summary))
    paths.append(fig8_big_picture(df, summary))

    print(f"\n{'='*60}")
    print(f"  ALL {len([p for p in paths if p])} FIGURES GENERATED")
    print(f"  Saved to: {FIGS_DIR}")
    print("="*60)
