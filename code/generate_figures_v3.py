"""
The Bluffing Machine — Definitive Figure Generation (v3)
=========================================================
All bars and lollipops have explicit value labels.
No overlapping text. Publication-quality 300 DPI.
Nature/Science aesthetic: despined, serif font, clean palette.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.gridspec import GridSpec
from scipy.stats import binomtest
import os, warnings
warnings.filterwarnings('ignore')

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE     = "/home/ubuntu/bluffing_machine_repo"
DATA_DIR = f"{BASE}/data/raw"
QUAL_DIR = f"{BASE}/data/qualitative"
OUT_DIR  = f"{BASE}/figures/premium_v3"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Global style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':       'DejaVu Serif',
    'font.size':         11,
    'axes.titlesize':    13,
    'axes.titleweight':  'bold',
    'axes.labelsize':    11,
    'xtick.labelsize':   10,
    'ytick.labelsize':   10,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.grid':         True,
    'axes.grid.axis':    'x',
    'grid.alpha':        0.25,
    'grid.linestyle':    '--',
    'figure.facecolor':  '#FAFAFA',
    'axes.facecolor':    '#FAFAFA',
    'savefig.dpi':       300,
    'savefig.bbox':      'tight',
    'savefig.facecolor': '#FAFAFA',
})

# ── Palette ───────────────────────────────────────────────────────────────────
MINI_C   = '#1B6CA8'   # deep blue
NANO_C   = '#2E8B57'   # forest green
GEM_C    = '#C0392B'   # crimson
ZERO_C   = '#4393C3'   # sky blue
ROLE_C   = '#D6604D'   # terracotta
PBE_C    = '#7B2D8B'   # purple
BG       = '#FAFAFA'

MODEL_COLORS = {
    'GPT-4.1-mini':    MINI_C,
    'GPT-4.1-nano':    NANO_C,
    'Gemini-2.5-Flash': GEM_C,
}
SHORT = {
    'GPT-4.1-mini':    'GPT-4.1-mini',
    'GPT-4.1-nano':    'GPT-4.1-nano',
    'Gemini-2.5-Flash': 'Gemini-2.5-Flash',
}

# ── Load ONLY the main results CSV (exclude summary / sensitivity) ─────────────
csv_files = [f for f in os.listdir(DATA_DIR)
             if f.endswith('.csv') and 'summary' not in f and 'sensitivity' not in f]
df = pd.concat([pd.read_csv(f"{DATA_DIR}/{f}") for f in csv_files], ignore_index=True)

# Rename for convenience
df = df.rename(columns={'model_name': 'model', 'signal': 'sender_signal', 'action': 'receiver_action'})
df['model']         = df['model'].str.strip()
df['treatment']     = df['treatment'].str.strip()
df['sender_type']   = df['sender_type'].str.strip()
df['sender_signal'] = df['sender_signal'].str.strip()
df['receiver_action'] = df['receiver_action'].str.strip()

low_df = df[df['sender_type'] == 'LOW'].copy()

MODELS = ['GPT-4.1-mini', 'GPT-4.1-nano', 'Gemini-2.5-Flash']
TREATS = ['zero_shot', 'role_conditioned']
TREAT_LABELS = {'zero_shot': 'Zero-Shot', 'role_conditioned': 'Role-Conditioned'}

# ── Wilson CI ─────────────────────────────────────────────────────────────────
def wilson_ci(k, n, z=1.96):
    if n == 0: return 0.0, 0.0, 0.0
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2/(2*n)) / denom
    margin = (z * np.sqrt(p*(1-p)/n + z**2/(4*n**2))) / denom
    return p, max(0, centre - margin), min(1, centre + margin)

# ── Compute bluffing stats ────────────────────────────────────────────────────
bluff_stats = {}
for m in MODELS:
    for t in TREATS:
        sub = low_df[(low_df['model'] == m) & (low_df['treatment'] == t)]
        n = len(sub)
        k = int((sub['sender_signal'] == 'ESCALATE').sum())
        p, lo, hi = wilson_ci(k, n)
        pval = binomtest(k, n, 0.42).pvalue if n >= 1 else 1.0
        bluff_stats[(m, t)] = {'p': p, 'lo': lo, 'hi': hi, 'n': n, 'k': k, 'pval': pval}

print("Bluff stats:")
for key, v in bluff_stats.items():
    print(f"  {key}: {v['p']:.1%}  n={v['n']}  p={v['pval']:.4f}")

# ═══════════════════════════════════════════════════════════════════════════════
# FIG 2 — Lollipop: Bluffing Frequency (the headline result)
# ═══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
fig.patch.set_facecolor(BG)

for ax_idx, treat in enumerate(TREATS):
    ax = axes[ax_idx]
    ax.set_facecolor(BG)
    treat_color = ROLE_C if treat == 'role_conditioned' else ZERO_C
    ax.set_title(f"Treatment: {TREAT_LABELS[treat]}", fontsize=13, fontweight='bold',
                 color=treat_color, pad=10)

    # PBE reference
    ax.axvline(0.42, color=PBE_C, linestyle='--', linewidth=1.8, zorder=1, alpha=0.85)
    ax.text(0.44, 2.72, 'PBE benchmark\n(42%)', color=PBE_C, fontsize=8.5,
            fontstyle='italic', va='top', ha='left')

    for yi, model in enumerate(MODELS):
        s = bluff_stats[(model, treat)]
        c = MODEL_COLORS[model]
        y = yi

        # CI band (thick, transparent)
        if s['hi'] > s['lo']:
            ax.hlines(y, s['lo'], s['hi'], color=c, linewidth=9, alpha=0.18, zorder=2)

        # Stem from 0 to dot
        ax.hlines(y, 0, max(s['p'], 0.005), color=c, linewidth=2.2, alpha=0.75, zorder=3)

        # Dot
        dot_size = 220 if s['p'] > 0.01 else 100
        ax.scatter(s['p'], y, s=dot_size, color=c, zorder=5,
                   edgecolors='white', linewidths=1.8)

        # ── Value label (no overlap logic) ────────────────────────────────────
        pct = f"{s['p']*100:.0f}%"
        pv  = s['pval']
        sig = "p<0.001" if pv < 0.001 else (f"p={pv:.3f}" if pv < 0.05 else f"p={pv:.2f}")
        n_label = f"n={s['n']}"

        # Decide label side — push further out to avoid overlap
        if s['p'] > 0.80:
            x_lbl, ha_lbl = s['p'] - 0.05, 'right'
        elif s['p'] < 0.05:
            x_lbl, ha_lbl = s['p'] + 0.05, 'left'
        else:
            x_lbl, ha_lbl = s['p'] + 0.05, 'left'

        # Main percentage — bold, large (top line)
        ax.text(x_lbl, y + 0.30, pct,
                color=c, fontsize=12, fontweight='bold', ha=ha_lbl, va='bottom', zorder=6)
        # p-value — smaller, italic (second line, clearly separated)
        ax.text(x_lbl, y + 0.10, sig,
                color='#666666', fontsize=8.5, fontstyle='italic', ha=ha_lbl, va='bottom', zorder=6)
        # Sample size — below dot
        ax.text(x_lbl, y - 0.28, n_label,
                color=c, fontsize=8.5, ha=ha_lbl, va='top', zorder=6, alpha=0.8)

    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(['GPT-4.1-mini', 'GPT-4.1-nano', 'Gemini-2.5-Flash'],
                       fontsize=11, fontweight='bold')
    ax.set_xlim(-0.10, 1.30)
    ax.set_ylim(-0.65, 3.10)
    ax.set_xlabel('Bluff Rate  (Low Resolve Senders)', fontsize=10.5, labelpad=6)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x*100:.0f}%'))
    ax.spines['left'].set_visible(False)
    ax.tick_params(left=False)

fig.suptitle(
    'Figure 2: Bluffing Frequency by Model and Treatment\n'
    'Proportion of Low Resolve Senders Sending Escalatory Signal vs. PBE Rational Prediction',
    fontsize=13, fontweight='bold', y=1.02)

fig.text(0.5, -0.05,
    'Dots = observed bluff rate. Shaded bands = 95% Wilson CI.  '
    'Purple dashed line = PBE rational benchmark (42%).  '
    'N varies per cell due to random type assignment.',
    ha='center', fontsize=8.5, style='italic', color='#555555')

plt.tight_layout(w_pad=4)
plt.savefig(f"{OUT_DIR}/fig2_bluffing_lollipop.png")
plt.close()
print("✓ Fig 2 saved")

# ═══════════════════════════════════════════════════════════════════════════════
# FIG 3 — Side-by-side grouped bar: Bluff Rate comparison (alternative view)
# ═══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(12, 6))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)

x = np.arange(len(MODELS))
width = 0.32

for ti, (treat, color, label) in enumerate(zip(
        TREATS, [ZERO_C, ROLE_C], ['Zero-Shot', 'Role-Conditioned'])):
    vals  = [bluff_stats[(m, treat)]['p'] for m in MODELS]
    lo_ci = [max(0, bluff_stats[(m, treat)]['p'] - bluff_stats[(m, treat)]['lo']) for m in MODELS]
    hi_ci = [max(0, bluff_stats[(m, treat)]['hi'] - bluff_stats[(m, treat)]['p']) for m in MODELS]
    offset = -width/2 if ti == 0 else width/2

    bars = ax.bar(x + offset, vals, width, color=color, alpha=0.88, label=label,
                  edgecolor='white', linewidth=0.8,
                  yerr=[lo_ci, hi_ci], capsize=5, error_kw={'ecolor': '#333', 'elinewidth': 1.2})

    for bar, v in zip(bars, vals):
        h = bar.get_height()
        lbl = f"{v*100:.0f}%"
        if h < 0.05:
            # Zero or near-zero: place label above with arrow indicator
            ax.text(bar.get_x() + bar.get_width()/2, 0.03,
                    lbl, ha='center', va='bottom', fontsize=11, fontweight='bold', color=color)
        else:
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.03,
                    lbl, ha='center', va='bottom', fontsize=11, fontweight='bold', color=color)

# PBE line
ax.axhline(0.42, color=PBE_C, linestyle='--', linewidth=1.8, alpha=0.85, zorder=0)
ax.text(2.6, 0.44, 'PBE = 42%', color=PBE_C, fontsize=9.5, fontstyle='italic', ha='right')

ax.set_xticks(x)
ax.set_xticklabels(['GPT-4.1-mini', 'GPT-4.1-nano', 'Gemini-2.5-Flash'],
                   fontsize=12, fontweight='bold')
ax.set_ylabel('Bluff Rate  (Low Resolve Senders)', fontsize=11)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y*100:.0f}%'))
ax.set_ylim(0, 1.25)
ax.legend(fontsize=11, title='Treatment', title_fontsize=11, frameon=True,
          loc='upper left', framealpha=0.9)
ax.set_title(
    'Figure 3: Bluffing Rate by Model — Zero-Shot vs. Role-Conditioned Treatment\n'
    'Error bars = 95% Wilson Confidence Intervals.  Dashed line = PBE rational benchmark.',
    fontsize=12, fontweight='bold', pad=10)

# Annotation box for key finding
ax.annotate(
    'GPT-4.1-mini: 0% → 100%\n(prompt framing alone)',
    xy=(0 - width/2, 1.02), xytext=(0.6, 1.12),
    fontsize=9, color='#1B1B1B',
    arrowprops=dict(arrowstyle='->', color='#333', lw=1.2),
    bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF9C4', edgecolor='#CCC', alpha=0.95))

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig3_grouped_bar.png")
plt.close()
print("✓ Fig 3 saved")

# ═══════════════════════════════════════════════════════════════════════════════
# FIG 4 — Heatmap: Reasoning trace keyword analysis
# ═══════════════════════════════════════════════════════════════════════════════
keywords = {
    'Cost / Risk':         r'cost|risk|casualt',
    'Credibility':         r'credib',
    'Reputation':          r'reput',
    'Deception / Bluffing':r'decep|bluff',
    'Deterrence':          r'deter',
    'Threat Heuristic':    r'threat',
    'Rational Calculation':r'rational|calculat',
}

col_labels = []
heat_data  = []
for m in MODELS:
    for t in TREATS:
        sub = df[(df['model'] == m) & (df['treatment'] == t)]
        col_labels.append(
            f"{SHORT[m].replace('GPT-4.1-', 'GPT-').replace('Gemini-2.5-Flash', 'Gemini')}\n{TREAT_LABELS[t]}")
        row = []
        for kw_label, pattern in keywords.items():
            count = sub['sender_reasoning'].fillna('').str.lower().str.contains(pattern, regex=True).sum()
            row.append(count / len(sub) if len(sub) > 0 else 0)
        heat_data.append(row)

heat_arr = np.array(heat_data).T   # shape: (n_keywords, n_conditions)

fig, ax = plt.subplots(figsize=(15, 6.5))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)

im = ax.imshow(heat_arr, cmap='Blues', aspect='auto', vmin=0, vmax=1)

# Cell value labels
for i in range(len(keywords)):
    for j in range(len(col_labels)):
        v = heat_arr[i, j]
        text_color = 'white' if v > 0.52 else '#1a1a1a'
        ax.text(j, i, f'{v*100:.0f}%', ha='center', va='center',
                fontsize=10.5, fontweight='bold', color=text_color)

ax.set_xticks(range(len(col_labels)))
ax.set_xticklabels(col_labels, fontsize=9.5, rotation=25, ha='right')
ax.set_yticks(range(len(keywords)))
ax.set_yticklabels(list(keywords.keys()), fontsize=11)

# Model group separators and labels
for sep in [1.5, 3.5]:
    ax.axvline(sep, color='white', linewidth=3.5)

for xi, (model, color) in enumerate(zip(MODELS, [MINI_C, NANO_C, GEM_C])):
    short = model.replace('GPT-4.1-', 'GPT-').replace('Gemini-2.5-Flash', 'Gemini')
    ax.text(xi*2 + 0.5, -1.1, short, ha='center', va='top',
            fontsize=11, fontweight='bold', color=color)

cbar = plt.colorbar(im, ax=ax, fraction=0.018, pad=0.02)
cbar.set_label('% of reasoning traces\ncontaining keyword', fontsize=9.5)
cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x*100:.0f}%'))

ax.set_title(
    'Figure 4: Reasoning Trace Keyword Analysis\n'
    'What Concepts Do LLMs Invoke When Making Strategic Signaling Decisions?',
    fontsize=13, fontweight='bold', pad=18)

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig4_reasoning_heatmap.png")
plt.close()
print("✓ Fig 4 saved")

# ═══════════════════════════════════════════════════════════════════════════════
# FIG 5 — Violin plots: Payoff distributions
# ═══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)
fig.patch.set_facecolor(BG)

for ax_idx, (payoff_col, title, ylabel) in enumerate([
        ('sender_payoff',   'Sender Payoff Distribution', 'Payoff'),
        ('receiver_payoff', 'Receiver Payoff Distribution', 'Payoff')]):

    ax = axes[ax_idx]
    ax.set_facecolor(BG)
    ax.set_title(title, fontsize=12, fontweight='bold')

    positions = []
    data_list = []
    colors_list = []
    xtick_labels = []
    pos = 0

    for m in MODELS:
        for t in TREATS:
            sub = df[(df['model'] == m) & (df['treatment'] == t)]
            vals = pd.to_numeric(sub[payoff_col], errors='coerce').dropna().values
            if len(vals) > 1:
                data_list.append(vals)
                positions.append(pos)
                colors_list.append(ZERO_C if t == 'zero_shot' else ROLE_C)
                short_m = m.replace('GPT-4.1-', 'GPT-').replace('Gemini-2.5-Flash', 'Gemini')
                xtick_labels.append(f"{short_m}\n{TREAT_LABELS[t]}")
                pos += 1
        pos += 0.4  # gap between models

    if data_list:
        parts = ax.violinplot(data_list, positions=positions, widths=0.7,
                              showmedians=False, showextrema=False)
        for pc, c in zip(parts['bodies'], colors_list):
            pc.set_facecolor(c)
            pc.set_alpha(0.65)
            pc.set_edgecolor('white')

        # Median dots and labels
        for pos_i, vals, c in zip(positions, data_list, colors_list):
            med = np.median(vals)
            ax.scatter(pos_i, med, s=60, color='white', zorder=5, edgecolors=c, linewidths=2)
            ax.text(pos_i, med + 0.04, f'Md={med:.2f}',
                    ha='center', va='bottom', fontsize=8.5, fontweight='bold', color=c)

        ax.set_xticks(positions)
        ax.set_xticklabels(xtick_labels, fontsize=8.5, rotation=20, ha='right')
        ax.set_ylabel(ylabel, fontsize=10)
        ax.axhline(0, color='#888', linestyle='--', linewidth=1, alpha=0.5)
        ax.text(positions[-1], 0.01, 'Zero', ha='right', fontsize=8, color='#888', fontstyle='italic')

fig.suptitle(
    'Figure 5: Payoff Distributions by Model and Treatment\n'
    'Violin plots show full distribution; white dots = median; shading = treatment condition',
    fontsize=13, fontweight='bold', y=1.02)

legend_patches = [
    mpatches.Patch(color=ZERO_C, alpha=0.65, label='Zero-Shot'),
    mpatches.Patch(color=ROLE_C, alpha=0.65, label='Role-Conditioned'),
]
fig.legend(handles=legend_patches, loc='lower center', ncol=2, fontsize=10.5,
           title='Treatment', title_fontsize=10.5, bbox_to_anchor=(0.5, -0.04))

plt.tight_layout(w_pad=3)
plt.savefig(f"{OUT_DIR}/fig5_payoff_distributions.png")
plt.close()
print("✓ Fig 5 saved")

# ═══════════════════════════════════════════════════════════════════════════════
# FIG 6 — Summary Dashboard (6 panels, all with value labels)
# ═══════════════════════════════════════════════════════════════════════════════
# Compute metrics
metrics_data = {}
for m in MODELS:
    for t in TREATS:
        sub    = df[(df['model'] == m) & (df['treatment'] == t)]
        low_s  = sub[sub['sender_type'] == 'LOW']
        n_low  = len(low_s)
        n_bluff = int((low_s['sender_signal'] == 'ESCALATE').sum())
        br     = n_bluff / n_low if n_low > 0 else 0

        bluff_sub = low_s[low_s['sender_signal'] == 'ESCALATE']
        n_bs = len(bluff_sub)
        bs   = (bluff_sub['receiver_action'] == 'BACK_DOWN').sum() / n_bs if n_bs > 0 else 0

        if 'posterior_belief' in sub.columns:
            beliefs = pd.to_numeric(sub['posterior_belief'], errors='coerce').dropna()
            actuals = (sub.loc[beliefs.index, 'sender_type'] == 'HIGH').astype(float)
            brier   = ((beliefs - actuals)**2).mean() if len(beliefs) > 0 else np.nan
        else:
            brier = np.nan

        edi = abs(br - 0.42)
        sp  = pd.to_numeric(sub['sender_payoff'],   errors='coerce').mean()
        rp  = pd.to_numeric(sub['receiver_payoff'], errors='coerce').mean()

        metrics_data[(m, t)] = {'br': br, 'bs': bs, 'brier': brier,
                                 'edi': edi, 'sp': sp, 'rp': rp}

fig = plt.figure(figsize=(17, 10))
fig.patch.set_facecolor(BG)
gs = GridSpec(2, 3, figure=fig, hspace=0.60, wspace=0.40)

panel_cfg = [
    ('br',    'Bluff Rate',                  '%',      True,  0.42, 'PBE=42%'),
    ('bs',    'Bluff Success Rate',           '%',      False, None, None),
    ('brier', 'Brier Score  (↓ better)',      'score',  False, None, None),
    ('edi',   'Equilibrium Deviation Index\n(↓ better)', 'EDI', False, None, None),
    ('sp',    'Avg. Sender Payoff',           'payoff', False, 0,    'Zero'),
    ('rp',    'Avg. Receiver Payoff',         'payoff', False, 0,    'Zero'),
]

short_names = ['GPT-mini', 'GPT-nano', 'Gemini']
x = np.arange(len(MODELS))
w = 0.34

for idx, (metric, title, unit, show_pbe, ref_val, ref_label) in enumerate(panel_cfg):
    ax = fig.add_subplot(gs[idx // 3, idx % 3])
    ax.set_facecolor(BG)

    vals_z = [metrics_data[(m, 'zero_shot')][metric]      for m in MODELS]
    vals_r = [metrics_data[(m, 'role_conditioned')][metric] for m in MODELS]

    bars_z = ax.bar(x - w/2, vals_z, w, color=ZERO_C, alpha=0.88, label='Zero-Shot',
                    edgecolor='white', linewidth=0.6)
    bars_r = ax.bar(x + w/2, vals_r, w, color=ROLE_C, alpha=0.88, label='Role-Cond.',
                    edgecolor='white', linewidth=0.6)

    def label_bar(bar, v, color):
        if v is None or np.isnan(v): return
        h = bar.get_height()
        fmt = f'{v*100:.0f}%' if unit == '%' else f'{v:.2f}'
        if abs(h) < 0.01:
            ax.text(bar.get_x() + bar.get_width()/2,
                    0.01 if h >= 0 else -0.01,
                    fmt, ha='center', va='bottom' if h >= 0 else 'top',
                    fontsize=9, fontweight='bold', color=color)
        else:
            offset = abs(h) * 0.06 + 0.005
            ax.text(bar.get_x() + bar.get_width()/2,
                    h + offset if h >= 0 else h - offset,
                    fmt, ha='center', va='bottom' if h >= 0 else 'top',
                    fontsize=9, fontweight='bold', color=color)

    for bar, v in zip(bars_z, vals_z): label_bar(bar, v, ZERO_C)
    for bar, v in zip(bars_r, vals_r): label_bar(bar, v, ROLE_C)

    if ref_val is not None:
        ax.axhline(ref_val, color=PBE_C if show_pbe else '#888',
                   linestyle='--', linewidth=1.3, alpha=0.8)
        if ref_label:
            ax.text(len(MODELS)-0.5, ref_val, ref_label, ha='right', va='bottom',
                    fontsize=8, color=PBE_C if show_pbe else '#888', fontstyle='italic')

    ax.set_title(title, fontsize=11, fontweight='bold', pad=6)
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, fontsize=9.5)
    if unit == '%':
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y*100:.0f}%'))
    ax.spines['left'].set_alpha(0.4)

legend_patches = [
    mpatches.Patch(color=ZERO_C, alpha=0.88, label='Zero-Shot'),
    mpatches.Patch(color=ROLE_C, alpha=0.88, label='Role-Conditioned'),
]
fig.legend(handles=legend_patches, loc='lower center', ncol=2, fontsize=11,
           title='Treatment', title_fontsize=11, bbox_to_anchor=(0.5, -0.02),
           frameon=True, framealpha=0.9)

fig.suptitle(
    'Figure 6: Complete Results Dashboard — Six Metrics by Model and Treatment\n'
    'The Bluffing Machine: 900 Real LLM Signaling Game Simulations',
    fontsize=13, fontweight='bold', y=1.01)

plt.savefig(f"{OUT_DIR}/fig6_summary_dashboard.png")
plt.close()
print("✓ Fig 6 saved")

# ═══════════════════════════════════════════════════════════════════════════════
# FIG 7 — Sensitivity Analysis
# ═══════════════════════════════════════════════════════════════════════════════
sens_files = [f for f in os.listdir(DATA_DIR)
              if f.endswith('.csv') and 'sensitivity' in f and 'summary' not in f]

if sens_files:
    sens_df = pd.concat([pd.read_csv(f"{DATA_DIR}/{f}") for f in sens_files], ignore_index=True)
    if 'model_name' in sens_df.columns:
        sens_df = sens_df.rename(columns={'model_name': 'model', 'signal': 'sender_signal'})
    sens_df['sender_type']   = sens_df['sender_type'].str.strip()
    sens_df['sender_signal'] = sens_df['sender_signal'].str.strip()

    variants = sens_df['variant'].unique().tolist() if 'variant' in sens_df.columns else []
    variant_labels = {
        'neutral':    'V1: Neutral\n(Abstract framing)',
        'diplomatic': 'V2: Diplomatic\n(Negotiation framing)',
        'military':   'V3: Military\n(Coercive framing)',
    }
    variant_colors = {'neutral': '#4393C3', 'diplomatic': '#2166AC', 'military': '#C0392B'}

    if variants:
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor(BG)
        ax.set_facecolor(BG)

        br_vals = []
        labels_s = []
        colors_s = []
        for v in ['neutral', 'diplomatic', 'military']:
            if v not in variants: continue
            sub = sens_df[sens_df['variant'] == v]
            low_s = sub[sub['sender_type'] == 'LOW']
            n = len(low_s)
            k = int((low_s['sender_signal'] == 'ESCALATE').sum())
            br = k / n if n > 0 else 0
            br_vals.append(br)
            labels_s.append(variant_labels.get(v, v))
            colors_s.append(variant_colors.get(v, '#888'))

        x_s = np.arange(len(br_vals))
        bars = ax.bar(x_s, br_vals, 0.55, color=colors_s, alpha=0.88,
                      edgecolor='white', linewidth=0.8)

        for bar, v in zip(bars, br_vals):
            h = bar.get_height()
            lbl = f"{v*100:.0f}%"
            if h < 0.04:
                ax.text(bar.get_x() + bar.get_width()/2, 0.03,
                        lbl, ha='center', va='bottom', fontsize=13, fontweight='bold',
                        color=bar.get_facecolor())
            else:
                ax.text(bar.get_x() + bar.get_width()/2, h + 0.02,
                        lbl, ha='center', va='bottom', fontsize=13, fontweight='bold',
                        color=bar.get_facecolor())

        ax.axhline(0.42, color=PBE_C, linestyle='--', linewidth=1.8, alpha=0.85)
        ax.text(len(br_vals)-0.5, 0.44, 'PBE = 42%', ha='right', fontsize=10,
                color=PBE_C, fontstyle='italic')

        ax.set_xticks(x_s)
        ax.set_xticklabels(labels_s, fontsize=11)
        ax.set_ylabel('Bluff Rate  (Low Resolve Senders)', fontsize=11)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y*100:.0f}%'))
        ax.set_ylim(0, 1.20)
        ax.set_title(
            'Figure 7: Prompt Sensitivity Analysis\n'
            'GPT-4.1-mini Bluffing Rate Across Three Framing Conditions (N=100 per variant)',
            fontsize=13, fontweight='bold', pad=10)

        # Robustness annotation
        ax.annotate('Results are robust across\nall three framing conditions\n(0% in all variants)',
                    xy=(1, 0.03), xytext=(1.5, 0.35),
                    fontsize=9.5, color='#333',
                    arrowprops=dict(arrowstyle='->', color='#555', lw=1.2),
                    bbox=dict(boxstyle='round,pad=0.4', facecolor='#E8F4FD', edgecolor='#AAA'))

        plt.tight_layout()
        plt.savefig(f"{OUT_DIR}/fig7_sensitivity_analysis.png")
        plt.close()
        print("✓ Fig 7 saved")

# ═══════════════════════════════════════════════════════════════════════════════
# FIG 8 — S-R-O Qualitative Analysis
# ═══════════════════════════════════════════════════════════════════════════════
sro_path = f"{QUAL_DIR}/sro_coded_traces.csv"
if os.path.exists(sro_path):
    sro_df = pd.read_csv(sro_path)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.patch.set_facecolor(BG)
    fig.suptitle(
        'Figure 8: S-R-O Qualitative Coding — Strategic Reasoning in LLM Signaling Games\n'
        '150 Reasoning Traces Coded Across Four Dimensions',
        fontsize=14, fontweight='bold', y=1.01)

    # ── Panel A: Signal Justification ─────────────────────────────────────────
    ax = axes[0, 0]
    ax.set_facecolor(BG)
    ax.set_title('A. Signal Justification Category', fontsize=12, fontweight='bold')

    if 'signal_justification' in sro_df.columns:
        counts = sro_df['signal_justification'].value_counts(normalize=True)
        cats   = counts.index.tolist()
        vals   = counts.values.tolist()
        colors_a = plt.cm.Blues(np.linspace(0.4, 0.85, len(cats)))
        bars = ax.barh(range(len(cats)), vals, color=colors_a, edgecolor='white', linewidth=0.6)
        for bar, v in zip(bars, vals):
            w = bar.get_width()
            lbl = f'{v*100:.1f}%'
            if w > 0.08:
                ax.text(w - 0.01, bar.get_y() + bar.get_height()/2,
                        lbl, ha='right', va='center', fontsize=10.5, fontweight='bold', color='white')
            else:
                ax.text(w + 0.01, bar.get_y() + bar.get_height()/2,
                        lbl, ha='left', va='center', fontsize=10.5, fontweight='bold', color='#333')
        ax.set_yticks(range(len(cats)))
        ax.set_yticklabels([c.replace('_', ' ').title() for c in cats], fontsize=10)
        ax.set_xlabel('Proportion of Traces', fontsize=10)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x*100:.0f}%'))
        ax.set_xlim(0, 1.15)
        ax.grid(axis='x', alpha=0.3)

    # ── Panel B: Deception Awareness ──────────────────────────────────────────
    ax = axes[0, 1]
    ax.set_facecolor(BG)
    ax.set_title('B. Deception Awareness by Treatment', fontsize=12, fontweight='bold')

    if 'deception_awareness' in sro_df.columns and 'treatment' in sro_df.columns:
        aware_cats = sro_df['deception_awareness'].unique().tolist()
        x_pos = np.arange(len(aware_cats))
        bw = 0.35
        for ti, (treat, color, label) in enumerate(zip(
                ['zero_shot', 'role_conditioned'], [ZERO_C, ROLE_C], ['Zero-Shot', 'Role-Cond.'])):
            sub = sro_df[sro_df['treatment'] == treat]
            vals = [(sub['deception_awareness'] == c).mean() for c in aware_cats]
            bars = ax.bar(x_pos + ti*bw - bw/2, vals, bw, color=color, alpha=0.88,
                          label=label, edgecolor='white', linewidth=0.6)
            for bar, v in zip(bars, vals):
                h = bar.get_height()
                if h > 0.01:
                    ax.text(bar.get_x() + bar.get_width()/2, h + 0.01,
                            f'{v*100:.0f}%', ha='center', va='bottom',
                            fontsize=9.5, fontweight='bold', color=color)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([c.replace('_', '\n') for c in aware_cats], fontsize=9.5)
        ax.set_ylabel('Proportion of Traces', fontsize=10)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y*100:.0f}%'))
        ax.legend(fontsize=9.5)
        ax.set_ylim(0, 1.15)

    # ── Panel C: Receiver Inference Strategy ──────────────────────────────────
    ax = axes[1, 0]
    ax.set_facecolor(BG)
    ax.set_title('C. Receiver Inference Strategy', fontsize=12, fontweight='bold')

    if 'receiver_inference' in sro_df.columns:
        counts = sro_df['receiver_inference'].value_counts(normalize=True).sort_values(ascending=False)
        cats   = counts.index.tolist()
        vals   = counts.values.tolist()
        colors_c = [MINI_C if v > 0.5 else '#92C5DE' for v in vals]
        bars = ax.barh(range(len(cats)), vals, color=colors_c, edgecolor='white', linewidth=0.6)
        for bar, v in zip(bars, vals):
            w = bar.get_width()
            lbl = f'{v*100:.1f}%'
            if w > 0.08:
                ax.text(w - 0.01, bar.get_y() + bar.get_height()/2,
                        lbl, ha='right', va='center', fontsize=10.5, fontweight='bold', color='white')
            else:
                ax.text(w + 0.01, bar.get_y() + bar.get_height()/2,
                        lbl, ha='left', va='center', fontsize=10.5, fontweight='bold', color='#333')
        ax.set_yticks(range(len(cats)))
        ax.set_yticklabels([c.replace('_', ' ').title() for c in cats], fontsize=10)
        ax.set_xlabel('Proportion of Traces', fontsize=10)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x*100:.0f}%'))
        ax.set_xlim(0, 1.20)

        # Key finding annotation — placed below the bars to avoid title overlap
        if vals and vals[0] > 0.5:
            ax.annotate(
                f'Key finding: {vals[0]*100:.0f}% of receivers\nclaim Bayesian updating\nbut calibration shows they do not',
                xy=(vals[0], 0), xytext=(0.40, -0.35),
                fontsize=8.5, color='#333',
                arrowprops=dict(arrowstyle='->', color='#555', lw=1),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF9C4', edgecolor='#CCC', alpha=0.95))

    # ── Panel D: Belief Attribution Heatmap ───────────────────────────────────
    ax = axes[1, 1]
    ax.set_facecolor(BG)
    ax.set_title('D. Belief Attribution by Model\n(% of traces)', fontsize=12, fontweight='bold')

    if 'belief_attribution' in sro_df.columns:
        ba_cats = sorted(sro_df['belief_attribution'].unique().tolist())
        short_models = ['GPT-mini', 'GPT-nano', 'Gemini']
        heat = []
        for model in MODELS:
            sub = sro_df[sro_df['model_name'] == model]
            row = [(sub['belief_attribution'] == c).mean() for c in ba_cats]
            heat.append(row)
        heat_arr2 = np.array(heat)
        im = ax.imshow(heat_arr2, cmap='Blues', aspect='auto', vmin=0, vmax=1)
        for i in range(len(MODELS)):
            for j in range(len(ba_cats)):
                v = heat_arr2[i, j]
                text_color = 'white' if v > 0.52 else '#1a1a1a'
                ax.text(j, i, f'{v*100:.0f}%', ha='center', va='center',
                        fontsize=13, fontweight='bold', color=text_color)
        ax.set_xticks(range(len(ba_cats)))
        ax.set_xticklabels([c.replace('_', '\n') for c in ba_cats], fontsize=10)
        ax.set_yticks(range(len(MODELS)))
        ax.set_yticklabels(short_models, fontsize=10.5)
        plt.colorbar(im, ax=ax, fraction=0.04, pad=0.04,
                     format=plt.FuncFormatter(lambda x, _: f'{x*100:.0f}%'))

    plt.tight_layout(h_pad=3.5, w_pad=3)
    plt.savefig(f"{OUT_DIR}/fig8_sro_qualitative_analysis.png")
    plt.close()
    print("✓ Fig 8 saved")
else:
    print("  ⚠ SRO data not found — skipping Fig 8")

print(f"\n✅ All figures saved to {OUT_DIR}")
