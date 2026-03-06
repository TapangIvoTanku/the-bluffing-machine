"""
S-R-O (Signal-Reason-Outcome) Qualitative Coding Framework
===========================================================
A novel qualitative analysis framework for signaling game reasoning traces.
Codes each reasoning trace across 4 dimensions grounded in the formal model:
  1. Signal Justification  - what reason did the Sender give for its signal?
  2. Belief Attribution    - did the Sender explicitly model opponent's beliefs?
  3. Deception Awareness   - did the Sender acknowledge bluffing/type concealment?
  4. Receiver Inference    - how did the Receiver interpret and update beliefs?

This framework is directly analogous to Payne's (2026) R-F-D architecture
but is specifically designed for signaling games and maps to formal model variables.
"""

import pandas as pd
import json
import os
import re
from openai import OpenAI
from collections import Counter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from tqdm import tqdm

client = OpenAI()

# ── Load data ──────────────────────────────────────────────────────────────────
DATA_DIR = "/home/ubuntu/bluffing_machine_repo/data/raw"
OUT_DIR  = "/home/ubuntu/bluffing_machine_repo/data/qualitative"
FIG_DIR  = "/home/ubuntu/bluffing_machine_repo/figures/premium_v2"
os.makedirs(OUT_DIR, exist_ok=True)

csv_files = [f for f in os.listdir(DATA_DIR) if f.startswith("main_results") and f.endswith(".csv")]
df = pd.concat([pd.read_csv(os.path.join(DATA_DIR, f)) for f in csv_files], ignore_index=True)

# Also load sensitivity data if available
sens_files = [f for f in os.listdir(DATA_DIR) if "sensitivity" in f and f.endswith(".csv")]
if sens_files:
    df_sens = pd.concat([pd.read_csv(os.path.join(DATA_DIR, f)) for f in sens_files], ignore_index=True)
    df_all = pd.concat([df, df_sens], ignore_index=True)
else:
    df_all = df.copy()

print(f"Total games loaded: {len(df_all)}")
print(f"Columns: {list(df_all.columns)}")

# ── Sample for coding (use a representative stratified sample) ─────────────────
# Sample 150 games: 50 per model, balanced across treatments and sender types
sample_rows = []
for model in df_all['model_key'].unique():
    for treatment in df_all['treatment'].unique():
        subset = df_all[(df_all['model_key'] == model) & (df_all['treatment'] == treatment)]
        n = min(25, len(subset))
        if n > 0:
            sample_rows.append(subset.sample(n=n, random_state=42))

sample_df = pd.concat(sample_rows, ignore_index=True)
print(f"\nStratified sample size: {len(sample_df)} games")

# ── S-R-O Coding Function ──────────────────────────────────────────────────────
SRO_SYSTEM_PROMPT = """You are an expert qualitative coder for political science research on AI strategic reasoning.
Your task is to code reasoning traces from a signaling game experiment using the S-R-O framework.

For each reasoning trace, code the following dimensions:

SENDER CODING:
1. signal_justification: The primary reason given for the signal choice. Choose ONE:
   - "capability_based": Justifies signal based on actual military/resolve strength
   - "strategic_deception": Explicitly mentions misleading or bluffing the opponent
   - "risk_avoidance": Justifies signal to avoid conflict or reduce risk
   - "reputation_management": Mentions credibility, reputation, or future interactions
   - "uncertainty_exploitation": Exploits opponent's uncertainty about sender's type
   - "other": Does not fit above categories

2. belief_attribution: Did the sender explicitly model the opponent's beliefs? 
   - "yes_explicit": Explicitly mentions opponent's beliefs, probability, or inference
   - "yes_implicit": Implicitly considers opponent's perspective without explicit mention
   - "no": No consideration of opponent's beliefs

3. deception_awareness: Did the sender acknowledge bluffing or type concealment?
   - "explicit_bluff": Explicitly states it is bluffing, deceiving, or concealing type
   - "implicit_bluff": Suggests deception without explicitly naming it
   - "no_bluff": No deceptive intent expressed
   - "honest_signal": Explicitly states the signal reflects true type

RECEIVER CODING:
4. receiver_inference: How did the receiver interpret and update beliefs?
   - "bayesian_update": Explicitly updates probability/posterior based on signal
   - "heuristic_update": Updates beliefs using a rule of thumb without probability
   - "signal_ignored": Ignores the signal and acts on prior only
   - "signal_overweighted": Treats signal as definitive proof of type
   - "confused": Reasoning is contradictory or incoherent

Return ONLY a valid JSON object with these exact keys:
{
  "signal_justification": "<category>",
  "belief_attribution": "<category>",
  "deception_awareness": "<category>",
  "receiver_inference": "<category>",
  "sender_quote": "<most revealing 1-2 sentence quote from sender reasoning>",
  "receiver_quote": "<most revealing 1-2 sentence quote from receiver reasoning>"
}"""

def code_trace(sender_reasoning, receiver_reasoning, model_name, treatment, sender_type, signal):
    """Code a single reasoning trace using GPT-4.1-mini."""
    user_msg = f"""Code this reasoning trace from a signaling game experiment.

Context: Model={model_name}, Treatment={treatment}, Sender Type={sender_type}, Signal Sent={signal}

SENDER REASONING:
{sender_reasoning}

RECEIVER REASONING:
{receiver_reasoning}

Apply the S-R-O coding framework and return a JSON object."""

    try:
        resp = client.chat.completions.create(
            model="gpt-4.1-nano",  # Use nano for speed/cost in coding
            messages=[
                {"role": "system", "content": SRO_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg}
            ],
            temperature=0.1,
            max_tokens=400
        )
        content = resp.choices[0].message.content.strip()
        # Extract JSON
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        return None
    except Exception as e:
        print(f"Coding error: {e}")
        return None

# ── Run coding ─────────────────────────────────────────────────────────────────
print("\nRunning S-R-O qualitative coding...")
coded_results = []

for idx, row in tqdm(sample_df.iterrows(), total=len(sample_df), desc="Coding traces"):
    result = code_trace(
        sender_reasoning=str(row.get('sender_reasoning', '')),
        receiver_reasoning=str(row.get('receiver_reasoning', '')),
        model_name=str(row.get('model_name', '')),
        treatment=str(row.get('treatment', '')),
        sender_type=str(row.get('sender_type', '')),
        signal=str(row.get('signal', ''))
    )
    if result:
        result['model_key'] = row.get('model_key', '')
        result['model_name'] = row.get('model_name', '')
        result['treatment'] = row.get('treatment', '')
        result['sender_type'] = row.get('sender_type', '')
        result['signal'] = row.get('signal', '')
        result['is_bluff'] = row.get('is_bluff', False)
        coded_results.append(result)

print(f"\nSuccessfully coded {len(coded_results)} traces")

# Save coded results
coded_df = pd.DataFrame(coded_results)
coded_df.to_csv(os.path.join(OUT_DIR, "sro_coded_traces.csv"), index=False)
print(f"Saved to {OUT_DIR}/sro_coded_traces.csv")

# ── Analysis ───────────────────────────────────────────────────────────────────
print("\n=== S-R-O ANALYSIS RESULTS ===\n")

# 1. Signal justification distribution
print("1. Signal Justification Distribution:")
sj_counts = coded_df['signal_justification'].value_counts()
print(sj_counts.to_string())

# 2. Belief attribution by model
print("\n2. Belief Attribution by Model:")
ba_by_model = coded_df.groupby(['model_name', 'belief_attribution']).size().unstack(fill_value=0)
print(ba_by_model.to_string())

# 3. Deception awareness by treatment
print("\n3. Deception Awareness by Treatment:")
da_by_treatment = coded_df.groupby(['treatment', 'deception_awareness']).size().unstack(fill_value=0)
print(da_by_treatment.to_string())

# 4. Receiver inference distribution
print("\n4. Receiver Inference Distribution:")
ri_counts = coded_df['receiver_inference'].value_counts()
print(ri_counts.to_string())

# 5. Deception awareness for actual bluffs (Low Resolve + Escalate)
actual_bluffs = coded_df[coded_df['is_bluff'] == True]
print(f"\n5. Deception Awareness for Actual Bluffs (n={len(actual_bluffs)}):")
if len(actual_bluffs) > 0:
    print(actual_bluffs['deception_awareness'].value_counts().to_string())

# 6. Key quotes - most revealing examples
print("\n6. Selected Key Quotes:")
# Find explicit bluffs
explicit_bluffs = coded_df[coded_df['deception_awareness'] == 'explicit_bluff']
if len(explicit_bluffs) > 0:
    print("\nExplicit Bluff Examples:")
    for _, row in explicit_bluffs.head(3).iterrows():
        print(f"  [{row['model_name']}, {row['treatment']}]: {row.get('sender_quote', 'N/A')}")

# Find Bayesian updaters
bayesian = coded_df[coded_df['receiver_inference'] == 'bayesian_update']
if len(bayesian) > 0:
    print("\nBayesian Update Examples:")
    for _, row in bayesian.head(3).iterrows():
        print(f"  [{row['model_name']}, {row['treatment']}]: {row.get('receiver_quote', 'N/A')}")

# ── Generate Visualizations ────────────────────────────────────────────────────
print("\nGenerating S-R-O visualizations...")

# Color palette
COLORS = {
    'GPT-4.1-mini': '#2166AC',
    'GPT-4.1-nano': '#4DAC26',
    'Gemini-2.5-Flash': '#D01C8B',
    'zero_shot': '#4393C3',
    'role_conditioned': '#D6604D',
}

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.patch.set_facecolor('#FAFAFA')

# Define category orders and colors
sj_order = ['capability_based', 'strategic_deception', 'risk_avoidance',
            'reputation_management', 'uncertainty_exploitation', 'other']
sj_colors = ['#2166AC', '#D01C8B', '#4DAC26', '#F4A582', '#92C5DE', '#AAAAAA']

da_order = ['explicit_bluff', 'implicit_bluff', 'no_bluff', 'honest_signal']
da_colors = ['#D01C8B', '#F4A582', '#4DAC26', '#2166AC']

ri_order = ['bayesian_update', 'heuristic_update', 'signal_ignored',
            'signal_overweighted', 'confused']
ri_colors = ['#2166AC', '#4DAC26', '#F4A582', '#D01C8B', '#AAAAAA']

ba_order = ['yes_explicit', 'yes_implicit', 'no']
ba_colors = ['#2166AC', '#92C5DE', '#AAAAAA']

def safe_pct(counts, order):
    total = sum(counts.get(k, 0) for k in order)
    if total == 0:
        return [0] * len(order)
    return [100 * counts.get(k, 0) / total for k in order]

# ── Panel 1: Signal Justification by Model ─────────────────────────────────────
ax1 = axes[0, 0]
ax1.set_facecolor('#FAFAFA')
models = coded_df['model_name'].unique()
x = np.arange(len(sj_order))
width = 0.25
model_colors_list = ['#2166AC', '#4DAC26', '#D01C8B']

for i, model in enumerate(sorted(models)):
    model_data = coded_df[coded_df['model_name'] == model]
    counts = model_data['signal_justification'].value_counts().to_dict()
    vals = safe_pct(counts, sj_order)
    bars = ax1.bar(x + i * width, vals, width, label=model,
                   color=model_colors_list[i % 3], alpha=0.85, edgecolor='white', linewidth=0.5)

ax1.set_xticks(x + width)
ax1.set_xticklabels([s.replace('_', '\n') for s in sj_order], fontsize=8, fontfamily='DejaVu Serif')
ax1.set_ylabel('Percentage of Traces (%)', fontsize=10, fontfamily='DejaVu Serif')
ax1.set_title('A. Signal Justification by Model', fontsize=12, fontweight='bold',
              fontfamily='DejaVu Serif', pad=10)
ax1.legend(fontsize=8, framealpha=0.9)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.set_ylim(0, 100)
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0f}%'))

# ── Panel 2: Deception Awareness by Treatment ──────────────────────────────────
ax2 = axes[0, 1]
ax2.set_facecolor('#FAFAFA')
treatments = ['zero_shot', 'role_conditioned']
treat_labels = ['Zero-Shot', 'Role-Conditioned']
x2 = np.arange(len(da_order))

for i, (treat, label) in enumerate(zip(treatments, treat_labels)):
    treat_data = coded_df[coded_df['treatment'] == treat]
    counts = treat_data['deception_awareness'].value_counts().to_dict()
    vals = safe_pct(counts, da_order)
    color = '#4393C3' if treat == 'zero_shot' else '#D6604D'
    ax2.bar(x2 + i * 0.35, vals, 0.35, label=label, color=color,
            alpha=0.85, edgecolor='white', linewidth=0.5)

ax2.set_xticks(x2 + 0.175)
ax2.set_xticklabels([s.replace('_', '\n') for s in da_order], fontsize=9, fontfamily='DejaVu Serif')
ax2.set_ylabel('Percentage of Traces (%)', fontsize=10, fontfamily='DejaVu Serif')
ax2.set_title('B. Deception Awareness by Treatment', fontsize=12, fontweight='bold',
              fontfamily='DejaVu Serif', pad=10)
ax2.legend(fontsize=9, framealpha=0.9)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.set_ylim(0, 100)
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0f}%'))

# ── Panel 3: Receiver Inference Distribution ───────────────────────────────────
ax3 = axes[1, 0]
ax3.set_facecolor('#FAFAFA')
ri_counts_dict = coded_df['receiver_inference'].value_counts().to_dict()
ri_vals = [ri_counts_dict.get(k, 0) for k in ri_order]
ri_pcts = [100 * v / sum(ri_vals) if sum(ri_vals) > 0 else 0 for v in ri_vals]

bars3 = ax3.barh([r.replace('_', ' ').title() for r in ri_order], ri_pcts,
                  color=ri_colors, alpha=0.85, edgecolor='white', linewidth=0.5, height=0.6)

for bar, pct in zip(bars3, ri_pcts):
    if pct > 2:
        ax3.text(pct + 0.5, bar.get_y() + bar.get_height() / 2,
                 f'{pct:.1f}%', va='center', fontsize=9, fontfamily='DejaVu Serif')

ax3.set_xlabel('Percentage of Traces (%)', fontsize=10, fontfamily='DejaVu Serif')
ax3.set_title('C. Receiver Inference Strategy', fontsize=12, fontweight='bold',
              fontfamily='DejaVu Serif', pad=10)
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.set_xlim(0, max(ri_pcts) * 1.2 + 5)

# ── Panel 4: Belief Attribution Heatmap by Model x Treatment ──────────────────
ax4 = axes[1, 1]
ax4.set_facecolor('#FAFAFA')

# Build matrix: rows = models, cols = belief attribution categories
model_order = sorted(coded_df['model_name'].unique())
ba_matrix = np.zeros((len(model_order), len(ba_order)))

for i, model in enumerate(model_order):
    model_data = coded_df[coded_df['model_name'] == model]
    counts = model_data['belief_attribution'].value_counts().to_dict()
    total = sum(counts.values())
    for j, ba in enumerate(ba_order):
        ba_matrix[i, j] = 100 * counts.get(ba, 0) / total if total > 0 else 0

im = ax4.imshow(ba_matrix, cmap='Blues', aspect='auto', vmin=0, vmax=100)

# Add text annotations
for i in range(len(model_order)):
    for j in range(len(ba_order)):
        val = ba_matrix[i, j]
        color = 'white' if val > 60 else 'black'
        ax4.text(j, i, f'{val:.0f}%', ha='center', va='center',
                 fontsize=11, fontweight='bold', color=color, fontfamily='DejaVu Serif')

ax4.set_xticks(range(len(ba_order)))
ax4.set_xticklabels([b.replace('_', '\n') for b in ba_order], fontsize=9, fontfamily='DejaVu Serif')
ax4.set_yticks(range(len(model_order)))
short_names = [m.replace('GPT-4.1-', 'GPT-').replace('Gemini-2.5-', 'Gemini-') for m in model_order]
ax4.set_yticklabels(short_names, fontsize=9, fontfamily='DejaVu Serif')
ax4.set_title('D. Belief Attribution by Model\n(% of traces)', fontsize=12, fontweight='bold',
              fontfamily='DejaVu Serif', pad=10)

plt.colorbar(im, ax=ax4, label='% of traces', shrink=0.8)

plt.suptitle('S-R-O Qualitative Coding Analysis: Strategic Reasoning in LLM Signaling Games',
             fontsize=14, fontweight='bold', fontfamily='DejaVu Serif', y=1.01)

plt.tight_layout(pad=2.0)
fig_path = os.path.join(FIG_DIR, 'fig8_sro_qualitative_analysis.png')
plt.savefig(fig_path, dpi=300, bbox_inches='tight', facecolor='#FAFAFA')
plt.close()
print(f"Saved S-R-O figure to {fig_path}")

# ── Save summary statistics ────────────────────────────────────────────────────
summary = {
    "total_coded": len(coded_results),
    "signal_justification": coded_df['signal_justification'].value_counts().to_dict(),
    "belief_attribution": coded_df['belief_attribution'].value_counts().to_dict(),
    "deception_awareness": coded_df['deception_awareness'].value_counts().to_dict(),
    "receiver_inference": coded_df['receiver_inference'].value_counts().to_dict(),
    "deception_awareness_by_treatment": coded_df.groupby('treatment')['deception_awareness'].value_counts().to_dict(),
    "belief_attribution_by_model": coded_df.groupby('model_name')['belief_attribution'].value_counts().to_dict(),
}

with open(os.path.join(OUT_DIR, "sro_summary.json"), 'w') as f:
    json.dump(summary, f, indent=2, default=str)

print("\n=== CODING COMPLETE ===")
print(f"Coded traces: {len(coded_results)}")
print(f"Results saved to: {OUT_DIR}")
print(f"Figure saved to: {fig_path}")
