'''
Generates the prompt sensitivity analysis figure.
'''

import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

# This script assumes the necessary fonts (like Palatino) are installed on the system.
# If they are not, matplotlib will fall back to a default serif font.

# Load the sensitivity analysis summary data
data_path = '/home/ubuntu/bluffing_machine_repo/data/sensitivity/sensitivity_summary_20260306_104758.csv'
df = pd.read_csv(data_path)

# --- Plotting ---

# Set the plot style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Palatino', 'Garamond', 'Computer Modern'] # Fallback fonts

# Create the figure and axes
fig, ax = plt.subplots(figsize=(10, 6))

# Create the bar plot
barplot = sns.barplot(
    x='variant_label',
    y='bluff_rate',
    data=df,
    ax=ax,
    palette='viridis',
    width=0.6
)

# Set titles and labels
ax.set_title('Prompt Sensitivity Analysis: Bluffing Rate by Framing Condition', fontsize=16, pad=20, weight='bold')
ax.set_xlabel('Prompt Framing Condition', fontsize=12, labelpad=15)
ax.set_ylabel('Bluff Rate (Low Resolve Senders)', fontsize=12, labelpad=15)
ax.set_ylim(0, 1.0)
ax.set_xticklabels(ax.get_xticklabels(), rotation=0) # Ensure labels are not rotated

# Add value labels on top of the bars
for p in ax.patches:
    ax.annotate(f'{p.get_height():.1%}', 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='center', 
                xytext=(0, 10), 
                textcoords='offset points', 
                fontsize=12, 
                weight='bold')

# Add a horizontal line for the PBE benchmark for context
ax.axhline(y=0.42, color='red', linestyle='--', linewidth=1.5, label='PBE Benchmark (42%)')
ax.legend()

# Despine the axes for a cleaner look
sns.despine()

# Add a footer with data source
fig.text(0.5, -0.02, f'Data source: {os.path.basename(data_path)}\nN=100 simulations per variant.', ha='center', fontsize=8, style='italic', color='gray')

# Save the figure
output_path = '/home/ubuntu/bluffing_machine_repo/figures/premium_v2/fig7_sensitivity_analysis.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')

print(f"Sensitivity analysis figure generated successfully and saved to {output_path}")
