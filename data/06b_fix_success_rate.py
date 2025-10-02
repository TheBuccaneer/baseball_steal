"""
Quick fix: Calculate success_rate for runner panel
Usage: python 06b_fix_success_rate.py
"""

import pandas as pd
from pathlib import Path

print("Loading runner panel...")
runner_panel = pd.read_csv("analysis/analysis_runner_panel.csv")

print(f"Before: {runner_panel['success_rate'].notna().sum()} non-null success_rate values")

# Calculate success_rate where attempts > 0
runner_panel['success_rate'] = runner_panel.apply(
    lambda row: row['sb'] / row['attempts'] if row['attempts'] > 0 else 0.0,
    axis=1
)

print(f"After: {runner_panel['success_rate'].notna().sum()} non-null success_rate values")
print(f"\nSuccess rate stats:")
print(runner_panel['success_rate'].describe())

# Save
runner_panel.to_csv("analysis/analysis_runner_panel.csv", index=False)
print("\nSaved updated file")