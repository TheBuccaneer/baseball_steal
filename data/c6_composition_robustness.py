"""
c6_composition_robustness_full.py
=================================
Complete composition robustness check with THREE specifications

Purpose:
Test whether treatment effects are driven by compositional changes using:
  (A) Baseline: No composition controls
  (B) Event-weighted: Controls from realized steal attempts (selection bias concern)
  (C) Opportunity-weighted: Controls from all opportunities (unbiased exposure)

Primary specification: (C) Opportunity-weighted
Robustness check: (B) Event-weighted shown in appendix

Acceptance criterion: Treatment effects stable within ±10% across all specs
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import statsmodels.api as sm
from statsmodels.genmod.families import Poisson
import sys

# ============================================================================
# CONFIGURATION
# ============================================================================

PANEL_FILE = Path("analysis/c_running_game/c_panel_with_baseline.csv")
COVARIATES_OPP = Path("analysis/intermediate/pitcher_covariates_2022_opportunities.csv")
COVARIATES_EVENT = Path("analysis/intermediate/pitcher_covariates_2022.csv")

OUTPUT_DIR = Path("analysis/c6_composition")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BASE_YEAR = 2022
EXCLUDE_2020 = True
STABILITY_THRESHOLD = 10.0  # % change threshold

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def estimate_ppml(df, outcome_col, label, control_cols=None):
    """
    Estimate FE-PPML model with optional composition controls
    
    Returns: dict with coefficients for 2023/2024, result object, sample size
    """
    
    years = sorted(df['season'].unique())
    years_non_base = [y for y in years if y != BASE_YEAR]
    
    # Create year dummies
    for year in years_non_base:
        df[f'year_{year}'] = (df['season'] == year).astype(int)
    
    # Create treatment interactions (T1/T3)
    interaction_cols = []
    group_col = "baseline_group" if "baseline_group" in df.columns else "baseline_tercile_2022"
    for year in years_non_base:
        for tercile in ['T1', 'T3']:
            int_col = f'y{year}_x_{tercile}'
            df[int_col] = (
                (df['season'] == year) & 
                (df[group_col] == tercile)
            ).astype(int)
            interaction_cols.append(int_col)
    
    year_dummy_cols = [f'year_{y}' for y in years_non_base]
    
    # Add controls if provided
    all_regressors = year_dummy_cols + interaction_cols
    if control_cols:
        # Drop rows with missing controls (no imputation)
        df = df.dropna(subset=control_cols)
        all_regressors = all_regressors + control_cols
    
    # Pitcher FE
    pitcher_dummies = pd.get_dummies(df['pitcher_id'], prefix='pitcher', drop_first=True)
    
    X = pd.concat([df[all_regressors], pitcher_dummies], axis=1)
    X = X.astype(np.float64)
    X = sm.add_constant(X, has_constant='add')
    
    y = df[outcome_col].values
    exposure = df['opportunities_2b'].values
    
    model = sm.GLM(endog=y, exog=X, family=Poisson(), exposure=exposure)
    result = model.fit(cov_type='cluster', cov_kwds={'groups': df['pitcher_id'].values})
    
    # Extract T2 (reference group) coefficients
    coefs = {}
    for year in [2023, 2024]:
        year_col = f'year_{year}'
        coefs[year] = {
            'coef': result.params.get(year_col, np.nan),
            'se': result.bse.get(year_col, np.nan),
            'pval': result.pvalues.get(year_col, np.nan),
            'rr': np.exp(result.params.get(year_col, 0.0)),
            'n_obs': len(df)
        }
    
    return coefs, result, len(df)

# ============================================================================
# LOAD DATA
# ============================================================================

print("=" * 70)
print("C6 COMPOSITION ROBUSTNESS (THREE SPECIFICATIONS)")
print("=" * 70)

print("\n" + "-" * 70)
print("LOADING DATA")
print("-" * 70)

# Load panel
df = pd.read_csv(PANEL_FILE)
df_baseline = df[df['in_baseline_2022'] == 1].copy()

if EXCLUDE_2020:
    df_baseline = df_baseline[df_baseline['season'] != 2020].copy()

df_baseline = df_baseline[df_baseline['opportunities_2b'] > 0].copy()

print(f"Panel: {len(df_baseline):,} pitcher-season observations")
print(f"  Pitchers: {df_baseline['pitcher_id'].nunique()}")
print(f"  Years: {sorted(df_baseline['season'].unique())}")

# Load covariates - Opportunity-weighted
try:
    df_cov_opp = pd.read_csv(COVARIATES_OPP)
    print(f"\nOpportunity-weighted covariates: {len(df_cov_opp):,} pitchers")
except FileNotFoundError:
    print(f"\nERROR: Opportunity-weighted covariates not found at {COVARIATES_OPP}")
    sys.exit(1)

# Load covariates - Event-weighted
try:
    df_cov_event = pd.read_csv(COVARIATES_EVENT)
    print(f"Event-weighted covariates: {len(df_cov_event):,} pitchers")
except FileNotFoundError:
    print(f"\nWARNING: Event-weighted covariates not found at {COVARIATES_EVENT}")
    df_cov_event = None

# Merge covariates to panel
df_baseline = df_baseline.merge(
    df_cov_opp[['pitcher_id', 'avg_speed_faced_2022_opp', 'avg_poptime_behind_2022_opp']],
    on='pitcher_id',
    how='left'
)

if df_cov_event is not None:
    df_baseline = df_baseline.merge(
        df_cov_event[['pitcher_id', 'avg_speed_faced_2022', 'avg_poptime_behind_2022']],
        on='pitcher_id',
        how='left'
    )

# Check coverage
print(f"\nCovariate coverage in panel:")
print(f"  Opportunity-weighted speed: {df_baseline['avg_speed_faced_2022_opp'].notna().sum():,} ({df_baseline['avg_speed_faced_2022_opp'].notna().mean()*100:.1f}%)")
print(f"  Opportunity-weighted pop time: {df_baseline['avg_poptime_behind_2022_opp'].notna().sum():,} ({df_baseline['avg_poptime_behind_2022_opp'].notna().mean()*100:.1f}%)")

if df_cov_event is not None:
    print(f"  Event-weighted speed: {df_baseline['avg_speed_faced_2022'].notna().sum():,} ({df_baseline['avg_speed_faced_2022'].notna().mean()*100:.1f}%)")
    print(f"  Event-weighted pop time: {df_baseline['avg_poptime_behind_2022'].notna().sum():,} ({df_baseline['avg_poptime_behind_2022'].notna().mean()*100:.1f}%)")

# ============================================================================
# SPECIFICATION A: BASELINE (NO CONTROLS)
# ============================================================================

print("\n" + "=" * 70)
print("SPECIFICATION A: BASELINE (NO COMPOSITION CONTROLS)")
print("=" * 70)

results_a = {}

for outcome_name, outcome_col in [('C3_Attempts', 'attempts_2b'), ('C5_Total', 'sb_2b')]:
    print(f"\nEstimating {outcome_name}...")
    coefs, result, n_obs = estimate_ppml(df_baseline.copy(), outcome_col, f'{outcome_name}_baseline')
    results_a[outcome_name] = coefs

print("\n" + "-" * 70)
print("RESULTS (SPEC A)")
print("-" * 70)
print(f"{'Model':<15} {'Year':<6} {'RR':<8} {'Coef':<10} {'SE':<8} {'p-value':<10} {'N':<8}")
print("-" * 70)

for model in ['C3_Attempts', 'C5_Total']:
    for year in [2023, 2024]:
        res = results_a[model][year]
        print(f"{model:<15} {year:<6} {res['rr']:.4f}   {res['coef']:+.4f}    {res['se']:.4f}  {res['pval']:.4f}   {res['n_obs']:,}")

# ============================================================================
# SPECIFICATION B: EVENT-WEIGHTED CONTROLS
# ============================================================================

print("\n" + "=" * 70)
print("SPECIFICATION B: EVENT-WEIGHTED CONTROLS")
print("=" * 70)

if df_cov_event is not None:
    control_cols_event = ['avg_speed_faced_2022', 'avg_poptime_behind_2022']
    
    results_b = {}
    
    for outcome_name, outcome_col in [('C3_Attempts', 'attempts_2b'), ('C5_Total', 'sb_2b')]:
        print(f"\nEstimating {outcome_name} with event-weighted controls...")
        coefs, result, n_obs = estimate_ppml(df_baseline.copy(), outcome_col, 
                                             f'{outcome_name}_event', control_cols_event)
        results_b[outcome_name] = coefs
    
    print("\n" + "-" * 70)
    print("RESULTS (SPEC B)")
    print("-" * 70)
    print(f"{'Model':<15} {'Year':<6} {'RR':<8} {'Coef':<10} {'SE':<8} {'p-value':<10} {'N':<8}")
    print("-" * 70)
    
    for model in ['C3_Attempts', 'C5_Total']:
        for year in [2023, 2024]:
            res = results_b[model][year]
            print(f"{model:<15} {year:<6} {res['rr']:.4f}   {res['coef']:+.4f}    {res['se']:.4f}  {res['pval']:.4f}   {res['n_obs']:,}")
else:
    print("\nSkipping Spec B - event-weighted covariates not available")
    results_b = None

# ============================================================================
# SPECIFICATION C: OPPORTUNITY-WEIGHTED CONTROLS (PRIMARY)
# ============================================================================

print("\n" + "=" * 70)
print("SPECIFICATION C: OPPORTUNITY-WEIGHTED CONTROLS (PRIMARY)")
print("=" * 70)

control_cols_opp = ['avg_speed_faced_2022_opp', 'avg_poptime_behind_2022_opp']

results_c = {}

for outcome_name, outcome_col in [('C3_Attempts', 'attempts_2b'), ('C5_Total', 'sb_2b')]:
    print(f"\nEstimating {outcome_name} with opportunity-weighted controls...")
    coefs, result, n_obs = estimate_ppml(df_baseline.copy(), outcome_col, 
                                         f'{outcome_name}_opp', control_cols_opp)
    results_c[outcome_name] = coefs

print("\n" + "-" * 70)
print("RESULTS (SPEC C - PRIMARY)")
print("-" * 70)
print(f"{'Model':<15} {'Year':<6} {'RR':<8} {'Coef':<10} {'SE':<8} {'p-value':<10} {'N':<8}")
print("-" * 70)

for model in ['C3_Attempts', 'C5_Total']:
    for year in [2023, 2024]:
        res = results_c[model][year]
        print(f"{model:<15} {year:<6} {res['rr']:.4f}   {res['coef']:+.4f}    {res['se']:.4f}  {res['pval']:.4f}   {res['n_obs']:,}")

# ============================================================================
# STABILITY COMPARISON
# ============================================================================

print("\n" + "=" * 70)
print("STABILITY TEST: COMPARING ALL SPECIFICATIONS")
print("=" * 70)

comparison = []

for model_name in ['C3_Attempts', 'C5_Total']:
    for year in [2023, 2024]:
        
        # Baseline
        baseline_rr = results_a[model_name][year]['rr']
        baseline_coef = results_a[model_name][year]['coef']
        
        # Event-weighted
        if results_b:
            event_rr = results_b[model_name][year]['rr']
            event_diff = ((event_rr - baseline_rr) / baseline_rr) * 100
            event_stable = abs(event_diff) < STABILITY_THRESHOLD
        else:
            event_rr = np.nan
            event_diff = np.nan
            event_stable = None
        
        # Opportunity-weighted
        opp_rr = results_c[model_name][year]['rr']
        opp_diff = ((opp_rr - baseline_rr) / baseline_rr) * 100
        opp_stable = abs(opp_diff) < STABILITY_THRESHOLD
        
        comparison.append({
            'model': model_name,
            'year': year,
            'baseline_rr': baseline_rr,
            'event_rr': event_rr,
            'opp_rr': opp_rr,
            'event_pct_change': event_diff,
            'opp_pct_change': opp_diff,
            'event_stable': event_stable,
            'opp_stable': opp_stable
        })

df_comparison = pd.DataFrame(comparison)

print(f"\nStability threshold: ±{STABILITY_THRESHOLD}%\n")
print(f"{'Model':<15} {'Year':<6} {'Baseline':<10} {'Event':<10} {'Event Δ%':<10} {'Opp':<10} {'Opp Δ%':<10} {'Status':<15}")
print("-" * 100)

for _, row in df_comparison.iterrows():
    event_status = "✓" if row['event_stable'] else ("✗" if row['event_stable'] == False else "N/A")
    opp_status = "✓" if row['opp_stable'] else "✗"
    
    event_rr_str = f"{row['event_rr']:.4f}" if not pd.isna(row['event_rr']) else "N/A"
    event_pct_str = f"{row['event_pct_change']:+.2f}%" if not pd.isna(row['event_pct_change']) else "N/A"
    
    print(f"{row['model']:<15} {row['year']:<6} {row['baseline_rr']:.4f}    {event_rr_str:<10} {event_pct_str:<10} "
          f"{row['opp_rr']:.4f}    {row['opp_pct_change']:+.2f}%     {event_status} Event / {opp_status} Opp")

# Overall assessment
opp_all_stable = df_comparison['opp_stable'].all()
event_all_stable = df_comparison['event_stable'].all() if results_b else None

print("\n" + "-" * 70)
print("INTERPRETATION")
print("-" * 70)

print(f"\nPrimary specification (Opportunity-weighted):")
if opp_all_stable:
    print(f"  ✓ All treatment effects stable (<{STABILITY_THRESHOLD}% change)")
    print(f"  → Composition changes do NOT drive the observed effects")
    print(f"  → Results reflect behavioral responses to rule changes")
else:
    print(f"  ✗ Some effects change >{STABILITY_THRESHOLD}% after adding controls")
    print(f"  → Composition may partially explain treatment effects")

if results_b:
    print(f"\nRobustness check (Event-weighted):")
    if event_all_stable:
        print(f"  ✓ Consistent with primary specification")
    else:
        print(f"  ✗ Some sensitivity detected (expected due to selection bias)")
        print(f"  → Reinforces need for opportunity-weighted primary spec")

# Maximum change
max_opp_change = df_comparison['opp_pct_change'].abs().max()
print(f"\nMaximum change (Opportunity-weighted): {max_opp_change:.2f}%")

# ============================================================================
# SAVE OUTPUTS
# ============================================================================

print("\n" + "-" * 70)
print("SAVING OUTPUTS")
print("-" * 70)

# Comparison table
output_comp = OUTPUT_DIR / "c6_full_comparison.csv"
df_comparison.to_csv(output_comp, index=False)
print(f"\nSaved: {output_comp}")

# Summary report
output_summary = OUTPUT_DIR / "c6_full_summary.txt"
with open(output_summary, 'w', encoding='utf-8') as f:
    f.write("C6 COMPOSITION ROBUSTNESS (THREE SPECIFICATIONS)\n")
    f.write("=" * 70 + "\n\n")
    
    f.write("Purpose: Test whether treatment effects are driven by composition\n")
    f.write("Primary specification: Opportunity-weighted controls\n")
    f.write(f"Stability threshold: ±{STABILITY_THRESHOLD}%\n\n")
    
    f.write("Specifications:\n")
    f.write("  (A) Baseline: No composition controls\n")
    f.write("  (B) Event-weighted: Controls from realized steal attempts\n")
    f.write("  (C) Opportunity-weighted: Controls from all opportunities (PRIMARY)\n\n")
    
    f.write("-" * 70 + "\n")
    f.write("COMPARISON TABLE\n")
    f.write("-" * 70 + "\n\n")
    f.write(df_comparison.to_string(index=False))
    f.write("\n\n")
    
    f.write("-" * 70 + "\n")
    f.write("CONCLUSION\n")
    f.write("-" * 70 + "\n\n")
    
    if opp_all_stable:
        f.write("✓ PRIMARY SPECIFICATION: Treatment effects robust to opportunity-weighted controls\n")
        f.write(f"  Maximum change: {max_opp_change:.2f}% (threshold: {STABILITY_THRESHOLD}%)\n")
        f.write("  → Results reflect behavioral responses, not compositional shifts\n\n")
    else:
        f.write("✗ Some sensitivity detected in opportunity-weighted controls\n\n")
    
    if results_b:
        if event_all_stable:
            f.write("✓ ROBUSTNESS: Event-weighted controls also show stability\n")
        else:
            f.write("⚠ ROBUSTNESS: Event-weighted shows some sensitivity (expected)\n")
            f.write("  → Selection bias in event-weighting may affect results\n")

print(f"Saved: {output_summary}")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

for idx, model_name in enumerate(['C3_Attempts', 'C5_Total']):
    df_model = df_comparison[df_comparison['model'] == model_name]
    
    # Plot 1: Rate Ratios
    ax1 = axes[idx, 0]
    x = np.arange(len(df_model))
    width = 0.25
    
    ax1.bar(x - width, df_model['baseline_rr'] - 1, width, 
           label='Baseline', alpha=0.8, color='steelblue')
    if results_b:
        ax1.bar(x, df_model['event_rr'] - 1, width, 
               label='Event-weighted', alpha=0.8, color='orange')
    ax1.bar(x + width, df_model['opp_rr'] - 1, width, 
           label='Opportunity-weighted', alpha=0.8, color='green')
    
    ax1.axhline(0, color='black', linestyle='--', linewidth=0.8)
    ax1.set_xlabel('Year', fontsize=11)
    ax1.set_ylabel('Rate Ratio - 1 (% Change)', fontsize=11)
    ax1.set_title(f'{model_name}: Rate Ratios', fontweight='bold', fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(df_model['year'])
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Plot 2: Percent Changes
    ax2 = axes[idx, 1]
    
    if results_b:
        ax2.bar(x - width/2, df_model['event_pct_change'], width, 
               label='Event vs Baseline', alpha=0.8, color='orange')
    ax2.bar(x + width/2, df_model['opp_pct_change'], width, 
           label='Opp vs Baseline', alpha=0.8, color='green')
    
    ax2.axhline(0, color='black', linestyle='-', linewidth=0.8)
    ax2.axhline(STABILITY_THRESHOLD, color='red', linestyle='--', linewidth=0.8, label=f'±{STABILITY_THRESHOLD}% threshold')
    ax2.axhline(-STABILITY_THRESHOLD, color='red', linestyle='--', linewidth=0.8)
    
    ax2.set_xlabel('Year', fontsize=11)
    ax2.set_ylabel('% Change from Baseline', fontsize=11)
    ax2.set_title(f'{model_name}: Stability Test', fontweight='bold', fontsize=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(df_model['year'])
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
output_plot = OUTPUT_DIR / "c6_full_comparison.png"
plt.savefig(output_plot, dpi=300, bbox_inches='tight')
print(f"Saved: {output_plot}")
plt.close()

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "=" * 70)
print("C6 COMPOSITION ROBUSTNESS COMPLETE")
print("=" * 70)

print(f"\nSpecs tested: 3 (Baseline, Event-weighted, Opportunity-weighted)")
print(f"Stability threshold: ±{STABILITY_THRESHOLD}%")

if opp_all_stable:
    print(f"\n✓ PRIMARY RESULT: Treatment effects are STABLE (opportunity-weighted)")
    print(f"  → Composition does not explain the rule change effects")
else:
    print(f"\n⚠ PRIMARY RESULT: Some SENSITIVITY detected (opportunity-weighted)")
    print(f"  → Composition may partially contribute")

if results_b:
    if event_all_stable:
        print(f"✓ ROBUSTNESS: Event-weighted also stable")
    else:
        print(f"⚠ ROBUSTNESS: Event-weighted shows sensitivity (expected)")

print(f"\nOutputs:")
print(f"  1. c6_full_comparison.csv")
print(f"  2. c6_full_summary.txt")
print(f"  3. c6_full_comparison.png")

print(f"\nReporting guidance:")
print(f"  - Main text: Use Spec C (Opportunity-weighted) as primary")
print(f"  - Appendix: Show Spec B (Event-weighted) for robustness")
print(f"  - Note: Sprint speed correlation 0.475 indicates selection bias in events")

print("\n" + "=" * 70)