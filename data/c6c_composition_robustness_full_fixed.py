"""
c6_composition_robustness_full_fixed.py
=======================================
Fixed version with proper ID handling and diagnostics
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
STABILITY_THRESHOLD = 10.0

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def estimate_ppml(df, outcome_col, label, control_cols=None):
    """Estimate FE-PPML with optional controls"""
    
    df = df.copy()  # Work on copy
    
    years = sorted(df['season'].unique())
    years_non_base = [y for y in years if y != BASE_YEAR]
    
    # Year dummies
    for year in years_non_base:
        df[f'year_{year}'] = (df['season'] == year).astype(int)
    
    # Treatment interactions
    interaction_cols = []
    group_col = "baseline_group" if "baseline_group" in df.columns else "baseline_tercile_2022"
    for year in years_non_base:
        for tercile in ['T1', 'T3']:
            int_col = f'y{year}_x_{tercile}'
            df[int_col] = ((df['season'] == year) & (df[group_col] == tercile)).astype(int)
            interaction_cols.append(int_col)
    
    year_dummy_cols = [f'year_{y}' for y in years_non_base]
    all_regressors = year_dummy_cols + interaction_cols
    
    # Handle controls
    if control_cols:
        # Check if controls exist
        missing_cols = [c for c in control_cols if c not in df.columns]
        if missing_cols:
            print(f"  WARNING: Missing control columns: {missing_cols}")
            return None, None, 0
        
        # Drop rows with missing controls
        n_before = len(df)
        df = df.dropna(subset=control_cols)
        n_after = len(df)
        
        if n_after < n_before:
            print(f"  INFO: Dropped {n_before - n_after} rows due to missing controls ({n_after} remain)")
        
        if n_after == 0:
            print(f"  ERROR: No observations remain after dropping missing controls!")
            return None, None, 0
        
        all_regressors = all_regressors + control_cols
    
    # Check that all regressors exist
    missing_regs = [r for r in all_regressors if r not in df.columns]
    if missing_regs:
        print(f"  ERROR: Missing regressors: {missing_regs}")
        return None, None, 0
    
    # Pitcher FE
    pitcher_dummies = pd.get_dummies(df['pitcher_id'], prefix='pitcher', drop_first=True)
    
    # Build X
    X = pd.concat([df[all_regressors], pitcher_dummies], axis=1)
    X = X.astype(np.float64)
    X = sm.add_constant(X, has_constant='add')
    
    y = df[outcome_col].values
    exposure = df['opportunities_2b'].values
    
    # Estimate
    model = sm.GLM(endog=y, exog=X, family=Poisson(), exposure=exposure)
    result = model.fit(cov_type='cluster', cov_kwds={'groups': df['pitcher_id'].values})
    
    # Extract coefficients
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
print("C6 COMPOSITION ROBUSTNESS (FIXED VERSION)")
print("=" * 70)

print("\n" + "-" * 70)
print("LOADING PANEL")
print("-" * 70)

df = pd.read_csv(PANEL_FILE)
df_baseline = df[df['in_baseline_2022'] == 1].copy()

if EXCLUDE_2020:
    df_baseline = df_baseline[df_baseline['season'] != 2020].copy()

df_baseline = df_baseline[df_baseline['opportunities_2b'] > 0].copy()

print(f"Panel: {len(df_baseline):,} pitcher-season observations")
print(f"  Pitchers: {df_baseline['pitcher_id'].nunique()}")
print(f"  Years: {sorted(df_baseline['season'].unique())}")

# Ensure pitcher_id is int
df_baseline['pitcher_id'] = df_baseline['pitcher_id'].astype(int)

# ============================================================================
# LOAD AND MERGE COVARIATES
# ============================================================================

print("\n" + "-" * 70)
print("LOADING COVARIATES")
print("-" * 70)

# Opportunity-weighted
df_cov_opp = pd.read_csv(COVARIATES_OPP)
df_cov_opp['pitcher_id'] = df_cov_opp['pitcher_id'].astype(int)
print(f"Opportunity-weighted: {len(df_cov_opp):,} pitchers")

# Event-weighted
try:
    df_cov_event = pd.read_csv(COVARIATES_EVENT)
    df_cov_event['pitcher_id'] = df_cov_event['pitcher_id'].astype(int)
    print(f"Event-weighted: {len(df_cov_event):,} pitchers")
except FileNotFoundError:
    print(f"WARNING: Event-weighted file not found")
    df_cov_event = None

print("\n" + "-" * 70)
print("MERGING COVARIATES TO PANEL")
print("-" * 70)

# Merge opportunity-weighted
n_before = len(df_baseline)
df_baseline = df_baseline.merge(
    df_cov_opp[['pitcher_id', 'avg_speed_faced_2022_opp', 'avg_poptime_behind_2022_opp']],
    on='pitcher_id',
    how='left'
)
print(f"Merged opportunity-weighted: {len(df_baseline):,} rows (expected {n_before:,})")

# Merge event-weighted
if df_cov_event is not None:
    df_baseline = df_baseline.merge(
        df_cov_event[['pitcher_id', 'avg_speed_faced_2022', 'avg_poptime_behind_2022']],
        on='pitcher_id',
        how='left'
    )
    print(f"Merged event-weighted: {len(df_baseline):,} rows")

# ============================================================================
# DIAGNOSTIC CHECK
# ============================================================================

print("\n" + "-" * 70)
print("COVARIATE DIAGNOSTIC")
print("-" * 70)

print(f"\nOpportunity-weighted controls:")
opp_speed_valid = df_baseline['avg_speed_faced_2022_opp'].notna().sum()
opp_pop_valid = df_baseline['avg_poptime_behind_2022_opp'].notna().sum()
print(f"  avg_speed_faced_2022_opp: {opp_speed_valid:,} / {len(df_baseline):,} ({opp_speed_valid/len(df_baseline)*100:.1f}%)")
print(f"  avg_poptime_behind_2022_opp: {opp_pop_valid:,} / {len(df_baseline):,} ({opp_pop_valid/len(df_baseline)*100:.1f}%)")

if opp_speed_valid > 0:
    print(f"\n  Speed stats:")
    print(df_baseline['avg_speed_faced_2022_opp'].describe())
    print(f"\n  Pop time stats:")
    print(df_baseline['avg_poptime_behind_2022_opp'].describe())
else:
    print(f"\n  ERROR: No valid opportunity-weighted covariates found!")
    print(f"  Check pitcher_id matching between panel and covariates file")
    sys.exit(1)

if df_cov_event is not None:
    print(f"\nEvent-weighted controls:")
    event_speed_valid = df_baseline['avg_speed_faced_2022'].notna().sum()
    event_pop_valid = df_baseline['avg_poptime_behind_2022'].notna().sum()
    print(f"  avg_speed_faced_2022: {event_speed_valid:,} / {len(df_baseline):,} ({event_speed_valid/len(df_baseline)*100:.1f}%)")
    print(f"  avg_poptime_behind_2022: {event_pop_valid:,} / {len(df_baseline):,} ({event_pop_valid/len(df_baseline)*100:.1f}%)")

# ============================================================================
# SPECIFICATION A: BASELINE
# ============================================================================

print("\n" + "=" * 70)
print("SPECIFICATION A: BASELINE (NO CONTROLS)")
print("=" * 70)

results_a = {}

for outcome_name, outcome_col in [('C3_Attempts', 'attempts_2b'), ('C5_Total', 'sb_2b')]:
    print(f"\nEstimating {outcome_name}...")
    coefs, result, n_obs = estimate_ppml(df_baseline.copy(), outcome_col, f'{outcome_name}_baseline')
    if coefs is not None:
        results_a[outcome_name] = coefs
    else:
        print(f"  ERROR: Estimation failed!")
        sys.exit(1)

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
# SPECIFICATION B: EVENT-WEIGHTED
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
        if coefs is not None:
            results_b[outcome_name] = coefs
        else:
            print(f"  WARNING: Estimation failed for event-weighted!")
            results_b = None
            break
    
    if results_b:
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
# SPECIFICATION C: OPPORTUNITY-WEIGHTED (PRIMARY)
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
    if coefs is not None:
        results_c[outcome_name] = coefs
    else:
        print(f"  ERROR: Estimation failed!")
        sys.exit(1)

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
print("STABILITY TEST")
print("=" * 70)

comparison = []

for model_name in ['C3_Attempts', 'C5_Total']:
    for year in [2023, 2024]:
        
        baseline_rr = results_a[model_name][year]['rr']
        
        if results_b:
            event_rr = results_b[model_name][year]['rr']
            event_diff = ((event_rr - baseline_rr) / baseline_rr) * 100
            event_stable = abs(event_diff) < STABILITY_THRESHOLD
        else:
            event_rr = np.nan
            event_diff = np.nan
            event_stable = None
        
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

opp_all_stable = df_comparison['opp_stable'].all()
event_all_stable = df_comparison['event_stable'].all() if results_b else None

print("\n" + "-" * 70)
print("INTERPRETATION")
print("-" * 70)

print(f"\nPrimary specification (Opportunity-weighted):")
if opp_all_stable:
    print(f"  ✓ All treatment effects stable (<{STABILITY_THRESHOLD}% change)")
    print(f"  → Composition changes do NOT drive the observed effects")
else:
    print(f"  ✗ Some effects change >{STABILITY_THRESHOLD}%")
    print(f"  → Composition may partially explain treatment effects")

max_opp_change = df_comparison['opp_pct_change'].abs().max()
print(f"\nMaximum change (Opportunity-weighted): {max_opp_change:.2f}%")

# Save outputs
output_comp = OUTPUT_DIR / "c6_full_comparison.csv"
df_comparison.to_csv(output_comp, index=False)

output_summary = OUTPUT_DIR / "c6_full_summary.txt"
with open(output_summary, 'w', encoding='utf-8') as f:
    f.write("C6 COMPOSITION ROBUSTNESS (FIXED VERSION)\n")
    f.write("=" * 70 + "\n\n")
    f.write(df_comparison.to_string(index=False))
    f.write("\n\n")
    if opp_all_stable:
        f.write("✓ Treatment effects robust to opportunity-weighted controls\n")
    else:
        f.write("⚠ Some sensitivity detected\n")

print(f"\n\nOutputs saved to {OUTPUT_DIR}/")
print("=" * 70)