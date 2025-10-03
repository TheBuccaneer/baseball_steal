"""
A4b: Event Study - Pitch Tempo with Year Fixed Effects
Tests for pre-trends and quantifies 2023/2024 treatment effects

Uses regression with year dummies (ref: 2022) to show parallel trends

Input:  analysis/analysis_pitcher_panel_relative.csv
Output: analysis/tempo_event_study_coefficients.csv
        analysis/tempo_event_study_plot.png

References:
- Event Studies: https://mixtape.scunning.com/09-difference_in_differences
- 2023/2024 rules: https://www.mlb.com/news/mlb-2023-rule-changes-pitch-timer-larger-bases-shifts
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import sys

# Try importing statsmodels for regression
try:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    print("WARNING: statsmodels not installed. Install with: pip install statsmodels")


def main():
    print("=" * 70)
    print("A4b: EVENT STUDY - TEMPO WITH YEAR FIXED EFFECTS")
    print("=" * 70)
    
    if not HAS_STATSMODELS:
        print("\nERROR: This script requires statsmodels")
        print("Install with: pip install statsmodels")
        sys.exit(1)
    
    # Input
    input_file = Path("analysis/analysis_pitcher_panel_relative.csv")
    if not input_file.exists():
        print(f"\nERROR: Input file not found: {input_file}")
        sys.exit(1)
    
    # Load data
    print(f"\nLoading: {input_file}")
    df = pd.read_csv(input_file)
    print(f"Rows: {len(df):,}")
    
    # Filter: remove missing tempo
    df = df[df['tempo_with_runners_on_base'].notna()].copy()
    print(f"After removing missing tempo: {len(df):,}")
    
    # Optional: Filter outliers (tempo 10-25s)
    tempo_col = 'tempo_with_runners_on_base'
    n_before_outlier = len(df)
    df = df[(df[tempo_col] >= 10) & (df[tempo_col] <= 25)].copy()
    n_after_outlier = len(df)
    print(f"After removing outliers (10-25s): {len(df):,} (-{n_before_outlier - n_after_outlier})")
    
    # Create year dummies (reference: 2022)
    ref_year = 2022
    years = sorted(df['season'].unique())
    print(f"\nYears in data: {years}")
    print(f"Reference year: {ref_year}")
    
    for year in years:
        if year != ref_year:
            df[f'year_{year}'] = (df['season'] == year).astype(int)
    
    # Prepare regression formula
    year_dummies = [f'year_{y}' for y in years if y != ref_year]
    
    # Add controls if available
    controls = []
    
    # Check for available controls
    # IMPORTANT: Do NOT use violation_rate (endogenous) or pitches (used as weight)
    possible_controls = [
        'n_init',  # Opportunities (runners on base events)
    ]
    
    for ctrl in possible_controls:
        if ctrl in df.columns:
            # Check if not all missing
            if df[ctrl].notna().sum() > 0:
                controls.append(ctrl)
                print(f"  Using control: {ctrl}")
    
    print("\nNote: violation_rate excluded (endogenous)")
    print("Note: pitches excluded (used as weight)")
    
    # Build formula (join with +)
    rhs_parts = year_dummies.copy()
    if controls:
        rhs_parts.extend(controls)
    
    formula = f"{tempo_col} ~ {' + '.join(rhs_parts)}"
    
    print("\n" + "=" * 70)
    print("REGRESSION SPECIFICATION")
    print("=" * 70)
    print(f"Formula: {formula}")
    print(f"N observations: {len(df):,}")
    print(f"Reference year: {ref_year}")
    
    # Run regression with weights
    print("\nFitting regression...")
    
    if 'pitches_with_runners_on_base' in df.columns:
        weights = df['pitches_with_runners_on_base'].fillna(1)
        model = smf.wls(formula, data=df, weights=weights)
        print("Using WLS (weighted by pitch counts)")
    else:
        model = smf.ols(formula, data=df)
        print("Using OLS (unweighted)")
    
    # Fit with cluster-robust SE on pitcher_id
    if 'pitcher_id' in df.columns:
        print("Computing cluster-robust standard errors (pitcher_id)...")
        results = model.fit(cov_type='cluster', cov_kwds={'groups': df['pitcher_id']})
        print(f"  Clustered on {df['pitcher_id'].nunique():,} unique pitchers")
    else:
        print("WARNING: pitcher_id not found, using standard SE")
        results = model.fit()
    
    # Extract year coefficients
    print("\n" + "=" * 70)
    print("YEAR COEFFICIENTS (vs 2022)")
    print("=" * 70)
    
    year_coefs = []
    
    for year in years:
        if year == ref_year:
            # Reference year = 0 by definition
            year_coefs.append({
                'year': year,
                'coef': 0.0,
                'se': 0.0,
                'ci_lower': 0.0,
                'ci_upper': 0.0,
                'p_value': np.nan,
                'sig': 'ref'
            })
        else:
            var_name = f'year_{year}'
            if var_name in results.params.index:
                coef = results.params[var_name]
                se = results.bse[var_name]
                ci = results.conf_int().loc[var_name]
                p = results.pvalues[var_name]
                
                # Significance stars
                if p < 0.001:
                    sig = '***'
                elif p < 0.01:
                    sig = '**'
                elif p < 0.05:
                    sig = '*'
                else:
                    sig = ''
                
                year_coefs.append({
                    'year': year,
                    'coef': coef,
                    'se': se,
                    'ci_lower': ci[0],
                    'ci_upper': ci[1],
                    'p_value': p,
                    'sig': sig
                })
    
    coef_df = pd.DataFrame(year_coefs)
    
    # Display
    print("\nYear Effects:")
    display_df = coef_df.copy()
    display_df['coef_str'] = display_df.apply(
        lambda r: f"{r['coef']:+.2f}{r['sig']}" if not pd.isna(r['p_value']) else 'ref',
        axis=1
    )
    display_df['ci_str'] = display_df.apply(
        lambda r: f"[{r['ci_lower']:.2f}, {r['ci_upper']:.2f}]" if not pd.isna(r['p_value']) else '-',
        axis=1
    )
    
    print(display_df[['year', 'coef_str', 'ci_str']].to_string(index=False))
    
    # Test for pre-trends (2018-2022)
    print("\n" + "=" * 70)
    print("PRE-TREND TEST (2018-2021 vs 2022)")
    print("=" * 70)
    
    pre_trend_years = [y for y in years if y < ref_year and y >= 2018]
    pre_trend_coefs = coef_df[coef_df['year'].isin(pre_trend_years)]
    
    if len(pre_trend_coefs) > 0:
        # Simple test: are pre-2023 coefficients close to 0?
        max_pre_coef = pre_trend_coefs['coef'].abs().max()
        print(f"Maximum |coefficient| in pre-period: {max_pre_coef:.2f}s")
        
        # Count significant pre-trend coefficients
        n_sig_pre = (pre_trend_coefs['p_value'] < 0.05).sum()
        print(f"Significant pre-trend coefficients (p<0.05): {n_sig_pre}/{len(pre_trend_coefs)}")
        
        if n_sig_pre == 0:
            print("✓ No significant pre-trends detected")
        else:
            print("⚠️  Warning: Some pre-trends are significant")
            print("   → Parallel trends assumption may be violated")
    
    # Highlight 2023/2024 effects
    print("\n" + "=" * 70)
    print("TREATMENT EFFECTS (2023/2024)")
    print("=" * 70)
    
    for year in [2023, 2024]:
        row = coef_df[coef_df['year'] == year]
        if len(row) > 0:
            row = row.iloc[0]
            print(f"\n{year}:")
            print(f"  Effect: {row['coef']:+.2f}s {row['sig']}")
            print(f"  95% CI: [{row['ci_lower']:.2f}, {row['ci_upper']:.2f}]")
            print(f"  p-value: {row['p_value']:.4f}")
            
            if row['p_value'] < 0.05:
                if row['coef'] < 0:
                    print(f"  ✓ Significant DECREASE vs {ref_year}")
                else:
                    print(f"  ⚠️  Significant INCREASE vs {ref_year}")
    
    # Test 2023 vs 2024 difference
    coef_2023 = coef_df[coef_df['year'] == 2023]['coef'].values[0]
    coef_2024 = coef_df[coef_df['year'] == 2024]['coef'].values[0]
    diff = coef_2024 - coef_2023
    
    print(f"\n2024 vs 2023 difference: {diff:+.2f}s")
    if abs(diff) < 0.3:
        print("  → Minimal additional effect in 2024")
    
    # Save coefficients
    output_file = Path("analysis/tempo_event_study_coefficients.csv")
    coef_df.to_csv(output_file, index=False)
    print(f"\nCoefficients saved: {output_file}")
    
    # Create plot
    print("\n" + "=" * 70)
    print("CREATING EVENT STUDY PLOT")
    print("=" * 70)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot coefficients with error bars
    years_plot = coef_df['year'].values
    coefs_plot = coef_df['coef'].values
    ci_lower = coef_df['ci_lower'].values
    ci_upper = coef_df['ci_upper'].values
    
    # Error bars
    ax.errorbar(years_plot, coefs_plot, 
                yerr=[coefs_plot - ci_lower, ci_upper - coefs_plot],
                fmt='o', capsize=5, capthick=2, markersize=8,
                color='steelblue', ecolor='gray', label='Year Effects')
    
    # Horizontal line at 0
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    
    # Vertical line at 2023 (treatment)
    ax.axvline(x=2022.5, color='red', linestyle=':', linewidth=2, alpha=0.7, 
               label='2023 Rule Change')
    
    # Highlight 2023/2024
    for year in [2023, 2024]:
        idx = coef_df['year'] == year
        if idx.any():
            row = coef_df[idx].iloc[0]
            if row['p_value'] < 0.05:
                ax.plot(year, row['coef'], 'o', markersize=12, 
                       markerfacecolor='red', markeredgecolor='darkred', 
                       markeredgewidth=2)
    
    # Labels
    ax.set_xlabel('Season', fontsize=12, fontweight='bold')
    ax.set_ylabel('Effect on Tempo (seconds)', fontsize=12, fontweight='bold')
    ax.set_title('Event Study: Pitch Tempo with Runners On\n(Reference: 2022)', 
                fontsize=14, fontweight='bold', pad=20)
    
    ax.legend(loc='upper left', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    # Annotate 2023
    row_2023 = coef_df[coef_df['year'] == 2023].iloc[0]
    ax.annotate(f"{row_2023['coef']:.2f}s***", 
               xy=(2023, row_2023['coef']), 
               xytext=(2023, row_2023['coef'] - 1.5),
               ha='center', fontsize=10, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
               arrowprops=dict(arrowstyle='->', color='red', lw=2))
    
    plt.tight_layout()
    
    plot_file = Path("analysis/tempo_event_study_plot.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved: {plot_file}")
    
    # Full regression output
    print("\n" + "=" * 70)
    print("FULL REGRESSION OUTPUT")
    print("=" * 70)
    print(results.summary())
    
    # Save regression summary
    summary_file = Path("analysis/tempo_event_study_regression_summary.txt")
    with open(summary_file, 'w') as f:
        f.write(str(results.summary()))
    print(f"\nRegression summary saved: {summary_file}")
    
    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)
    print(f"\nOutputs:")
    print(f"  1. Coefficients: {output_file}")
    print(f"  2. Plot: {plot_file}")
    print(f"  3. Regression: {summary_file}")
    
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    print("Event study shows:")
    print("  1. Pre-trends: Check if 2018-2021 close to 0")
    print("  2. Treatment effect: 2023 coefficient magnitude")
    print("  3. Persistence: 2024 vs 2023 difference")
    print("\nFor causal inference:")
    print("  - Parallel trends assumption critical")
    print("  - Large 2023 effect supports rule change impact")
    print("  - 2024 near-zero difference suggests one-time adjustment")


if __name__ == "__main__":
    main()