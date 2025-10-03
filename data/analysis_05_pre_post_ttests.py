"""
A5: Pre/Post t-Tests (PITCHER-LEVEL) - Test for changes after 2023 rules
Uses individual pitcher-seasons as units for high statistical power

Compares Pre (2018-2022) vs Post (2023-2025) for:
- Pitch Tempo (with runners on)

Input:  analysis/analysis_pitcher_panel_relative.csv
Output: analysis/pre_post_ttests_pitcher_level.csv

References:
- 2023/2024 rules: https://www.mlb.com/news/mlb-2023-rule-changes-pitch-timer-larger-bases-shifts
- Welch's t-test: https://en.wikipedia.org/wiki/Welch%27s_t-test
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import sys


def welch_ttest_summary(pre_data, post_data, metric_name):
    """
    Perform Welch's t-test and return comprehensive summary
    """
    # Remove NaNs
    pre_clean = pre_data[~np.isnan(pre_data)]
    post_clean = post_data[~np.isnan(post_data)]
    
    # Calculate statistics
    mean_pre = np.mean(pre_clean)
    mean_post = np.mean(post_clean)
    std_pre = np.std(pre_clean, ddof=1)
    std_post = np.std(post_clean, ddof=1)
    delta = mean_post - mean_pre
    pct_change = (delta / mean_pre) * 100
    
    # Welch's t-test (unequal variances)
    t_stat, p_value = stats.ttest_ind(post_clean, pre_clean, equal_var=False)
    
    # Sample sizes
    n_pre = len(pre_clean)
    n_post = len(post_clean)
    
    # Degrees of freedom (Welch-Satterthwaite)
    df = (std_pre**2/n_pre + std_post**2/n_post)**2 / \
         ((std_pre**2/n_pre)**2/(n_pre-1) + (std_post**2/n_post)**2/(n_post-1))
    
    # 95% Confidence Interval for difference
    se = np.sqrt(std_pre**2/n_pre + std_post**2/n_post)
    t_crit = stats.t.ppf(0.975, df)
    ci_lower = delta - t_crit * se
    ci_upper = delta + t_crit * se
    
    # Effect size (Cohen's d - pooled)
    pooled_std = np.sqrt(((n_pre-1)*std_pre**2 + (n_post-1)*std_post**2) / (n_pre + n_post - 2))
    cohens_d = delta / pooled_std
    
    # Significance stars
    if p_value < 0.001:
        sig = '***'
    elif p_value < 0.01:
        sig = '**'
    elif p_value < 0.05:
        sig = '*'
    else:
        sig = ''
    
    return {
        'metric': metric_name,
        'mean_pre': mean_pre,
        'mean_post': mean_post,
        'std_pre': std_pre,
        'std_post': std_post,
        'delta': delta,
        'delta_pct': pct_change,
        't_stat': t_stat,
        'df': df,
        'p_value': p_value,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'n_pre': n_pre,
        'n_post': n_post,
        'cohens_d': cohens_d,
        'sig': sig
    }


def main():
    print("=" * 70)
    print("A5: PRE/POST T-TESTS (PITCHER-LEVEL)")
    print("=" * 70)
    
    # Input
    input_file = Path("analysis/analysis_pitcher_panel_relative.csv")
    if not input_file.exists():
        print(f"\nERROR: Input file not found: {input_file}")
        sys.exit(1)
    
    # Load data
    print(f"\nLoading: {input_file}")
    df = pd.read_csv(input_file)
    print(f"Rows: {len(df):,}")
    print(f"Seasons: {df['season'].min()} - {df['season'].max()}")
    
    # Define periods
    print("\n" + "=" * 70)
    print("PERIOD DEFINITIONS")
    print("=" * 70)
    print("Pre-period:  2018-2022 (before rule changes)")
    print("Post-period: 2023-2025 (after rule changes)")
    print("\n2023 rule changes:")
    print("  - Pitch timer: 15s (empty), 20s (runners)")
    print("  - Larger bases: 18 inches (from 15)")
    print("  - Disengagement limit: 2 max, 3rd = balk")
    print("\n2024 adjustment:")
    print("  - Pitch timer: 18s with runners (down from 20s)")
    
    # Create period indicator
    df['period'] = df['season'].apply(lambda x: 'Pre' if x < 2023 else 'Post')
    
    print(f"\nPre period:  {(df['period'] == 'Pre').sum():,} pitcher-seasons")
    print(f"Post period: {(df['period'] == 'Post').sum():,} pitcher-seasons")
    
    # Metrics to test
    metrics = {
        'tempo_with_runners_on_base': 'Pitch Tempo (Runners On)',
    }
    
    # Run tests
    print("\n" + "=" * 70)
    print("RUNNING WELCH'S T-TESTS")
    print("=" * 70)
    
    results = []
    
    for col, label in metrics.items():
        if col not in df.columns:
            print(f"\n⚠️  Skipping {label}: column '{col}' not found")
            continue
        
        print(f"\n{label} ({col}):")
        
        # Split data
        pre_data = df[df['period'] == 'Pre'][col].values
        post_data = df[df['period'] == 'Post'][col].values
        
        # Remove outliers (optional: 10-25s for tempo)
        if 'tempo' in col.lower():
            pre_data = pre_data[(pre_data >= 10) & (pre_data <= 25)]
            post_data = post_data[(post_data >= 10) & (post_data <= 25)]
            print("  Filtered outliers (10-25s range)")
        
        # Test
        result = welch_ttest_summary(pre_data, post_data, label)
        results.append(result)
        
        # Display
        print(f"  Pre:  {result['mean_pre']:.2f} ± {result['std_pre']:.2f} (n={result['n_pre']:,})")
        print(f"  Post: {result['mean_post']:.2f} ± {result['std_post']:.2f} (n={result['n_post']:,})")
        print(f"  Delta: {result['delta']:+.2f} ({result['delta_pct']:+.1f}%)")
        print(f"  t({result['df']:.0f}) = {result['t_stat']:.3f}, p = {result['p_value']:.6f} {result['sig']}")
        print(f"  95% CI: [{result['ci_lower']:.2f}, {result['ci_upper']:.2f}]")
        print(f"  Cohen's d: {result['cohens_d']:.3f}")
        
        # Interpretation
        if result['p_value'] < 0.05:
            direction = "DECREASE" if result['delta'] < 0 else "INCREASE"
            print(f"  ✓ Significant {direction}")
        else:
            print(f"  ○ No significant difference")
    
    # Create output DataFrame
    output_df = pd.DataFrame(results)
    
    # Round for readability
    output_df['mean_pre'] = output_df['mean_pre'].round(3)
    output_df['mean_post'] = output_df['mean_post'].round(3)
    output_df['std_pre'] = output_df['std_pre'].round(3)
    output_df['std_post'] = output_df['std_post'].round(3)
    output_df['delta'] = output_df['delta'].round(3)
    output_df['delta_pct'] = output_df['delta_pct'].round(2)
    output_df['t_stat'] = output_df['t_stat'].round(3)
    output_df['df'] = output_df['df'].round(1)
    output_df['ci_lower'] = output_df['ci_lower'].round(3)
    output_df['ci_upper'] = output_df['ci_upper'].round(3)
    output_df['cohens_d'] = output_df['cohens_d'].round(3)
    
    # Display summary table
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    
    display_cols = ['metric', 'mean_pre', 'mean_post', 'delta', 't_stat', 'p_value', 'sig']
    print(output_df[display_cols].to_string(index=False))
    
    # Effect sizes
    print("\n" + "=" * 70)
    print("EFFECT SIZES (COHEN'S D)")
    print("=" * 70)
    
    for _, row in output_df.iterrows():
        d = row['cohens_d']
        if abs(d) < 0.2:
            size = "negligible"
        elif abs(d) < 0.5:
            size = "small"
        elif abs(d) < 0.8:
            size = "medium"
        else:
            size = "large"
        
        print(f"{row['metric']}: d = {d:.3f} ({size})")
    
    # Save
    output_file = Path("analysis/pre_post_ttests_pitcher_level.csv")
    output_file.parent.mkdir(exist_ok=True)
    output_df.to_csv(output_file, index=False)
    
    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)
    print(f"\nSaved: {output_file}")
    print(f"Tests: {len(output_df)}")
    
    print("\n" + "=" * 70)
    print("NOTES")
    print("=" * 70)
    print("Statistical approach:")
    print("  - Unit of analysis: Individual pitcher-seasons")
    print("  - Test: Welch's t-test (unequal variances)")
    print("  - High power from large n (thousands vs 8 seasons)")
    print("\nAdvantages over season-level tests:")
    print("  - Proper inference (n_pre ≈ 4000, n_post ≈ 2500)")
    print("  - Robust to individual pitcher variation")
    print("  - Complements A4b event study")
    print("\nLimitations:")
    print("  - Assumes independence of pitcher-seasons")
    print("  - Does not account for within-pitcher correlation")
    print("  - For clustered SE, see A4b regression approach")
    print("\nInterpretation:")
    print("  - Significant effects indicate rule impact at individual level")
    print("  - Effect sizes quantify practical magnitude")
    print("  - Consistent with aggregate trends (A1) and event study (A4b)")


if __name__ == "__main__":
    main()