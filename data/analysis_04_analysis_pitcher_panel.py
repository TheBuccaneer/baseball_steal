"""
A4: Pitch Tempo WITH Runners On - Analyze tempo changes across seasons
Uses weighted means by pitch count to avoid aggregation bias

Expected: 2023 decrease (20s timer), 2024 further decrease (18s timer)

Input:  analysis/analysis_pitcher_panel_relative.csv
Output: analysis/pitch_tempo_runners_on_by_season.csv
        analysis/tempo_welch_test_2023_2024.csv

References:
- Pitch Tempo Definition: https://baseballsavant.mlb.com/leaderboard/pitch-tempo
  (Median release-to-release time, takes only, same batter)
- 2023 rules: https://www.mlb.com/news/mlb-2023-rule-changes-pitch-timer-larger-bases-shifts
  (15s bases empty, 20s runners on)
- 2024 rules: https://www.mlb.com/news/mlb-rule-changes-for-2024
  (18s runners on, 15s empty unchanged)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import sys


def welch_ttest(group1, group2, weights1=None, weights2=None):
    """
    Perform Welch's t-test for two groups with optional weights.
    """
    if weights1 is not None and weights2 is not None:
        # Weighted statistics
        mean1 = np.average(group1, weights=weights1)
        mean2 = np.average(group2, weights=weights2)
        
        # Weighted variance
        var1 = np.average((group1 - mean1)**2, weights=weights1)
        var2 = np.average((group2 - mean2)**2, weights=weights2)
        
        n1 = len(group1)
        n2 = len(group2)
    else:
        # Unweighted
        mean1 = np.mean(group1)
        mean2 = np.mean(group2)
        var1 = np.var(group1, ddof=1)
        var2 = np.var(group2, ddof=1)
        n1 = len(group1)
        n2 = len(group2)
    
    # Welch's t-statistic
    se = np.sqrt(var1/n1 + var2/n2)
    t_stat = (mean1 - mean2) / se
    
    # Degrees of freedom (Welch-Satterthwaite)
    df = (var1/n1 + var2/n2)**2 / ((var1/n1)**2/(n1-1) + (var2/n2)**2/(n2-1))
    
    # Two-tailed p-value
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))
    
    # 95% CI
    t_crit = stats.t.ppf(0.975, df)
    ci_lower = (mean1 - mean2) - t_crit * se
    ci_upper = (mean1 - mean2) + t_crit * se
    
    return {
        'mean1': mean1,
        'mean2': mean2,
        'delta': mean1 - mean2,
        't_stat': t_stat,
        'df': df,
        'p_value': p_value,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'n1': n1,
        'n2': n2
    }


def main():
    print("=" * 70)
    print("A4: PITCH TEMPO WITH RUNNERS ON (WEIGHTED)")
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
    print(f"Columns: {len(df.columns)}")
    
    # Check required columns
    required = ['season', 'tempo_with_runners_on_base', 'pitches_with_runners_on_base']
    missing = [col for col in required if col not in df.columns]
    
    if missing:
        print(f"\nERROR: Missing required columns: {missing}")
        sys.exit(1)
    
    print(f"Seasons: {df['season'].min()} - {df['season'].max()}")
    
    # Filter: remove missing tempo values
    n_before = len(df)
    df = df[df['tempo_with_runners_on_base'].notna()].copy()
    n_after = len(df)
    print(f"Removed {n_before - n_after:,} rows with missing tempo ({(n_before-n_after)/n_before*100:.1f}%)")
    
    # QC: tempo range
    tempo_min = df['tempo_with_runners_on_base'].min()
    tempo_max = df['tempo_with_runners_on_base'].max()
    tempo_median = df['tempo_with_runners_on_base'].median()
    
    print(f"\nTempo range: {tempo_min:.1f}s - {tempo_max:.1f}s (median: {tempo_median:.1f}s)")
    
    if tempo_min < 10 or tempo_max > 40:
        print("⚠️  WARNING: Unusual tempo values detected")
    
    # Aggregate with WEIGHTED means
    print("\n" + "=" * 70)
    print("AGGREGATING BY SEASON (WEIGHTED BY PITCH COUNT)")
    print("=" * 70)
    
    results = []
    
    for season in sorted(df['season'].unique()):
        subset = df[df['season'] == season].copy()
        
        # Use pitch count as weights
        weights = subset['pitches_with_runners_on_base'].fillna(1)
        
        # Weighted mean
        mean_tempo = np.average(subset['tempo_with_runners_on_base'], weights=weights)
        
        # Total pitches and pitchers
        n_pitchers = len(subset)
        total_pitches = int(weights.sum())
        
        results.append({
            'season': int(season),
            'mean_tempo': round(mean_tempo, 2),
            'n_pitchers': n_pitchers,
            'total_pitches': total_pitches
        })
    
    # Create DataFrame
    agg_df = pd.DataFrame(results)
    agg_df = agg_df.sort_values('season')
    
    # Display
    print("\nResults:")
    print(agg_df.to_string(index=False))
    
    # Calculate year-over-year changes
    agg_df['yoy_change'] = agg_df['mean_tempo'].diff()
    agg_df['yoy_pct'] = (agg_df['yoy_change'] / agg_df['mean_tempo'].shift(1)) * 100
    
    print("\nYear-over-Year Changes:")
    print(agg_df[['season', 'mean_tempo', 'yoy_change', 'yoy_pct']].to_string(index=False))
    
    # Pre/Post 2023 Analysis
    print("\n" + "=" * 70)
    print("PRE/POST 2023 COMPARISON")
    print("=" * 70)
    
    pre = agg_df[agg_df['season'] < 2023]
    post = agg_df[agg_df['season'] >= 2023]
    
    if len(pre) > 0 and len(post) > 0:
        pre_mean = pre['mean_tempo'].mean()
        post_mean = post['mean_tempo'].mean()
        delta = post_mean - pre_mean
        pct = (delta / pre_mean) * 100
        
        print(f"\nPre-2023 average:  {pre_mean:.2f}s (n={len(pre)} seasons)")
        print(f"Post-2023 average: {post_mean:.2f}s (n={len(post)} seasons)")
        print(f"Change:            {delta:+.2f}s ({pct:+.1f}%)")
        
        if delta < 0:
            print("✓ Tempo DECREASED (faster pace) after 2023 rule change")
        else:
            print("⚠️  Tempo INCREASED (slower pace) - unexpected!")
        
        # Check if 2023 had biggest drop
        biggest_drop_year = agg_df.loc[agg_df['yoy_change'].idxmin(), 'season']
        biggest_drop_value = agg_df['yoy_change'].min()
        
        print(f"\nBiggest YoY decrease: {biggest_drop_year} ({biggest_drop_value:+.2f}s)")
        if biggest_drop_year == 2023:
            print("✓ 2023 had the largest single-year decrease")
    
    # Welch's t-test: 2023 vs 2024
    print("\n" + "=" * 70)
    print("WELCH'S T-TEST: 2023 vs 2024 (18s TIMER EFFECT)")
    print("=" * 70)
    
    df_2023 = df[df['season'] == 2023].copy()
    df_2024 = df[df['season'] == 2024].copy()
    
    if len(df_2023) > 0 and len(df_2024) > 0:
        weights_2023 = df_2023['pitches_with_runners_on_base'].fillna(1).values
        weights_2024 = df_2024['pitches_with_runners_on_base'].fillna(1).values
        
        test_result = welch_ttest(
            df_2023['tempo_with_runners_on_base'].values,
            df_2024['tempo_with_runners_on_base'].values,
            weights_2023,
            weights_2024
        )
        
        print(f"\n2023 mean: {test_result['mean1']:.2f}s (n={test_result['n1']} pitchers)")
        print(f"2024 mean: {test_result['mean2']:.2f}s (n={test_result['n2']} pitchers)")
        print(f"\nDifference: {test_result['delta']:.2f}s")
        print(f"t-statistic: {test_result['t_stat']:.3f}")
        print(f"df: {test_result['df']:.1f}")
        print(f"p-value: {test_result['p_value']:.6f}")
        print(f"95% CI: [{test_result['ci_lower']:.2f}, {test_result['ci_upper']:.2f}]")
        
        # Interpretation
        if test_result['p_value'] < 0.001:
            sig = "p < 0.001 ***"
        elif test_result['p_value'] < 0.01:
            sig = "p < 0.01 **"
        elif test_result['p_value'] < 0.05:
            sig = "p < 0.05 *"
        else:
            sig = "p >= 0.05 (n.s.)"
        
        print(f"\nSignificance: {sig}")
        
        if test_result['p_value'] < 0.05:
            if test_result['delta'] < 0:
                print("✓ Significant DECREASE in 2024")
                print("  → 18s timer (down from 20s) had measurable effect")
            else:
                print("⚠️  Significant INCREASE in 2024 (unexpected!)")
        else:
            print("○ No significant difference between 2023 and 2024")
            print("  → 18s timer may not have had additional effect")
        
        # Effect size (Cohen's d)
        if test_result['delta'] != 0:
            pooled_std = np.sqrt((np.var(df_2023['tempo_with_runners_on_base'], ddof=1) + 
                                  np.var(df_2024['tempo_with_runners_on_base'], ddof=1)) / 2)
            cohens_d = test_result['delta'] / pooled_std
            print(f"\nEffect size (Cohen's d): {cohens_d:.3f}")
            
            if abs(cohens_d) < 0.2:
                print("  (small effect)")
            elif abs(cohens_d) < 0.5:
                print("  (medium effect)")
            else:
                print("  (large effect)")
        
        # Save test results
        test_df = pd.DataFrame([{
            'comparison': '2023_vs_2024_runners_on',
            'mean_2023': round(test_result['mean1'], 3),
            'mean_2024': round(test_result['mean2'], 3),
            'delta': round(test_result['delta'], 3),
            't_stat': round(test_result['t_stat'], 3),
            'df': round(test_result['df'], 1),
            'p_value': test_result['p_value'],
            'ci_lower': round(test_result['ci_lower'], 3),
            'ci_upper': round(test_result['ci_upper'], 3),
            'n_2023': test_result['n1'],
            'n_2024': test_result['n2'],
            'cohens_d': round(cohens_d, 3) if test_result['delta'] != 0 else None
        }])
        
        test_output = Path("analysis/tempo_welch_test_2023_2024.csv")
        test_df.to_csv(test_output, index=False)
        print(f"\nTest saved: {test_output}")
    else:
        print("\n⚠️  Cannot perform test: missing 2023 or 2024 data")
    
    # Save aggregated results
    output_file = Path("analysis/pitch_tempo_runners_on_by_season.csv")
    output_file.parent.mkdir(exist_ok=True)
    
    # Save without yoy columns (cleaner)
    agg_df[['season', 'mean_tempo', 'n_pitchers', 'total_pitches']].to_csv(output_file, index=False)
    
    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)
    print(f"\nSaved: {output_file}")
    print(f"Rows: {len(agg_df)}")
    
    print("\n" + "=" * 70)
    print("NOTES")
    print("=" * 70)
    print("Data: Tempo WITH RUNNERS ON only (bases empty not available)")
    print("Weighting: By pitcher pitch counts (avoids Simpson's paradox)")
    print("Statistical test: Welch's t-test (unequal variances)")
    print("\nPitch Tempo definition (Statcast):")
    print("  - Median release-to-release time")
    print("  - Takes only (no swings)")
    print("  - Same batter")
    print("\nRule changes:")
    print("  2023: Pitch timer 20s with runners (NEW)")
    print("  2024: Pitch timer 18s with runners (down from 20s)")
    print("\nReferences:")
    print("  https://www.mlb.com/news/mlb-2023-rule-changes-pitch-timer-larger-bases-shifts")
    print("  https://www.mlb.com/news/mlb-rule-changes-for-2024")


if __name__ == "__main__":
    main()