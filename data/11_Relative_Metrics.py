"""
Adds relative metrics (z-scores, percentiles) to all panels
Makes cross-year comparisons easier by standardizing metrics

Output: 
- analysis_runner_panel_relative.csv
- analysis_catcher_panel_relative.csv  
- analysis_pitcher_panel_relative.csv

Usage: python 11_add_relative_metrics.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

def calculate_relative_metrics(df, metrics, group_col='season', min_samples=10):
    """
    Calculate z-scores and percentiles for specified metrics
    
    Args:
        df: DataFrame
        metrics: List of column names to standardize
        group_col: Column to group by (typically 'season')
        min_samples: Minimum non-null values required to calculate stats
    
    Returns:
        DataFrame with added _z and _pct columns
    """
    df_out = df.copy()
    
    for metric in metrics:
        if metric not in df.columns:
            print(f"  WARNING: {metric} not found in data, skipping")
            continue
        
        # Initialize new columns
        z_col = f"{metric}_z"
        pct_col = f"{metric}_pct"
        df_out[z_col] = np.nan
        df_out[pct_col] = np.nan
        
        # Calculate per group
        for group_val, group_df in df.groupby(group_col):
            valid = group_df[metric].notna()
            n_valid = valid.sum()
            
            if n_valid < min_samples:
                print(f"  WARNING: {metric} in {group_col}={group_val} has only {n_valid} valid values, skipping")
                continue
            
            # Z-score: (x - mean) / std
            mean = group_df.loc[valid, metric].mean()
            std = group_df.loc[valid, metric].std()
            
            if std > 0:
                z_scores = (group_df.loc[valid, metric] - mean) / std
                df_out.loc[group_df.loc[valid].index, z_col] = z_scores
            
            # Percentile rank (0-1)
            percentiles = group_df.loc[valid, metric].rank(pct=True)
            df_out.loc[group_df.loc[valid].index, pct_col] = percentiles
    
    return df_out

def main():
    print("=" * 80)
    print("ADDING RELATIVE METRICS TO PANELS")
    print("=" * 80)
    print(f"\nCurrent directory: {Path.cwd()}")
    
    analysis_path = Path("analysis")
    
    if not analysis_path.exists():
        print(f"\nERROR: {analysis_path.absolute()} not found!")
        print("Please run this script from the data/ directory")
        sys.exit(1)
    
    # ============================================================================
    # RUNNER PANEL
    # ============================================================================
    print("\n" + "=" * 80)
    print("RUNNER PANEL")
    print("=" * 80)
    
    runner_file = analysis_path / "analysis_runner_panel.csv"
    if not runner_file.exists():
        print(f"  ERROR: {runner_file} not found!")
    else:
        print(f"\nLoading {runner_file.name}...")
        runner = pd.read_csv(runner_file)
        print(f"  Loaded {len(runner):,} rows")
        
        # Define metrics to standardize
        runner_metrics = [
            'sb', 'cs', 'attempts', 'success_rate',
            'brv_total', 'brv_xb', 'brv_steal',
            'r_primary_lead', 'r_secondary_lead',
            'sprint_speed_fps',
            'opponent_avg_poptime', 'opponent_avg_tempo'
        ]
        
        print("\nCalculating relative metrics...")
        runner_rel = calculate_relative_metrics(runner, runner_metrics, group_col='season')
        
        # Save
        output_file = analysis_path / "analysis_runner_panel_relative.csv"
        runner_rel.to_csv(output_file, index=False)
        
        print(f"\n✓ Saved: {output_file}")
        print(f"  Original columns: {len(runner.columns)}")
        print(f"  New columns: {len(runner_rel.columns)} (+{len(runner_rel.columns) - len(runner.columns)})")
    
    # ============================================================================
    # CATCHER PANEL
    # ============================================================================
    print("\n" + "=" * 80)
    print("CATCHER PANEL")
    print("=" * 80)
    
    catcher_file = analysis_path / "analysis_catcher_panel.csv"
    if not catcher_file.exists():
        print(f"  ERROR: {catcher_file} not found!")
    else:
        print(f"\nLoading {catcher_file.name}...")
        catcher = pd.read_csv(catcher_file)
        print(f"  Loaded {len(catcher):,} rows")
        
        # Define metrics to standardize
        catcher_metrics = [
            'sb_attempts', 'n_cs', 'n_sb',
            'cs_rate', 'sb_success_rate',
            'pop_time', 'exchange_time', 'arm_strength',
            'catcher_stealing_runs', 'caught_stealing_above_average',
            'seasonal_runner_speed', 'runner_distance_from_second'
        ]
        
        print("\nCalculating relative metrics...")
        catcher_rel = calculate_relative_metrics(catcher, catcher_metrics, group_col='season')
        
        # Save
        output_file = analysis_path / "analysis_catcher_panel_relative.csv"
        catcher_rel.to_csv(output_file, index=False)
        
        print(f"\n✓ Saved: {output_file}")
        print(f"  Original columns: {len(catcher.columns)}")
        print(f"  New columns: {len(catcher_rel.columns)} (+{len(catcher_rel.columns) - len(catcher.columns)})")
    
    # ============================================================================
    # PITCHER PANEL
    # ============================================================================
    print("\n" + "=" * 80)
    print("PITCHER PANEL")
    print("=" * 80)
    
    pitcher_file = analysis_path / "analysis_pitcher_panel.csv"
    if not pitcher_file.exists():
        print(f"  ERROR: {pitcher_file} not found!")
    else:
        print(f"\nLoading {pitcher_file.name}...")
        pitcher = pd.read_csv(pitcher_file)
        print(f"  Loaded {len(pitcher):,} rows")
        
        # Define metrics to standardize
        pitcher_metrics = [
            'n_init', 'total_sb_attempts', 'n_sb', 'n_cs', 'n_pk',
            'sb_success_rate', 'cs_rate', 'pickoff_rate',
            'r_primary_lead', 'r_secondary_lead', 'lead_gain',
            'runs_prevented_on_running_attr', 'n_pitcher_cs_aa',
            'tempo_with_runners_on_base'
        ]
        
        print("\nCalculating relative metrics...")
        pitcher_rel = calculate_relative_metrics(pitcher, pitcher_metrics, group_col='season')
        
        # Save
        output_file = analysis_path / "analysis_pitcher_panel_relative.csv"
        pitcher_rel.to_csv(output_file, index=False)
        
        print(f"\n✓ Saved: {output_file}")
        print(f"  Original columns: {len(pitcher.columns)}")
        print(f"  New columns: {len(pitcher_rel.columns)} (+{len(pitcher_rel.columns) - len(pitcher.columns)})")
    
    # ============================================================================
    # SUMMARY
    # ============================================================================
    print("\n" + "=" * 80)
    print("COMPLETE")
    print("=" * 80)
    print("\nRelative metrics added to all panels:")
    print("  - _z suffix: z-score (season-adjusted)")
    print("  - _pct suffix: percentile rank (0-1, season-adjusted)")
    print("\nExample usage:")
    print("  # Find elite catchers (top 10% CS rate)")
    print("  elite = df[df['cs_rate_pct'] > 0.90]")
    print("  ")
    print("  # Compare across years (z-scores)")
    print("  df['cs_rate_z'] # standardized, comparable 2018-2025")
    print("\nFiles created:")
    print("  - analysis_runner_panel_relative.csv")
    print("  - analysis_catcher_panel_relative.csv")
    print("  - analysis_pitcher_panel_relative.csv")

if __name__ == "__main__":
    main()