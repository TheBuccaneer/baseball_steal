"""
c1_steal_opportunities_build.py
================================
Build Stolen Base Opportunities panel from Baseball Savant data.

Creates pitcher×season panel (2018-2024) with:
- Opportunities (runner on 1B with 2B open - steal situation to 2B)
- Attempts (steal attempts to 2B: SB + CS)
- Successes (successful steals to 2B)
- Success rate (SB / attempts, NaN if attempts=0)

CRITICAL DEFINITIONS (for replicability):
- Opportunity: Runner on 1B with 2B open (1B→2B steal situation)
- Target base: 2B ONLY (not "All" bases)
- Source: Savant "Pitcher Running Game" leaderboard (target_base=2B)
- NO qualifier filter (includes all pitchers, missing = 0)
- Years: 2018-2024 (2025 excluded due to 18s rule change mid-implementation)

RULE CONTEXT:
- 2023: Pitch Timer 15s (bases empty) / 20s (runners on)
- 2024: Changed to 18s (runners on)
- 2023+: Disengagement limits (max 2 pickoffs/stepoffs per PA)
These rule changes affect steal opportunities and attempts.

Data Sources:
- Baseball Savant Pitcher Running Game Leaderboard
  https://baseballsavant.mlb.com/leaderboard/pitcher-running-game
- Download separate CSV for each year 2018-2024, target_base=2B
- Save as: ./data/leaderboards/pitcher_running_game_2B_20{YY}.csv

Expected columns:
- player_id: pitcher ID
- start_year: season year
- key_target_base: should be "2B"
- n_init: opportunities (runner on 1B, 2B open)
- n_sb: successful steals to 2B
- n_cs: caught stealing at 2B
- n_pk: pickoffs at 1B

Outputs:
- c_opportunities_panel.csv (pitcher×season panel, all years combined)
- c1_coverage_check.txt (data quality report)
- c1_summary_stats.csv (descriptive statistics by year)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# ============================================================================
# CONFIGURATION
# ============================================================================

# Paths
if Path.cwd().name == 'data':
    INPUT_DIR = Path("leaderboards")
    OUTPUT_DIR = Path("analysis/c_running_game")
else:
    INPUT_DIR = Path("./data/leaderboards")
    OUTPUT_DIR = Path("./analysis/c_running_game")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Years to process (2025 excluded - rule changed to 18s mid-season)
YEARS = list(range(2018, 2025))  # 2018-2024

# Minimum opportunities for descriptive stats (not for filtering)
MIN_OPP_DESCRIPTIVE = 10

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_savant_running_game(year, input_dir):
    """
    Load Savant Pitcher Running Game CSV for a given year.
    
    Expected filename: pitcher_running_game_2B_20{YY}.csv
    
    Returns DataFrame with standardized columns:
    - pitcher_id (int)
    - season (int)
    - opportunities_2b (int)
    - attempts_2b (int): n_sb + n_cs
    - sb_2b (int)
    - cs_2b (int)
    """
    # Construct filename: 2018 → pitcher_running_game_2B_2018.csv
    yy = str(year)
    filename = input_dir / f"pitcher_running_game_2B_{yy}.csv"
    
    if not filename.exists():
        print(f"⚠️  File not found: {filename}")
        print(f"    Expected location: {filename.absolute()}")
        return None
    
    try:
        df = pd.read_csv(filename)
        print(f"✓ Loaded {filename.name}: {len(df):,} rows")
    except Exception as e:
        print(f"⚠️  Error reading {filename}: {e}")
        return None
    
    # Verify required columns exist
    required_cols = ['player_id', 'start_year', 'n_init', 'n_sb', 'n_cs']
    missing = [col for col in required_cols if col not in df.columns]
    
    if missing:
        print(f"⚠️  Missing columns in {year}: {missing}")
        print(f"    Available columns: {df.columns.tolist()}")
        return None
    
    # Check target base (should be "2B")
    if 'key_target_base' in df.columns:
        unique_bases = df['key_target_base'].unique()
        if len(unique_bases) == 1 and unique_bases[0] == '2B':
            print(f"  ✓ Confirmed: target_base = 2B")
        else:
            print(f"  ⚠️  Warning: unexpected target_base values: {unique_bases}")
    
    # Standardize columns
    df_std = pd.DataFrame({
        'pitcher_id': df['player_id'].astype(int),
        'season': df['start_year'].astype(int),
        'opportunities_2b': df['n_init'].fillna(0).astype(int),
        'sb_2b': df['n_sb'].fillna(0).astype(int),
        'cs_2b': df['n_cs'].fillna(0).astype(int)
    })
    
    # Calculate attempts
    df_std['attempts_2b'] = df_std['sb_2b'] + df_std['cs_2b']
    
    # Deduplicate (keep first occurrence if multiple records per pitcher)
    n_before = len(df_std)
    df_std = df_std.groupby(['pitcher_id', 'season'], as_index=False).first()
    n_after = len(df_std)
    
    if n_before > n_after:
        print(f"  ⚠️  Deduped: {n_before - n_after} duplicate pitcher-season records")
    
    return df_std


def validate_data(df, year):
    """
    Apply data logic checks and corrections.
    
    Checks:
    - attempts_2b <= opportunities_2b
    - sb_2b + cs_2b = attempts_2b (already enforced by calculation)
    
    Returns validated DataFrame with corrections logged.
    """
    issues = []
    
    # Check: attempts <= opportunities
    bad_attempts = df['attempts_2b'] > df['opportunities_2b']
    if bad_attempts.any():
        n_bad = bad_attempts.sum()
        issues.append(f"{n_bad} records: attempts > opportunities (clipped)")
        df.loc[bad_attempts, 'attempts_2b'] = df.loc[bad_attempts, 'opportunities_2b']
        
        # Also clip sb/cs proportionally
        excess_ratio = df.loc[bad_attempts, 'opportunities_2b'] / df.loc[bad_attempts, 'attempts_2b']
        df.loc[bad_attempts, 'sb_2b'] = (df.loc[bad_attempts, 'sb_2b'] * excess_ratio).astype(int)
        df.loc[bad_attempts, 'cs_2b'] = (df.loc[bad_attempts, 'cs_2b'] * excess_ratio).astype(int)
        df.loc[bad_attempts, 'attempts_2b'] = df.loc[bad_attempts, 'sb_2b'] + df.loc[bad_attempts, 'cs_2b']
    
    if issues:
        print(f"  ⚠️  {year} validation issues:")
        for issue in issues:
            print(f"      - {issue}")
    
    return df


def calculate_rates(df):
    """
    Calculate attempt rate and success rate.
    
    - attempt_rate: attempts / max(opportunities, 1)
    - sb_pct: sb / attempts (NaN if attempts=0)
    """
    # Attempt rate (defensive: avoid division by zero)
    df['attempt_rate'] = df['attempts_2b'] / np.maximum(df['opportunities_2b'], 1)
    
    # Success rate (NaN if no attempts)
    df['sb_pct'] = np.where(
        df['attempts_2b'] > 0,
        df['sb_2b'] / df['attempts_2b'],
        np.nan
    )
    
    return df


# ============================================================================
# MAIN PROCESSING
# ============================================================================

def main():
    print("=" * 70)
    print("C1: BUILD STOLEN BASE OPPORTUNITIES PANEL (TARGET BASE = 2B)")
    print("=" * 70)
    
    print(f"\nInput directory: {INPUT_DIR.absolute()}")
    print(f"Output directory: {OUTPUT_DIR.absolute()}")
    print(f"Years to process: {min(YEARS)}-{max(YEARS)}")
    print(f"Target base: 2B (1B→2B steals only)")
    
    # Load all years
    print("\n" + "-" * 70)
    print("LOADING DATA")
    print("-" * 70)
    
    dfs = []
    coverage_info = []
    
    for year in YEARS:
        df = load_savant_running_game(year, INPUT_DIR)
        
        if df is not None:
            # Validate
            df = validate_data(df, year)
            
            # Track coverage
            coverage_info.append({
                'year': year,
                'n_pitchers_csv': len(df),
                'n_zero_opp': (df['opportunities_2b'] == 0).sum(),
                'pct_zero_opp': 100 * (df['opportunities_2b'] == 0).sum() / len(df)
            })
            
            dfs.append(df)
        else:
            print(f"⚠️  Skipping {year} due to load failure")
    
    if len(dfs) == 0:
        print("\n❌ ERROR: No data loaded. Check file locations and try again.")
        sys.exit(1)
    
    # Combine all years
    print("\n" + "-" * 70)
    print("COMBINING DATA")
    print("-" * 70)
    
    df_panel = pd.concat(dfs, ignore_index=True)
    
    print(f"\n✓ Combined panel: {len(df_panel):,} pitcher-season observations")
    print(f"  Pitchers: {df_panel['pitcher_id'].nunique():,}")
    print(f"  Years: {sorted(df_panel['season'].unique())}")
    
    # Calculate rates
    df_panel = calculate_rates(df_panel)
    
    # ========================================================================
    # COVERAGE REPORT
    # ========================================================================
    
    print("\n" + "-" * 70)
    print("COVERAGE REPORT")
    print("-" * 70)
    
    df_coverage = pd.DataFrame(coverage_info)
    
    print("\nPitchers per year:")
    print(df_coverage[['year', 'n_pitchers_csv', 'n_zero_opp', 'pct_zero_opp']].to_string(index=False))
    
    # Save coverage report
    coverage_file = OUTPUT_DIR / "c1_coverage_check.txt"
    with open(coverage_file, 'w', encoding='utf-8') as f:
        f.write("C1 COVERAGE REPORT\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Data source: Savant Pitcher Running Game (target_base=2B)\n")
        f.write(f"Years: {min(YEARS)}-{max(YEARS)}\n")
        f.write(f"Total observations: {len(df_panel):,}\n")
        f.write(f"Unique pitchers: {df_panel['pitcher_id'].nunique():,}\n\n")
        f.write("Pitchers per year:\n")
        f.write(df_coverage.to_string(index=False))
        f.write("\n\nNote: Zero opportunities = pitcher had no 1B→2B steal situations\n")
        f.write("      These are kept in panel (opportunities=0, attempts=0, sb=0)\n")
    
    print(f"\n✓ Saved: {coverage_file}")
    
    # ========================================================================
    # SUMMARY STATISTICS
    # ========================================================================
    
    print("\n" + "-" * 70)
    print("SUMMARY STATISTICS")
    print("-" * 70)
    
    # Filter for descriptive stats (min opportunities)
    df_qualified = df_panel[df_panel['opportunities_2b'] >= MIN_OPP_DESCRIPTIVE]
    
    summary_data = []
    
    for year in sorted(df_panel['season'].unique()):
        df_year = df_panel[df_panel['season'] == year]
        df_year_qual = df_qualified[df_qualified['season'] == year]
        
        summary_data.append({
            'year': year,
            'n_pitchers': len(df_year),
            'n_qualified': len(df_year_qual),
            'total_opportunities': df_year['opportunities_2b'].sum(),
            'total_attempts': df_year['attempts_2b'].sum(),
            'total_sb': df_year['sb_2b'].sum(),
            'total_cs': df_year['cs_2b'].sum(),
            'league_attempt_rate': df_year['attempts_2b'].sum() / max(df_year['opportunities_2b'].sum(), 1),
            'league_sb_pct': df_year['sb_2b'].sum() / max(df_year['attempts_2b'].sum(), 1),
            'mean_attempt_rate_qual': df_year_qual['attempt_rate'].mean(),
            'mean_sb_pct_qual': df_year_qual['sb_pct'].mean()
        })
    
    df_summary = pd.DataFrame(summary_data)
    
    print(f"\nLeague-wide rates (all pitchers, 1B→2B only):")
    display_cols = ['year', 'total_opportunities', 'total_attempts', 'league_attempt_rate', 'league_sb_pct']
    print(df_summary[display_cols].to_string(index=False))
    
    print(f"\nPitcher-average rates (≥{MIN_OPP_DESCRIPTIVE} opp):")
    display_cols = ['year', 'n_qualified', 'mean_attempt_rate_qual', 'mean_sb_pct_qual']
    print(df_summary[display_cols].to_string(index=False))
    
    # Save summary
    summary_file = OUTPUT_DIR / "c1_summary_stats.csv"
    df_summary.to_csv(summary_file, index=False)
    print(f"\n✓ Saved: {summary_file}")
    
    # ========================================================================
    # SAVE PANEL
    # ========================================================================
    
    print("\n" + "-" * 70)
    print("SAVING PANEL")
    print("-" * 70)
    
    panel_file = OUTPUT_DIR / "c_opportunities_panel.csv"
    df_panel.to_csv(panel_file, index=False)
    
    print(f"\n✓ Saved: {panel_file}")
    print(f"  Columns: {df_panel.columns.tolist()}")
    print(f"  Rows: {len(df_panel):,}")
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    
    print("\n" + "=" * 70)
    print("C1 COMPLETE")
    print("=" * 70)
    
    print(f"\nPanel built: {min(YEARS)}-{max(YEARS)} (1B→2B steals only)")
    print(f"  Observations: {len(df_panel):,}")
    print(f"  Pitchers: {df_panel['pitcher_id'].nunique():,}")
    
    # Quality checks
    n_with_opp = (df_panel['opportunities_2b'] > 0).sum()
    n_with_att = (df_panel['attempts_2b'] > 0).sum()
    
    print(f"\nData quality:")
    print(f"  Records with opportunities: {n_with_opp:,} ({100*n_with_opp/len(df_panel):.1f}%)")
    print(f"  Records with attempts: {n_with_att:,} ({100*n_with_att/len(df_panel):.1f}%)")
    
    # Show 2023 jump
    if 2022 in df_panel['season'].values and 2023 in df_panel['season'].values:
        rate_2022 = df_summary[df_summary['year'] == 2022]['league_attempt_rate'].values[0]
        rate_2023 = df_summary[df_summary['year'] == 2023]['league_attempt_rate'].values[0]
        pct_change = 100 * (rate_2023 - rate_2022) / rate_2022
        
        print(f"\n2023 rule change impact:")
        print(f"  2022 attempt rate: {rate_2022:.3f}")
        print(f"  2023 attempt rate: {rate_2023:.3f}")
        print(f"  Change: {pct_change:+.1f}%")
    
    print(f"\nOutputs saved to: {OUTPUT_DIR}/")
    print(f"  1. {panel_file.name} (main panel)")
    print(f"  2. {coverage_file.name} (quality report)")
    print(f"  3. {summary_file.name} (descriptive stats)")
    
    print("\n" + "=" * 70)
    print("Next step: Run c2_join_pitcher_baseline.py")
    print("=" * 70)


if __name__ == "__main__":
    main()