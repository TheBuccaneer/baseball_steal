"""
Validate MiLB data with proper per-game rate calculations
Accounts for different season lengths across years

Usage: python c7e_validate_milb_pergame.py
"""

import pandas as pd
from pathlib import Path

def load_and_validate():
    data_path = Path("milb_data")
    
    # Load schedule to get game counts
    schedule_file = data_path / "milb_schedule.csv"
    if not schedule_file.exists():
        print("ERROR: milb_schedule.csv not found!")
        return
    
    schedule = pd.read_csv(schedule_file)
    
    # Count games by level and year
    games_count = schedule.groupby(['level_name', 'season']).size().reset_index(name='total_games')
    
    # Load all pitcher opportunities files
    files = sorted(data_path.glob("milb_pitcher_opportunities_*.csv"))
    
    if not files:
        print("ERROR: No pitcher opportunities files found!")
        return
    
    print("=" * 80)
    print("MILB DATA VALIDATION (PER-GAME RATES)")
    print("=" * 80)
    
    # Load and combine all data
    all_data = []
    for file in files:
        df = pd.read_csv(file)
        all_data.append(df)
        print(f"\nLoaded: {file.name}")
    
    combined = pd.concat(all_data, ignore_index=True)
    
    # Aggregate by level and year
    aggregated = combined.groupby(['level_name', 'season']).agg({
        'pitcher_id': 'count',
        'sb': 'sum',
        'cs': 'sum',
        'attempts': 'sum'
    }).reset_index()
    
    aggregated = aggregated.rename(columns={'pitcher_id': 'pitchers'})
    aggregated['success_rate'] = aggregated['sb'] / aggregated['attempts']
    
    # Merge with game counts
    results = aggregated.merge(games_count, on=['level_name', 'season'], how='left')
    
    # Calculate per-game rates
    results['sb_per_game'] = results['sb'] / results['total_games']
    results['cs_per_game'] = results['cs'] / results['total_games']
    results['attempts_per_game'] = results['attempts'] / results['total_games']
    
    # Display
    print("\n" + "=" * 80)
    print("AGGREGATE STATISTICS (WITH PER-GAME RATES)")
    print("=" * 80)
    
    for _, row in results.iterrows():
        print(f"\n{row['level_name']} {int(row['season'])}:")
        print(f"  Games: {int(row['total_games']):,}")
        print(f"  Pitchers: {int(row['pitchers']):,}")
        print(f"  Total Attempts: {int(row['attempts']):,}")
        print(f"  Success Rate: {row['success_rate']:.1%}")
        print(f"  Attempts/Game: {row['attempts_per_game']:.2f}")
        print(f"  SB/Game: {row['sb_per_game']:.2f}")
        print(f"  CS/Game: {row['cs_per_game']:.2f}")
    
    # Summary table
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    
    display_cols = ['level_name', 'season', 'total_games', 'pitchers', 
                    'attempts_per_game', 'success_rate', 'sb_per_game', 'cs_per_game']
    
    summary = results[display_cols].copy()
    summary['season'] = summary['season'].astype(int)
    summary['total_games'] = summary['total_games'].astype(int)
    summary['pitchers'] = summary['pitchers'].astype(int)
    summary = summary.round(3)
    
    print(summary.to_string(index=False))
    
    # Validation checks
    print("\n" + "=" * 80)
    print("VALIDATION CHECKS")
    print("=" * 80)
    
    for level in ['AAA', 'AA']:
        level_data = results[results['level_name'] == level].sort_values('season')
        
        if len(level_data) < 2:
            continue
        
        print(f"\n{level}:")
        
        for i in range(len(level_data) - 1):
            y1 = level_data.iloc[i]
            y2 = level_data.iloc[i + 1]
            
            yr1 = int(y1['season'])
            yr2 = int(y2['season'])
            
            pct_change = ((y2['attempts_per_game'] - y1['attempts_per_game']) / y1['attempts_per_game']) * 100
            
            print(f"\n  {yr1} → {yr2}:")
            print(f"    Attempts/Game: {y1['attempts_per_game']:.2f} → {y2['attempts_per_game']:.2f} ({pct_change:+.1f}%)")
            print(f"    Success Rate: {y1['success_rate']:.1%} → {y2['success_rate']:.1%}")
            
            # Check for 2021→2022 treatment effect
            if yr1 == 2021 and yr2 == 2022:
                if pct_change > 20:
                    print(f"    ✓ Strong treatment effect detected")
                elif pct_change > 10:
                    print(f"    ✓ Moderate treatment effect detected")
                else:
                    print(f"    ⚠️  Treatment effect weaker than expected")
    
    print("\n" + "=" * 80)
    print("EXTERNAL BENCHMARK COMPARISON")
    print("=" * 80)
    
    # Find 2019 and 2022 for comparison
    for level in ['AAA', 'AA']:
        level_data = results[results['level_name'] == level]
        
        data_2019 = level_data[level_data['season'] == 2019]
        data_2022 = level_data[level_data['season'] == 2022]
        
        if len(data_2019) > 0 and len(data_2022) > 0:
            y19 = data_2019.iloc[0]
            y22 = data_2022.iloc[0]
            
            change = ((y22['attempts_per_game'] - y19['attempts_per_game']) / y19['attempts_per_game']) * 100
            
            print(f"\n{level} 2019→2022:")
            print(f"  Attempts/Game: {y19['attempts_per_game']:.2f} → {y22['attempts_per_game']:.2f} ({change:+.1f}%)")
            print(f"  Expected: +15-30% increase")
            
            if 15 <= change <= 35:
                print(f"  ✓ Within expected range")
            else:
                print(f"  ⚠️  Outside expected range")
            
            print(f"\n  Success Rate 2022: {y22['success_rate']:.1%}")
            print(f"  Expected: 75-83% (AAA higher than lower minors)")
            
            if 0.73 <= y22['success_rate'] <= 0.87:
                print(f"  ✓ Within expected range")
            else:
                print(f"  ⚠️  Outside expected range")

if __name__ == "__main__":
    load_and_validate()