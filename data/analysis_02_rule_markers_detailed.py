"""
A2: Create Rule Markers CSV
Documents MLB rule changes for 2023-2024

Output: analysis/rule_markers_detailed.csv
"""

import pandas as pd
from pathlib import Path

def main():
    print("=" * 60)
    print("CREATING RULE MARKERS")
    print("=" * 60)
    
    # Define rule changes
    rules = [
        {
            'season': 2023,
            'rule_category': 'pitch_timer',
            'rule_detail': 'Pitch timer 15s (bases empty) / 20s (runners on)'
        },
        {
            'season': 2023,
            'rule_category': 'bases',
            'rule_detail': 'Larger bases 18 inches (from 15)'
        },
        {
            'season': 2023,
            'rule_category': 'pickoff',
            'rule_detail': 'Two disengagement limit; third unsuccessful attempt = balk'
        },
        {
            'season': 2024,
            'rule_category': 'pitch_timer',
            'rule_detail': 'Pitch timer 18s with runners on (15s empty unchanged)'
        }
    ]
    
    # Create DataFrame
    df = pd.DataFrame(rules)
    
    # Display
    print("\nRule Markers:")
    print(df.to_string(index=False))
    
    # Save
    output_path = Path("analysis")
    output_path.mkdir(exist_ok=True)
    output_file = output_path / "rule_markers_detailed.csv"
    
    df.to_csv(output_file, index=False)
    
    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)
    print(f"\nSaved: {output_file}")
    print(f"Rows: {len(df)}")
    
    print("\nSources:")
    print("  2023 rules: https://www.mlb.com/news/mlb-2023-rule-changes-pitch-timer-larger-bases-shifts")
    print("  2024 rules: https://www.mlb.com/news/mlb-rule-changes-for-2024")

if __name__ == "__main__":
    main()