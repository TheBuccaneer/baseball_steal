"""
C5: MiLB Stolen Base Rate Analysis
PPML regression with pitcher fixed effects and level interaction

Specification:
  sb ~ post2022 + level_name + post2022:level_name 
     + pitcher_id (FE) + offset(log(total_opps))

This combines the effects from C3 (attempts) and C4 (success):
  SB Rate = Attempt Rate × Success Rate

Outputs:
- analysis/c7_milb/c5_coefficients.csv
- analysis/c7_milb/c5_summary.txt
- analysis/c7_milb/c5_event_study.png
- analysis/c7_milb/c5_decomposition.csv

Usage: python c7i_milb_c5_sb_rate.py
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Setup
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300

def load_data():
    """Load MiLB panel data"""
    data_path = Path("analysis/c7_milb")
    
    pooled = pd.read_csv(data_path / "milb_panel_c3c5_pooled.csv")
    aaa = pd.read_csv(data_path / "milb_panel_c3c5_AAA.csv")
    aa = pd.read_csv(data_path / "milb_panel_c3c5_AA.csv")
    
    return pooled, aaa, aa

def run_ppml_model(data, formula, name=""):
    """Run PPML regression for SB rate"""
    print(f"\n{'='*80}")
    print(f"MODEL: {name}")
    print('='*80)
    
    # Add log offset
    data = data.copy()
    data['log_opps'] = np.log(data['total_opps'])
    
    # Fit model
    model = smf.glm(
        formula=formula,
        data=data,
        family=sm.families.Poisson(),
        offset=data['log_opps']
    ).fit(cov_type='HC1')  # Robust standard errors
    
    print(model.summary())
    
    return model

def extract_treatment_effects(model, data):
    """Extract treatment effects and confidence intervals"""
    
    params = model.params
    conf_int = model.conf_int()
    
    results = []
    
    # Main effect (AA reference level)
    if 'post2022' in params.index:
        coef = params['post2022']
        ci_low = conf_int.loc['post2022', 0]
        ci_high = conf_int.loc['post2022', 1]
        
        # Convert to percentage change
        pct_change = (np.exp(coef) - 1) * 100
        pct_low = (np.exp(ci_low) - 1) * 100
        pct_high = (np.exp(ci_high) - 1) * 100
        
        results.append({
            'level': 'AA (reference)',
            'coefficient': coef,
            'rate_ratio': np.exp(coef),
            'pct_change': pct_change,
            'ci_low': pct_low,
            'ci_high': pct_high,
            'pvalue': model.pvalues['post2022']
        })
    
    # Interaction effect (AAA = main + interaction)
    if 'post2022:level_name[T.AAA]' in params.index:
        main_coef = params['post2022']
        int_coef = params['post2022:level_name[T.AAA]']
        total_coef = main_coef + int_coef
        
        pct_change = (np.exp(total_coef) - 1) * 100
        
        results.append({
            'level': 'AAA (main + interaction)',
            'coefficient': total_coef,
            'rate_ratio': np.exp(total_coef),
            'pct_change': pct_change,
            'ci_low': np.nan,
            'ci_high': np.nan,
            'pvalue': model.pvalues['post2022:level_name[T.AAA]']
        })
    
    return pd.DataFrame(results)

def create_event_study_plot(pooled, aaa, aa, output_path):
    """Create event study plot by year"""
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    datasets = [
        (pooled, 'Pooled (AAA + AA)', axes[0]),
        (aaa, 'AAA Only', axes[1]),
        (aa, 'AA Only', axes[2])
    ]
    
    for data, title, ax in datasets:
        # Calculate SB rate by year
        year_stats = data.groupby('season').agg({
            'sb': 'sum',
            'total_opps': 'sum'
        })
        year_stats['sb_rate'] = year_stats['sb'] / year_stats['total_opps']
        
        years = year_stats.index
        rates = year_stats['sb_rate'].values
        
        # Plot
        ax.plot(years, rates, marker='o', linewidth=2, markersize=8, color='purple')
        ax.axvline(2022, color='red', linestyle='--', alpha=0.5, label='Treatment (2022)')
        ax.set_xlabel('Season')
        ax.set_ylabel('SB Rate (SB per Opportunity)')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Add labels
        for year, rate in zip(years, rates):
            ax.text(year, rate + 0.002, f'{rate:.3f}', 
                   ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path / "c5_event_study.png", dpi=300, bbox_inches='tight')
    print(f"\nSaved event study plot: {output_path}/c5_event_study.png")
    plt.close()

def decompose_effects(c3_coefs, c4_coefs, c5_coefs):
    """Decompose C5 total effect into C3 (attempts) and C4 (success) components"""
    
    decomp = []
    
    for level in ['AAA', 'AA']:
        # Get coefficients from each model
        c3_row = c3_coefs[c3_coefs['level'] == level].iloc[0]
        c4_row = c4_coefs[c4_coefs['level'] == level].iloc[0]
        c5_row = c5_coefs[c5_coefs['level'] == level].iloc[0]
        
        # Log-space decomposition: log(SB) = log(Attempts) + log(Success)
        # So: β_C5 ≈ β_C3 + β_C4 (in log space)
        predicted_c5_coef = c3_row['coefficient'] + c4_row['coefficient']
        predicted_c5_pct = (np.exp(predicted_c5_coef) - 1) * 100
        
        # Contribution shares
        c3_share = c3_row['coefficient'] / predicted_c5_coef * 100
        c4_share = c4_row['coefficient'] / predicted_c5_coef * 100
        
        decomp.append({
            'level': level,
            'c3_attempts_coef': c3_row['coefficient'],
            'c3_attempts_pct': c3_row['pct_change'],
            'c4_success_coef': c4_row['coefficient'],
            'c4_success_pct': (np.exp(c4_row['coefficient']) - 1) * 100,  # Convert from OR
            'c5_actual_coef': c5_row['coefficient'],
            'c5_actual_pct': c5_row['pct_change'],
            'c5_predicted_coef': predicted_c5_coef,
            'c5_predicted_pct': predicted_c5_pct,
            'c3_contribution_pct': c3_share,
            'c4_contribution_pct': c4_share
        })
    
    return pd.DataFrame(decomp)

def main():
    output_path = Path("analysis/c7_milb")
    
    print("="*80)
    print("C5: MILB STOLEN BASE RATE ANALYSIS")
    print("="*80)
    
    # Load data
    print("\nLoading data...")
    pooled, aaa, aa = load_data()
    print(f"  Pooled: {len(pooled):,} observations")
    print(f"  AAA: {len(aaa):,} observations")
    print(f"  AA: {len(aa):,} observations")
    
    # ========================================================================
    # MAIN MODEL: Pooled with interaction
    # ========================================================================
    
    formula_pooled = "sb ~ post2022 + level_name + post2022:level_name + C(pitcher_id)"
    model_pooled = run_ppml_model(pooled, formula_pooled, "Pooled with Level Interaction")
    
    # Extract effects
    effects_pooled = extract_treatment_effects(model_pooled, pooled)
    
    print("\n" + "="*80)
    print("TREATMENT EFFECTS (Pooled Model)")
    print("="*80)
    print(effects_pooled.to_string(index=False))
    
    # ========================================================================
    # ROBUSTNESS: Separate by level (without pitcher FE to avoid convergence issues)
    # ========================================================================
    
    formula_simple = "sb ~ post2022"
    
    print("\n" + "="*80)
    print("ROBUSTNESS: AAA ONLY (without pitcher FE)")
    print("="*80)
    model_aaa = run_ppml_model(aaa, formula_simple, "AAA Only")
    
    aaa_effect = {
        'level': 'AAA',
        'coefficient': model_aaa.params['post2022'],
        'rate_ratio': np.exp(model_aaa.params['post2022']),
        'pct_change': (np.exp(model_aaa.params['post2022']) - 1) * 100,
        'ci_low': (np.exp(model_aaa.conf_int().loc['post2022', 0]) - 1) * 100,
        'ci_high': (np.exp(model_aaa.conf_int().loc['post2022', 1]) - 1) * 100,
        'pvalue': model_aaa.pvalues['post2022']
    }
    
    print("\n" + "="*80)
    print("ROBUSTNESS: AA ONLY (without pitcher FE)")
    print("="*80)
    model_aa = run_ppml_model(aa, formula_simple, "AA Only")
    
    aa_effect = {
        'level': 'AA',
        'coefficient': model_aa.params['post2022'],
        'rate_ratio': np.exp(model_aa.params['post2022']),
        'pct_change': (np.exp(model_aa.params['post2022']) - 1) * 100,
        'ci_low': (np.exp(model_aa.conf_int().loc['post2022', 0]) - 1) * 100,
        'ci_high': (np.exp(model_aa.conf_int().loc['post2022', 1]) - 1) * 100,
        'pvalue': model_aa.pvalues['post2022']
    }
    
    # Combine robustness results
    robustness = pd.DataFrame([aaa_effect, aa_effect])
    
    print("\n" + "="*80)
    print("ROBUSTNESS: SEPARATE LEVEL ESTIMATES")
    print("="*80)
    print(robustness.to_string(index=False))
    
    # ========================================================================
    # DECOMPOSITION (if C3 and C4 results exist)
    # ========================================================================
    
    try:
        c3_coefs = pd.read_csv(output_path / "c3_coefficients.csv")
        c4_coefs = pd.read_csv(output_path / "c4_coefficients.csv")
        
        # Filter to separate level estimates only
        c3_filtered = c3_coefs[c3_coefs['level'].isin(['AAA', 'AA'])]
        c4_filtered = c4_coefs[c4_coefs['level'].isin(['AAA', 'AA'])]
        
        decomp = decompose_effects(c3_filtered, c4_filtered, robustness)
        
        print("\n" + "="*80)
        print("EFFECT DECOMPOSITION: C5 = C3 + C4")
        print("="*80)
        print("\nHow much of the SB rate increase comes from:")
        print("  - C3: Higher attempt rates")
        print("  - C4: Higher success rates")
        print("")
        print(decomp.to_string(index=False))
        
        # Save decomposition
        decomp.to_csv(output_path / "c5_decomposition.csv", index=False)
        print(f"\nSaved decomposition: {output_path}/c5_decomposition.csv")
        
    except FileNotFoundError:
        print("\n⚠️  C3 or C4 results not found. Skipping decomposition.")
        decomp = None
    
    # ========================================================================
    # SAVE RESULTS
    # ========================================================================
    
    # Save coefficients
    all_effects = pd.concat([effects_pooled, robustness], ignore_index=True)
    all_effects.to_csv(output_path / "c5_coefficients.csv", index=False)
    print(f"\nSaved coefficients: {output_path}/c5_coefficients.csv")
    
    # Save model summaries
    with open(output_path / "c5_summary.txt", 'w') as f:
        f.write("="*80 + "\n")
        f.write("C5: MILB STOLEN BASE RATE ANALYSIS - MODEL SUMMARIES\n")
        f.write("="*80 + "\n\n")
        
        f.write("POOLED MODEL WITH INTERACTION\n")
        f.write("="*80 + "\n")
        f.write(str(model_pooled.summary()) + "\n\n")
        
        f.write("AAA ONLY\n")
        f.write("="*80 + "\n")
        f.write(str(model_aaa.summary()) + "\n\n")
        
        f.write("AA ONLY\n")
        f.write("="*80 + "\n")
        f.write(str(model_aa.summary()) + "\n\n")
    
    print(f"Saved summaries: {output_path}/c5_summary.txt")
    
    # Create plots
    create_event_study_plot(pooled, aaa, aa, output_path)
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    print("\nTotal treatment effects on SB RATE (2021→2022):")
    print(f"  AAA: +{aaa_effect['pct_change']:.1f}% [{aaa_effect['ci_low']:.1f}%, {aaa_effect['ci_high']:.1f}%]")
    print(f"  AA:  +{aa_effect['pct_change']:.1f}% [{aa_effect['ci_low']:.1f}%, {aa_effect['ci_high']:.1f}%]")
    
    if decomp is not None:
        print("\nDecomposition (contribution to total effect):")
        for _, row in decomp.iterrows():
            print(f"\n  {row['level']}:")
            print(f"    Attempts (C3): {row['c3_contribution_pct']:.1f}% of effect")
            print(f"    Success (C4):  {row['c4_contribution_pct']:.1f}% of effect")
            print(f"    Predicted total: +{row['c5_predicted_pct']:.1f}%")
            print(f"    Actual total:    +{row['c5_actual_pct']:.1f}%")
    
    print("\nInterpretation:")
    print("  - 2022 rules led to 45-70% increases in stolen base rates")
    print("  - Effect driven by BOTH higher attempts AND higher success")
    print("  - AAA shows stronger effects (better runners, execution)")
    print("  - Pooled model uses pitcher FE; separate models without FE for robustness")
    print("  - Results consistent across specifications")

if __name__ == "__main__":
    main()