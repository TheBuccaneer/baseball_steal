"""
C8: AAA 2021 Larger Bases Effect (DiD vs AA)
Classic 2x2 DiD: AAA vs AA, 2019 vs 2021

Treatment: AAA got 18" bases in 2021; AA did not (AA had defensive positioning rules)
Outcome: Stolen base rate (SB per opportunity)

Model: PPML with cluster-robust SE
Specification: sb ~ treat + post + treat:post + offset(log(opportunities))
Clustering: pitcher_id

Outputs:
- analysis/c8_bases/did_coefficients.csv
- analysis/c8_bases/did_summary.txt
- analysis/c8_bases/did_plot.png

Usage: python c8_aaa2021_bases_did.py
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

def load_and_prepare_data():
    """Load and filter data for AAA vs AA, 2019 vs 2021"""
    data_path = Path("milb_data")
    
    # Load relevant files
    files = [
        "milb_pitcher_opportunities_AAA_2019_2019.csv",
        "milb_pitcher_opportunities_AAA_2021_2021.csv",
        "milb_pitcher_opportunities_AA_2019_2019.csv",
        "milb_pitcher_opportunities_AA_2021_2021.csv"
    ]
    
    all_data = []
    for fname in files:
        fpath = data_path / fname
        if fpath.exists():
            df = pd.read_csv(fpath)
            all_data.append(df)
            print(f"Loaded: {fname}")
        else:
            print(f"WARNING: {fname} not found!")
    
    if not all_data:
        raise FileNotFoundError("No data files found!")
    
    # Combine
    panel = pd.concat(all_data, ignore_index=True)
    
    # Create treatment indicators
    panel['treat'] = (panel['level_name'] == 'AAA').astype(int)
    panel['post'] = (panel['season'] == 2021).astype(int)
    panel['did'] = panel['treat'] * panel['post']
    
    # Filter: minimum opportunities
    panel = panel[panel['total_opps'] >= 10].copy()
    
    # Add log offset
    panel['log_opps'] = np.log(panel['total_opps'])
    
    return panel

def run_did_model(data, formula, name="", use_cluster=True):
    """Run PPML DiD with optional cluster-robust SE"""
    print(f"\n{'='*80}")
    print(f"MODEL: {name}")
    print('='*80)
    
    # Fit model with cluster-robust SE
    if use_cluster:
        print("\nUsing cluster-robust standard errors (pitcher_id)...")
        model = smf.glm(
            formula=formula,
            data=data,
            family=sm.families.Poisson(),
            offset=data['log_opps']
        ).fit(cov_type='cluster', cov_kwds={'groups': data['pitcher_id']})
    else:
        model = smf.glm(
            formula=formula,
            data=data,
            family=sm.families.Poisson(),
            offset=data['log_opps']
        ).fit()
    
    print(model.summary())
    
    return model

def extract_did_effect(model):
    """Extract DiD coefficient and convert to percentage"""
    
    # Get DiD coefficient
    did_coef = model.params['treat:post']
    did_se = model.bse['treat:post']
    did_pval = model.pvalues['treat:post']
    
    # Confidence interval
    ci = model.conf_int().loc['treat:post']
    
    # Convert to percentage change
    rate_ratio = np.exp(did_coef)
    pct_change = (rate_ratio - 1) * 100
    ci_low = (np.exp(ci[0]) - 1) * 100
    ci_high = (np.exp(ci[1]) - 1) * 100
    
    result = {
        'coefficient': did_coef,
        'se': did_se,
        'rate_ratio': rate_ratio,
        'pct_change': pct_change,
        'ci_low': ci_low,
        'ci_high': ci_high,
        'pvalue': did_pval
    }
    
    return result

def create_did_plot(data, output_path):
    """Create 2-point event study plot"""
    
    # Calculate mean SB rate by level and year
    plot_data = data.groupby(['level_name', 'season']).agg({
        'sb': 'sum',
        'total_opps': 'sum'
    }).reset_index()
    
    plot_data['sb_rate'] = plot_data['sb'] / plot_data['total_opps']
    
    # Separate by level
    aaa_data = plot_data[plot_data['level_name'] == 'AAA'].sort_values('season')
    aa_data = plot_data[plot_data['level_name'] == 'AA'].sort_values('season')
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # AAA (treatment)
    ax.plot(aaa_data['season'], aaa_data['sb_rate'], 
            marker='o', linewidth=2, markersize=10, 
            label='AAA (Treatment: Larger Bases)', color='blue')
    
    # AA (control)
    ax.plot(aa_data['season'], aa_data['sb_rate'], 
            marker='s', linewidth=2, markersize=10, 
            label='AA (Control: No Base Change)', color='red')
    
    # Vertical line at treatment
    ax.axvline(2020, color='gray', linestyle='--', alpha=0.5, 
               label='Treatment Year (2021)')
    
    # Labels and formatting
    ax.set_xlabel('Season', fontsize=12)
    ax.set_ylabel('SB Rate (SB per Opportunity)', fontsize=12)
    ax.set_title('DiD: Effect of Larger Bases on Stolen Base Rate\n(AAA vs AA, 2019→2021)', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks([2019, 2021])
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    
    # Add value labels
    for _, row in aaa_data.iterrows():
        ax.text(row['season'], row['sb_rate'] + 0.003, 
                f"{row['sb_rate']:.3f}", ha='center', va='bottom', 
                fontsize=9, color='blue')
    
    for _, row in aa_data.iterrows():
        ax.text(row['season'], row['sb_rate'] - 0.003, 
                f"{row['sb_rate']:.3f}", ha='center', va='top', 
                fontsize=9, color='red')
    
    plt.tight_layout()
    plt.savefig(output_path / "did_plot.png", dpi=300, bbox_inches='tight')
    print(f"\nSaved plot: {output_path}/did_plot.png")
    plt.close()

def main():
    output_path = Path("analysis/c8_bases")
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("C8: AAA 2021 LARGER BASES EFFECT (DiD)")
    print("="*80)
    
    # Load data
    print("\nLoading and preparing data...")
    panel = load_and_prepare_data()
    
    print(f"\nFinal sample: {len(panel):,} pitcher-season observations")
    print(f"  AAA 2019: {len(panel[(panel['level_name']=='AAA') & (panel['season']==2019)]):,}")
    print(f"  AAA 2021: {len(panel[(panel['level_name']=='AAA') & (panel['season']==2021)]):,}")
    print(f"  AA 2019: {len(panel[(panel['level_name']=='AA') & (panel['season']==2019)]):,}")
    print(f"  AA 2021: {len(panel[(panel['level_name']=='AA') & (panel['season']==2021)]):,}")
    
    # ========================================================================
    # MAIN MODEL: Classic 2x2 DiD with cluster-robust SE
    # ========================================================================
    
    formula_did = "sb ~ treat + post + treat:post"
    model_main = run_did_model(panel, formula_did, "Classic DiD with Cluster-Robust SE", use_cluster=True)
    
    # Extract effect
    did_effect = extract_did_effect(model_main)
    
    print("\n" + "="*80)
    print("DiD TREATMENT EFFECT (Larger Bases)")
    print("="*80)
    print(f"\nCoefficient: {did_effect['coefficient']:.4f}")
    print(f"Standard Error: {did_effect['se']:.4f}")
    print(f"Rate Ratio: {did_effect['rate_ratio']:.3f}")
    print(f"Percentage Change: {did_effect['pct_change']:+.1f}%")
    print(f"95% CI: [{did_effect['ci_low']:+.1f}%, {did_effect['ci_high']:+.1f}%]")
    print(f"p-value: {did_effect['pvalue']:.4f}")
    
    # ========================================================================
    # ROBUSTNESS: With pitcher fixed effects
    # ========================================================================
    
    print("\n" + "="*80)
    print("ROBUSTNESS: DiD with Pitcher Fixed Effects")
    print("="*80)
    
    formula_fe = "sb ~ post + treat:post + C(pitcher_id)"
    
    try:
        model_fe = run_did_model(panel, formula_fe, "DiD with Pitcher FE + Cluster SE", use_cluster=True)
        did_effect_fe = extract_did_effect(model_fe)
        
        print("\n" + "="*80)
        print("DiD EFFECT (with Pitcher FE)")
        print("="*80)
        print(f"\nCoefficient: {did_effect_fe['coefficient']:.4f}")
        print(f"Percentage Change: {did_effect_fe['pct_change']:+.1f}%")
        print(f"95% CI: [{did_effect_fe['ci_low']:+.1f}%, {did_effect_fe['ci_high']:+.1f}%]")
        print(f"p-value: {did_effect_fe['pvalue']:.4f}")
        
    except Exception as e:
        print(f"\nWarning: Pitcher FE model did not converge: {e}")
        print("This is common with many fixed effects. Main model (without FE) is still valid.")
        did_effect_fe = None
    
    # ========================================================================
    # SAVE RESULTS
    # ========================================================================
    
    # Save coefficients
    results_df = pd.DataFrame([{
        'model': 'Classic DiD (Cluster SE)',
        **did_effect
    }])
    
    if did_effect_fe is not None:
        results_df = pd.concat([results_df, pd.DataFrame([{
            'model': 'DiD + Pitcher FE (Cluster SE)',
            **did_effect_fe
        }])], ignore_index=True)
    
    results_df.to_csv(output_path / "did_coefficients.csv", index=False)
    print(f"\nSaved coefficients: {output_path}/did_coefficients.csv")
    
    # Save full summary
    with open(output_path / "did_summary.txt", 'w') as f:
        f.write("="*80 + "\n")
        f.write("C8: AAA 2021 LARGER BASES EFFECT - MODEL SUMMARY\n")
        f.write("="*80 + "\n\n")
        f.write("Treatment: AAA received 18\" bases in 2021; AA did not\n")
        f.write("Identification: 2x2 DiD (AAA vs AA, 2019 vs 2021)\n")
        f.write("Model: PPML with cluster-robust SE (pitcher_id)\n\n")
        f.write("="*80 + "\n")
        f.write("MAIN MODEL\n")
        f.write("="*80 + "\n")
        f.write(str(model_main.summary()) + "\n\n")
        
        if did_effect_fe is not None:
            f.write("="*80 + "\n")
            f.write("ROBUSTNESS: WITH PITCHER FIXED EFFECTS\n")
            f.write("="*80 + "\n")
            f.write(str(model_fe.summary()) + "\n\n")
    
    print(f"Saved summary: {output_path}/did_summary.txt")
    
    # Create plot
    create_did_plot(panel, output_path)
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    print("\nEffect of Larger Bases (18\" vs 15\") on SB Rate:")
    print(f"  Point estimate: {did_effect['pct_change']:+.1f}%")
    print(f"  95% CI: [{did_effect['ci_low']:+.1f}%, {did_effect['ci_high']:+.1f}%]")
    print(f"  p-value: {did_effect['pvalue']:.4f}")
    
    if did_effect['pvalue'] < 0.05:
        print(f"\n  ✓ Statistically significant at 5% level")
    else:
        print(f"\n  ✗ Not statistically significant")
    
    print("\nInterpretation:")
    print("  - Larger bases reduced 1B-2B and 2B-3B distance by 4.5 inches")
    print("  - This DiD isolates the base size effect from other rule changes")
    print("  - AA is valid control (had defensive positioning rules, not base changes)")
    
    print("\nContext for Paper 0:")
    print("  - 2022 package effect (C3-C5): +27% to +47%")
    print(f"  - Bases-only effect (C8): {did_effect['pct_change']:+.1f}%")
    print("  - Remaining effect ≈ Timer + Pickoff combined")

if __name__ == "__main__":
    main()