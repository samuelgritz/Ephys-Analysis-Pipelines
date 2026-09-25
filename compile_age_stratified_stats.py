"""
compile_age_stratified_stats.py
================================
Compiles an Age-Stratified Master Statistics table (6-9 weeks vs 10-12 weeks)
combining two-group Mann-Whitney U tests and Linear Mixed Effects (LME) models
across electrophysiology figures.

Outputs:
    paper_data/Master_Stats_Age_Stratified.csv
    paper_data/Master_Stats_Age_Stratified.xlsx
"""

import os
import pandas as pd
import numpy as np

def p_to_sig(p):
    try:
        p = float(p)
    except (ValueError, TypeError):
        return str(p)
    if p < 0.001:  return "***"
    elif p < 0.01: return "**"
    elif p < 0.05: return "*"
    else:          return "ns"

def fmt_val(x, sig_figs=4):
    try:
        v = float(x)
        if pd.isna(v): return ""
        if v == 0: return "0"
        return f"{v:.{sig_figs}g}"
    except (ValueError, TypeError):
        return str(x)

def fmt_p(p):
    try:
        v = float(p)
        if pd.isna(v): return ""
        return v
    except (ValueError, TypeError):
        return p

def fmt_p_str(p):
    try:
        v = float(p)
        if pd.isna(v): return ""
        if v >= 0.0001:
            return f"p = {v:.4f}"
        else:
            return "p < 0.0001"
    except (ValueError, TypeError):
        return str(p)

def compile_age_stats():
    base_dir = "paper_data" if os.path.exists("paper_data") else "../paper_data"
    r_stats_path = os.path.join(base_dir, "Age_Stratified_R_Stats_Results.csv")
    
    if not os.path.exists(r_stats_path):
        print(f"Error: {r_stats_path} not found. Run Age_Stratified_Ephys_Stats.R first.")
        return
        
    df_r = pd.read_csv(r_stats_path)
    
    out_rows = []
    for idx, row in df_r.iterrows():
        fig = row["Figure"]
        sub = row["Subpanel"]
        met = row["Metric"]
        pw = row["Pathway"]
        cond = row["Condition"]
        ag = row["Age_Group"]
        
        wt_m = row["WT_Mean"]
        wt_s = row["WT_SEM"]
        wt_n = row["WT_N"]
        
        mut_m = row["I80T_Mean"]
        mut_s = row["I80T_SEM"]
        mut_n = row["I80T_N"]
        
        anv_f_g = row.get("ANOVA_F_Genotype", row.get("LME_F_Genotype", np.nan))
        anv_p_g = row.get("ANOVA_P_Genotype", row.get("LME_P_Genotype", np.nan))
        anv_f_a = row.get("ANOVA_F_Age", row.get("LME_F_Age", np.nan))
        anv_p_a = row.get("ANOVA_P_Age", row.get("LME_P_Age", np.nan))
        anv_f_i = row.get("ANOVA_F_Interaction", row.get("LME_F_Interaction", np.nan))
        anv_p_i = row.get("ANOVA_P_Interaction", row.get("LME_P_Interaction", np.nan))
        
        tuk_p_within = row.get("Tukey_P_WithinAge", np.nan)
        tuk_p_wt_age = row.get("Tukey_P_WT_Age", np.nan)
        tuk_p_mut_age = row.get("Tukey_P_Mutant_Age", np.nan)
        
        mw_u = row["PostHoc_MW_U"]
        mw_p = row["PostHoc_P_Value"]
        mw_sig = p_to_sig(tuk_p_within) if pd.notna(tuk_p_within) else row["PostHoc_Sig"]
        
        paper_fmt = f"Tukey HSD p_adj = {fmt_p_str(tuk_p_within)}; 2-Way ANOVA Geno F = {fmt_val(anv_f_g, 4)}, {fmt_p_str(anv_p_g)}; Age F = {fmt_val(anv_f_a, 4)}, {fmt_p_str(anv_p_a)}; Interaction F = {fmt_val(anv_f_i, 4)}, {fmt_p_str(anv_p_i)}"
        
        out_rows.append({
            "Figure": fig,
            "Subpanel": sub,
            "Metric": met,
            "Pathway": pw,
            "Condition": cond,
            "Age_Group": ag,
            "WT_Mean": wt_m,
            "WT_SEM": wt_s,
            "WT_N": wt_n,
            "I80T_Mean": mut_m,
            "I80T_SEM": mut_s,
            "I80T_N": mut_n,
            "Test_Used": "2-Way ANOVA with Post-Hoc Tukey HSD",
            "Tukey_P_WithinAge": fmt_p(tuk_p_within),
            "Tukey_P_WT_Age": fmt_p(tuk_p_wt_age),
            "Tukey_P_Mutant_Age": fmt_p(tuk_p_mut_age),
            "Significance": mw_sig,
            "ANOVA_Genotype_F": anv_f_g,
            "ANOVA_Genotype_P": fmt_p(anv_p_g),
            "ANOVA_Age_F": anv_f_a,
            "ANOVA_Age_P": fmt_p(anv_p_a),
            "ANOVA_Interaction_F": anv_f_i,
            "ANOVA_Interaction_P": fmt_p(anv_p_i),
            "Mann_Whitney_U": mw_u,
            "Mann_Whitney_P": fmt_p(mw_p),
            "Paper_Formatted": paper_fmt
        })
        
    master_df = pd.DataFrame(out_rows)
    
    out_csv = os.path.join(base_dir, "Master_Stats_Age_Stratified.csv")
    out_xlsx = os.path.join(base_dir, "Master_Stats_Age_Stratified.xlsx")
    
    master_df.to_csv(out_csv, index=False)
    print(f"✓ Saved CSV to: {out_csv} ({len(master_df)} rows)")
    
    with pd.ExcelWriter(out_xlsx, engine='openpyxl') as writer:
        master_df.to_excel(writer, sheet_name="Age Stratified Stats", index=False)
    print(f"✓ Saved Excel to: {out_xlsx}")

if __name__ == "__main__":
    compile_age_stats()
