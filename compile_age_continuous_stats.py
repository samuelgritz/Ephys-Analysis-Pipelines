"""
compile_age_continuous_stats.py
================================
Compiles an Age-Continuous Master Statistics table (Mouse Age in Weeks vs Measurements)
combining linear regressions and ANCOVA models across electrophysiology properties:
  1. Somatic Input Resistance (Input_Resistance_MOhm)
  2. F-I Curve Midpoint (Midpoint)
  3. Perforant GABAB Area (GABAB_Area)
  4. Dendritic Plateau Area (Plateau_Area)

Outputs:
    paper_data/Master_Stats_Age_Continuous.csv
    paper_data/Master_Stats_Age_Continuous.xlsx
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

def compile_age_continuous_stats():
    base_dir = "paper_data" if os.path.exists("paper_data") else "../paper_data"
    r_stats_path = os.path.join(base_dir, "Age_Continuous_R_Stats_Results.csv")
    
    if not os.path.exists(r_stats_path):
        print(f"Error: {r_stats_path} not found. Run Age_Continuous_Ephys_Stats.R first.")
        return
        
    df_r = pd.read_csv(r_stats_path)
    
    out_rows = []
    for idx, row in df_r.iterrows():
        fig = row["Figure"]
        sub = row["Subpanel"]
        met = row["Metric"]
        pw = row["Pathway"]
        cond = row["Condition"]
        
        wt_n = row["WT_N"]
        wt_slope = row["WT_Slope"]
        wt_r2 = row["WT_R2"]
        wt_p = row["WT_P_Value"]
        
        mut_n = row["I80T_N"]
        mut_slope = row["I80T_Slope"]
        mut_r2 = row["I80T_R2"]
        mut_p = row["I80T_P_Value"]
        
        anc_p_g = row["ANCOVA_P_Genotype"]
        anc_p_a = row["ANCOVA_P_Age"]
        anc_f_i = row["ANCOVA_F_Interaction"]
        anc_p_i = row["ANCOVA_P_Interaction"]
        
        paper_fmt = (f"WT Regression: Slope = {fmt_val(wt_slope, 3)}, R² = {fmt_val(wt_r2, 3)}, {fmt_p_str(wt_p)}; "
                     f"I80T/+ Regression: Slope = {fmt_val(mut_slope, 3)}, R² = {fmt_val(mut_r2, 3)}, {fmt_p_str(mut_p)}; "
                     f"ANCOVA Age × Genotype Interaction F = {fmt_val(anc_f_i, 3)}, {fmt_p_str(anc_p_i)}")
        
        out_rows.append({
            "Figure": fig,
            "Subpanel": sub,
            "Metric": met,
            "Pathway": pw,
            "Condition": cond,
            "WT_N": wt_n,
            "WT_Slope": wt_slope,
            "WT_R2": wt_r2,
            "WT_P_Value": fmt_p(wt_p),
            "WT_Significance": p_to_sig(wt_p),
            "I80T_N": mut_n,
            "I80T_Slope": mut_slope,
            "I80T_R2": mut_r2,
            "I80T_P_Value": fmt_p(mut_p),
            "I80T_Significance": p_to_sig(mut_p),
            "ANCOVA_Genotype_P": fmt_p(anc_p_g),
            "ANCOVA_Age_P": fmt_p(anc_p_a),
            "ANCOVA_Interaction_F": anc_f_i,
            "ANCOVA_Interaction_P": fmt_p(anc_p_i),
            "ANCOVA_Interaction_Sig": p_to_sig(anc_p_i),
            "Paper_Formatted": paper_fmt
        })
        
    master_df = pd.DataFrame(out_rows)
    
    out_csv = os.path.join(base_dir, "Master_Stats_Age_Continuous.csv")
    out_xlsx = os.path.join(base_dir, "Master_Stats_Age_Continuous.xlsx")
    
    master_df.to_csv(out_csv, index=False)
    print(f"✓ Saved CSV to: {out_csv} ({len(master_df)} rows)")
    
    with pd.ExcelWriter(out_xlsx, engine='openpyxl') as writer:
        master_df.to_excel(writer, sheet_name="Continuous Age Regressions", index=False)
    print(f"✓ Saved Excel to: {out_xlsx}")

if __name__ == "__main__":
    compile_age_continuous_stats()
