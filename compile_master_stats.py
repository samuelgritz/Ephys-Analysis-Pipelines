"""
compile_master_stats.py
=======================
Compiles a single master statistics table from all figure-level stats files
and raw data CSVs across the GNB1 manuscript.

Output columns (per row):
    Figure          – e.g. "Figure 1", "Figure 4", "Supplemental Figure 1"
    Subpanel        – e.g. "B", "C – Perforant", "E – ANOVA"
    Metric          – human-readable metric name
    Pathway         – anatomical pathway or "N/A"
    Condition       – drug/stimulus condition or "N/A"
    WT_Mean         – WT group mean
    WT_SEM          – WT group SEM
    WT_N            – WT sample size
    I80T_Mean       – Gnb1^I80T/+ group mean
    I80T_SEM        – Gnb1^I80T/+ group SEM
    I80T_N          – Gnb1^I80T/+ sample size
    Test_Used       – statistical test
    Statistic       – test statistic value
    P_Value         – raw p-value (or FDR-corrected for post-hocs)
    Significance    – symbol (ns / * / ** / ***)
    Notes           – extra context (effect term, ISI, interaction p, etc.)

Run:
    python compile_master_stats.py

Outputs:
    paper_data/Master_Stats_Summary.csv
    paper_data/Master_Stats_Summary.xlsx
"""

import os
import pandas as pd
import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def p_to_sig(p):
    try:
        p = float(p)
    except (ValueError, TypeError):
        return str(p)
    if p < 0.001:  return "***"
    elif p < 0.01: return "**"
    elif p < 0.05: return "*"
    else:          return "ns"


def row(figure, subpanel, metric, pathway, condition,
        wt_mean, wt_sem, wt_n, i80t_mean, i80t_sem, i80t_n,
        test_used, statistic, p_value, significance,
        notes=""):
    def _f(x): return round(float(x), 4) if pd.notna(x) else np.nan
    def _i(x): return int(x) if pd.notna(x) else np.nan
    return dict(
        Figure=figure, Subpanel=subpanel, Metric=metric,
        Pathway=pathway, Condition=condition,
        WT_Mean=_f(wt_mean), WT_SEM=_f(wt_sem),
        WT_N=_i(wt_n),
        I80T_Mean=_f(i80t_mean), I80T_SEM=_f(i80t_sem),
        I80T_N=_i(i80t_n),
        Test_Used=str(test_used),
        Statistic=statistic,
        P_Value=float(p_value) if pd.notna(p_value) else np.nan,
        Significance=str(significance),
        Notes=str(notes),
    )


rows = []


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 – Behaviour
# ══════════════════════════════════════════════════════════════════════════════
print("Building Figure 1 rows …")

df_w    = pd.read_csv("paper_data/Behavior_Analysis/Mouse_Weights_Processed.csv")
df_loc  = pd.read_csv("paper_data/Behavior_Analysis/Open_Field_Locomotion_Trial1.csv")
df_anx  = pd.read_csv("paper_data/Behavior_Analysis/Open_Field_Anxiety_Processed.csv")
df_dvc  = pd.read_csv("paper_data/DVC_Analysis/Aggregated_DVC_Data_Master.csv")
df_tmaze= pd.read_csv("paper_data/Behavior_Analysis/T_Maze_Alternations.csv")
df_s1   = pd.read_csv("paper_data/Behavior_Analysis/Stats_Results_Figure_1.csv")

# Fig 1B – weights (three timepoints)
for tp in ["P8-P10", "P28", "Adult"]:
    sub = df_w[df_w["Timepoint_Label"] == tp]
    wt  = sub[sub["Genotype"] == "WT"]["Weight_g"]
    gnb = sub[sub["Genotype"] == "GNB1"]["Weight_g"]
    st_rows = df_s1[(df_s1["Figure_Panel"] == "Fig 1B") &
                    (df_s1["Comparison"].str.contains(tp, na=False))]
    if len(st_rows) == 0: continue
    st = st_rows.iloc[0]
    rows.append(row(
        "Figure 1", f"B – {tp}", f"Body Weight ({tp})", "N/A", "N/A",
        wt.mean(), wt.sem(), len(wt), gnb.mean(), gnb.sem(), len(gnb),
        st["Test_Used"], st["Statistic"], st["P_Value"], st["Significance"],
        notes="Weight in grams"
    ))

# Fig 1C – locomotion
wt_loc  = df_loc[df_loc["Genotype"] == "WT"]["Distance (m)"]
gnb_loc = df_loc[df_loc["Genotype"] == "GNB1"]["Distance (m)"]
st = df_s1[df_s1["Figure_Panel"] == "Fig 1C"].iloc[0]
rows.append(row(
    "Figure 1", "C", "Open Field Total Distance (m)", "N/A", "N/A",
    wt_loc.mean(), wt_loc.sem(), len(wt_loc),
    gnb_loc.mean(), gnb_loc.sem(), len(gnb_loc),
    st["Test_Used"], st["Statistic"], st["P_Value"], st["Significance"]
))

# Fig 1D – anxiety ratio
wt_anx  = df_anx[df_anx["Genotype"] == "WT"]["Center_Outer_Time_Ratio"]
gnb_anx = df_anx[df_anx["Genotype"] == "GNB1"]["Center_Outer_Time_Ratio"]
st = df_s1[df_s1["Figure_Panel"] == "Fig 1D"].iloc[0]
rows.append(row(
    "Figure 1", "D", "Open Field Center:Outer Time Ratio", "N/A", "N/A",
    wt_anx.mean(), wt_anx.sem(), len(wt_anx),
    gnb_anx.mean(), gnb_anx.sem(), len(gnb_anx),
    st["Test_Used"], st["Statistic"], st["P_Value"], st["Significance"]
))

# Fig 1G – DVC dark phase
dark = df_dvc[(df_dvc["Hour"] >= 20) | (df_dvc["Hour"] < 8)]
cage_sum = dark.groupby(["Cage_ID","Genotype"])["Activity_Value"].mean().reset_index()
wt_dvc  = cage_sum[cage_sum["Genotype"] == "WT"]["Activity_Value"]
gnb_dvc = cage_sum[cage_sum["Genotype"] == "GNB1"]["Activity_Value"]
st = df_s1[df_s1["Figure_Panel"] == "Fig 1G"].iloc[0]
rows.append(row(
    "Figure 1", "G", "DVC Dark Phase Activity (a.u.)", "N/A", "N/A",
    wt_dvc.mean(), wt_dvc.sem(), len(wt_dvc),
    gnb_dvc.mean(), gnb_dvc.sem(), len(gnb_dvc),
    st["Test_Used"], st["Statistic"], st["P_Value"], st["Significance"],
    notes="Mean activity per cage across dark-phase hours"
))

# Fig 1I, 1J, 1K – load zone-entries file for distance & arm counts
df_tmaze_entries = pd.read_csv("paper_data/Behavior_Analysis/T_Maze_Zone_Entries.csv") \
    if os.path.exists("paper_data/Behavior_Analysis/T_Maze_Zone_Entries.csv") else pd.DataFrame()

if not df_tmaze_entries.empty:
    df_tmaze_entries["Total_Arm_Entries"] = (
        df_tmaze_entries["Left Arm : entries"] + df_tmaze_entries["Right Arm : entries"]
    )

for panel_key, metric_label, src_df, col in [
    ("Fig 1I", "T-Maze Distance Traveled (m)", df_tmaze_entries, "Distance (m)"),
    ("Fig 1J", "T-Maze Total Arm Entries",     df_tmaze_entries, "Total_Arm_Entries"),
    ("Fig 1K", "T-Maze Alternation %",         df_tmaze,         "Percent_Alternations"),
]:
    st_rows = df_s1[df_s1["Figure_Panel"] == panel_key]
    if len(st_rows) == 0: continue
    st = st_rows.iloc[0]
    if src_df is not None and not src_df.empty and col in src_df.columns:
        wt_v  = src_df[src_df["Genotype"] == "WT"][col].dropna()
        gnb_v = src_df[src_df["Genotype"] == "GNB1"][col].dropna()
    else:
        wt_v = gnb_v = pd.Series(dtype=float)
    rows.append(row(
        "Figure 1", panel_key.replace("Fig 1",""), metric_label, "N/A", "N/A",
        wt_v.mean()  if len(wt_v)  else np.nan,
        wt_v.sem()   if len(wt_v) > 1 else np.nan,
        len(wt_v)    if len(wt_v)  else np.nan,
        gnb_v.mean() if len(gnb_v) else np.nan,
        gnb_v.sem()  if len(gnb_v) > 1 else np.nan,
        len(gnb_v)   if len(gnb_v) else np.nan,
        st["Test_Used"], st["Statistic"], st["P_Value"], st["Significance"]
    ))

# Supplemental OLM
for _, st in df_s1[df_s1["Figure_Panel"] == "Supplemental Fig"].iterrows():
    rows.append(row(
        "Supplemental Figure (OLM)", "OLM", st["Comparison"],
        "N/A", "N/A", np.nan,np.nan,np.nan, np.nan,np.nan,np.nan,
        st["Test_Used"], st["Statistic"], st["P_Value"], st["Significance"]
    ))


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 – Intrinsic Physiology
# ══════════════════════════════════════════════════════════════════════════════
print("Building Figure 2 rows …")

df_phys = pd.read_csv("paper_data/Physiology_Analysis/intrinsic_properties.csv")
df_ap   = pd.read_csv("paper_data/Physiology_Analysis/combined_AP_AHP_rheobase_analysis.csv")
df_fi   = pd.read_csv("paper_data/Firing_Rate/Firing_Rates_midpoints.csv")
df_fi_isi = pd.read_csv("paper_data/Firing_Rate/FI_ISI_Stats_Complete.csv")
df_isi_a  = pd.read_csv("paper_data/Firing_Rate/ISI_Adaptation_2way_ANOVA.csv")
df_s2   = pd.read_csv("paper_data/Physiology_Analysis/Stats_Results_Figure_2.csv")

phys_map = {
    "Input Resistance": ("Input_Resistance_MOhm", df_phys, "A", "Input Resistance (MΩ)"),
    "Voltage Sag":      ("Voltage_sag",           df_phys, "A", "Voltage Sag (mV)"),
    "Vm Rest":          ("Vm rest/start (mV)",    df_phys, "A", "Resting Vm (mV)"),
    "Rheobase":         ("Rheobase_Current",       df_ap,   "C", "Rheobase Current (pA)"),
    "AP Threshold":     ("AP_threshold",           df_ap,   "C", "AP Threshold (mV)"),
    "AP Size":          ("AP_size",                df_ap,   "C", "AP Amplitude (mV)"),
    "AP Halfwidth":     ("AP_halfwidth",           df_ap,   "C", "AP Halfwidth (ms)"),
    "AHP Amplitude":    ("AHP_size",               df_ap,   "E", "AHP Amplitude (mV)"),
    "AHP Decay":        ("decay_area",             df_ap,   "E", "AHP Decay Area (mV·ms)"),
    "Access Resistance": ("Access Resistance (From Whole Cell V-Clamp)", df_phys, "QC", "Access Resistance (MΩ)"),
}
for comp_key, (col, df_src, subp, mlabel) in phys_map.items():
    st_rows = df_s2[df_s2["Comparison"] == comp_key]
    if len(st_rows) == 0: continue
    st  = st_rows.iloc[0]
    wt  = df_src[df_src["Genotype"] == "WT"][col].dropna()
    gnb = df_src[df_src["Genotype"] == "GNB1"][col].dropna()
    rows.append(row(
        "Figure 2", subp, mlabel, "N/A", "N/A",
        wt.mean(), wt.sem(), len(wt), gnb.mean(), gnb.sem(), len(gnb),
        st["Test_Used"], st["Statistic"], st["P_Value"], st["Significance"]
    ))

# F-I midpoint
st_fi = df_s2[df_s2["Comparison"] == "F-I Curve Midpoint: WT vs GNB1"].iloc[0]
wt_fi  = df_fi[df_fi["Genotype"] == "WT"]["FI_Midpoint"].dropna()
gnb_fi = df_fi[df_fi["Genotype"] == "GNB1"]["FI_Midpoint"].dropna()
rows.append(row(
    "Figure 2", "F", "F-I Curve Midpoint (pA)", "N/A", "N/A",
    wt_fi.mean(), wt_fi.sem(), len(wt_fi), gnb_fi.mean(), gnb_fi.sem(), len(gnb_fi),
    st_fi["Test_Used"], st_fi["Statistic"], st_fi["P_Value"], st_fi["Significance"]
))

# F-I 2-way RM ANOVA model terms
for _, st in df_fi_isi[df_fi_isi["Analysis"] == "F-I Curve"].iterrows():
    rows.append(row(
        "Figure 2", "E", f"F-I Curve ANOVA: {st['Comparison']}", "N/A", "N/A",
        np.nan,np.nan,np.nan, np.nan,np.nan,np.nan,
        st["Test"], st.get("F_value",np.nan), st["p_value"], st["significance"],
        notes="2-way RM ANOVA model term"
    ))

# ISI adaptation ANOVA
for _, st in df_isi_a.iterrows():
    rows.append(row(
        "Figure 2", "I", f"ISI Adaptation ANOVA: {st['Term']}", "N/A", "N/A",
        np.nan,np.nan,np.nan, np.nan,np.nan,np.nan,
        "2-way RM ANOVA", st["F_value"], st["p_value"], st["significance"],
        notes="Model term"
    ))


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 – Morphology
# ══════════════════════════════════════════════════════════════════════════════
print("Building Figure 3 rows …")

df_morph = pd.read_csv("paper_data/Morphology_Analysis/Dendrite_Properties_All.csv")
df_s3    = pd.read_csv("paper_data/Morphology_Analysis/Stats_Results_Figure_3.csv")

morph_map = [
    ("Fig 3D",        "Basal Sholl Distribution (KS): WT vs GNB1",  "Basal Sholl (KS test)",            "Basal",  "branch_sum"),
    ("Fig 3E",        "Apical Sholl Distribution (KS): WT vs GNB1", "Apical Sholl (KS test)",           "Apical", "branch_sum"),
    ("Fig 3F (Left)", "Basal Total Branch Length",                   "Total Dendritic Length (μm)",      "Basal",  "branch_sum"),
    ("Fig 3F (Right)","Apical Total Branch Length",                  "Total Dendritic Length (μm)",      "Apical", "branch_sum"),
    ("Fig 3G (Left)", "Basal Terminal Branches",                     "Number of Terminal Branches",      "Basal",  "N_terminal_branches"),
    ("Fig 3G (Right)","Apical Terminal Branches",                    "Number of Terminal Branches",      "Apical", "N_terminal_branches"),
]
for panel_key, comp, mlabel, dtype, col in morph_map:
    st_rows = df_s3[df_s3["Figure_Panel"] == panel_key]
    if len(st_rows) == 0: continue
    st  = st_rows.iloc[0]
    sub = df_morph[df_morph["Dendrite_Type"] == dtype]
    wt  = sub[sub["Genotype"] == "WT"][col].dropna()
    gnb = sub[sub["Genotype"] == "GNB1"][col].dropna()
    rows.append(row(
        "Figure 3", panel_key.replace("Fig 3","").strip(),
        mlabel, dtype, "N/A",
        wt.mean(), wt.sem(), len(wt), gnb.mean(), gnb.sem(), len(gnb),
        st["Test_Used"], st["Statistic"], st["P_Value"], st["Significance"]
    ))


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 4 – Unitary E:I (ISI 300 ms)
# ══════════════════════════════════════════════════════════════════════════════
print("Building Figure 4 rows …")

df_ei = pd.read_csv("paper_data/E_I_data/E_I_amplitudes.csv")
df_s4 = pd.read_csv("paper_data/E_I_data/Stats_Results_Figure_4.csv")
uni   = df_ei[df_ei["ISI"] == 300].copy()

panel_map4 = {
    "Gabazine_Amplitude":             ("C", "EPSP Amplitude – Gabazine condition (mV)"),
    "Estimated_Inhibition_Amplitude": ("D", "GABAA-mediated Inhibition Amplitude (mV)"),
    "GABAB_Area":                     ("E", "GABAB-mediated Slow IPSP Area (mV·ms)"),
}
for _, st in df_s4.iterrows():
    mcol = st["Metric"]
    if mcol not in panel_map4: continue
    subp, mlabel = panel_map4[mcol]
    pw  = st["Pathway"]
    sub = uni[uni["Pathway"] == pw].dropna(subset=[mcol])
    wt  = sub[sub["Genotype"] == "WT"][mcol]
    gnb = sub[sub["Genotype"] == "GNB1"][mcol]
    rows.append(row(
        "Figure 4", f"{subp} – {pw}", mlabel, pw, "Unitary (ISI 300 ms)",
        wt.mean(), wt.sem(), len(wt), gnb.mean(), gnb.sem(), len(gnb),
        st["Test_Used"], st["Statistic"], st["P_Value"], st["Significance"]
    ))


# ══════════════════════════════════════════════════════════════════════════════
# FIGURES 5 & 6 – Frequency-Dependent E:I + Supralinearity
# THREE LAYERS:
#   (1) LME Type III ANOVA model terms (Genotype, ISI_Time, Genotype:ISI_Time)
#   (2) FDR-corrected per-ISI post-hoc genotype contrasts
# ══════════════════════════════════════════════════════════════════════════════
print("Building Figure 5 & 6 rows …")

df_anova56 = pd.read_csv("paper_data/E_I_data/Figure_5_6_All_Stats_ANOVA.csv")
df_fdr56   = pd.read_csv("paper_data/E_I_data/Figure_5_6_All_Stats_FDR_Corrected.csv")

# Metric → (Figure, Subpanel)
METRIC_FIG56 = {
    "Gabazine_Amplitude":             ("Figure 5", "B"),
    "Estimated_Inhibition_Amplitude": ("Figure 5", "C"),
    "Inhibition_Amplitude":           ("Figure 5", "C"),   # name used in ANOVA/FDR files
    "GABAB_Area":                     ("Figure 5", "D"),
    "Gabazine_Supralinearity":        ("Figure 6", "C/D/E"),
}
METRIC_LABEL56 = {
    "Gabazine_Amplitude":             "EPSP Amplitude – Gabazine condition (mV)",
    "Estimated_Inhibition_Amplitude": "GABAA Inhibition Amplitude (mV)",
    "Inhibition_Amplitude":           "GABAA Inhibition Amplitude (mV)",
    "GABAB_Area":                     "GABAB Slow IPSP Area (mV·ms)",
    "Gabazine_Supralinearity":        "Supralinearity (Measured − Expected, mV)",
}
RAW_COL56 = {
    "Gabazine_Amplitude":             "Gabazine_Amplitude",
    "Estimated_Inhibition_Amplitude": "Estimated_Inhibition_Amplitude",
    "Inhibition_Amplitude":           "Estimated_Inhibition_Amplitude",  # maps to raw data column
    "GABAB_Area":                     "GABAB_Area",
    "Gabazine_Supralinearity":        "Gabazine_Supralinearity",
}
EFFECT_LABEL56 = {
    "Genotype":          "Main Effect – Genotype",
    "ISI_Time":          "Main Effect – ISI",
    "Genotype:ISI_Time": "Interaction – Genotype × ISI",
}

# ── LAYER 1: LME Type III ANOVA model terms ──────────────────────────────────
# Exact columns in ANOVA file:
#   Analysis, Pathway, Comparison, Effect,
#   Mean_GNB1, Mean_WT, SEM_GNB1, SEM_WT,
#   NumDF, DenDF, "F value", P_Value, Significant
for _, st in df_anova56.iterrows():
    metric = st["Analysis"]
    pw     = st["Pathway"]
    comp   = st["Comparison"]
    effect = st["Effect"]

    if metric not in METRIC_FIG56:           continue
    if "WT_vs_GNB1" not in str(comp):        continue
    if effect not in EFFECT_LABEL56:          continue

    fig_label, subp = METRIC_FIG56[metric]
    num_df = st["NumDF"]
    den_df = st["DenDF"]
    f_val  = st["F value"]    # <── column name has a space
    p_val  = st["P_Value"]
    sig    = st.get("Significant", p_to_sig(p_val))

    rows.append(row(
        fig_label,
        f"{subp} – {pw} – ANOVA",
        METRIC_LABEL56[metric],
        pw,
        f"All ISIs – {EFFECT_LABEL56[effect]}",
        st.get("Mean_WT",   np.nan), st.get("SEM_WT",   np.nan), np.nan,
        st.get("Mean_GNB1", np.nan), st.get("SEM_GNB1", np.nan), np.nan,
        "LME Type III ANOVA (lmerTest)",
        f"F({int(num_df) if pd.notna(num_df) else '?'},{round(den_df,1) if pd.notna(den_df) else '?'})={round(f_val,3) if pd.notna(f_val) else '?'}",
        p_val, sig,
        notes=f"ANOVA term: {effect}"
    ))

# ── LAYER 2: FDR-corrected per-ISI post-hoc contrasts ────────────────────────
# Exact columns in FDR file:
#   Analysis, Pathway, Comparison, ISI, FDR_Group,
#   estimate, SE, df, t_ratio,
#   Main_Effect_p, Interaction_p,
#   p_value_uncorrected, Significant_Uncorrected,
#   p_value_FDR, Significant_FDR
for _, st in df_fdr56.iterrows():
    metric  = st["Analysis"]
    pw      = st["Pathway"]
    comp    = st["Comparison"]
    isi_raw = str(st["ISI"])

    if metric not in METRIC_FIG56:           continue
    if "WT_vs_GNB1" not in str(comp):        continue

    fig_label, subp = METRIC_FIG56[metric]

    isi_val = None
    for v in [10, 25, 50, 100, 300]:
        if str(v) in isi_raw:
            isi_val = v
            break

    raw_col = RAW_COL56.get(metric, metric)
    if isi_val is not None and raw_col in df_ei.columns:
        sub_ei  = df_ei[(df_ei["Pathway"] == pw) & (df_ei["ISI"] == isi_val)].dropna(subset=[raw_col])
        wt_sub  = sub_ei[sub_ei["Genotype"] == "WT"][raw_col]
        gnb_sub = sub_ei[sub_ei["Genotype"] == "GNB1"][raw_col]
    else:
        wt_sub = gnb_sub = pd.Series(dtype=float)

    t_ratio  = st["t_ratio"]           # exact name
    df_val   = st["df"]
    p_fdr    = st["p_value_FDR"]       # exact name (capital FDR)
    sig_fdr  = st["Significant_FDR"]   # exact name
    main_p   = st["Main_Effect_p"]
    inter_p  = st["Interaction_p"]

    rows.append(row(
        fig_label,
        f"{subp} – {pw} – ISI {isi_val} ms",
        METRIC_LABEL56[metric],
        pw, f"ISI {isi_val} ms",
        wt_sub.mean()  if len(wt_sub)  else np.nan,
        wt_sub.sem()   if len(wt_sub) > 1 else np.nan,
        len(wt_sub)    if len(wt_sub)  else np.nan,
        gnb_sub.mean() if len(gnb_sub) else np.nan,
        gnb_sub.sem()  if len(gnb_sub) > 1 else np.nan,
        len(gnb_sub)   if len(gnb_sub) else np.nan,
        "LME FDR-corrected post-hoc (R lmerTest)",
        f"t({round(df_val,1) if pd.notna(df_val) else '?'})={round(t_ratio,3) if pd.notna(t_ratio) else '?'}",
        p_fdr, sig_fdr,
        notes=(
            f"FDR post-hoc at ISI {isi_val} ms | "
            f"Genotype main effect p={round(main_p,4) if pd.notna(main_p) else 'NA'} | "
            f"Interaction p={round(inter_p,4) if pd.notna(inter_p) else 'NA'}"
        )
    ))


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 7 – Theta Burst / Dendritic Excitability
# ══════════════════════════════════════════════════════════════════════════════
print("Building Figure 7 rows …")

df_s7 = pd.read_csv("paper_data/Plateau_data/Stats_Results_Figure_7.csv")

panel_map7 = {"Fig 6C": "C", "Fig 6F": "E", "Fig 6G": "E"}

for _, st in df_s7.iterrows():
    comp = st["Comparison"]
    subp = panel_map7.get(st.get("Figure_Panel",""), "?")

    pw = "N/A"
    for kw in ["Both","Schaffer","Perforant","CA3","ECIII"]:
        if kw in comp:
            pw = kw; break

    if "Plateau Area" in comp:
        mlabel = "Plateau Area (mV·s)"; subp = "C"
    elif "Supralinear Total AUC" in comp:
        mlabel = "Supralinear Total AUC (mV·s)"; subp = "E"
    elif "Cycle" in comp:
        mlabel = "Supralinear AUC – Per Cycle (mV·s)"; subp = "E"
    else:
        mlabel = comp

    cond = ("TBS – Simultaneous" if "Both" in comp
            else "TBS – ECIII/Perforant only" if ("ECIII" in comp or "Perforant" in comp)
            else "TBS – Schaffer only")

    rows.append(row(
        "Figure 7", f"{subp} – {comp.split(':')[0].strip()}",
        mlabel, pw, cond,
        st["Mean_WT"], st["SEM_WT"],
        st["N_WT"] if pd.notna(st["N_WT"]) else np.nan,
        st["Mean_GNB1"], st["SEM_GNB1"],
        st["N_GNB1"] if pd.notna(st["N_GNB1"]) else np.nan,
        st["Test_Used"], st["Statistic"], st["P_Value"], st["Significance"]
    ))


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 8 – GIRK / GABAB Pharmacology
# ══════════════════════════════════════════════════════════════════════════════
print("Building Figure 8 rows …")

df_s8     = pd.read_csv("paper_data/Plateau_data/Stats_Results_Figure_8.csv")
df_bac    = pd.read_csv("paper_data/gabab_analysis/Baclofen_Vm_Change.csv")
df_plat_d = pd.read_csv("paper_data/Plateau_data/Plateau_Delta_GIRK.csv")
df_uni_d  = pd.read_csv("paper_data/Plateau_data/GIRK_Unitary_GABAB_Deltas.csv")

def _sem(df, g_col, g_val, v_col):
    g = df[df[g_col] == g_val][v_col].dropna()
    return g.sem() if len(g) > 1 else np.nan

for _, st in df_s8.iterrows():
    drug = st["Drug"]
    pw   = st.get("Pathway","N/A")
    comp = st["Comparison"]

    if "ΔVm" in comp:
        wt_s  = _sem(df_bac, "Genotype","WT","Voltage Change")
        gnb_s = _sem(df_bac, "Genotype","GNB1","Voltage Change")
        subp, mlabel, cond = "C", "Baclofen ΔVm (mV)", "Baclofen 10 μM"
    elif "Plateau Delta" in comp:
        sub_p = df_plat_d[(df_plat_d["Drug"]==drug)&(df_plat_d["Pathway"]==pw)]
        wt_s  = _sem(sub_p, "Genotype","WT","Delta_Area")
        gnb_s = _sem(sub_p, "Genotype","GNB1","Delta_Area")
        subp  = "I" if drug=="ML297" else "L"
        mlabel = f"Δ Plateau Area – {drug} (mV·s)"
        cond   = f"{drug} – TBS"
    elif "Unitary Delta" in comp:
        sub_u = df_uni_d[(df_uni_d["Drug"]==drug)&(df_uni_d["Pathway"]==pw)]
        wt_s  = _sem(sub_u, "Genotype","WT","Delta_GABAB_Area")
        gnb_s = _sem(sub_u, "Genotype","GNB1","Delta_GABAB_Area")
        subp = "F"
        mlabel = f"Δ Unitary GABAB Area – {drug}"
        cond   = f"{drug} – Unitary"
    else:
        wt_s = gnb_s = np.nan
        subp, mlabel, cond = "?", comp, drug

    rows.append(row(
        "Figure 8", f"{subp} – {drug} ({pw})",
        mlabel, pw, cond,
        st["WT_mean"], wt_s, st["WT_n"],
        st["GNB1_mean"], gnb_s, st["GNB1_n"],
        st["test_type"], st["test_stat"], st["p_value"], st["Significance"]
    ))


# ══════════════════════════════════════════════════════════════════════════════
# SUPPLEMENTAL FIGURE 1 – E:I Imbalance Index
# ══════════════════════════════════════════════════════════════════════════════
print("Building Supplemental Figure 1 rows …")

df_anova_ei = pd.read_csv("paper_data/E_I_data/Figure_5_6_All_Stats_ANOVA.csv")
imb = df_anova_ei[df_anova_ei["Analysis"] == "E_I_Imbalance"]

for _, st in imb.iterrows():
    pw  = st["Pathway"]
    eff = st["Effect"]
    
    if eff not in EFFECT_LABEL56: continue
    cond_label = f"All ISIs – {EFFECT_LABEL56[eff]}"
    
    sub = df_ei[df_ei["Pathway"] == pw].dropna(
        subset=["Gabazine_Amplitude","Estimated_Inhibition_Amplitude"]).copy()
    sub["EI_ratio"] = (sub["Gabazine_Amplitude"] /
                       (sub["Gabazine_Amplitude"] + sub["Estimated_Inhibition_Amplitude"].abs()))
    wt_cells  = sub[sub["Genotype"]=="WT"]["Cell_ID"].unique()
    gnb_cells = sub[sub["Genotype"]=="GNB1"]["Cell_ID"].unique()
    wt_data   = sub[sub["Genotype"]=="WT"]["EI_ratio"]
    gnb_data  = sub[sub["Genotype"]=="GNB1"]["EI_ratio"]
    
    f_val  = st["F value"]
    num_df = st["NumDF"]
    den_df = st["DenDF"]
    stat_str = f"F({int(num_df) if pd.notna(num_df) else '?'},{round(den_df,1) if pd.notna(den_df) else '?'})={round(f_val,3) if pd.notna(f_val) else '?'}"
    
    rows.append(row(
        "Supplemental Figure 1", f"E:I Imbalance – {pw}",
        "E:I Imbalance Index (EPSP / (EPSP+|IPSP|))", pw, cond_label,
        wt_data.mean() if eff=="Genotype" else np.nan, 
        wt_data.sem() if eff=="Genotype" else np.nan, 
        len(wt_cells),
        gnb_data.mean() if eff=="Genotype" else np.nan, 
        gnb_data.sem() if eff=="Genotype" else np.nan, 
        len(gnb_cells),
        "LME Type III ANOVA (lmerTest)", stat_str,
        st["P_Value"], st["Significant"],
        notes=f"ANOVA term: {eff}"
    ))


# ══════════════════════════════════════════════════════════════════════════════
# SUPPLEMENTAL FIGURE 3 – GNB1 Protein Levels
# ══════════════════════════════════════════════════════════════════════════════
print("Building Supplemental Figure 3 rows …")

df_supp3 = pd.read_csv("paper_data/Stats_Results_Supplemental_Figure_3.csv")
for _, st in df_supp3.iterrows():
    rows.append(row(
        "Supplemental Figure 3", st["Figure_Panel"],
        st["Comparison"], "Hippocampus", "N/A",
        np.nan,np.nan,np.nan, np.nan,np.nan,np.nan,
        st["Test_Used"], st["Statistic"], st["P_Value"], st["Significance"],
        notes="GNB1 protein levels (Western blot)"
    ))


# ══════════════════════════════════════════════════════════════════════════════
# BUILD & SAVE
# ══════════════════════════════════════════════════════════════════════════════
print("Assembling master table …")

COLUMNS = [
    "Figure","Subpanel","Metric","Pathway","Condition",
    "WT_Mean","WT_SEM","WT_N",
    "I80T_Mean","I80T_SEM","I80T_N",
    "Test_Used","Statistic","P_Value","Significance",
    "Notes",
]

df_master = pd.DataFrame(rows, columns=COLUMNS)
for c in ["WT_Mean","WT_SEM","I80T_Mean","I80T_SEM"]:
    df_master[c] = pd.to_numeric(df_master[c], errors="coerce").round(4)

fig_order = {
    "Figure 1":1,"Figure 2":2,"Figure 3":3,
    "Figure 4":4,"Figure 5":5,"Figure 6":6,
    "Figure 7":7,"Figure 8":8,
    "Supplemental Figure 1":9,
    "Supplemental Figure (OLM)":10,
    "Supplemental Figure 3":11,
}
df_master["_s"] = df_master["Figure"].map(lambda x: fig_order.get(x,99))
df_master = (df_master.sort_values(["_s","Subpanel"])
             .drop(columns=["_s"]).reset_index(drop=True))

out_csv  = "paper_data/Master_Stats_Summary.csv"
out_xlsx = "paper_data/Master_Stats_Summary.xlsx"

df_master.to_csv(out_csv, index=False)
print(f"  Saved CSV  → {out_csv}  ({len(df_master)} rows)")

try:
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        df_master.to_excel(writer, sheet_name="All Figures", index=False)
        for fig_name in df_master["Figure"].unique():
            sub_df    = df_master[df_master["Figure"] == fig_name]
            safe_name = (fig_name
                         .replace("Supplemental ","Supp ")
                         .replace("Figure ","Fig ")
                         .replace("/","_"))[:31]
            sub_df.to_excel(writer, sheet_name=safe_name, index=False)
        for sheet in writer.sheets.values():
            for col_cells in sheet.columns:
                max_len = max((len(str(c.value or "")) for c in col_cells), default=10)
                sheet.column_dimensions[col_cells[0].column_letter].width = min(max_len+2, 50)
    print(f"  Saved XLSX → {out_xlsx}  ({len(df_master['Figure'].unique())} sheets)")
except ImportError:
    print("  openpyxl not installed – run: pip install openpyxl")

print("\nDone.")
print(df_master[["Figure","Subpanel","Metric","Statistic","P_Value","Significance"]].to_string())
