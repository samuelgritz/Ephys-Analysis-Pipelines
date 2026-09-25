"""
compile_master_stats.py
=======================
Compiles a single master statistics table from all figure-level stats files
and raw data CSVs across the GNB1 manuscript.

Output columns (per row):
    Figure              – e.g. "Figure 1", "Figure 4", "Supplemental Figure 1"
    Subpanel            – e.g. "B", "C – Perforant", "E – ANOVA"
    Metric              – human-readable metric name
    Pathway             – anatomical pathway or "N/A"
    Condition           – drug/stimulus condition or "N/A"
    WT_Mean             – WT group mean
    WT_SEM              – WT group SEM
    WT_N                – WT sample size (number of cells or mice)
    WT_N_Animals        – WT number of animals (unique mice / dates)
    I80T_Mean           – Gnb1^I80T/+ group mean
    I80T_SEM            – Gnb1^I80T/+ group SEM
    I80T_N              – Gnb1^I80T/+ sample size (number of cells or mice)
    I80T_N_Animals      – Gnb1^I80T/+ number of animals (unique mice / dates)
    Test_Used           – statistical test
    Statistic           – test statistic value
    Degrees_of_Freedom  – degrees of freedom (n1+n2-2 for MWU; NumDF,DenDF for ANOVA)
    P_Value             – raw p-value (or FDR-corrected for post-hocs)
    Significance        – symbol (ns / * / ** / ***)
    Notes               – extra context (effect term, ISI, interaction p, etc.)

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


def _4sig(x):
    """Return a number formatted to 4 significant figures as a string."""
    try:
        v = float(x)
    except (ValueError, TypeError):
        return str(x)
    if v == 0:
        return "0"
    return f"{v:.4g}"


def fmt_stat(s):
    """Normalise a statistic value for display."""
    if pd.isna(s) or s is None:
        return s
    s = str(s).strip()

    import re
    m = re.match(r'^([Fft]\(.*?\)=)(.+)$', s)
    if m:
        prefix, num_part = m.group(1), m.group(2)
        return prefix + _4sig(num_part)

    try:
        v = float(s)
        frac = v - int(v)
        if frac == 0.0:
            return str(int(v))
        elif frac == 0.5:
            return f"{v:.1f}"
        else:
            return _4sig(v)
    except ValueError:
        return s


def fmt_p(p):
    """Store the raw p-value as a float — no rounding."""
    try:
        return float(p)
    except (ValueError, TypeError):
        return p


def fmt_p_display(p):
    """Return a human-readable p-value string — always 4 decimal places."""
    try:
        v = float(p)
        if v >= 0.0001:
            return f"{v:.4f}"
        else:
            return f"{v:.4f} (p<0.0001)"
    except (ValueError, TypeError):
        return str(p)


def _jn_p(p):
    """Format p-value for JNeurosci Paper_Formatted column."""
    try:
        v = float(p)
        if v >= 0.0001:
            return f"p = {v:.4f}"
        else:
            return "p < 0.0001"
    except (ValueError, TypeError):
        return f"p = {p}"


def fmt_paper_str(test_used, statistic, p_value):
    """Build a complete JNeurosci manuscript-ready string."""
    import re
    if pd.isna(p_value) or pd.isna(statistic):
        return ""

    stat_s = str(statistic).strip()
    p_str  = _jn_p(p_value)

    m = re.match(r'^([Fft]\(.*?\))=(.+)$', stat_s)
    if m:
        prefix, val = m.group(1), m.group(2)
        return f"{prefix} = {val}, {p_str}"

    test_lower = str(test_used).lower()
    if 'wilcoxon' in test_lower:
        prefix = 'W'
    elif 'mann-whitney' in test_lower or 'mann whitney' in test_lower:
        prefix = 'U'
    elif 'ks' in test_lower or 'kolmogorov' in test_lower:
        prefix = 'D'
    else:
        prefix = 'stat'

    return f"{prefix} = {stat_s}, {p_str}"


def get_n_animals(series_or_df, cell_col='Cell_ID'):
    """
    Extract count of unique animals from a Series of Cell_IDs or a DataFrame.
    For ephys data, each cell ID has format YYYYMMDD_cN where YYYYMMDD is the animal (recording date).
    For behavior data where each row is one animal, returns length or unique animal IDs.
    """
    if series_or_df is None:
        return np.nan
    if isinstance(series_or_df, pd.DataFrame):
        if cell_col in series_or_df.columns:
            s = series_or_df[cell_col].dropna().astype(str)
        elif 'Subject' in series_or_df.columns:
            s = series_or_df['Subject'].dropna().astype(str)
        elif 'Animal' in series_or_df.columns:
            s = series_or_df['Animal'].dropna().astype(str)
        else:
            return len(series_or_df)
    else:
        s = pd.Series(series_or_df).dropna().astype(str)

    if len(s) == 0:
        return np.nan
    animals = s.str.replace(r'_c[0-9]+$', '', regex=True)
    return int(animals.nunique())


def row(figure, subpanel, metric, pathway, condition,
        wt_mean, wt_sem, wt_n, wt_n_animals,
        i80t_mean, i80t_sem, i80t_n, i80t_n_animals,
        test_used, statistic, p_value, significance,
        notes="", degrees_of_freedom=None):
    def _f(x):
        try:
            return float(x)
        except (ValueError, TypeError):
            return np.nan
    def _s(x):
        try:
            return float(x)
        except (ValueError, TypeError):
            return np.nan
    def _i(x):
        if isinstance(x, str):
            return x
        return int(x) if pd.notna(x) else np.nan

    return dict(
        Figure=figure, Subpanel=subpanel, Metric=metric,
        Pathway=pathway, Condition=condition,
        WT_Mean=_f(wt_mean), WT_SEM=_s(wt_sem),
        WT_N=_i(wt_n), WT_N_Animals=_i(wt_n_animals),
        I80T_Mean=_f(i80t_mean), I80T_SEM=_s(i80t_sem),
        I80T_N=_i(i80t_n), I80T_N_Animals=_i(i80t_n_animals),
        Test_Used=str(test_used),
        Statistic=fmt_stat(statistic),
        Degrees_of_Freedom=degrees_of_freedom if degrees_of_freedom is not None else np.nan,
        P_Value=fmt_p(p_value) if pd.notna(p_value) else np.nan,
        P_Value_Formatted=fmt_p_display(p_value) if pd.notna(p_value) else "",
        Paper_Formatted=fmt_paper_str(test_used, fmt_stat(statistic), p_value),
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
df_dvc_cages = pd.read_csv("paper_data/DVC_Analysis/Cage_Specific_Hours_Summary.csv")
df_tmaze= pd.read_csv("paper_data/Behavior_Analysis/T_Maze_Alternations.csv")
df_s1   = pd.read_csv("paper_data/Behavior_Analysis/Stats_Results_Figure_1.csv")

# Fig 1B – weights (three timepoints)
for tp in ["P8-P10", "P28", "Adult"]:
    sub = df_w[df_w["Timepoint_Label"] == tp]
    wt  = sub[sub["Genotype"] == "WT"]["Weight_g"].dropna()
    gnb = sub[sub["Genotype"] == "GNB1"]["Weight_g"].dropna()
    st_rows = df_s1[(df_s1["Figure_Panel"] == "Fig 1B") &
                    (df_s1["Comparison"].str.contains(tp, na=False))]
    if len(st_rows) == 0: continue
    st = st_rows.iloc[0]
    rows.append(row(
        "Figure 1", f"B – {tp}", f"Body Weight ({tp})", "N/A", "N/A",
        wt.mean(), wt.sem(), len(wt), len(wt),
        gnb.mean(), gnb.sem(), len(gnb), len(gnb),
        st["Test_Used"], st["Statistic"], st["P_Value"], st["Significance"],
        notes="Weight in grams"
    ))

# Fig 1D – locomotion
wt_loc  = df_loc[df_loc["Genotype"] == "WT"]["Distance (m)"].dropna()
gnb_loc = df_loc[df_loc["Genotype"] == "GNB1"]["Distance (m)"].dropna()
st = df_s1[df_s1["Figure_Panel"] == "Fig 1C"].iloc[0]
rows.append(row(
    "Figure 1", "D", "Open Field Total Distance (m)", "N/A", "N/A",
    wt_loc.mean(), wt_loc.sem(), len(wt_loc), len(wt_loc),
    gnb_loc.mean(), gnb_loc.sem(), len(gnb_loc), len(gnb_loc),
    st["Test_Used"], st["Statistic"], st["P_Value"], st["Significance"]
))

# Fig 1E – anxiety ratio
wt_anx  = df_anx[df_anx["Genotype"] == "WT"]["Center_Outer_Time_Ratio"].dropna()
gnb_anx = df_anx[df_anx["Genotype"] == "GNB1"]["Center_Outer_Time_Ratio"].dropna()
st = df_s1[df_s1["Figure_Panel"] == "Fig 1D"].iloc[0]
rows.append(row(
    "Figure 1", "E", "Open Field % Time in Center", "N/A", "N/A",
    wt_anx.mean(), wt_anx.sem(), len(wt_anx), len(wt_anx),
    gnb_anx.mean(), gnb_anx.sem(), len(gnb_anx), len(gnb_anx),
    st["Test_Used"], st["Statistic"], st["P_Value"], st["Significance"]
))

# Fig 1G – DVC total activity in dark phase
wt_dvc  = df_dvc_cages[df_dvc_cages["Genotype"] == "WT"]["Sum_All_Dark"].dropna()
gnb_dvc = df_dvc_cages[df_dvc_cages["Genotype"] == "GNB1"]["Sum_All_Dark"].dropna()
st = df_s1[df_s1["Figure_Panel"] == "Fig 1G"].iloc[0]
rows.append(row(
    "Figure 1", "G", "Total Activity in Dark Phase (m)", "N/A", "N/A",
    wt_dvc.mean(), wt_dvc.sem(), len(wt_dvc), len(wt_dvc),
    gnb_dvc.mean(), gnb_dvc.sem(), len(gnb_dvc), len(gnb_dvc),
    st["Test_Used"], st["Statistic"], st["P_Value"], st["Significance"],
    notes="Summed cage activity across dark-phase hours"
))

# Fig 1I, 1J, 1K – T-Maze panels
df_tmaze_entries = pd.read_csv("paper_data/Behavior_Analysis/T_Maze_Zone_Entries.csv") \
    if os.path.exists("paper_data/Behavior_Analysis/T_Maze_Zone_Entries.csv") else pd.DataFrame()

if not df_tmaze_entries.empty:
    df_tmaze_entries["Total_Arm_Entries"] = (
        df_tmaze_entries["Left Arm : entries"] + df_tmaze_entries["Right Arm : entries"]
    )

TMAZE_PANEL_MAP = {"Fig 1I": "I", "Fig 1J": "J", "Fig 1K": "K"}

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
        "Figure 1", TMAZE_PANEL_MAP[panel_key], metric_label, "N/A", "N/A",
        wt_v.mean()  if len(wt_v)  else np.nan,
        wt_v.sem()   if len(wt_v) > 1 else np.nan,
        len(wt_v)    if len(wt_v)  else np.nan,
        len(wt_v)    if len(wt_v)  else np.nan,
        gnb_v.mean() if len(gnb_v) else np.nan,
        gnb_v.sem()  if len(gnb_v) > 1 else np.nan,
        len(gnb_v)   if len(gnb_v) else np.nan,
        len(gnb_v)   if len(gnb_v) else np.nan,
        st["Test_Used"], st["Statistic"], st["P_Value"], st["Significance"]
    ))

# Supplemental OLM
for _, st in df_s1[df_s1["Figure_Panel"] == "Supplemental Fig"].iterrows():
    rows.append(row(
        "Supplemental Figure (OLM)", "OLM", st["Comparison"],
        "N/A", "N/A", np.nan,np.nan,np.nan,np.nan, np.nan,np.nan,np.nan,np.nan,
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
    "AHP Decay":        ("decay_area",             df_ap,   "E", "AHP Decay Area (mV·s)"),
    "Access Resistance": ("Access Resistance (From Whole Cell V-Clamp)", df_phys, "QC", "Access Resistance (MΩ)"),
}
for comp_key, (col, df_src, subp, mlabel) in phys_map.items():
    st_rows = df_s2[df_s2["Comparison"] == comp_key]
    if len(st_rows) == 0: continue
    st  = st_rows.iloc[0]
    wt_sub = df_src[(df_src["Genotype"] == "WT") & df_src[col].notna()]
    gnb_sub = df_src[(df_src["Genotype"] == "GNB1") & df_src[col].notna()]
    wt = wt_sub[col]
    gnb = gnb_sub[col]
    rows.append(row(
        "Figure 2", subp, mlabel, "N/A", "N/A",
        wt.mean(), wt.sem(), len(wt), get_n_animals(wt_sub),
        gnb.mean(), gnb.sem(), len(gnb), get_n_animals(gnb_sub),
        st["Test_Used"], st["Statistic"], st["P_Value"], st["Significance"]
    ))

# F-I midpoint
st_fi = df_s2[df_s2["Comparison"] == "F-I Curve Midpoint: WT vs GNB1"].iloc[0]
wt_fi_sub  = df_fi[(df_fi["Genotype"] == "WT") & df_fi["FI_Midpoint"].notna()]
gnb_fi_sub = df_fi[(df_fi["Genotype"] == "GNB1") & df_fi["FI_Midpoint"].notna()]
wt_fi = wt_fi_sub["FI_Midpoint"]
gnb_fi = gnb_fi_sub["FI_Midpoint"]
rows.append(row(
    "Figure 2", "F", "F-I Curve Midpoint (pA)", "N/A", "N/A",
    wt_fi.mean(), wt_fi.sem(), len(wt_fi), get_n_animals(wt_fi_sub),
    gnb_fi.mean(), gnb_fi.sem(), len(gnb_fi), get_n_animals(gnb_fi_sub),
    st_fi["Test_Used"], st_fi["Statistic"], st_fi["P_Value"], st_fi["Significance"]
))

# F-I 2-way RM ANOVA model terms (LME with nested random effects)
wt_fi_all = df_fi[df_fi["Genotype"] == "WT"]
gnb_fi_all = df_fi[df_fi["Genotype"] == "GNB1"]
wt_fi_anim = get_n_animals(wt_fi_all)
gnb_fi_anim = get_n_animals(gnb_fi_all)

for _, st in df_fi_isi[df_fi_isi["Analysis"] == "F-I Curve"].iterrows():
    if pd.isna(st.get("F_value")): continue
    fi_df1 = st.get("df1", np.nan)
    fi_df2 = st.get("df2", np.nan)
    fi_df_str = f"{int(fi_df1)},{round(fi_df2, 1)}" if pd.notna(fi_df1) and pd.notna(fi_df2) else np.nan
    f_val = st.get("F_value", np.nan)
    stat_str = f"F({int(fi_df1) if pd.notna(fi_df1) else '?'},{round(fi_df2, 1) if pd.notna(fi_df2) else '?'})={_4sig(f_val) if pd.notna(f_val) else '?'}"
    rows.append(row(
        "Figure 2", "E", f"F-I Curve ANOVA: {st['Comparison']}", "N/A", "N/A",
        np.nan,np.nan, 68, wt_fi_anim,
        np.nan,np.nan, 70, gnb_fi_anim,
        "LME Type III ANOVA (lmerTest)", stat_str, st["p_value"], st["significance"],
        notes="LME 2-way RM ANOVA model term",
        degrees_of_freedom=fi_df_str
    ))

# ISI adaptation ANOVA (LME with nested random effects)
for _, st in df_isi_a.iterrows():
    isi_df1 = st.get("df1", np.nan)
    isi_df2 = st.get("df2", np.nan)
    isi_df_str = f"{int(isi_df1)},{round(isi_df2, 1)}" if pd.notna(isi_df1) and pd.notna(isi_df2) else np.nan
    f_val = st.get("F_value", np.nan)
    stat_str = f"F({int(isi_df1) if pd.notna(isi_df1) else '?'},{round(isi_df2, 1) if pd.notna(isi_df2) else '?'})={_4sig(f_val) if pd.notna(f_val) else '?'}"
    rows.append(row(
        "Figure 2", "I", f"ISI Adaptation ANOVA: {st['Term']}", "N/A", "N/A",
        np.nan,np.nan, 60, wt_fi_anim,
        np.nan,np.nan, 60, gnb_fi_anim,
        "LME Type III ANOVA (lmerTest)", stat_str, st["p_value"], st["significance"],
        notes="LME 2-way RM ANOVA model term",
        degrees_of_freedom=isi_df_str
    ))


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 – Morphology
# ══════════════════════════════════════════════════════════════════════════════
print("Building Figure 3 rows …")

df_morph = pd.read_csv("paper_data/Morphology_Analysis/Dendrite_Properties_All.csv")
df_s3    = pd.read_csv("paper_data/Morphology_Analysis/Stats_Results_Figure_3.csv")
df_sholl = pd.read_csv("paper_data/Morphology_Analysis/Sholl_Intersections_Raw.csv")

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
    wt_sub = sub[(sub["Genotype"] == "WT") & sub[col].notna()]
    gnb_sub = sub[(sub["Genotype"] == "GNB1") & sub[col].notna()]
    wt  = wt_sub[col]
    gnb = gnb_sub[col]
    rows.append(row(
        "Figure 3", panel_key.replace("Fig 3","").strip(),
        mlabel, dtype, "N/A",
        wt.mean(), wt.sem(), len(wt), get_n_animals(wt_sub),
        gnb.mean(), gnb.sem(), len(gnb), get_n_animals(gnb_sub),
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
    "GABAB_Area":                     ("E", "GABAB-mediated Slow IPSP Area (mV·s)"),
}
for _, st in df_s4.iterrows():
    mcol = st["Metric"]
    if mcol not in panel_map4: continue
    subp, mlabel = panel_map4[mcol]
    pw  = st["Pathway"]
    sub = uni[uni["Pathway"] == pw].dropna(subset=[mcol])
    wt_sub  = sub[sub["Genotype"] == "WT"]
    gnb_sub = sub[sub["Genotype"] == "GNB1"]
    wt = wt_sub[mcol]
    gnb = gnb_sub[mcol]
    rows.append(row(
        "Figure 4", f"{subp} – {pw}", mlabel, pw, "Unitary (ISI 300 ms)",
        wt.mean(), wt.sem(), len(wt), get_n_animals(wt_sub),
        gnb.mean(), gnb.sem(), len(gnb), get_n_animals(gnb_sub),
        st["Test_Used"], st["Statistic"], st["P_Value"], st["Significance"]
    ))


# ══════════════════════════════════════════════════════════════════════════════
# FIGURES 5 & 6 – Frequency-Dependent E:I + Supralinearity
# ══════════════════════════════════════════════════════════════════════════════
print("Building Figure 5 & 6 rows …")

df_anova56 = pd.read_csv("paper_data/E_I_data/Figure_5_6_All_Stats_ANOVA.csv")
df_fdr56   = pd.read_csv("paper_data/E_I_data/Figure_5_6_All_Stats_FDR_Corrected.csv")
df_n_summary56 = pd.read_csv("paper_data/E_I_data/Figure_5_6_Stats_Summary.csv")

_N_RANGE_COL = {
    "Gabazine_Amplitude":             "Figure_5_B_D_Range (per ISI)",
    "Estimated_Inhibition_Amplitude": "Figure_5_B_D_Range (per ISI)",
    "Inhibition_Amplitude":           "Figure_5_B_D_Range (per ISI)",
    "GABAB_Area":                     "Figure_5_E_GABAB_Range",
    "Gabazine_Supralinearity":        "Figure_6_Supralinearity_Range",
    "E_I_Imbalance":                  "Figure_5_B_D_Range (per ISI)",
}

def _get_n_range(pathway, genotype, metric):
    range_col = _N_RANGE_COL.get(metric)
    if range_col is None:
        return np.nan
    mask = (df_n_summary56["Pathway"] == pathway) & (df_n_summary56["Genotype"] == genotype)
    rows_match = df_n_summary56.loc[mask]
    if len(rows_match) == 0 or range_col not in rows_match.columns:
        return np.nan
    return str(rows_match.iloc[0][range_col])

METRIC_FIG56 = {
    "Gabazine_Amplitude":             ("Figure 5", "B"),
    "Estimated_Inhibition_Amplitude": ("Figure 5", "C"),
    "Inhibition_Amplitude":           ("Figure 5", "C"),
    "GABAB_Area":                     ("Figure 5", "D"),
    "Gabazine_Supralinearity":        ("Figure 6", "C/D/E"),
}
METRIC_LABEL56 = {
    "Gabazine_Amplitude":             "EPSP Amplitude – Gabazine condition (mV)",
    "Estimated_Inhibition_Amplitude": "GABAA Inhibition Amplitude (mV)",
    "Inhibition_Amplitude":           "GABAA Inhibition Amplitude (mV)",
    "GABAB_Area":                     "GABAB Slow IPSP Area (mV·s)",
    "Gabazine_Supralinearity":        "Supralinearity (Measured − Expected, mV)",
}
RAW_COL56 = {
    "Gabazine_Amplitude":             "Gabazine_Amplitude",
    "Estimated_Inhibition_Amplitude": "Estimated_Inhibition_Amplitude",
    "Inhibition_Amplitude":           "Estimated_Inhibition_Amplitude",
    "GABAB_Area":                     "GABAB_Area",
    "Gabazine_Supralinearity":        "Gabazine_Supralinearity",
}
EFFECT_LABEL56 = {
    "Genotype":          "Main Effect – Genotype",
    "ISI_Time":          "Main Effect – ISI",
    "Genotype:ISI_Time": "Interaction – Genotype × ISI",
}

# ── LAYER 1: LME Type III ANOVA model terms ──────────────────────────────────
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
    f_val  = st["F value"]
    p_val  = st["P_Value"]
    sig    = st.get("Significant", p_to_sig(p_val))

    anova_df_str = f"{int(num_df)},{round(den_df,1)}" if pd.notna(num_df) and pd.notna(den_df) else np.nan
    wt_n_range  = _get_n_range(pw, "WT", metric)
    gnb_n_range = _get_n_range(pw, "GNB1", metric)
    
    raw_col = RAW_COL56.get(metric, metric)
    wt_all = df_ei[(df_ei["Pathway"]==pw) & (df_ei["Genotype"]=="WT") & df_ei[raw_col].notna()]
    gnb_all = df_ei[(df_ei["Pathway"]==pw) & (df_ei["Genotype"]=="GNB1") & df_ei[raw_col].notna()]
    wt_anim_n = get_n_animals(wt_all)
    gnb_anim_n = get_n_animals(gnb_all)

    rows.append(row(
        fig_label,
        f"{subp} – {pw} – ANOVA",
        METRIC_LABEL56[metric],
        pw,
        f"All ISIs – {EFFECT_LABEL56[effect]}",
        st.get("Mean_WT",   np.nan), st.get("SEM_WT",   np.nan), wt_n_range, wt_anim_n,
        st.get("Mean_GNB1", np.nan), st.get("SEM_GNB1", np.nan), gnb_n_range, gnb_anim_n,
        "LME Type III ANOVA (lmerTest)",
        f"F({int(num_df) if pd.notna(num_df) else '?'},{round(den_df,1) if pd.notna(den_df) else '?'})={_4sig(f_val) if pd.notna(f_val) else '?'}",
        p_val, sig,
        notes=f"ANOVA term: {effect}",
        degrees_of_freedom=anova_df_str
    ))

# ── LAYER 2: FDR-corrected per-ISI post-hoc contrasts ────────────────────────
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
        if isi_raw == f"ISI{v}":
            isi_val = v
            break

    raw_col = RAW_COL56.get(metric, metric)
    if isi_val is not None and raw_col in df_ei.columns:
        sub_ei  = df_ei[(df_ei["Pathway"] == pw) & (df_ei["ISI"] == isi_val)].dropna(subset=[raw_col])
        wt_sub  = sub_ei[sub_ei["Genotype"] == "WT"]
        gnb_sub = sub_ei[sub_ei["Genotype"] == "GNB1"]
        wt_v  = wt_sub[raw_col]
        gnb_v = gnb_sub[raw_col]
        wt_anim_n = get_n_animals(wt_sub)
        gnb_anim_n = get_n_animals(gnb_sub)
    else:
        wt_v = gnb_v = pd.Series(dtype=float)
        wt_anim_n = gnb_anim_n = np.nan

    t_ratio  = st["t_ratio"]
    df_val   = st["df"]
    p_fdr    = st["p_value_FDR"]
    sig_fdr  = st["Significant_FDR"]
    main_p   = st["Main_Effect_p"]
    inter_p  = st["Interaction_p"]

    wt_n_range  = len(wt_v) if len(wt_v) else _get_n_range(pw, "WT", metric)
    gnb_n_range = len(gnb_v) if len(gnb_v) else _get_n_range(pw, "GNB1", metric)
    fdr_df_val = float(f"{df_val:.4g}") if pd.notna(df_val) else np.nan
    rows.append(row(
        fig_label,
        f"{subp} – {pw} – ISI {isi_val} ms",
        METRIC_LABEL56[metric],
        pw, f"ISI {isi_val} ms",
        wt_v.mean()  if len(wt_v)  else np.nan,
        wt_v.sem()   if len(wt_v) > 1 else np.nan,
        wt_n_range, wt_anim_n,
        gnb_v.mean() if len(gnb_v) else np.nan,
        gnb_v.sem()  if len(gnb_v) > 1 else np.nan,
        gnb_n_range, gnb_anim_n,
        "LME FDR-corrected post-hoc (R lmerTest)",
        f"t({round(df_val,1) if pd.notna(df_val) else '?'})={_4sig(t_ratio) if pd.notna(t_ratio) else '?'}",
        p_fdr, sig_fdr,
        notes=(
            f"FDR post-hoc at ISI {isi_val} ms | "
            f"Genotype main effect p={round(main_p,4) if pd.notna(main_p) else 'NA'} | "
            f"Interaction p={round(inter_p,4) if pd.notna(inter_p) else 'NA'}"
        ),
        degrees_of_freedom=fdr_df_val
    ))


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 7 – Theta Burst / Dendritic Excitability
# ══════════════════════════════════════════════════════════════════════════════
print("Building Figure 7 rows …")

df_s7 = pd.read_csv("paper_data/Plateau_data/Stats_Results_Figure_7.csv")
df_plateau_raw = pd.read_csv("paper_data/Plateau_data/Plateau_data.csv") if os.path.exists("paper_data/Plateau_data/Plateau_data.csv") else None
df_supralin_raw = pd.read_csv("paper_data/supralinearity/Supralinear_AUC_Total.csv") if os.path.exists("paper_data/supralinearity/Supralinear_AUC_Total.csv") else None

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

    wt_anim = np.nan
    gnb_anim = np.nan
    if df_plateau_raw is not None and "Plateau Area" in comp:
        pw_match = 'Both' if 'Both' in comp else ('Schaffer' if 'Schaffer' in comp else 'Perforant')
        wt_sub = df_plateau_raw[(df_plateau_raw["Pathway"]==pw_match) & (df_plateau_raw["Genotype"]=="WT")]
        gnb_sub = df_plateau_raw[(df_plateau_raw["Pathway"]==pw_match) & (df_plateau_raw["Genotype"]=="GNB1")]
        wt_anim = get_n_animals(wt_sub)
        gnb_anim = get_n_animals(gnb_sub)
    elif df_supralin_raw is not None and "Supralinear" in comp:
        pathway_info = {'Schaffer': 'Schaffer', 'Perforant': 'Perforant', 'CA3': 'Schaffer', 'ECIII': 'Perforant', 'Both': 'Both Pathways'}
        sup_pw = pathway_info.get(pw, 'Both Pathways')
        wt_sub = df_supralin_raw[(df_supralin_raw["Pathway"]==sup_pw) & (df_supralin_raw["Genotype"]=="WT")]
        gnb_sub = df_supralin_raw[(df_supralin_raw["Pathway"]==sup_pw) & (df_supralin_raw["Genotype"]=="GNB1")]
        wt_anim = get_n_animals(wt_sub)
        gnb_anim = get_n_animals(gnb_sub)

    rows.append(row(
        "Figure 7", f"{subp} – {comp.split(':')[0].strip()}",
        mlabel, pw, cond,
        st["Mean_WT"], st["SEM_WT"],
        st["N_WT"] if pd.notna(st["N_WT"]) else np.nan, wt_anim,
        st["Mean_GNB1"], st["SEM_GNB1"],
        st["N_GNB1"] if pd.notna(st["N_GNB1"]) else np.nan, gnb_anim,
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

    wt_anim = np.nan
    gnb_anim = np.nan

    if "ΔVm" in comp:
        wt_s  = _sem(df_bac, "Genotype","WT","Voltage Change")
        gnb_s = _sem(df_bac, "Genotype","GNB1","Voltage Change")
        subp, mlabel, cond = "C", "Baclofen ΔVm (mV)", "Baclofen 10 μM"
        wt_anim = get_n_animals(df_bac[df_bac["Genotype"]=="WT"])
        gnb_anim = get_n_animals(df_bac[df_bac["Genotype"]=="GNB1"])
    elif "Plateau Delta" in comp:
        sub_p = df_plat_d[(df_plat_d["Drug"]==drug)&(df_plat_d["Pathway"]==pw)]
        wt_s  = _sem(sub_p, "Genotype","WT","Delta_Area")
        gnb_s = _sem(sub_p, "Genotype","GNB1","Delta_Area")
        subp  = "I" if drug=="ML297" else "L"
        mlabel = f"Δ Plateau Area – {drug} (mV·s)"
        cond   = f"{drug} – TBS"
        wt_anim = get_n_animals(sub_p[sub_p["Genotype"]=="WT"])
        gnb_anim = get_n_animals(sub_p[sub_p["Genotype"]=="GNB1"])
    elif "Unitary Delta" in comp:
        sub_u = df_uni_d[(df_uni_d["Drug"]==drug)&(df_uni_d["Pathway"]==pw)]
        wt_s  = _sem(sub_u, "Genotype","WT","Delta_GABAB_Area")
        gnb_s = _sem(sub_u, "Genotype","GNB1","Delta_GABAB_Area")
        subp = "F"
        mlabel = f"Δ Unitary GABAB Area – {drug}"
        cond   = f"{drug} – Unitary"
        wt_anim = get_n_animals(sub_u[sub_u["Genotype"]=="WT"])
        gnb_anim = get_n_animals(sub_u[sub_u["Genotype"]=="GNB1"])
    elif "Pre vs Post" in comp and "Plateau" in comp:
        sub_p = df_plat_d[(df_plat_d["Drug"]==drug)&(df_plat_d["Pathway"]==pw)]
        wt_s = gnb_s = np.nan
        subp  = "I" if drug=="ML297" else "L"
        mlabel = f"Pre vs Post {drug} – Plateau Area (mV·s) [Paired]"
        cond   = f"{drug} – TBS (Paired)"
        wt_anim = get_n_animals(sub_p[sub_p["Genotype"]=="WT"]) if pd.notna(st["WT_n"]) else np.nan
        gnb_anim = get_n_animals(sub_p[sub_p["Genotype"]=="GNB1"]) if pd.notna(st["GNB1_n"]) else np.nan
    elif "Pre vs Post" in comp and "Unitary" in comp:
        sub_u = df_uni_d[(df_uni_d["Drug"]==drug)&(df_uni_d["Pathway"]==pw)]
        wt_s = gnb_s = np.nan
        subp = "F"
        mlabel = f"Pre vs Post {drug} – Unitary GABAB Area [Paired]"
        cond   = f"{drug} – Unitary (Paired)"
        wt_anim = get_n_animals(sub_u[sub_u["Genotype"]=="WT"]) if pd.notna(st["WT_n"]) else np.nan
        gnb_anim = get_n_animals(sub_u[sub_u["Genotype"]=="GNB1"]) if pd.notna(st["GNB1_n"]) else np.nan
    else:
        wt_s = gnb_s = np.nan
        subp, mlabel, cond = "?", comp, drug

    rows.append(row(
        "Figure 8", f"{subp} – {drug} ({pw})",
        mlabel, pw, cond,
        st["WT_mean"], wt_s, st["WT_n"], wt_anim,
        st["GNB1_mean"], gnb_s, st["GNB1_n"], gnb_anim,
        st["test_type"], st["test_stat"], st["p_value"], st["Significance"]
    ))


# ══════════════════════════════════════════════════════════════════════════════
# SUPPLEMENTAL FIGURE 1 – E:I Imbalance Index
# ══════════════════════════════════════════════════════════════════════════════
print("Building Supplemental Figure 1 rows …")

df_anova_ei = pd.read_csv("paper_data/E_I_data/Figure_5_6_All_Stats_ANOVA.csv")
imb = df_anova_ei[df_anova_ei["Analysis"] == "E_I_Imbalance"]

df_n_summary = pd.read_csv("paper_data/E_I_data/Figure_5_6_Stats_Summary.csv")
ei_n_ranges = {}
for pw in df_n_summary["Pathway"].unique():
    wt_row  = df_n_summary[(df_n_summary["Pathway"]==pw) & (df_n_summary["Genotype"]=="WT")]
    gnb_row = df_n_summary[(df_n_summary["Pathway"]==pw) & (df_n_summary["Genotype"]=="GNB1")]
    wt_range  = wt_row["Figure_5_B_D_Range (per ISI)"].values[0]  if len(wt_row)  else "?"
    gnb_range = gnb_row["Figure_5_B_D_Range (per ISI)"].values[0] if len(gnb_row) else "?"
    ei_n_ranges[pw] = {"WT": wt_range, "GNB1": gnb_range}

for _, st in imb.iterrows():
    pw  = st["Pathway"]
    eff = st["Effect"]

    if eff not in EFFECT_LABEL56: continue
    cond_label = f"All ISIs – {EFFECT_LABEL56[eff]}"

    n_note = ei_n_ranges.get(pw, {})
    n_range_str = f"WT n={n_note.get('WT','?')}, GNB1 n={n_note.get('GNB1','?')} cells per ISI"

    f_val  = st["F value"]
    num_df = st["NumDF"]
    den_df = st["DenDF"]
    stat_str = f"F({int(num_df) if pd.notna(num_df) else '?'},{round(den_df,1) if pd.notna(den_df) else '?'})={round(f_val,3) if pd.notna(f_val) else '?'}"
    ei_anova_df_str = f"{int(num_df)},{round(den_df,1)}" if pd.notna(num_df) and pd.notna(den_df) else np.nan
    wt_n_range  = _get_n_range(pw, "WT", "E_I_Imbalance")
    gnb_n_range = _get_n_range(pw, "GNB1", "E_I_Imbalance")

    wt_sub = df_ei[(df_ei["Pathway"]==pw) & (df_ei["Genotype"]=="WT") & df_ei["E_I_Imbalance"].notna()]
    gnb_sub = df_ei[(df_ei["Pathway"]==pw) & (df_ei["Genotype"]=="GNB1") & df_ei["E_I_Imbalance"].notna()]
    wt_anim = get_n_animals(wt_sub)
    gnb_anim = get_n_animals(gnb_sub)

    rows.append(row(
        "Supplemental Figure 1", f"E:I Imbalance – {pw}",
        "E:I Imbalance Index (EPSP / (EPSP+|IPSP|))", pw, cond_label,
        np.nan, np.nan, wt_n_range, wt_anim,
        np.nan, np.nan, gnb_n_range, gnb_anim,
        "LME Type III ANOVA (lmerTest)", stat_str,
        st["P_Value"], st["Significant"],
        notes=f"ANOVA term: {eff}; {n_range_str}",
        degrees_of_freedom=ei_anova_df_str
    ))


# ══════════════════════════════════════════════════════════════════════════════
# SUPPLEMENTAL FIGURE 3 – GNB1 Protein Levels
# ══════════════════════════════════════════════════════════════════════════════
print("Building Supplemental Figure 3 rows …")

df_supp3 = pd.read_csv("paper_data/Stats_Results_Supplemental_Figure_3.csv")
df_protein = pd.read_csv("paper_data/GNB1_Protein_Levels_Hippocampus.csv")

wt_abs_mean  = df_protein["WT_Average_Signal"].values[0]
i80t_abs_mean = df_protein["I80T/+_Average_Signal"].values[0]

wt_abs_reps  = df_protein[["WT GNB1 Absolute Protein Signal Rep 1",
                            "WT GNB1 Absolute Protein Signal Rep 2",
                            "WT GNB1 Absolute Protein Signal Rep 3"]].values.flatten()
i80t_abs_reps = df_protein[["I80T/+ GNB1 Absolute Protein Signal Rep 1",
                             "I80T/+ GNB1 Absolute Protein Signal Rep 2",
                             "I80T/+ GNB1 Absolute Protein Signal Rep 3"]].values.flatten()
wt_abs_sem   = np.std(wt_abs_reps, ddof=1) / np.sqrt(len(wt_abs_reps))
i80t_abs_sem = np.std(i80t_abs_reps, ddof=1) / np.sqrt(len(i80t_abs_reps))

wt_rel_mean   = df_protein["WT_Relative"].values[0]
wt_rel_sem    = df_protein["WT_Relative_SEM"].values[0]
i80t_rel_mean = df_protein["I80T/+_Relative"].values[0]
i80t_rel_sem  = df_protein["I80T/+_Relative_SEM"].values[0]

_protein_summary = {
    "Supp Fig 3 (Top)":    (wt_abs_mean, wt_abs_sem, 3, 3, i80t_abs_mean, i80t_abs_sem, 3, 3),
    "Supp Fig 3 (Bottom)": (wt_rel_mean, wt_rel_sem, 3, 3, i80t_rel_mean, i80t_rel_sem, 3, 3),
}

for _, st in df_supp3.iterrows():
    panel = st["Figure_Panel"]
    wm, ws, wn, wna, im, is_, in_, ina = _protein_summary.get(panel, (np.nan,)*8)
    stat_val = st["Statistic"]
    stat_str = f"t(4)={_4sig(stat_val)}" if pd.notna(stat_val) else np.nan
    rows.append(row(
        "Supplemental Figure 3", panel,
        st["Comparison"], "Hippocampus", "N/A",
        wm, ws, wn, wna, im, is_, in_, ina,
        st["Test_Used"], stat_str, st["P_Value"], st["Significance"],
        notes="GNB1 protein levels (Western blot)",
        degrees_of_freedom=4
    ))


# ══════════════════════════════════════════════════════════════════════════════
# SUPPLEMENTAL FIGURE – Sex Stratified
# ══════════════════════════════════════════════════════════════════════════════
print("Building Supplemental Figure (Sex Stratified) rows …")

from scipy import stats

df_w_sex    = pd.read_csv("paper_data/Behavior_Analysis/Mouse_Weights_Processed.csv")
df_dvc_sex  = pd.read_csv("paper_data/DVC_Analysis/Cage_Specific_Hours_Summary.csv")
df_loc_sex  = pd.read_csv("paper_data/Behavior_Analysis/Open_Field_Locomotion_Trial1.csv")
df_anx_sex  = pd.read_csv("paper_data/Behavior_Analysis/Open_Field_Anxiety_Processed.csv")
df_fi_sex   = pd.read_csv("paper_data/Firing_Rate/Sigmoid_Fit_Params.csv")
df_plat_sex = pd.read_csv("paper_data/Plateau_data/Plateau_data.csv")
df_ei_sex   = pd.read_csv("paper_data/E_I_data/E_I_amplitudes.csv")
df_olm_sex  = pd.read_csv("paper_data/Behavior_Analysis/OLM_Summary_Deltas.csv")
master_df_sex = pd.read_csv("master_df.csv", low_memory=False) if os.path.exists("master_df.csv") else None

sex_map = {"M": "M", "Male": "M", "F": "F", "Female": "F"}

# 1. Body Weights
df_w_sex["SexGeno"] = df_w_sex["Genotype"].replace({"GNB1": "I80T/+"}) + "-" + df_w_sex["Sex"].astype(str).str.strip().map(sex_map)
for tp in ["P8-P10", "P28", "Adult"]:
    sub = df_w_sex[df_w_sex["Timepoint_Label"] == tp]
    for comp, sex_lbl in [("WT-M vs I80T/+-M", "Male"), ("WT-F vs I80T/+-F", "Female")]:
        ga, gb = comp.split(" vs ")
        va = sub[sub["SexGeno"] == ga]["Weight_g"].dropna()
        vb = sub[sub["SexGeno"] == gb]["Weight_g"].dropna()
        u, p = stats.mannwhitneyu(va, vb)
        rows.append(row(
            "Supplemental Figure (Sex Stratified)", f"Body Weight – {tp}", f"Body Weight ({tp}) [{sex_lbl}]", "N/A", tp,
            va.mean(), va.sem(), len(va), len(va),
            vb.mean(), vb.sem(), len(vb), len(vb),
            "Mann-Whitney U", u, p, p_to_sig(p),
            notes=f"Sex-stratified: {comp}", degrees_of_freedom=len(va) + len(vb) - 2
        ))

# 2. DVC Circadian
df_dvc_sex["SexGeno"] = df_dvc_sex["Genotype"].replace({"GNB1": "I80T/+"}) + "-" + df_dvc_sex["Sex"].astype(str).str.strip().map(sex_map)
for comp, sex_lbl in [("WT-M vs I80T/+-M", "Male"), ("WT-F vs I80T/+-F", "Female")]:
    ga, gb = comp.split(" vs ")
    va = df_dvc_sex[df_dvc_sex["SexGeno"] == ga]["Sum_All_Dark"].dropna()
    vb = df_dvc_sex[df_dvc_sex["SexGeno"] == gb]["Sum_All_Dark"].dropna()
    u, p = stats.mannwhitneyu(va, vb)
    rows.append(row(
        "Supplemental Figure (Sex Stratified)", "Circadian Activity", f"DVC Summed Dark Activity [{sex_lbl}]", "N/A", "All Dark Hours",
        va.mean(), va.sem(), len(va), len(va),
        vb.mean(), vb.sem(), len(vb), len(vb),
        "Mann-Whitney U", u, p, p_to_sig(p),
        notes=f"Sex-stratified: {comp}", degrees_of_freedom=len(va) + len(vb) - 2
    ))

# 3. Open Field Locomotion
df_loc_sex["SexGeno"] = df_loc_sex["Genotype"].replace({"GNB1": "I80T/+"}) + "-" + df_loc_sex["Sex"].astype(str).str.strip().map(sex_map)
for comp, sex_lbl in [("WT-M vs I80T/+-M", "Male"), ("WT-F vs I80T/+-F", "Female")]:
    ga, gb = comp.split(" vs ")
    va = df_loc_sex[df_loc_sex["SexGeno"] == ga]["Distance (m)"].dropna()
    vb = df_loc_sex[df_loc_sex["SexGeno"] == gb]["Distance (m)"].dropna()
    u, p = stats.mannwhitneyu(va, vb)
    rows.append(row(
        "Supplemental Figure (Sex Stratified)", "Open Field Locomotion", f"Open Field Distance [{sex_lbl}]", "N/A", "Trial 1",
        va.mean(), va.sem(), len(va), len(va),
        vb.mean(), vb.sem(), len(vb), len(vb),
        "Mann-Whitney U", u, p, p_to_sig(p),
        notes=f"Sex-stratified: {comp}", degrees_of_freedom=len(va) + len(vb) - 2
    ))

# 4. Open Field Anxiety
df_anx_sex["SexGeno"] = df_anx_sex["Genotype"].replace({"GNB1": "I80T/+"}) + "-" + df_anx_sex["Sex"].astype(str).str.strip().map(sex_map)
for comp, sex_lbl in [("WT-M vs I80T/+-M", "Male"), ("WT-F vs I80T/+-F", "Female")]:
    ga, gb = comp.split(" vs ")
    va = df_anx_sex[df_anx_sex["SexGeno"] == ga]["Center_Outer_Time_Ratio"].dropna()
    vb = df_anx_sex[df_anx_sex["SexGeno"] == gb]["Center_Outer_Time_Ratio"].dropna()
    u, p = stats.mannwhitneyu(va, vb)
    rows.append(row(
        "Supplemental Figure (Sex Stratified)", "Open Field Anxiety", f"Center:Outer Time Ratio [{sex_lbl}]", "N/A", "N/A",
        va.mean(), va.sem(), len(va), len(va),
        vb.mean(), vb.sem(), len(vb), len(vb),
        "Mann-Whitney U", u, p, p_to_sig(p),
        notes=f"Sex-stratified: {comp}", degrees_of_freedom=len(va) + len(vb) - 2
    ))

# 5. Somatic Excitability (FI Midpoint)
if master_df_sex is not None:
    master_df_sex["Cell_ID"] = master_df_sex["Cell_ID"].astype(str).str.strip()
    master_df_sex["Sex"]     = master_df_sex["Sex"].astype(str).str.strip()
    df_fi_sex["Cell_ID"]     = df_fi_sex["Cell_ID"].astype(str).str.strip()
    df_fi_m = df_fi_sex.merge(master_df_sex[["Cell_ID", "Sex"]].drop_duplicates(), on="Cell_ID", how="left")
    df_fi_m["SexGeno"] = df_fi_m["Genotype"].replace({"GNB1": "I80T/+"}) + "-" + df_fi_m["Sex"].map({"Male": "M", "Female": "F", "M": "M", "F": "F"})
    for comp, sex_lbl in [("WT-M vs I80T/+-M", "Male"), ("WT-F vs I80T/+-F", "Female")]:
        ga, gb = comp.split(" vs ")
        sub_a = df_fi_m[df_fi_m["SexGeno"] == ga]
        sub_b = df_fi_m[df_fi_m["SexGeno"] == gb]
        va = sub_a["Midpoint"].dropna()
        vb = sub_b["Midpoint"].dropna()
        u, p = stats.mannwhitneyu(va, vb)
        rows.append(row(
            "Supplemental Figure (Sex Stratified)", "Somatic Excitability", f"F-I Midpoint (pA) [{sex_lbl}]", "N/A", "N/A",
            va.mean(), va.sem(), len(va), get_n_animals(sub_a["Cell_ID"]),
            vb.mean(), vb.sem(), len(vb), get_n_animals(sub_b["Cell_ID"]),
            "Mann-Whitney U", u, p, p_to_sig(p),
            notes=f"Sex-stratified: {comp}", degrees_of_freedom=len(va) + len(vb) - 2
        ))

# 5b. Somatic Input Resistance
df_intr_sex = pd.read_csv("paper_data/Physiology_Analysis/Intrinsic_properties.csv")
df_intr_sex["SexGeno"] = df_intr_sex["Genotype"].replace({"GNB1": "I80T/+"}) + "-" + df_intr_sex["Sex"].astype(str).str.strip().map({"Male": "M", "Female": "F", "M": "M", "F": "F"})
for comp, sex_lbl in [("WT-M vs I80T/+-M", "Male"), ("WT-F vs I80T/+-F", "Female")]:
    ga, gb = comp.split(" vs ")
    sub_a = df_intr_sex[df_intr_sex["SexGeno"] == ga]
    sub_b = df_intr_sex[df_intr_sex["SexGeno"] == gb]
    va = sub_a["Input_Resistance_MOhm"].dropna()
    vb = sub_b["Input_Resistance_MOhm"].dropna()
    u, p = stats.mannwhitneyu(va, vb)
    rows.append(row(
        "Supplemental Figure (Sex Stratified)", "Somatic Input Resistance", f"Input Resistance (MΩ) [{sex_lbl}]", "N/A", "N/A",
        va.mean(), va.sem(), len(va), get_n_animals(sub_a["Cell_ID"]),
        vb.mean(), vb.sem(), len(vb), get_n_animals(sub_b["Cell_ID"]),
        "Mann-Whitney U", u, p, p_to_sig(p),
        notes=f"Sex-stratified: {comp}", degrees_of_freedom=len(va) + len(vb) - 2
    ))

# 6. Dendritic Excitability (Plateau Area)
df_plat_gab = df_plat_sex[df_plat_sex["Condition"].str.contains("Gabazine", na=False, case=False)].copy()
df_plat_gab["SexGeno"] = df_plat_gab["Genotype"].replace({"GNB1": "I80T/+"}) + "-" + df_plat_gab["Sex"].astype(str).str.strip().map({"Male": "M", "Female": "F", "M": "M", "F": "F"})
for comp, sex_lbl in [("WT-M vs I80T/+-M", "Male"), ("WT-F vs I80T/+-F", "Female")]:
    ga, gb = comp.split(" vs ")
    sub_a = df_plat_gab[df_plat_gab["SexGeno"] == ga]
    sub_b = df_plat_gab[df_plat_gab["SexGeno"] == gb]
    va = sub_a["Plateau_Area"].dropna()
    vb = sub_b["Plateau_Area"].dropna()
    u, p = stats.mannwhitneyu(va, vb)
    rows.append(row(
        "Supplemental Figure (Sex Stratified)", "Dendritic Excitability", f"Plateau Area (mV·s) [{sex_lbl}]", "Both Pathways", "Gabazine condition",
        va.mean(), va.sem(), len(va), get_n_animals(sub_a["Cell_ID"]),
        vb.mean(), vb.sem(), len(vb), get_n_animals(sub_b["Cell_ID"]),
        "Mann-Whitney U", u, p, p_to_sig(p),
        notes=f"Sex-stratified: {comp}", degrees_of_freedom=len(va) + len(vb) - 2
    ))

# 7. E/I Imbalance Index (All Pathways & ISIs)
df_ei_sex["SexGeno"] = df_ei_sex["Genotype"].replace({"GNB1": "I80T/+"}) + "-" + df_ei_sex["Sex"].astype(str).str.strip().map(sex_map)
for pw in ["Perforant", "Schaffer", "Basal_Stratum_Oriens"]:
    df_pw = df_ei_sex[df_ei_sex["Pathway"] == pw]
    for isi in [300, 100, 50, 25, 10]:
        df_isi = df_pw[df_pw["ISI"] == isi]
        for comp, sex_lbl in [("WT-M vs I80T/+-M", "Male"), ("WT-F vs I80T/+-F", "Female")]:
            ga, gb = comp.split(" vs ")
            sub_a = df_isi[df_isi["SexGeno"] == ga]
            sub_b = df_isi[df_isi["SexGeno"] == gb]
            va = sub_a["E_I_Imbalance"].dropna()
            vb = sub_b["E_I_Imbalance"].dropna()
            if len(va) >= 2 and len(vb) >= 2:
                u, p = stats.mannwhitneyu(va, vb)
                rows.append(row(
                    "Supplemental Figure (Sex Stratified)", f"E/I Imbalance – {pw}", f"E/I Imbalance Index [{sex_lbl}]", pw, f"ISI {isi} ms",
                    va.mean(), va.sem(), len(va), get_n_animals(sub_a["Cell_ID"]),
                    vb.mean(), vb.sem(), len(vb), get_n_animals(sub_b["Cell_ID"]),
                    "Mann-Whitney U", u, p, p_to_sig(p),
                    notes=f"Sex-stratified: {comp} | ISI {isi} ms", degrees_of_freedom=len(va) + len(vb) - 2
                ))

# -----------------------------------------------------------------------------
# STIMULATION AMPLITUDES (Figures 4 & 5)
# -----------------------------------------------------------------------------
mdf_amps = pd.read_csv("master_df.csv", low_memory=False)
mdf_amps["Genotype"] = mdf_amps["Genotype"].replace({"GNB1": "I80T/+"})
inc_amps = mdf_amps[mdf_amps["Inclusion"].astype(str).str.contains("Yes", case=False, na=False)].copy()

amp_specs = [
    ("Figure 4", "Perforant Path", "Perforant_Stim_Amp", "Perforant Path Stimulus Amplitude (mA)"),
    ("Figure 4", "Schaffer Collateral", "Schaffer_Stim_Amp", "Schaffer Collateral Stimulus Amplitude (mA)"),
    ("Figure 4", "Stratum Oriens (Basal)", "Stratum_Oriens_Stim_Amp", "Stratum Oriens Stimulus Amplitude (mA)"),
    ("Figure 5", "Channel 1 Overall", "channel_1_amp", "Channel 1 Stimulus Amplitude (mA)"),
    ("Figure 5", "Channel 2 Overall", "channel_2_amp", "Channel 2 Stimulus Amplitude (mA)"),
]

for fig_lbl, subp_name, col_name, metric_lbl in amp_specs:
    sub_wt = inc_amps[(inc_amps["Genotype"] == "WT") & inc_amps[col_name].notna()]
    sub_mut = inc_amps[(inc_amps["Genotype"] == "I80T/+") & inc_amps[col_name].notna()]
    
    v_wt = sub_wt[col_name].dropna().values
    v_mut = sub_mut[col_name].dropna().values
    
    if len(v_wt) > 0 and len(v_mut) > 0:
        u, p = stats.mannwhitneyu(v_wt, v_mut, alternative="two-sided")
        rows.append(row(
            fig_lbl, subp_name, metric_lbl, subp_name, "Control",
            v_wt.mean(), v_wt.std(ddof=1)/np.sqrt(len(v_wt)) if len(v_wt) > 1 else 0.0, len(v_wt), get_n_animals(sub_wt["Cell_ID"]),
            v_mut.mean(), v_mut.std(ddof=1)/np.sqrt(len(v_mut)) if len(v_mut) > 1 else 0.0, len(v_mut), get_n_animals(sub_mut["Cell_ID"]),
            "Mann-Whitney U", u, p, p_to_sig(p),
            notes=f"Stimulation amplitude check: WT vs I80T/+", degrees_of_freedom=len(v_wt) + len(v_mut) - 2
        ))

# 8. T-Maze
df_t_ent_sex = pd.read_csv("paper_data/Behavior_Analysis/T_Maze_Zone_Entries.csv")
df_t_alt_sex = pd.read_csv("paper_data/Behavior_Analysis/T_Maze_Alternations.csv")
df_t_ent_sex["SexGeno"] = df_t_ent_sex["Genotype"].replace({"GNB1": "I80T/+"}) + "-" + df_t_ent_sex["Sex"].astype(str).str.strip().map(sex_map)
df_t_alt_sex["SexGeno"] = df_t_alt_sex["Genotype"].replace({"GNB1": "I80T/+"}) + "-" + df_t_alt_sex["Sex"].astype(str).str.strip().map(sex_map)
df_t_ent_sex["Total_Arm_Entries"] = df_t_ent_sex["Start : entries"] + df_t_ent_sex["Left Arm : entries"] + df_t_ent_sex["Right Arm : entries"]

for comp, sex_lbl in [("WT-M vs I80T/+-M", "Male"), ("WT-F vs I80T/+-F", "Female")]:
    ga, gb = comp.split(" vs ")
    va = df_t_ent_sex[df_t_ent_sex["SexGeno"] == ga]["Distance (m)"].dropna()
    vb = df_t_ent_sex[df_t_ent_sex["SexGeno"] == gb]["Distance (m)"].dropna()
    u, p = stats.mannwhitneyu(va, vb)
    rows.append(row(
        "Supplemental Figure (Sex Stratified)", "T-Maze Distance", f"T-Maze Distance [{sex_lbl}]", "N/A", "Trial 1",
        va.mean(), va.sem(), len(va), len(va),
        vb.mean(), vb.sem(), len(vb), len(vb),
        "Mann-Whitney U", u, p, p_to_sig(p),
        notes=f"Sex-stratified: {comp}", degrees_of_freedom=len(va) + len(vb) - 2
    ))

    v_ent_a = df_t_ent_sex[df_t_ent_sex["SexGeno"] == ga]["Total_Arm_Entries"].dropna()
    v_ent_b = df_t_ent_sex[df_t_ent_sex["SexGeno"] == gb]["Total_Arm_Entries"].dropna()
    u_ent, p_ent = stats.mannwhitneyu(v_ent_a, v_ent_b)
    rows.append(row(
        "Supplemental Figure (Sex Stratified)", "T-Maze Entries", f"T-Maze Total Arm Entries [{sex_lbl}]", "N/A", "Trial 1",
        v_ent_a.mean(), v_ent_a.sem(), len(v_ent_a), len(v_ent_a),
        v_ent_b.mean(), v_ent_b.sem(), len(v_ent_b), len(v_ent_b),
        "Mann-Whitney U", u_ent, p_ent, p_to_sig(p_ent),
        notes=f"Sex-stratified: {comp}", degrees_of_freedom=len(v_ent_a) + len(v_ent_b) - 2
    ))

    v_alt_a = df_t_alt_sex[df_t_alt_sex["SexGeno"] == ga]["Percent_Alternations"].dropna()
    v_alt_b = df_t_alt_sex[df_t_alt_sex["SexGeno"] == gb]["Percent_Alternations"].dropna()
    u_alt, p_alt = stats.mannwhitneyu(v_alt_a, v_alt_b)
    rows.append(row(
        "Supplemental Figure (Sex Stratified)", "T-Maze Alternations", f"T-Maze Percent Alternations [{sex_lbl}]", "N/A", "Trial 1",
        v_alt_a.mean(), v_alt_a.sem(), len(v_alt_a), len(v_alt_a),
        v_alt_b.mean(), v_alt_b.sem(), len(v_alt_b), len(v_alt_b),
        "Mann-Whitney U", u_alt, p_alt, p_to_sig(p_alt),
        notes=f"Sex-stratified: {comp}", degrees_of_freedom=len(v_alt_a) + len(v_alt_b) - 2
    ))

# 9. OLM
df_olm_sex["SexGeno"] = df_olm_sex["Genotype"].replace({"GNB1": "I80T/+"}) + "-" + df_olm_sex["Sex"].astype(str).str.strip().map(sex_map)
for comp, sex_lbl in [("WT-M vs I80T/+-M", "Male"), ("WT-F vs I80T/+-F", "Female")]:
    ga, gb = comp.split(" vs ")
    va = df_olm_sex[df_olm_sex["SexGeno"] == ga]["Testing_DI"].dropna()
    vb = df_olm_sex[df_olm_sex["SexGeno"] == gb]["Testing_DI"].dropna()
    u, p = stats.mannwhitneyu(va, vb)
    rows.append(row(
        "Supplemental Figure (Sex Stratified)", "OLM", f"OLM Discrimination Index [{sex_lbl}]", "N/A", "Testing Stage",
        va.mean(), va.sem(), len(va), len(va),
        vb.mean(), vb.sem(), len(vb), len(vb),
        "Mann-Whitney U", u, p, p_to_sig(p),
        notes=f"Sex-stratified: {comp}", degrees_of_freedom=len(va) + len(vb) - 2
    ))


# ══════════════════════════════════════════════════════════════════════════════
# BUILD & SAVE
# ══════════════════════════════════════════════════════════════════════════════
print("Assembling master table …")

COLUMNS = [
    "Figure","Subpanel","Metric","Pathway","Condition",
    "WT_Mean","WT_SEM","WT_N","WT_N_Animals",
    "I80T_Mean","I80T_SEM","I80T_N","I80T_N_Animals",
    "Test_Used","Statistic","Degrees_of_Freedom",
    "P_Value","P_Value_Formatted","Paper_Formatted","Significance",
    "Notes",
]

df_master = pd.DataFrame(rows, columns=COLUMNS)
for c in ["WT_Mean", "I80T_Mean", "WT_SEM", "I80T_SEM"]:
    df_master[c] = pd.to_numeric(df_master[c], errors="coerce")

fig_order = {
    "Figure 1":1,"Figure 2":2,"Figure 3":3,
    "Figure 4":4,"Figure 5":5,"Figure 6":6,
    "Figure 7":7,"Figure 8":8,
    "Supplemental Figure 1":9,
    "Supplemental Figure (OLM)":10,
    "Supplemental Figure 3":11,
    "Supplemental Figure (Sex Stratified)":12,
}
df_master["_s"] = df_master["Figure"].map(lambda x: fig_order.get(x,99))
df_master = (df_master.sort_values(["_s","Subpanel"])
             .drop(columns=["_s"]).reset_index(drop=True))

out_csv  = "paper_data/Master_Stats_Summary.csv"
out_xlsx = "paper_data/Master_Stats_Summary.xlsx"

df_master.to_csv(out_csv, index=False)
print(f"  Saved CSV  → {out_csv}  ({len(df_master)} rows)")

try:
    from openpyxl.styles import numbers as xl_numbers

    COL_FORMATS = {
        "WT_Mean":            "0.000",
        "I80T_Mean":          "0.000",
        "WT_SEM":             "0.000",
        "I80T_SEM":           "0.000",
        "WT_N":               "@",
        "WT_N_Animals":       "@",
        "I80T_N":             "@",
        "I80T_N_Animals":     "@",
        "P_Value":            "0.0000",
        "P_Value_Formatted":  "@",
        "Paper_Formatted":    "@",
        "Degrees_of_Freedom": "0.000",
    }

    def _apply_formats(ws):
        header = {cell.value: cell.column_letter for cell in ws[1]}
        for col_name, fmt in COL_FORMATS.items():
            if col_name not in header:
                continue
            col_letter = header[col_name]
            for row_idx in range(2, ws.max_row + 1):
                cell = ws[f"{col_letter}{row_idx}"]
                if cell.value is not None:
                    cell.number_format = fmt

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
            _apply_formats(sheet)

    print(f"  Saved XLSX → {out_xlsx}  ({len(df_master['Figure'].unique())} sheets)")
except ImportError:
    print("  openpyxl not installed – run: pip install openpyxl")

print("\nDone.")
print(df_master[["Figure","Subpanel","Metric","WT_N","WT_N_Animals","I80T_N","I80T_N_Animals","Statistic","P_Value","Significance"]].to_string())
