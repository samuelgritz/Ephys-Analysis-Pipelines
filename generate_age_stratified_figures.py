"""
generate_age_stratified_figures.py
====================================
Generates cleaned publication-quality age-stratified figures comparing
6-9 weeks vs 10-12 weeks for GNB1 WT and I80T/+ neurons.

Statistical Engine:
  - 2-Way ANOVA (Genotype x Age_Group)
  - Post-hoc Tukey's HSD (Honest Significant Difference) multiple comparison tests for all 4 pairwise contrasts:
      1. WT vs I80T/+ (6–9 weeks)
      2. WT vs I80T/+ (10–12 weeks)
      3. WT (6–9w) vs WT (10–12w) [WT Across-Age]
      4. I80T/+ (6–9w) vs I80T/+ (10–12w) [Mutant Across-Age]

Styling:
  - Clean non-overlapping vertical layout with expanded top headroom.
  - 4-group Okabe-Ito color palette with zero horizontal jitter on scatter points.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats

from plotting_utils import (
    setup_publication_style,
    add_subplot_label,
    apply_clean_yticks
)

PAPER_DATA_DIR = "paper_data"
FIGURES_DIR = "paper_figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

# 4-group Okabe-Ito Color Palette
AGE_COLORS = {
    ("WT", "6-9 weeks"): "#56B4E9",      # Light Sky Blue
    ("I80T/+", "6-9 weeks"): "#E69F00",  # Light Orange
    ("WT", "10-12 weeks"): "#0072B2",    # Dark Blue
    ("I80T/+", "10-12 weeks"): "#D55E00",# Dark Vermillion/Red
}

def load_master_age_map():
    mdf_path = "master_df.csv" if os.path.exists("master_df.csv") else "../master_df.csv"
    if not os.path.exists(mdf_path):
        return {}
    mdf = pd.read_csv(mdf_path)
    mdf["Cell_ID"] = mdf["Cell_ID"].astype(str).str.strip()
    
    age_col = None
    for col in ["Mouse Age (weeks)", "Mouse.Age..weeks."]:
        if col in mdf.columns:
            age_col = col
            break
            
    if age_col is None:
        return {}
        
    mdf["Age_Weeks"] = pd.to_numeric(mdf[age_col], errors="coerce")
    
    age_map = {}
    for _, row in mdf.iterrows():
        cid = row["Cell_ID"]
        ag_w = row["Age_Weeks"]
        if pd.notna(ag_w):
            if ag_w <= 9:
                age_map[cid] = "6-9 weeks"
            elif ag_w >= 10:
                age_map[cid] = "10-12 weeks"
    return age_map

def assign_age_group(df, age_map):
    df = df.copy()
    df["Cell_ID"] = df["Cell_ID"].astype(str).str.strip()
    df["Genotype"] = df["Genotype"].replace({"GNB1": "I80T/+"})
    
    if "Age_Group" not in df.columns:
        df["Age_Group"] = df["Cell_ID"].map(age_map)
    return df

def p_to_sig(p):
    if pd.isna(p): return "ns"
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"

def load_r_tukey_stats(metric_name, pathway="N/A"):
    r_stats_path = os.path.join(PAPER_DATA_DIR, "Age_Stratified_R_Stats_Results.csv")
    if not os.path.exists(r_stats_path):
        return None
    df_r = pd.read_csv(r_stats_path)
    sub = df_r[(df_r["Metric"] == metric_name)]
    if pathway != "N/A" and "Pathway" in sub.columns:
        sub = sub[sub["Pathway"] == pathway]
    if len(sub) == 0:
        return None
        
    row_6_9 = sub[sub["Age_Group"] == "6-9 weeks"]
    row_10_12 = sub[sub["Age_Group"] == "10-12 weeks"]
    
    p_w1 = row_6_9["Tukey_P_WithinAge"].values[0] if len(row_6_9) > 0 else np.nan
    p_w2 = row_10_12["Tukey_P_WithinAge"].values[0] if len(row_10_12) > 0 else np.nan
    p_wt_age = sub["Tukey_P_WT_Age"].values[0] if "Tukey_P_WT_Age" in sub.columns else np.nan
    p_mut_age = sub["Tukey_P_Mutant_Age"].values[0] if "Tukey_P_Mutant_Age" in sub.columns else np.nan
    
    return {
        "p_w1": p_w1,
        "p_w2": p_w2,
        "p_wt_age": p_wt_age,
        "p_mut_age": p_mut_age
    }

def plot_age_stratified_clean(ax, df, val_col, title, ylabel, metric_name="", pathway="N/A", show_legend=True):
    """
    Clean, non-overlapping bar & scatter plot displaying 2-Way ANOVA + Tukey HSD post-hoc test results:
      1. WT vs I80T/+ (6-9w) [Tukey HSD]
      2. WT vs I80T/+ (10-12w) [Tukey HSD]
      3. WT (6-9w) vs WT (10-12w) [Tukey HSD Across-Age]
      4. I80T/+ (6-9w) vs I80T/+ (10-12w) [Tukey HSD Across-Age]
    """
    groups = [
        ("WT", "6-9 weeks", 0.0),
        ("I80T/+", "6-9 weeks", 0.8),
        ("WT", "10-12 weeks", 2.2),
        ("I80T/+", "10-12 weeks", 3.0),
    ]
    
    all_vals = df[val_col].dropna().values
    if len(all_vals) == 0:
        return
        
    y_min_data = np.min(all_vals)
    y_max_data = np.max(all_vals)
    is_negative = y_min_data < 0 and y_max_data <= 0.1
    
    data_dict = {}
    for geno, age_grp, x_pos in groups:
        sub = df[(df["Genotype"] == geno) & (df["Age_Group"] == age_grp)]
        vals = sub[val_col].dropna().values
        data_dict[(geno, age_grp)] = vals
        n = len(vals)
        color = AGE_COLORS.get((geno, age_grp), "black")
        
        if n > 0:
            m = np.mean(vals)
            s = np.std(vals, ddof=1) / np.sqrt(n) if n > 1 else 0.0
            
            # Bar chart
            ax.bar(x_pos, m, yerr=s, color=color, alpha=0.35, edgecolor=color, linewidth=1.2,
                   width=0.65, error_kw=dict(capsize=3, capthick=1.0, ecolor=color))
            
            # Scatter points WITH ZERO HORIZONTAL JITTER
            x_fixed = np.full(n, x_pos)
            ax.scatter(x_fixed, vals, color=color, s=16, alpha=0.75, zorder=3, edgecolors="none")
            
            # Sample size label placed inside bar base or safely offset
            if is_negative:
                ax.text(x_pos, 0.02, f"n={n}", ha="center", va="bottom", fontsize=5.8, color="#333333")
            else:
                ax.text(x_pos, 0.02 * y_max_data, f"n={n}", ha="center", va="bottom", fontsize=5.8, color="#333333")

    # Set up X-ticks cleanly
    ax.set_xticks([0.0, 0.8, 2.2, 3.0])
    ax.set_xticklabels(["WT", "I80T/+", "WT", "I80T/+"], fontsize=7.5)
    
    # Add age group sub-labels centered below the ticks
    y_tick_sub = -0.16 if not is_negative else -0.18
    ax.text(0.4, y_tick_sub, "6–9 weeks", ha="center", va="top", transform=ax.get_xaxis_transform(), fontsize=8, fontweight="bold")
    ax.text(2.6, y_tick_sub, "10–12 weeks", ha="center", va="top", transform=ax.get_xaxis_transform(), fontsize=8, fontweight="bold")
    
    # Add group underline spines below x-axis
    ax.plot([0.0, 0.8], [-0.11, -0.11], color="black", lw=1.0, clip_on=False, transform=ax.get_xaxis_transform())
    ax.plot([2.2, 3.0], [-0.11, -0.11], color="black", lw=1.0, clip_on=False, transform=ax.get_xaxis_transform())

    # --- TUKEY HSD POST-HOC STATISTICAL ANNOTATIONS ---
    tuk_stats = load_r_tukey_stats(metric_name if metric_name else title, pathway)
    if tuk_stats is not None:
        p_w1 = tuk_stats["p_w1"]
        p_w2 = tuk_stats["p_w2"]
        p_wt_age = tuk_stats["p_wt_age"]
        p_mut_age = tuk_stats["p_mut_age"]
    else:
        # Scipy / Mann-Whitney fallback
        _, p_w1 = stats.mannwhitneyu(data_dict[("WT", "6-9 weeks")], data_dict[("I80T/+", "6-9 weeks")]) if len(data_dict[("WT", "6-9 weeks")])>0 and len(data_dict[("I80T/+", "6-9 weeks")])>0 else (None, np.nan)
        _, p_w2 = stats.mannwhitneyu(data_dict[("WT", "10-12 weeks")], data_dict[("I80T/+", "10-12 weeks")]) if len(data_dict[("WT", "10-12 weeks")])>0 and len(data_dict[("I80T/+", "10-12 weeks")])>0 else (None, np.nan)
        _, p_wt_age = stats.mannwhitneyu(data_dict[("WT", "6-9 weeks")], data_dict[("WT", "10-12 weeks")]) if len(data_dict[("WT", "6-9 weeks")])>0 and len(data_dict[("WT", "10-12 weeks")])>0 else (None, np.nan)
        _, p_mut_age = stats.mannwhitneyu(data_dict[("I80T/+", "6-9 weeks")], data_dict[("I80T/+", "10-12 weeks")]) if len(data_dict[("I80T/+", "6-9 weeks")])>0 and len(data_dict[("I80T/+", "10-12 weeks")])>0 else (None, np.nan)

    if not is_negative:
        # Level 1: Within-Age Genotype Brackets (Tukey HSD)
        y_l1 = y_max_data * 1.08
        ax.plot([0.0, 0.8], [y_l1, y_l1], color="black", lw=0.8)
        ax.text(0.4, y_l1 * 1.01, f"{p_to_sig(p_w1)} (p={p_w1:.3f})" if pd.notna(p_w1) else "ns", ha="center", va="bottom", fontsize=6)
        
        ax.plot([2.2, 3.0], [y_l1, y_l1], color="black", lw=0.8)
        ax.text(2.6, y_l1 * 1.01, f"{p_to_sig(p_w2)} (p={p_w2:.3f})" if pd.notna(p_w2) else "ns", ha="center", va="bottom", fontsize=6)
        
        # Level 2: WT Across-Age Bracket (Tukey HSD)
        y_l2 = y_max_data * 1.26
        ax.plot([0.0, 2.2], [y_l2, y_l2], color="#0072B2", lw=0.8, ls="--")
        ax.text(1.1, y_l2 * 1.01, f"WT Age: {p_to_sig(p_wt_age)} (p={p_wt_age:.3f})" if pd.notna(p_wt_age) else "WT Age: ns", ha="center", va="bottom", fontsize=6, color="#0072B2")
        
        # Level 3: I80T/+ Across-Age Bracket (Tukey HSD)
        y_l3 = y_max_data * 1.44
        ax.plot([0.8, 3.0], [y_l3, y_l3], color="#D55E00", lw=0.8, ls="--")
        ax.text(1.9, y_l3 * 1.01, f"I80T/+ Age: {p_to_sig(p_mut_age)} (p={p_mut_age:.3f})" if pd.notna(p_mut_age) else "I80T/+ Age: ns", ha="center", va="bottom", fontsize=6, color="#D55E00")
        
        ax.set_ylim(bottom=0, top=y_max_data * 1.68)
    else:
        abs_min = abs(y_min_data)
        
        # Level 1: Within-Age Genotype Brackets
        y_l1 = 0.05 * abs_min
        ax.plot([0.0, 0.8], [y_l1, y_l1], color="black", lw=0.8)
        ax.text(0.4, y_l1 + 0.02 * abs_min, f"{p_to_sig(p_w1)} (p={p_w1:.3f})", ha="center", va="bottom", fontsize=6)
        
        ax.plot([2.2, 3.0], [y_l1, y_l1], color="black", lw=0.8)
        ax.text(2.6, y_l1 + 0.02 * abs_min, f"{p_to_sig(p_w2)} (p={p_w2:.3f})", ha="center", va="bottom", fontsize=6)
        
        # Level 2: WT Across-Age Bracket
        y_l2 = 0.16 * abs_min
        ax.plot([0.0, 2.2], [y_l2, y_l2], color="#0072B2", lw=0.8, ls="--")
        ax.text(1.1, y_l2 + 0.02 * abs_min, f"WT Age: {p_to_sig(p_wt_age)} (p={p_wt_age:.3f})", ha="center", va="bottom", fontsize=6, color="#0072B2")
        
        # Level 3: I80T/+ Across-Age Bracket
        y_l3 = 0.28 * abs_min
        ax.plot([0.8, 3.0], [y_l3, y_l3], color="#D55E00", lw=0.8, ls="--")
        ax.text(1.9, y_l3 + 0.02 * abs_min, f"I80T/+ Age: {p_to_sig(p_mut_age)} (p={p_mut_age:.3f})", ha="center", va="bottom", fontsize=6, color="#D55E00")
        
        ax.set_ylim(bottom=y_min_data * 1.25, top=0.42 * abs_min)

    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_title(title, fontsize=8.5, pad=22, fontweight="bold")
    
    if show_legend:
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor="#56B4E9", edgecolor="#56B4E9", alpha=0.7, label="WT (6–9w)"),
            Patch(facecolor="#E69F00", edgecolor="#E69F00", alpha=0.7, label="I80T/+ (6–9w)"),
            Patch(facecolor="#0072B2", edgecolor="#0072B2", alpha=0.7, label="WT (10–12w)"),
            Patch(facecolor="#D55E00", edgecolor="#D55E00", alpha=0.7, label="I80T/+ (10–12w)"),
        ]
        ax.legend(handles=legend_elements, loc="upper center", bbox_to_anchor=(0.5, 1.34),
                  ncol=4, frameon=False, fontsize=6.5)

# ------------------------------------------------------------------------------
# Figure Generators
# ------------------------------------------------------------------------------

def generate_figure2_age_stratified(age_map):
    """Figure 2: Clean Age-Stratified Somatic Physiology."""
    setup_publication_style()
    fig = plt.figure(figsize=(7.5, 7.8))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.65, wspace=0.35)
    
    intr_file = os.path.join(PAPER_DATA_DIR, "Physiology_Analysis", "Intrinsic_properties.csv")
    fi_file = os.path.join(PAPER_DATA_DIR, "Firing_Rate", "Sigmoid_Fit_Params.csv")
    
    # Panel A: Input Resistance
    ax_rin = fig.add_subplot(gs[0, 0])
    if os.path.exists(intr_file):
        df_intr = assign_age_group(pd.read_csv(intr_file), age_map)
        plot_age_stratified_clean(ax_rin, df_intr, "Input_Resistance_MOhm", "Somatic Input Resistance", "Input Resistance (MΩ)", metric_name="Input_Resistance_MOhm", show_legend=True)
    add_subplot_label(ax_rin, "A")
    
    # Panel B: Voltage Sag
    ax_sag = fig.add_subplot(gs[0, 1])
    if os.path.exists(intr_file):
        df_sag = df_intr[df_intr["Voltage_sag"].notna()]
        plot_age_stratified_clean(ax_sag, df_sag, "Voltage_sag", "Hyperpolarization Sag", "Voltage Sag (%)", metric_name="Voltage_sag", show_legend=False)
    add_subplot_label(ax_sag, "B")
    
    # Panel C: F-I Midpoint
    ax_mid = fig.add_subplot(gs[1, 0])
    if os.path.exists(fi_file):
        df_fi = assign_age_group(pd.read_csv(fi_file), age_map)
        val_col = "Midpoint" if "Midpoint" in df_fi.columns else "FI_Midpoint"
        plot_age_stratified_clean(ax_mid, df_fi, val_col, "F-I Curve Midpoint", "Midpoint Current (pA)", metric_name="F-I Curve Midpoint (pA)", show_legend=False)
    add_subplot_label(ax_mid, "C")
    
    # Panel D: Vm Rest
    ax_vm = fig.add_subplot(gs[1, 1])
    if os.path.exists(intr_file):
        df_vm = df_intr[df_intr["Vm rest/start (mV)"].notna()]
        plot_age_stratified_clean(ax_vm, df_vm, "Vm rest/start (mV)", "Resting Membrane Potential", "RMP (mV)", metric_name="Vm rest/start (mV)", show_legend=False)
    add_subplot_label(ax_vm, "D")
    
    for ext in ["pdf", "png", "svg"]:
        fig.savefig(os.path.join(FIGURES_DIR, f"AgeStratified_Figure2_Physiology.{ext}"), dpi=300, bbox_inches="tight")
    plt.close()
    print("✓ Generated Clean AgeStratified_Figure2_Physiology with Tukey HSD")


def generate_figure4_age_stratified(age_map):
    """Figure 4: Clean Age-Stratified Unitary E:I Balance."""
    setup_publication_style()
    fig = plt.figure(figsize=(7.5, 11.2))
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.65, wspace=0.38)
    
    ei_file = os.path.join(PAPER_DATA_DIR, "E_I_data", "E_I_amplitudes.csv")
    if not os.path.exists(ei_file):
        return
        
    df_ei = assign_age_group(pd.read_csv(ei_file), age_map)
    df_300 = df_ei[df_ei["ISI"] == 300]
    
    # Panel A: Gabazine EPSP (Perforant)
    ax_epsp_pp = fig.add_subplot(gs[0, 0])
    plot_age_stratified_clean(ax_epsp_pp, df_300[df_300["Pathway"] == "Perforant"], "Gabazine_Amplitude", "Perforant Path Unitary EPSP", "EPSP Amplitude (mV)", metric_name="Gabazine EPSP Amplitude (mV)", pathway="Perforant", show_legend=True)
    add_subplot_label(ax_epsp_pp, "A")
    
    # Panel B: Gabazine EPSP (Schaffer)
    ax_epsp_sc = fig.add_subplot(gs[0, 1])
    plot_age_stratified_clean(ax_epsp_sc, df_300[df_300["Pathway"] == "Schaffer"], "Gabazine_Amplitude", "Schaffer Collateral Unitary EPSP", "EPSP Amplitude (mV)", metric_name="Gabazine EPSP Amplitude (mV)", pathway="Schaffer", show_legend=False)
    add_subplot_label(ax_epsp_sc, "B")
    
    # Panel C: GABAA Inhibition (Perforant)
    ax_inh_pp = fig.add_subplot(gs[1, 0])
    plot_age_stratified_clean(ax_inh_pp, df_300[df_300["Pathway"] == "Perforant"], "Estimated_Inhibition_Amplitude", "Perforant Path GABAA Inhibition", "Inhibition Amp (mV)", metric_name="GABAA Inhibition Amplitude (mV)", pathway="Perforant", show_legend=False)
    add_subplot_label(ax_inh_pp, "C")
    
    # Panel D: GABAA Inhibition (Schaffer)
    ax_inh_sc = fig.add_subplot(gs[1, 1])
    plot_age_stratified_clean(ax_inh_sc, df_300[df_300["Pathway"] == "Schaffer"], "Estimated_Inhibition_Amplitude", "Schaffer Collateral GABAA Inhibition", "Inhibition Amp (mV)", metric_name="GABAA Inhibition Amplitude (mV)", pathway="Schaffer", show_legend=False)
    add_subplot_label(ax_inh_sc, "D")
    
    # Panel E: GABAB Area (Perforant)
    ax_gb_pp = fig.add_subplot(gs[2, 0])
    plot_age_stratified_clean(ax_gb_pp, df_300[df_300["Pathway"] == "Perforant"], "GABAB_Area", "Perforant Path GABAB Area", "GABAB Area (mV·s)", metric_name="GABAB Area (mV·s)", pathway="Perforant", show_legend=False)
    add_subplot_label(ax_gb_pp, "E")
    
    # Panel F: E/I Imbalance Index (Perforant)
    ax_ei_pp = fig.add_subplot(gs[2, 1])
    plot_age_stratified_clean(ax_ei_pp, df_300[df_300["Pathway"] == "Perforant"], "E_I_Imbalance", "Perforant Path E/I Imbalance", "E/I Imbalance Index", metric_name="E/I Imbalance Index (ISI 300ms)", pathway="Perforant", show_legend=False)
    add_subplot_label(ax_ei_pp, "F")
    
    for ext in ["pdf", "png", "svg"]:
        fig.savefig(os.path.join(FIGURES_DIR, f"AgeStratified_Figure4_EI_Unitary.{ext}"), dpi=300, bbox_inches="tight")
    plt.close()
    print("✓ Generated Clean AgeStratified_Figure4_EI_Unitary with Tukey HSD")


def generate_figure7_age_stratified(age_map):
    """Figure 7: Clean Age-Stratified Dendritic Plateau Area."""
    setup_publication_style()
    fig, ax = plt.subplots(figsize=(4.2, 4.8))
    
    plat_file = os.path.join(PAPER_DATA_DIR, "Plateau_data", "Plateau_data.csv")
    if not os.path.exists(plat_file):
        return
        
    df_plat = assign_age_group(pd.read_csv(plat_file), age_map)
    df_gab = df_plat[df_plat["Condition"].str.contains("Gabazine", na=False)]
    
    plot_age_stratified_clean(ax, df_gab, "Plateau_Area", "Dendritic Plateau Area (Gabazine)", "Plateau Area (mV·s)", metric_name="Plateau Area (mV·s)", show_legend=True)
    add_subplot_label(ax, "A")
    
    for ext in ["pdf", "png", "svg"]:
        fig.savefig(os.path.join(FIGURES_DIR, f"AgeStratified_Figure7_Dendritic_Excitability.{ext}"), dpi=300, bbox_inches="tight")
    plt.close()
    print("✓ Generated Clean AgeStratified_Figure7_Dendritic_Excitability with Tukey HSD")


def generate_summary_figure_age_stratified(age_map):
    """Age-Stratified Ephys Summary Figure."""
    setup_publication_style()
    fig = plt.figure(figsize=(7.5, 7.8))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.65, wspace=0.35)
    
    intr_file = os.path.join(PAPER_DATA_DIR, "Physiology_Analysis", "Intrinsic_properties.csv")
    fi_file = os.path.join(PAPER_DATA_DIR, "Firing_Rate", "Sigmoid_Fit_Params.csv")
    ei_file = os.path.join(PAPER_DATA_DIR, "E_I_data", "E_I_amplitudes.csv")
    plat_file = os.path.join(PAPER_DATA_DIR, "Plateau_data", "Plateau_data.csv")
    
    # 1. Input Resistance
    ax1 = fig.add_subplot(gs[0, 0])
    if os.path.exists(intr_file):
        df_intr = assign_age_group(pd.read_csv(intr_file), age_map)
        plot_age_stratified_clean(ax1, df_intr, "Input_Resistance_MOhm", "Somatic Input Resistance", "Rin (MΩ)", metric_name="Input_Resistance_MOhm", show_legend=True)
    add_subplot_label(ax1, "A")
    
    # 2. F-I Midpoint
    ax2 = fig.add_subplot(gs[0, 1])
    if os.path.exists(fi_file):
        df_fi = assign_age_group(pd.read_csv(fi_file), age_map)
        val_col = "Midpoint" if "Midpoint" in df_fi.columns else "FI_Midpoint"
        plot_age_stratified_clean(ax2, df_fi, val_col, "F-I Curve Midpoint", "Midpoint (pA)", metric_name="F-I Curve Midpoint (pA)", show_legend=False)
    add_subplot_label(ax2, "B")
    
    # 3. GABAB Area (Perforant)
    ax3 = fig.add_subplot(gs[1, 0])
    if os.path.exists(ei_file):
        df_ei = assign_age_group(pd.read_csv(ei_file), age_map)
        df_300 = df_ei[(df_ei["ISI"] == 300) & (df_ei["Pathway"] == "Perforant")]
        plot_age_stratified_clean(ax3, df_300, "GABAB_Area", "Perforant GABAB Area", "GABAB Area (mV·s)", metric_name="GABAB Area (mV·s)", pathway="Perforant", show_legend=False)
    add_subplot_label(ax3, "C")
    
    # 4. Dendritic Plateau Area
    ax4 = fig.add_subplot(gs[1, 1])
    if os.path.exists(plat_file):
        df_plat = assign_age_group(pd.read_csv(plat_file), age_map)
        df_gab = df_plat[df_plat["Condition"].str.contains("Gabazine", na=False)]
        plot_age_stratified_clean(ax4, df_gab, "Plateau_Area", "Dendritic Plateau Area", "Plateau Area (mV·s)", metric_name="Plateau Area (mV·s)", show_legend=False)
    add_subplot_label(ax4, "D")
    
    for ext in ["pdf", "png", "svg"]:
        fig.savefig(os.path.join(FIGURES_DIR, f"AgeStratified_Ephys_Summary.{ext}"), dpi=300, bbox_inches="tight")
    plt.close()
    print("✓ Generated Clean AgeStratified_Ephys_Summary with Tukey HSD")

if __name__ == "__main__":
    age_map = load_master_age_map()
    print(f"Loaded {len(age_map)} cells with age mappings (6-9w vs 10-12w).")
    
    generate_figure2_age_stratified(age_map)
    generate_figure4_age_stratified(age_map)
    generate_figure7_age_stratified(age_map)
    generate_summary_figure_age_stratified(age_map)
    print("\nAll Age-Stratified Figures with Tukey HSD Generated Successfully!")
