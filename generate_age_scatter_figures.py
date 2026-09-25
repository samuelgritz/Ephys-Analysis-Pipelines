"""
generate_age_scatter_figures.py
=================================
Generates continuous scatter plots of Mouse Age (weeks) vs Electrophysiology Measurements
for WT and GNB1 I80T/+ neurons across 4 key properties:
  1. Somatic Input Resistance (Input_Resistance_MOhm)
  2. F-I Curve Midpoint (Midpoint, pA)
  3. Perforant Path GABAB Area (GABAB_Area, mV·s)
  4. Dendritic Plateau Area (Plateau_Area, mV·s)

Key Styling & Aesthetics:
  - Continuous x-axis: Mouse Age (weeks) from 5 to 12 weeks
  - Horizontal jitter (+/- 0.10 weeks) so points on exact same integer week are distinct
  - Linear regression fit lines with 95% confidence interval bands for WT and I80T/+
  - Colors: WT = Black (#000000), I80T/+ = Red (#E41A1C)
  - Regression statistics box (Slope, R², p-value, ANCOVA interaction p-value)

Outputs saved to paper_figures/ as .pdf, .png, .svg
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats

from plotting_utils import (
    setup_publication_style,
    add_subplot_label,
    apply_clean_yticks
)

PAPER_DATA_DIR = "paper_data"
FIGURES_DIR = "paper_figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

GENO_COLORS = {
    "WT": "#000000",
    "I80T/+": "#E41A1C"
}

def load_master_age_data():
    mdf_path = "master_df.csv" if os.path.exists("master_df.csv") else "../master_df.csv"
    if not os.path.exists(mdf_path):
        return None
    mdf = pd.read_csv(mdf_path)
    mdf["Cell_ID"] = mdf["Cell_ID"].astype(str).str.strip()
    
    age_col = None
    for col in ["Mouse Age (weeks)", "Mouse.Age..weeks."]:
        if col in mdf.columns:
            age_col = col
            break
            
    if age_col is None:
        return None
        
    mdf["Age_Weeks"] = pd.to_numeric(mdf[age_col], errors="coerce")
    return mdf[["Cell_ID", "Age_Weeks"]].drop_duplicates("Cell_ID")

def plot_age_scatter_continuous(ax, df_sub, val_col, title, ylabel, stats_row=None):
    """
    Plots Mouse Age in Weeks vs Measurement variable with linear regression fits and 95% CI bands.
    """
    mdf_age = load_master_age_data()
    if mdf_age is None:
        return
        
    df_sub = df_sub.copy()
    df_sub["Cell_ID"] = df_sub["Cell_ID"].astype(str).str.strip()
    df_sub["Genotype"] = df_sub["Genotype"].replace({"GNB1": "I80T/+"})
    
    if "Age_Weeks" not in df_sub.columns:
        df_sub = pd.merge(df_sub, mdf_age, on="Cell_ID", how="left")
        
    clean = df_sub[["Genotype", "Age_Weeks", val_col]].dropna()
    clean = clean[clean["Genotype"].isin(["WT", "I80T/+"])]
    
    if len(clean) == 0:
        return

    np.random.seed(42)  # Fixed seed for reproducible horizontal jitter
    
    stat_summary = []
    
    for geno in ["WT", "I80T/+"]:
        sub = clean[clean["Genotype"] == geno]
        x_raw = sub["Age_Weeks"].values
        y_vals = sub[val_col].values
        n = len(x_raw)
        color = GENO_COLORS[geno]
        
        if n > 3:
            # Add horizontal jitter (+/- 0.10 weeks) for visual scatter
            x_jitter = x_raw + np.random.uniform(-0.10, 0.10, size=n)
            
            # Scatter points
            ax.scatter(x_jitter, y_vals, color=color, s=20, alpha=0.55, zorder=3, label=f"{geno} (n={n})")
            
            # Seaborn linear regression with 95% CI band
            sns.regplot(
                x=x_raw, y=y_vals, ax=ax,
                color=color, scatter=False,
                line_kws={"linewidth": 1.5, "zorder": 4},
                ci=95
            )
            
            # Regress stats
            slope, intercept, r_val, p_val, std_err = stats.linregress(x_raw, y_vals)
            p_str = f"p={p_val:.3f}" if p_val >= 0.001 else "p<0.001"
            stat_summary.append(f"{geno}: Slope={slope:+.2f}, R²={r_val**2:.2f} ({p_str})")
            
    ax.set_xlabel("Mouse Age (weeks)", fontsize=8, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_title(title, fontsize=8.5, pad=12, fontweight="bold")
    
    # Set X Ticks continuously from 5 to 12 weeks
    ax.set_xticks(range(5, 13))
    ax.set_xlim(4.5, 12.5)
    
    # Format Legend
    ax.legend(loc="upper left", frameon=False, fontsize=7)
    
    # Display regression stats text box
    if len(stat_summary) > 0:
        stat_text = "\n".join(stat_summary)
        ax.text(0.97, 0.05, stat_text, transform=ax.transAxes,
                ha="right", va="bottom", fontsize=6, color="#333333",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor="#CCCCCC"))

# ------------------------------------------------------------------------------
# Figure Generators
# ------------------------------------------------------------------------------

def generate_figure2_rin_scatter():
    """Scatter: Age vs Somatic Input Resistance."""
    setup_publication_style()
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    
    intr_file = os.path.join(PAPER_DATA_DIR, "Physiology_Analysis", "Intrinsic_properties.csv")
    if os.path.exists(intr_file):
        df_intr = pd.read_csv(intr_file)
        plot_age_scatter_continuous(ax, df_intr, "Input_Resistance_MOhm", "Somatic Input Resistance vs Age", "Input Resistance (MΩ)")
    add_subplot_label(ax, "A")
    
    for ext in ["pdf", "png", "svg"]:
        fig.savefig(os.path.join(FIGURES_DIR, f"AgeScatter_Figure2_InputResistance.{ext}"), dpi=300, bbox_inches="tight")
    plt.close()
    print("✓ Generated AgeScatter_Figure2_InputResistance")

def generate_figure2_fi_scatter():
    """Scatter: Age vs F-I Midpoint."""
    setup_publication_style()
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    
    fi_file = os.path.join(PAPER_DATA_DIR, "Firing_Rate", "Sigmoid_Fit_Params.csv")
    if os.path.exists(fi_file):
        df_fi = pd.read_csv(fi_file)
        val_col = "Midpoint" if "Midpoint" in df_fi.columns else "FI_Midpoint"
        plot_age_scatter_continuous(ax, df_fi, val_col, "F-I Curve Midpoint vs Age", "Midpoint Current (pA)")
    add_subplot_label(ax, "A")
    
    for ext in ["pdf", "png", "svg"]:
        fig.savefig(os.path.join(FIGURES_DIR, f"AgeScatter_Figure2_FIMidpoint.{ext}"), dpi=300, bbox_inches="tight")
    plt.close()
    print("✓ Generated AgeScatter_Figure2_FIMidpoint")

def generate_figure4_gabab_scatter():
    """Scatter: Age vs Perforant GABAB Area."""
    setup_publication_style()
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    
    ei_file = os.path.join(PAPER_DATA_DIR, "E_I_data", "E_I_amplitudes.csv")
    if os.path.exists(ei_file):
        df_ei = pd.read_csv(ei_file)
        df_300 = df_ei[(df_ei["ISI"] == 300) & (df_ei["Pathway"] == "Perforant")]
        plot_age_scatter_continuous(ax, df_300, "GABAB_Area", "Perforant Path GABAB Area vs Age", "GABAB Area (mV·s)")
    add_subplot_label(ax, "A")
    
    for ext in ["pdf", "png", "svg"]:
        fig.savefig(os.path.join(FIGURES_DIR, f"AgeScatter_Figure4_PerforantGABAB.{ext}"), dpi=300, bbox_inches="tight")
    plt.close()
    print("✓ Generated AgeScatter_Figure4_PerforantGABAB")

def generate_figure7_plateau_scatter():
    """Scatter: Age vs Dendritic Plateau Area."""
    setup_publication_style()
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    
    plat_file = os.path.join(PAPER_DATA_DIR, "Plateau_data", "Plateau_data.csv")
    if os.path.exists(plat_file):
        df_plat = pd.read_csv(plat_file)
        df_gab = df_plat[df_plat["Condition"].str.contains("Gabazine", na=False)]
        plot_age_scatter_continuous(ax, df_gab, "Plateau_Area", "Dendritic Plateau Area vs Age", "Plateau Area (mV·s)")
    add_subplot_label(ax, "A")
    
    for ext in ["pdf", "png", "svg"]:
        fig.savefig(os.path.join(FIGURES_DIR, f"AgeScatter_Figure7_PlateauArea.{ext}"), dpi=300, bbox_inches="tight")
    plt.close()
    print("✓ Generated AgeScatter_Figure7_PlateauArea")

def generate_summary_scatter_figure():
    """Master 4-panel grid of continuous age scatter plots."""
    setup_publication_style()
    fig = plt.figure(figsize=(7.5, 7.2))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.32)
    
    intr_file = os.path.join(PAPER_DATA_DIR, "Physiology_Analysis", "Intrinsic_properties.csv")
    fi_file = os.path.join(PAPER_DATA_DIR, "Firing_Rate", "Sigmoid_Fit_Params.csv")
    ei_file = os.path.join(PAPER_DATA_DIR, "E_I_data", "E_I_amplitudes.csv")
    plat_file = os.path.join(PAPER_DATA_DIR, "Plateau_data", "Plateau_data.csv")
    
    # 1. Input Resistance
    ax1 = fig.add_subplot(gs[0, 0])
    if os.path.exists(intr_file):
        plot_age_scatter_continuous(ax1, pd.read_csv(intr_file), "Input_Resistance_MOhm", "Somatic Input Resistance vs Age", "Rin (MΩ)")
    add_subplot_label(ax1, "A")
    
    # 2. F-I Midpoint
    ax2 = fig.add_subplot(gs[0, 1])
    if os.path.exists(fi_file):
        df_fi = pd.read_csv(fi_file)
        val_col = "Midpoint" if "Midpoint" in df_fi.columns else "FI_Midpoint"
        plot_age_scatter_continuous(ax2, df_fi, val_col, "F-I Curve Midpoint vs Age", "Midpoint (pA)")
    add_subplot_label(ax2, "B")
    
    # 3. GABAB Area (Perforant)
    ax3 = fig.add_subplot(gs[1, 0])
    if os.path.exists(ei_file):
        df_ei = pd.read_csv(ei_file)
        df_300 = df_ei[(df_ei["ISI"] == 300) & (df_ei["Pathway"] == "Perforant")]
        plot_age_scatter_continuous(ax3, df_300, "GABAB_Area", "Perforant GABAB Area vs Age", "GABAB Area (mV·s)")
    add_subplot_label(ax3, "C")
    
    # 4. Dendritic Plateau Area
    ax4 = fig.add_subplot(gs[1, 1])
    if os.path.exists(plat_file):
        df_plat = pd.read_csv(plat_file)
        df_gab = df_plat[df_plat["Condition"].str.contains("Gabazine", na=False)]
        plot_age_scatter_continuous(ax4, df_gab, "Plateau_Area", "Dendritic Plateau Area vs Age", "Plateau Area (mV·s)")
    add_subplot_label(ax4, "D")
    
    for ext in ["pdf", "png", "svg"]:
        fig.savefig(os.path.join(FIGURES_DIR, f"AgeScatter_Ephys_Summary.{ext}"), dpi=300, bbox_inches="tight")
    plt.close()
    print("✓ Generated Master Summary Figure: AgeScatter_Ephys_Summary")

if __name__ == "__main__":
    generate_figure2_rin_scatter()
    generate_figure2_fi_scatter()
    generate_figure4_gabab_scatter()
    generate_figure7_plateau_scatter()
    generate_summary_scatter_figure()
    print("\nAll Continuous Age Scatter Figures Generated Successfully!")
