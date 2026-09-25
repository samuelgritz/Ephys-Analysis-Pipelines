import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats

from plotting_utils import (
    setup_publication_style,
    COLORS,
    add_subplot_label,
    apply_clean_yticks
)

PAPER_DATA_DIR = "paper_data"
FIGURES_DIR = "paper_figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

def mann_whitney(a, b):
    a_clean = np.array(a)[~np.isnan(a)]
    b_clean = np.array(b)[~np.isnan(b)]
    if len(a_clean) == 0 or len(b_clean) == 0:
        return np.nan, np.nan
    u, p = stats.mannwhitneyu(a_clean, b_clean, alternative="two-sided")
    return u, p

def p_to_sig(p):
    if pd.isna(p): return "ns"
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"

def get_n_animals(cell_ids):
    dates = set()
    for cid in cell_ids.dropna():
        s = str(cid).strip()
        if "_" in s:
            dates.add(s.split("_")[0])
        else:
            dates.add(s)
    return len(dates)

def plot_stim_amp_bar_scatter(ax, df, val_col, title, ylabel):
    """Plot bar + scatter plot with zero horizontal jitter for WT vs I80T/+ stimulation amplitudes."""
    geno_order = ["WT", "I80T/+"]
    geno_colors = {"WT": "#000000", "I80T/+": "#E41A1C"}
    
    means = []
    sems = []
    ns_cells = []
    ns_animals = []
    
    for i, g in enumerate(geno_order):
        sub = df[df["Genotype"] == g]
        vals = sub[val_col].dropna().values
        cell_ids = sub[val_col].dropna().index.map(lambda idx: sub.loc[idx, "Cell_ID"] if "Cell_ID" in sub.columns else idx)
        
        n_c = len(vals)
        n_a = get_n_animals(sub[sub[val_col].notna()]["Cell_ID"]) if "Cell_ID" in sub.columns else n_c
        ns_cells.append(n_c)
        ns_animals.append(n_a)
        
        if n_c > 0:
            m = np.mean(vals)
            s = np.std(vals, ddof=1) / np.sqrt(n_c) if n_c > 1 else 0.0
            means.append(m)
            sems.append(s)
            
            color = geno_colors.get(g, "black")
            # Bar plot
            ax.bar(i, m, yerr=s, color=color, alpha=0.3, edgecolor=color, linewidth=1.2,
                   width=0.55, error_kw=dict(capsize=3, capthick=1.0))
            
            # Scatter points WITH ZERO HORIZONTAL JITTER (fixed x-coordinate)
            x_fixed = np.full(n_c, i)
            ax.scatter(x_fixed, vals, color=color, s=16, alpha=0.7, zorder=3, edgecolors="none")
        else:
            means.append(0.0)
            sems.append(0.0)

    # Perform Mann-Whitney U test
    v_wt = df[df["Genotype"] == "WT"][val_col].dropna().values
    v_mut = df[df["Genotype"] == "I80T/+"][val_col].dropna().values
    u, p = mann_whitney(v_wt, v_mut)
    sig = p_to_sig(p)

    # Annotate p-value and stars above plot
    all_vals = df[val_col].dropna().values
    max_y = np.max(all_vals) if len(all_vals) > 0 else 1.0
    y_line = max_y * 1.12
    y_text = max_y * 1.18

    ax.plot([0, 1], [y_line, y_line], color="black", lw=0.8)
    if sig != "ns":
        ax.text(0.5, y_text, f"{sig}\n(p={p:.4f})", ha="center", va="bottom", fontsize=7.5, fontweight="bold")
    else:
        ax.text(0.5, y_text, f"ns\n(p={p:.4f})", ha="center", va="bottom", fontsize=7, color="gray")

    ax.set_xticks([0, 1])
    ax.set_xticklabels([f"WT\n(n={ns_cells[0]})", f"I80T/+\n(n={ns_cells[1]})"], fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_title(title, fontsize=9, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(direction="out", length=3, width=0.8, labelsize=8)
    ax.set_ylim(0, max_y * 1.35)
    apply_clean_yticks(ax)

    return {
        "Metric": title,
        "WT_Mean": means[0], "WT_SEM": sems[0], "WT_N_Cells": ns_cells[0], "WT_N_Animals": ns_animals[0],
        "I80T_Mean": means[1], "I80T_SEM": sems[1], "I80T_N_Cells": ns_cells[1], "I80T_N_Animals": ns_animals[1],
        "U": u, "P": p, "Sig": sig
    }

def plot_figure_4_stim_amplitudes(mdf):
    """Figure 4 Stimulation Amplitudes (Perforant Path, Schaffer Collateral, Stratum Oriens)."""
    print("  → Generating Figure 4 Stimulation Amplitudes")
    fig, axes = plt.subplots(1, 3, figsize=(6.93, 3.2))
    fig.subplots_adjust(wspace=0.38, left=0.10, right=0.96, top=0.86, bottom=0.22)

    st_p = plot_stim_amp_bar_scatter(axes[0], mdf, "Perforant_Stim_Amp", "Perforant Path", "Stim Amplitude (mA)")
    st_s = plot_stim_amp_bar_scatter(axes[1], mdf, "Schaffer_Stim_Amp", "Schaffer Collateral", "Stim Amplitude (mA)")
    st_o = plot_stim_amp_bar_scatter(axes[2], mdf, "Stratum_Oriens_Stim_Amp", "Stratum Oriens (Basal)", "Stim Amplitude (mA)")

    add_subplot_label(axes[0], "A", x=-0.14, y=1.08, fontsize=10, fontweight="bold")
    add_subplot_label(axes[1], "B", x=-0.14, y=1.08, fontsize=10, fontweight="bold")
    add_subplot_label(axes[2], "C", x=-0.14, y=1.08, fontsize=10, fontweight="bold")

    return fig, [st_p, st_s, st_o]

def plot_figure_5_stim_amplitudes(mdf):
    """Figure 5 Stimulation Amplitudes (Channel 1 & Channel 2)."""
    print("  → Generating Figure 5 Stimulation Amplitudes")
    fig, axes = plt.subplots(1, 2, figsize=(5.0, 3.2))
    fig.subplots_adjust(wspace=0.40, left=0.15, right=0.95, top=0.86, bottom=0.22)

    st_c1 = plot_stim_amp_bar_scatter(axes[0], mdf, "channel_1_amp", "Channel 1 Stimulus Amp", "Stim Amplitude (mA)")
    st_c2 = plot_stim_amp_bar_scatter(axes[1], mdf, "channel_2_amp", "Channel 2 Stimulus Amp", "Stim Amplitude (mA)")

    add_subplot_label(axes[0], "A", x=-0.14, y=1.08, fontsize=10, fontweight="bold")
    add_subplot_label(axes[1], "B", x=-0.14, y=1.08, fontsize=10, fontweight="bold")

    return fig, [st_c1, st_c2]

def plot_stim_amplitudes_summary(mdf):
    """Combined Multi-Panel Summary Figure for all channels and pathways."""
    print("  → Generating Stimulation Amplitudes Summary (Combined)")
    fig, axes = plt.subplots(2, 2, figsize=(6.93, 5.8))
    fig.subplots_adjust(wspace=0.38, hspace=0.42, left=0.10, right=0.96, top=0.90, bottom=0.12)

    st_p  = plot_stim_amp_bar_scatter(axes[0, 0], mdf, "Perforant_Stim_Amp", "Perforant Path", "Stim Amplitude (mA)")
    st_s  = plot_stim_amp_bar_scatter(axes[0, 1], mdf, "Schaffer_Stim_Amp", "Schaffer Collateral", "Stim Amplitude (mA)")
    st_c1 = plot_stim_amp_bar_scatter(axes[1, 0], mdf, "channel_1_amp", "Channel 1 Overall", "Stim Amplitude (mA)")
    st_c2 = plot_stim_amp_bar_scatter(axes[1, 1], mdf, "channel_2_amp", "Channel 2 Overall", "Stim Amplitude (mA)")

    add_subplot_label(axes[0, 0], "A", x=-0.14, y=1.08, fontsize=10, fontweight="bold")
    add_subplot_label(axes[0, 1], "B", x=-0.14, y=1.08, fontsize=10, fontweight="bold")
    add_subplot_label(axes[1, 0], "C", x=-0.14, y=1.08, fontsize=10, fontweight="bold")
    add_subplot_label(axes[1, 1], "D", x=-0.14, y=1.08, fontsize=10, fontweight="bold")

    return fig, [st_p, st_s, st_c1, st_c2]

def save_fig(fig, name):
    for ext in [".pdf", ".png", ".svg"]:
        fig.savefig(os.path.join(FIGURES_DIR, f"{name}{ext}"), dpi=300, bbox_inches="tight")
    print(f"✓ Saved Figure: {name} (.pdf, .svg, .png)")

def main():
    setup_publication_style()
    print("\n=== Generating Stimulation Amplitude Figures ===\n")

    mdf = pd.read_csv("master_df.csv", low_memory=False)
    mdf["Genotype"] = mdf["Genotype"].replace({"GNB1": "I80T/+"})
    mdf_inc = mdf[mdf["Inclusion"].astype(str).str.contains("Yes", case=False, na=False)].copy()

    # 1. Figure 4 Stimulation Amplitudes
    fig4, stats4 = plot_figure_4_stim_amplitudes(mdf_inc)
    save_fig(fig4, "Figure_4_Stimulation_Amplitudes")

    # 2. Figure 5 Stimulation Amplitudes
    fig5, stats5 = plot_figure_5_stim_amplitudes(mdf_inc)
    save_fig(fig5, "Figure_5_Stimulation_Amplitudes")

    # 3. Combined Summary
    fig_sum, stats_sum = plot_stim_amplitudes_summary(mdf_inc)
    save_fig(fig_sum, "Stimulation_Amplitudes_Summary")

    print("\n✓ Done. All stimulation amplitude figures saved to paper_figures/")

if __name__ == "__main__":
    main()
