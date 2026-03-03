import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import os
import sys
import numpy as np
import pickle
import box_utils

# Import Plotting Functions
from plotting_utils import *

# IMPORT BOX UTILITIES (Crucial for Example Traces)
try:
    import box_utils
except ImportError:
    print("Warning: box_utils not found. Raw data locating features will be disabled.")
    box_utils = None

# ==================================================================================================
# GLOBAL CONFIG & MASTER DF LOADING
# ==================================================================================================

# 1. RAW DATA SOURCE (For Trace Examples Only - Panels B & D)
#    This is the folder name in Box, NOT the local paper_data folder.
RAW_DATA_BOX_FOLDER = "All_Combined_Data"

# 2. PROCESSED DATA SOURCE (For Statistical Plots - Everything else)
#    This is local in the repository.
PAPER_DATA_DIR = "paper_data"

def load_master_df():
    """Locates and loads the master_df.csv from current or parent directories."""
    candidates = [
        'master_df.csv',
        '../master_df.csv',
        '../../master_df.csv'
    ]
    
    for path in candidates:
        if os.path.exists(path):
            print(f"✓ Found Master DF at: {path}")
            try:
                df = pd.read_csv(path, low_memory=False)
                # Ensure Cell_ID is string and clean for matching
                if 'Cell_ID' in df.columns:
                    df['Cell_ID'] = df['Cell_ID'].astype(str).str.strip()
                return df
            except Exception as e:
                print(f"Error loading Master DF: {e}")
                return None
    
    print("❌ Critical Warning: 'master_df.csv' not found. Example traces cannot be indexed.")
    return None

# Load immediately so it is available globally
master_df = load_master_df()


"""Sets matplotlib params for publication-quality figures."""
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 8
plt.rcParams['axes.labelsize'] = 8
plt.rcParams['axes.titlesize'] = 9
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['legend.fontsize'] = 8

# ==================================================================================================
# FIGURE 1: BEHAVIOR
# ==================================================================================================

def plot_figure_1_behavior():
    print("\n--- Generating Figure 1: Behavior ---")
    setup_publication_style()
    
    # Load Dataframes from LOCAL paper_data
    df_weights = load_data('Behavior_Analysis', 'Mouse_Weights_Processed.csv')
    df_of_loco = load_data('Behavior_Analysis', 'Open_Field_Locomotion_Trial1.csv')
    df_of_anx = load_data('Behavior_Analysis', 'Open_Field_Anxiety_Processed.csv')
    df_dvc_hourly = load_data('DVC_Analysis', 'Hourly_Stats_By_Genotype.csv')
    df_dvc_cages = load_data('DVC_Analysis', 'Cage_Specific_Hours_Summary.csv')
    df_tmaze = load_data('Behavior_Analysis', 'T_Maze_Alternations.csv')
    df_tmaze_entries = load_data('Behavior_Analysis', 'T_Maze_Zone_Entries.csv')
    df_stats = load_data('Behavior_Analysis', 'Stats_Results_Figure_1.csv')

    # Calculate Total Entries for T-Maze
    if df_tmaze_entries is not None:
        df_tmaze_entries['Total_Entries'] = df_tmaze_entries[['Start : entries', 'Left Arm : entries', 'Right Arm : entries']].sum(axis=1, skipna=True)

    # Rename GNB1 → I80T/+ for display
    df_weights = rename_genotype(df_weights)
    df_of_loco = rename_genotype(df_of_loco)
    df_of_anx = rename_genotype(df_of_anx)
    df_dvc_hourly = rename_genotype(df_dvc_hourly)
    df_dvc_cages = rename_genotype(df_dvc_cages)
    df_tmaze = rename_genotype(df_tmaze)
    df_tmaze_entries = rename_genotype(df_tmaze_entries)

    # Genotype order for all bar plots
    geno_order = ['WT', 'I80T/+']

    # =========================================================================
    # FIGURE LAYOUT (matching reference):
    # ROW 1: A (Mouse Image placeholder) | B (Body Weight - wider)
    # ROW 2: C (Tracing placeholder + Locomotion bar) | D (Tracing placeholder + Anxiety bar)
    # ROW 3: E (Circadian Activity - wide) | F (Dark Phase bar)
    # ROW 4: G (T-maze tracing placeholder) | H (Distance) | I (Entries) | J (Alternation)
    # =========================================================================
    fig = plt.figure(figsize=(6.93, 9))  # 17.6cm width
    
    gs = fig.add_gridspec(4, 1, hspace=0.6,
                         height_ratios=[1, 1, 1, 1])

    # ===== ROW 1: A (Mouse Image) | B (Body Weight) =====
    gs_row1 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0], wspace=0.4, 
                                               width_ratios=[1, 1.5])
    
    # Panel A: Mouse Image placeholder
    ax_a = fig.add_subplot(gs_row1[0])
    add_subplot_label(ax_a, "A")
    ax_a.text(0.5, 0.5, 'Mouse Image\n(To be added)', 
              ha='center', va='center', fontsize=8, color='gray', style='italic')
    ax_a.set_xticks([])
    ax_a.set_yticks([])
    ax_a.set_facecolor('#f8f8f8')
    for spine in ax_a.spines.values():
        spine.set_visible(False)
    
    # Panel B: Developmental Body Weight
    ax_b = fig.add_subplot(gs_row1[1])
    add_subplot_label(ax_b, "B")
    if df_weights is not None:
        summary = df_weights.groupby(['Timepoint_Label', 'Genotype'])['Weight_g'].agg(['mean', 'sem']).reset_index()
        summary['top'] = summary['mean'] + summary['sem']
        
        plot_longitudinal_lines(ax_b, df_weights, 'Timepoint_Label', 'Weight_g', 'Genotype', 
                                time_order=['P8-P10', 'P28', 'Adult'])
        ax_b.set_title('Developmental Body Weight', fontsize=8)
        # Relabel x-ticks for display
        ax_b.set_xticklabels(['P8-P10', 'P28', 'Adult (P49)'])
        ax_b.legend(frameon=False, fontsize=7)
        ax_b.set_ylabel('Weight (g)')
        
        def get_y_for_time(tp):
            sub = summary[summary['Timepoint_Label'] == tp]
            return sub['top'].max() * 1.1 if not sub.empty else 30

        annotate_from_stats(ax_b, df_stats, "Fig 1B", "P8-P10", x1=0, x2=0, y_pos=get_y_for_time('P8-P10'), bracket=True)
        annotate_from_stats(ax_b, df_stats, "Fig 1B", "P28", x1=1, x2=1, y_pos=get_y_for_time('P28'), bracket=True)
        annotate_from_stats(ax_b, df_stats, "Fig 1B", "Adult", x1=2, x2=2, y_pos=get_y_for_time('Adult'), bracket=True)

    # ===== ROW 2: C (Tracing + Locomotion) | D (Tracing + Anxiety) =====
    gs_row2 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[1], wspace=0.4)
    
    # Panel C: Open Field Locomotion (tracing placeholder + bar)
    gs_c = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_row2[0], wspace=0.2, 
                                            width_ratios=[0.6, 1])
    # C-left: tracing placeholder
    ax_c_trace = fig.add_subplot(gs_c[0])
    add_subplot_label(ax_c_trace, "C")
    ax_c_trace.text(0.5, 0.75, 'WT', ha='center', va='center', fontsize=8, fontweight='bold')
    ax_c_trace.text(0.5, 0.25, 'I80T/+', ha='center', va='center', fontsize=8, fontweight='bold', color='red')
    ax_c_trace.set_facecolor('#f8f8f8')
    ax_c_trace.set_xticks([])
    ax_c_trace.set_yticks([])
    for spine in ax_c_trace.spines.values():
        spine.set_visible(False)
    # C-right: bar plot
    ax_c = fig.add_subplot(gs_c[1])
    if df_of_loco is not None:
        plot_bar_scatter(ax_c, df_of_loco, 'Genotype', 'Distance (m)', 'Genotype', order=geno_order)
        ax_c.set_title('Locomotion (Open Field)', fontsize=8)
        annotate_from_stats(ax_c, df_stats, "Fig 1C", "Locomotion", x1=0, x2=1, y_pos=get_safe_y(df_of_loco['Distance (m)']))

    # Panel D: Anxiety (tracing placeholder + bar)
    gs_d = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_row2[1], wspace=0.2,
                                            width_ratios=[0.6, 1])
    # D-left: tracing placeholder
    ax_d_trace = fig.add_subplot(gs_d[0])
    add_subplot_label(ax_d_trace, "D")
    ax_d_trace.text(0.5, 0.5, 'Open Field\nAnxiety Tracing\n(To be added)', 
              ha='center', va='center', fontsize=7, color='gray', style='italic')
    ax_d_trace.set_facecolor('#f8f8f8')
    ax_d_trace.set_xticks([])
    ax_d_trace.set_yticks([])
    for spine in ax_d_trace.spines.values():
        spine.set_visible(False)
    # D-right: bar plot
    ax_d = fig.add_subplot(gs_d[1])
    if df_of_anx is not None:
        plot_bar_scatter(ax_d, df_of_anx, 'Genotype', 'Center_Outer_Time_Ratio', 'Genotype', order=geno_order)
        ax_d.set_title('Anxiety Ratio', fontsize=8)
        ax_d.set_ylabel('Center Outer Time Ratio')
        annotate_from_stats(ax_d, df_stats, "Fig 1D", "Anxiety", x1=0, x2=1, y_pos=get_safe_y(df_of_anx['Center_Outer_Time_Ratio']))

    # ===== ROW 3: E (Circadian Activity) | F (Total Dark Phase) =====
    gs_row3 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[2], wspace=0.4, 
                                               width_ratios=[1.2, 1])
    
    # Panel E: Circadian Activity
    ax_e = fig.add_subplot(gs_row3[0])
    add_subplot_label(ax_e, "E")
    if df_dvc_hourly is not None:
        plot_dvc_hourly(ax_e, df_dvc_hourly)
        ax_e.set_title('Circadian Activity', fontsize=8)
        # Add Dark Phase label to legend
        from matplotlib.patches import Patch
        handles, labels = ax_e.get_legend_handles_labels()
        handles.append(Patch(facecolor='blue', alpha=0.1, label='Dark Phase'))
        labels.append('Dark Phase')
        ax_e.legend(handles=handles, labels=labels, frameon=False, loc='upper left', fontsize=7)

    # Panel F: Total Activity (Dark Phase)
    ax_f = fig.add_subplot(gs_row3[1])
    add_subplot_label(ax_f, "F")
    if df_dvc_cages is not None:
        plot_bar_scatter(ax_f, df_dvc_cages, 'Genotype', 'Sum_All_Dark', 'Genotype', order=geno_order)
        ax_f.set_title('Total Activity (Dark Phase)', fontsize=8)
        ax_f.set_ylabel('Summed Activity (m)')
        if df_stats is not None:
            annotate_from_stats(ax_f, df_stats, "Fig 1G", "DVC Dark Phase", x1=0, x2=1, y_pos=get_safe_y(df_dvc_cages['Sum_All_Dark']))

    # ===== ROW 4: G (T-maze tracing) | H (Distance) | I (Entries) | J (Alternation) =====
    gs_row4 = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=gs[3], wspace=0.5, 
                                               width_ratios=[1, 1, 1, 1])
    
    # Panel G: T-Maze Tracing Placeholder
    ax_g = fig.add_subplot(gs_row4[0])
    add_subplot_label(ax_g, "G")
    ax_g.text(0.5, 0.75, 'WT', ha='center', va='center', fontsize=8, fontweight='bold')
    ax_g.text(0.5, 0.25, 'I80T/+', ha='center', va='center', fontsize=8, fontweight='bold', color='red')
    ax_g.set_xticks([])
    ax_g.set_yticks([])
    ax_g.set_facecolor('#f5e6c8')
    for spine in ax_g.spines.values():
        spine.set_visible(False)
    
    # Panel H: Distance Traveled
    ax_h = fig.add_subplot(gs_row4[1])
    add_subplot_label(ax_h, "H")
    if df_tmaze_entries is not None:
        plot_bar_scatter(ax_h, df_tmaze_entries, 'Genotype', 'Distance (m)', 'Genotype', order=geno_order)
        ax_h.set_title('Distance Traveled', fontsize=8)
        ax_h.set_ylabel('Distance (m)')
        ax_h.yaxis.set_major_locator(plt.MaxNLocator(6))
        annotate_from_stats(ax_h, df_stats, "Fig 1I", "Distance", x1=0, x2=1, y_pos=get_safe_y(df_tmaze_entries['Distance (m)']))
    
    # Panel I: Total Port Entries
    ax_i = fig.add_subplot(gs_row4[2])
    add_subplot_label(ax_i, "I")
    if df_tmaze_entries is not None:
        plot_bar_scatter(ax_i, df_tmaze_entries, 'Genotype', 'Total_Entries', 'Genotype', order=geno_order)
        ax_i.set_title('Total Port Entries', fontsize=8)
        ax_i.set_ylabel('Total Entries')
        annotate_from_stats(ax_i, df_stats, "Fig 1J", "Total Entries", x1=0, x2=1, y_pos=get_safe_y(df_tmaze_entries['Total_Entries']))
    
    # Panel J: Spontaneous Alternation
    ax_j = fig.add_subplot(gs_row4[3])
    add_subplot_label(ax_j, "J")
    if df_tmaze is not None:
        plot_bar_scatter(ax_j, df_tmaze, 'Genotype', 'Percent_Alternations', 'Genotype', order=geno_order)
        ax_j.set_title('Spontaneous Alternation', fontsize=8)
        ax_j.set_ylabel('Percent Alternations')
        ax_j.set_ylim(-1, 100)
        annotate_from_stats(ax_j, df_stats, "Fig 1K", "Alternation", x1=0, x2=1, y_pos=90)

    save_current_fig('Figure_1_Behavior')


# ==================================================================================================
# FIGURE 2: PHYSIOLOGY
# ==================================================================================================

def plot_figure_2_physiology():
    print("\n--- Generating Figure 2: Physiology ---")
    setup_publication_style()
    
    # ---------------------------------------------------------
    # 1. Load Processed Data (FROM REPO paper_data)
    # ---------------------------------------------------------
    df_intrinsic = load_data('Physiology_Analysis', 'Intrinsic_properties.csv')
    df_ap_ahp = load_data('Physiology_Analysis', 'combined_AP_AHP_rheobase_analysis.csv')
    df_stats = load_data('Physiology_Analysis', 'Stats_Results_Figure_2.csv')

    # Firing Rate & ISI (Unified CSV)
    df_FI = load_data('Firing_Rate', 'Firing_Rates_plotting_format.csv')
    
    # Load FI Midpoints (from sigmoid fitting)
    fi_midpoints_df = load_data('Firing_Rate', 'Sigmoid_Fit_Params.csv')
    if fi_midpoints_df is not None and 'Midpoint' in fi_midpoints_df.columns:
        fi_midpoints_df = fi_midpoints_df.rename(columns={'Midpoint': 'FI_Midpoint'})
    
    # Process FI Curves and ISI Curves
    fi_df_final, fi_df_long = get_FI_data(df_FI)
    isi_df_final = prepare_isi_curve_data(df_FI)

    # Standardize Genotypes
    for df in [df_intrinsic, df_ap_ahp, df_FI, fi_df_final, isi_df_final, fi_midpoints_df]:
        if df is not None and 'Genotype' in df.columns:
            df['Genotype'] = df['Genotype'].str.strip()

    # Rename GNB1 -> I80T/+ for display
    df_intrinsic = rename_genotype(df_intrinsic)
    df_ap_ahp = rename_genotype(df_ap_ahp)
    df_FI = rename_genotype(df_FI)
    fi_df_final = rename_genotype(fi_df_final)
    isi_df_final = rename_genotype(isi_df_final)
    fi_midpoints_df = rename_genotype(fi_midpoints_df)

    # ---------------------------------------------------------
    # 2. Locate Raw Data (FROM BOX for Trace Panels)
    # ---------------------------------------------------------
    raw_traces_path = None
    
    if box_utils:
        raw_traces_path = box_utils.get_data_path(target_folder_name=RAW_DATA_BOX_FOLDER)
    
    if not raw_traces_path and os.path.exists(RAW_DATA_BOX_FOLDER):
        raw_traces_path = RAW_DATA_BOX_FOLDER

    if not raw_traces_path:
        print(f"⚠ Warning: Raw Data Folder '{RAW_DATA_BOX_FOLDER}' not found via Box or locally.") 
        print("   Panel A will be a placeholder.")

    # Export summary table to CSV
    from plotting_utils import export_physiology_summary_table
    export_physiology_summary_table(df_intrinsic, df_ap_ahp, df_stats,
                                   'paper_figures/Figure_2_Summary_Table.csv')
    print('✓ Exported summary table to paper_figures/Figure_2_Summary_Table.csv')

    # ---------------------------------------------------------
    # 3. Setup Layout
    # ---------------------------------------------------------
    # Row 1: A (Input Resistance bar) | B (Vsag traces + Vsag bar) | C (AP traces + AHP bar)
    # Row 2: D (FI traces stacked)    | E (F-I curve)
    # Row 3: F (FI Midpoint)          | G (Rheobase) | H (ISI traces) | I (ISI adaptation)

    fig = plt.figure(figsize=(6.89, 8.5))  # slightly taller to accommodate 3 rows
    outer_grid = gridspec.GridSpec(3, 1, height_ratios=[0.55, 0.50, 0.55], hspace=0.55)

    # ==========================================================
    # ROW 1: A | B | C
    # A = Input Resistance bar (narrow)
    # B = Vsag traces stacked (top) + Vsag bar (bottom)
    # C = AP traces stacked   (top) + AHP Decay bar (bottom)
    # ==========================================================
    gs_row1 = gridspec.GridSpecFromSubplotSpec(
        1, 3, subplot_spec=outer_grid[0],
        width_ratios=[0.22, 0.39, 0.39], wspace=0.45)

    # --- Panel A: Input Resistance bar ---
    ax_a = fig.add_subplot(gs_row1[0])
    add_subplot_label(ax_a, "A")
    if df_intrinsic is not None and 'Input_Resistance_MOhm' in df_intrinsic.columns:
        plot_bar_scatter(ax_a, df_intrinsic, 'Genotype', 'Input_Resistance_MOhm',
                         'Genotype', order=['WT', 'I80T/+'])
        ax_a.set_ylabel('Input Resistance (MΩ)', fontsize=7)
        ax_a.set_title('Input Resistance', fontsize=8)
        ax_a.set_box_aspect(1.4)
        if df_stats is not None:
            annotate_from_stats(ax_a, df_stats, "Fig 2A", "Input Resistance",
                                x1=0, x2=1,
                                y_pos=get_safe_y(df_intrinsic['Input_Resistance_MOhm']))
    else:
        ax_a.text(0.5, 0.5, 'Rin Missing', ha='center', color='red')

    # --- Panel B: Voltage Sag traces (stacked) + Voltage Sag bar ---
    gs_B = gridspec.GridSpecFromSubplotSpec(
        2, 2, subplot_spec=gs_row1[1],
        height_ratios=[0.55, 0.45], hspace=0.35, wspace=0.25)

    # B-top: two trace axes side by side (WT left, I80T/+ right)
    ax_b_wt  = fig.add_subplot(gs_B[0, 0])
    ax_b_gnb = fig.add_subplot(gs_B[0, 1])
    add_subplot_label(ax_b_wt, "B")

    if raw_traces_path and master_df is not None:
        from plotting_utils import plot_voltage_sag_comparison
        plot_voltage_sag_comparison(ax_b_wt, ax_b_gnb, raw_traces_path, master_df,
                                    target_wt='03142024_c2', target_gnb1='02132024_c1')
        ax_b_wt.set_ylim(-95, -50)
        ax_b_gnb.set_ylim(-95, -50)
    else:
        plot_trace_placeholder(ax_b_wt, "Unavailable")
        plot_trace_placeholder(ax_b_gnb, "Unavailable")

    # B-bottom: Voltage Sag bar plot spanning both columns
    ax_b_bar = fig.add_subplot(gs_B[1, :])
    if df_intrinsic is not None and 'Voltage_sag' in df_intrinsic.columns:
        plot_bar_scatter(ax_b_bar, df_intrinsic, 'Genotype', 'Voltage_sag',
                         'Genotype', order=['WT', 'I80T/+'])
        ax_b_bar.set_ylabel('Voltage Sag (%)', fontsize=7)
        ax_b_bar.set_title('Voltage Sag', fontsize=8)
        ax_b_bar.set_box_aspect(0.7)
        if df_stats is not None:
            annotate_from_stats(ax_b_bar, df_stats, "Fig 2A", "Voltage Sag",
                                x1=0, x2=1,
                                y_pos=get_safe_y(df_intrinsic['Voltage_sag']))
    else:
        ax_b_bar.text(0.5, 0.5, 'Sag Missing', ha='center', color='red')

    # --- Panel C: AP traces (stacked) + AHP Decay bar ---
    gs_C = gridspec.GridSpecFromSubplotSpec(
        2, 2, subplot_spec=gs_row1[2],
        height_ratios=[0.55, 0.45], hspace=0.35, wspace=0.25)

    ax_c_wt  = fig.add_subplot(gs_C[0, 0])
    ax_c_gnb = fig.add_subplot(gs_C[0, 1])
    add_subplot_label(ax_c_wt, "C")

    if raw_traces_path and master_df is not None:
        target_wt_rheo  = '03142024_c2'
        target_gnb_rheo = '05092024_c3'
        sweep_idx_wt  = get_sweep_index_from_master(master_df, target_wt_rheo)
        sweep_idx_gnb = get_sweep_index_from_master(master_df, target_gnb_rheo)

        if sweep_idx_wt is not None:
            plot_example_rheobase_and_sweeps(ax_c_wt, raw_traces_path, master_df=master_df,
                                             target_cell_id=target_wt_rheo,
                                             sweep_idx=sweep_idx_wt,
                                             analysis_df=df_ap_ahp, show_values=False)
            ax_c_wt.axis('off')
            ax_c_wt.text(0.02, 0.95, 'WT', transform=ax_c_wt.transAxes,
                         fontsize=8, fontweight='bold', va='top')
        else:
            plot_trace_placeholder(ax_c_wt, "WT")

        if sweep_idx_gnb is not None:
            plot_example_rheobase_and_sweeps(ax_c_gnb, raw_traces_path, master_df=master_df,
                                             target_cell_id=target_gnb_rheo,
                                             sweep_idx=sweep_idx_gnb,
                                             analysis_df=df_ap_ahp, show_values=False)
            ax_c_gnb.axis('off')
            ax_c_gnb.text(0.02, 0.95, 'I80T/+', transform=ax_c_gnb.transAxes,
                          fontsize=8, fontweight='bold', va='top', color='red')
            add_scale_bar(ax_c_gnb, 5, 20, x_pos=0.8, y_pos=0.1)
        else:
            plot_trace_placeholder(ax_c_gnb, "I80T/+")

        ax_c_wt.set_ylim(-80, 50)
        ax_c_gnb.set_ylim(-80, 50)
    else:
        plot_trace_placeholder(ax_c_wt, "Unavailable")
        plot_trace_placeholder(ax_c_gnb, "Unavailable")

    # C-bottom: AHP Decay bar spanning both columns
    ax_c_bar = fig.add_subplot(gs_C[1, :])
    if df_ap_ahp is not None and 'decay_area' in df_ap_ahp.columns:
        plot_bar_scatter(ax_c_bar, df_ap_ahp, 'Genotype', 'decay_area',
                         'Genotype', order=['WT', 'I80T/+'])
        ax_c_bar.set_ylabel('AHP Area\n(mV·ms)', fontsize=7)
        ax_c_bar.set_title('AHP Decay', fontsize=8)
        ax_c_bar.set_box_aspect(0.7)
        if df_stats is not None:
            annotate_from_stats(ax_c_bar, df_stats, "Fig 2E", "AHP Decay",
                                x1=0, x2=1,
                                y_pos=get_safe_y(df_ap_ahp['decay_area']))
    else:
        ax_c_bar.text(0.5, 0.5, 'AHP Missing', ha='center', color='red')

    # ==========================================================
    # ROW 2: D (FI example traces stacked) | E (F-I curve)
    # ==========================================================
    gs_row2 = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=outer_grid[1],
        width_ratios=[0.30, 0.70], wspace=0.35)

    # Panel D: FI traces stacked
    gs_d = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_row2[0], hspace=0.1)
    ax_d_wt  = fig.add_subplot(gs_d[0])
    ax_d_gnb = fig.add_subplot(gs_d[1])
    add_subplot_label(ax_d_wt, "D")

    if raw_traces_path and master_df is not None:
        master_df_copy = master_df.copy()
        master_df_copy['File_Cell_ID'] = master_df_copy['Cell_ID'].apply(convert_cell_id_format)

        wt_cells   = master_df_copy[master_df_copy['Genotype'] == 'WT']['File_Cell_ID'].tolist()
        gnb1_cells = master_df_copy[master_df_copy['Genotype'] == 'GNB1']['File_Cell_ID'].tolist()

        wt_trace, wt_current     = None, None
        gnb1_trace, gnb1_current = None, None

        preferred_wt   = ['04042024_c1', '02262024_c2', '03142024_c1', '02132024_c2']
        preferred_gnb1 = ['02262024_c1', '04042024_c2', '02132024_c1']
        for cell in preferred_wt + wt_cells[:30]:
            t, c = find_200pA_trace_direct(cell, raw_traces_path)
            if t is not None:
                wt_trace, wt_current = t, c; break
        for cell in preferred_gnb1 + gnb1_cells[:30]:
            t, c = find_200pA_trace_direct(cell, raw_traces_path)
            if t is not None:
                gnb1_trace, gnb1_current = t, c; break

        common_baseline = -65
        if wt_trace is not None:
            time = np.arange(len(wt_trace)) / 20
            bl = np.mean(wt_trace[int(0.1*len(wt_trace)):int(0.15*len(wt_trace))])
            ax_d_wt.plot(time, wt_trace - bl + common_baseline, 'k-', linewidth=0.8)
            ax_d_wt.text(0.02, 0.95, 'WT', transform=ax_d_wt.transAxes,
                         fontsize=8, fontweight='bold', va='top')
        ax_d_wt.set_title('~200 pA', fontsize=8)

        if gnb1_trace is not None:
            time = np.arange(len(gnb1_trace)) / 20
            bl = np.mean(gnb1_trace[int(0.1*len(gnb1_trace)):int(0.15*len(gnb1_trace))])
            ax_d_gnb.plot(time, gnb1_trace - bl + common_baseline, 'r-', linewidth=0.8)
            ax_d_gnb.text(0.02, 0.95, 'I80T/+', transform=ax_d_gnb.transAxes,
                          fontsize=8, fontweight='bold', va='top', color='red')

        max_time = max(len(wt_trace) if wt_trace is not None else 0,
                       len(gnb1_trace) if gnb1_trace is not None else 0) / 20
        xlim_start, xlim_end = 150, max_time
        wt_aligned   = wt_trace   - np.mean(wt_trace  [int(0.1*len(wt_trace  )):int(0.15*len(wt_trace  ))]) + common_baseline if wt_trace is not None else None
        gnb1_aligned = gnb1_trace - np.mean(gnb1_trace[int(0.1*len(gnb1_trace)):int(0.15*len(gnb1_trace))]) + common_baseline if gnb1_trace is not None else None
        ylim_min = min(wt_aligned.min()   if wt_aligned   is not None else -100,
                       gnb1_aligned.min() if gnb1_aligned is not None else -100) - 10
        ylim_max = max(wt_aligned.max()   if wt_aligned   is not None else 50,
                       gnb1_aligned.max() if gnb1_aligned is not None else 50) + 10
        for ax in (ax_d_wt, ax_d_gnb):
            ax.set_xlim(xlim_start, xlim_end)
            ax.set_ylim(ylim_min, ylim_max)
            ax.axis('off')
        add_scale_bar(ax_d_gnb, 100, 20, x_pos=0.85, y_pos=0.1)
    else:
        ax_d_wt.text(0.5, 0.5, 'Data unavailable', ha='center', va='center', fontsize=8)
        ax_d_wt.axis('off')
        ax_d_gnb.axis('off')

    # Panel E: F-I curve
    ax_e = fig.add_subplot(gs_row2[1])
    add_subplot_label(ax_e, "E")
    if fi_df_final is not None:
        for genotype in fi_df_final['Genotype'].unique():
            subset = fi_df_final[fi_df_final['Genotype'] == genotype]
            n_cells = len(df_FI[df_FI['Genotype'] == genotype])
            color = COLORS.get(genotype, 'gray')
            ax_e.errorbar(
                subset['Current'].to_numpy(), subset['mean_rate'].to_numpy(),
                yerr=subset['sem_rate'].to_numpy(),
                label=f"{genotype} (n={n_cells})",
                color=color, marker='o', capsize=2, markersize=3)
        ax_e.set_xlabel('Current (pA)')
        ax_e.set_ylabel('Firing Rate (Hz)')
        ax_e.spines['top'].set_visible(False)
        ax_e.spines['right'].set_visible(False)
        ax_e.set_box_aspect(1)
        ax_e.legend(loc='upper left', frameon=False, fontsize=8)
    else:
        ax_e.text(0.5, 0.5, 'Curve Data Missing', ha='center', color='red')

    # ==========================================================
    # ROW 3: F (FI Midpoint) | G (Rheobase) | H (ISI traces) | I (ISI adaptation)
    # ==========================================================
    gs_row3 = gridspec.GridSpecFromSubplotSpec(
        1, 4, subplot_spec=outer_grid[2],
        width_ratios=[0.22, 0.22, 0.26, 0.30], wspace=0.4)

    # Panel F: F-I Midpoint
    ax_f = fig.add_subplot(gs_row3[0])
    add_subplot_label(ax_f, "F")
    if fi_midpoints_df is not None:
        plot_bar_scatter(ax_f, fi_midpoints_df, 'Genotype', 'FI_Midpoint',
                         'Genotype', order=['WT', 'I80T/+'])
        ax_f.set_ylabel('F-I Curve Midpoint (pA)')
        ax_f.set_title('F-I Midpoint', fontsize=8)
        ax_f.set_box_aspect(1)
        if df_stats is not None:
            annotate_from_stats(ax_f, df_stats, 'Fig 2F', 'F-I Curve Midpoint',
                                x1=0, x2=1, y_pos=get_safe_y(fi_midpoints_df['FI_Midpoint']))
    else:
        ax_f.text(0.5, 0.5, 'Midpoint Data Missing', ha='center', color='red')

    # Panel G: Rheobase
    ax_rheo = fig.add_subplot(gs_row3[1])
    add_subplot_label(ax_rheo, "G")
    if df_ap_ahp is not None and 'Rheobase_Current' in df_ap_ahp.columns:
        plot_data = df_ap_ahp.dropna(subset=['Rheobase_Current'])
        plot_bar_scatter(ax_rheo, plot_data, 'Genotype', 'Rheobase_Current',
                         'Genotype', order=['WT', 'I80T/+'])
        ax_rheo.set_ylabel('Rheobase (pA)')
        ax_rheo.set_title('Rheobase', fontsize=8)
        ax_rheo.set_box_aspect(1)
        if df_stats is not None:
            annotate_from_stats(ax_rheo, df_stats, "Fig 2C", "Rheobase",
                                x1=0, x2=1, y_pos=get_safe_y(plot_data['Rheobase_Current']))
    else:
        ax_rheo.text(0.5, 0.5, 'Rheobase Data Missing', ha='center', color='red')

    # Panel H: ISI example traces
    gs_h = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_row3[2], hspace=0.1)
    ax_h_wt  = fig.add_subplot(gs_h[0])
    ax_h_gnb = fig.add_subplot(gs_h[1])
    add_subplot_label(ax_h_wt, "H")
    if raw_traces_path and master_df is not None:
        from plotting_utils import plot_isi_example_traces
        plot_isi_example_traces(ax_h_wt, ax_h_gnb, raw_traces_path, master_df, df_ap_ahp)
    else:
        ax_h_wt.text(0.5, 0.5, 'ISI Traces Unavailable', ha='center', color='red')
        ax_h_wt.axis('off')
        ax_h_gnb.axis('off')

    # Panel I: ISI adaptation curve
    ax_i = fig.add_subplot(gs_row3[3])
    add_subplot_label(ax_i, "I")
    if isi_df_final is not None and not isi_df_final.empty:
        for genotype in isi_df_final['Genotype'].unique():
            subset = isi_df_final[isi_df_final['Genotype'] == genotype]
            subset = subset[(subset['Spike_Index'] >= 2) & (subset['Spike_Index'] <= 6)]
            n_cells = len(df_FI[df_FI['Genotype'] == genotype])
            color = COLORS.get(genotype, 'gray')
            y_col = 'mean_isi' if 'mean_isi' in subset.columns else 'mean'
            y_err = 'sem_isi'  if 'sem_isi'  in subset.columns else 'sem'
            ax_i.errorbar(
                subset['Spike_Index'].to_numpy(), subset[y_col].to_numpy(),
                yerr=subset[y_err].to_numpy(),
                label=f"{genotype} (n={n_cells})",
                color=color, marker='o', capsize=2, markersize=3)
        ax_i.set_xticks([2, 3, 4, 5, 6])
        ax_i.set_xticklabels(['2', '3', '4', '5', '6'])
        ax_i.set_xlim(1.5, 6.5)
        ax_i.set_xlabel('AP Spike Number')
        ax_i.set_ylabel('ISI (ms)')
        ax_i.spines['top'].set_visible(False)
        ax_i.spines['right'].set_visible(False)
        ax_i.set_box_aspect(1)
        ax_i.legend(loc='upper left', frameon=False, fontsize=8)
    else:
        ax_i.text(0.5, 0.5, 'ISI Data Missing', ha='center', color='red')
        ax_i.set_title('Spike Rate Adaptation')

    save_current_fig('Figure_2_Physiology')



# ==================================================================================================
# FIGURE 3: MORPHOLOGY
# ==================================================================================================


def plot_figure_3_morphology():
    print("\n--- Generating Figure 3: Morphology ---")
    setup_publication_style()
    
    # Load Sholl Intersections Data
    df_sholl = load_data('Morphology_Analysis', 'Sholl_Intersections_Raw.csv')
    df_cdf = load_data('Morphology_Analysis', 'Sholl_Cumulative_Distributions.csv')
    df_dend_props = load_data('Morphology_Analysis', 'Dendrite_Properties_All.csv')
    
    if df_sholl is None:
        print("❌ Error: Could not load Sholl data")
        return
    
    if df_cdf is None:
        print("⚠ Warning: Could not load cumulative distribution data, panels D and E will be empty")
    
    if df_dend_props is None:
        print("⚠ Warning: Could not load dendrite properties data, panels F and G will be empty")
    
    
    # Load Stats
    df_stats = load_data('Morphology_Analysis', 'Stats_Results_Figure_3.csv')
        
    # Separate data by genotype
    # Rename GNB1 -> I80T/+ for display
    df_sholl = rename_genotype(df_sholl)
    df_cdf = rename_genotype(df_cdf)
    df_dend_props = rename_genotype(df_dend_props)

    df_wt = df_sholl[df_sholl['Genotype'] == 'WT'].copy()
    df_gnb1 = df_sholl[df_sholl['Genotype'] == 'I80T/+'].copy()
    
    # Create figure with 7 panels: A (reconstructions), B (Basal Sholl), C (Apical Sholl), 
    #                               D (Basal CDF), E (Apical CDF), F (Branch Sum), G (Terminal Branches)
    fig = plt.figure(figsize=(6.93, 7))  # 17.6cm width, taller for morphology
    gs = fig.add_gridspec(3, 3, wspace=0.5, hspace=0.6, height_ratios=[1, 1, 1])
    
    # Panel A: Placeholder for cell reconstructions (spans 3 rows)
    ax_reconstructions = fig.add_subplot(gs[:, 0])
    add_subplot_label(ax_reconstructions, "A")
    ax_reconstructions.text(0.5, 0.5, 'Cell Reconstructions\n(To be added)', 
                           ha='center', va='center', fontsize=8, color='gray')
    ax_reconstructions.set_xticks([])
    ax_reconstructions.set_yticks([])
    ax_reconstructions.set_facecolor('#f8f8f8')
    ax_reconstructions.spines['top'].set_visible(False)
    ax_reconstructions.spines['right'].set_visible(False)
    ax_reconstructions.spines['bottom'].set_visible(False)
    ax_reconstructions.spines['left'].set_visible(False)
    
    # Panel B: Basal Dendrites (Sholl)
    ax_basal = fig.add_subplot(gs[0, 1])
    add_subplot_label(ax_basal, "B")
    
    # Panel C: Apical Dendrites (Sholl)
    ax_apical = fig.add_subplot(gs[0, 2])
    add_subplot_label(ax_apical, "C")
    
    # Panel D: Basal Cumulative Distribution
    ax_basal_cdf = fig.add_subplot(gs[1, 1])
    add_subplot_label(ax_basal_cdf, "D")
    
    # Panel E: Apical Cumulative Distribution
    ax_apical_cdf = fig.add_subplot(gs[1, 2])
    add_subplot_label(ax_apical_cdf, "E")
    
    # Panels F and G will be created as subgrids later when plotting dendritic properties
    
    
    # Plot Basal Dendrites
    plot_sholl_data(ax_basal, df_wt, 'WT', 'Basal', COLORS['WT'])
    plot_sholl_data(ax_basal, df_gnb1, 'I80T/+', 'Basal', COLORS['GNB1'])
    ax_basal.set_title('Basal Dendrites', fontsize=9)
    ax_basal.set_xlabel('Distance from Soma (μm)', fontsize=8)
    ax_basal.set_ylabel('Number of Intersections', fontsize=8)
    ax_basal.spines['top'].set_visible(False)
    ax_basal.spines['right'].set_visible(False)
    ax_basal.legend(frameon=False, loc='upper right')
    
    # Plot Apical Dendrites
    plot_sholl_data(ax_apical, df_wt, 'WT', 'Apical', COLORS['WT'])
    plot_sholl_data(ax_apical, df_gnb1, 'I80T/+', 'Apical', COLORS['GNB1'])
    ax_apical.set_title('Apical Dendrites', fontsize=9)
    ax_apical.set_xlabel('Distance from Soma (μm)', fontsize=8)
    ax_apical.set_ylabel('Number of Intersections', fontsize=8)
    ax_apical.spines['top'].set_visible(False)
    ax_apical.spines['right'].set_visible(False)
    ax_apical.legend(frameon=False, loc='upper right')
    
    # Plot Cumulative Distributions (Combined across sexes)
    if df_cdf is not None:
        error_multiplier = 5
        
        # Panel D: Basal Cumulative Distribution
        basal_cdf = df_cdf[df_cdf['Dendrite_Type'] == 'Basal']
        # Aggregate across sexes for each genotype
        for genotype, group_df in basal_cdf.groupby('Genotype'):
            color = COLORS.get(genotype, 'gray')
            
            # Average across sexes for each CDF_Bin
            agg_data = group_df.groupby('CDF_Bin').agg({
                'Radius_Quantile': 'mean',
                'Radius_SEM': lambda x: np.sqrt(np.sum(x**2)) / len(x)  # Combined SEM
            }).reset_index()
            
            x = agg_data['Radius_Quantile'].values
            y = agg_data['CDF_Bin'].values
            xerr = agg_data['Radius_SEM'].values * error_multiplier
            xerr = np.nan_to_num(xerr, nan=0.0)
            
            ax_basal_cdf.errorbar(x, y, xerr=xerr, fmt='o-', color=color, 
                                 markersize=1, capsize=1, label=genotype, alpha=0.8, linewidth=0.5)
        
        ax_basal_cdf.set_xlabel('Distance from Soma (μm)', fontsize=8)
        ax_basal_cdf.set_ylabel('Cumulative Probability', fontsize=8)
        #ax_basal_cdf.set_title('Basal Dendrites - Cumulative Distribution', fontsize=9)
        ax_basal_cdf.set_ylim(0, 1.05)
        ax_basal_cdf.set_xlim(0, ax_basal_cdf.get_xlim()[1])
        ax_basal_cdf.spines['top'].set_visible(False)
        ax_basal_cdf.spines['right'].set_visible(False)
        ax_basal_cdf.legend(frameon=False, loc='lower right', fontsize=8)
        
        # Add KS Test Stat
        # Using bracket=False for simple text annotation
        annotate_from_stats(ax_basal_cdf, df_stats, 'Fig 3D', 'Basal Sholl', 0.1, 0.1, 1.0, bracket=False)
        
        # Panel E: Apical Cumulative Distribution
        apical_cdf = df_cdf[df_cdf['Dendrite_Type'] == 'Apical']
        # Aggregate across sexes for each genotype
        for genotype, group_df in apical_cdf.groupby('Genotype'):
            color = COLORS.get(genotype, 'gray')
            
            # Average across sexes for each CDF_Bin
            agg_data = group_df.groupby('CDF_Bin').agg({
                'Radius_Quantile': 'mean',
                'Radius_SEM': lambda x: np.sqrt(np.sum(x**2)) / len(x)  # Combined SEM
            }).reset_index()
            
            x = agg_data['Radius_Quantile'].values
            y = agg_data['CDF_Bin'].values
            xerr = agg_data['Radius_SEM'].values * error_multiplier
            xerr = np.nan_to_num(xerr, nan=0.0)
            
            ax_apical_cdf.errorbar(x, y, xerr=xerr, fmt='o-', color=color, 
                                  markersize=1, capsize=1, label=genotype, alpha=0.8, linewidth=0.5)
        
        ax_apical_cdf.set_xlabel('Distance from Soma (μm)', fontsize=8)
        ax_apical_cdf.set_ylabel('Cumulative Probability', fontsize=8)
        #ax_apical_cdf.set_title('Apical Dendrites ', fontsize=9)
        ax_apical_cdf.set_ylim(0, 1.05)
        ax_apical_cdf.set_xlim(0, ax_apical_cdf.get_xlim()[1])
        ax_apical_cdf.spines['top'].set_visible(False)
        ax_apical_cdf.spines['right'].set_visible(False)
        ax_apical_cdf.legend(frameon=False, loc='lower right', fontsize=8)
        
        # Add KS Test Stat
        annotate_from_stats(ax_apical_cdf, df_stats, 'Fig 3E', 'Apical Sholl', 0.1, 0.1, 1.0, bracket=False)
    else:
        # Add placeholders if CDF data is not available
        ax_basal_cdf.text(0.5, 0.5, 'CDF Data Unavailable', ha='center', va='center', color='gray')
        ax_basal_cdf.set_xticks([])
        ax_basal_cdf.set_yticks([])
        
        ax_apical_cdf.text(0.5, 0.5, 'CDF Data Unavailable', ha='center', va='center', color='gray')
        ax_apical_cdf.set_xticks([])
        ax_apical_cdf.set_yticks([])
    
    # Synchronize X-axis max values for panels that show the same data
    # B (ax_basal) and D (ax_basal_cdf) should have the same X-max
    # C (ax_apical) and E (ax_apical_cdf) should have the same X-max
    basal_xmax = max(ax_basal.get_xlim()[1], ax_basal_cdf.get_xlim()[1])
    ax_basal.set_xlim(0, basal_xmax)
    ax_basal_cdf.set_xlim(0, basal_xmax)
    
    apical_xmax = max(ax_apical.get_xlim()[1], ax_apical_cdf.get_xlim()[1])
    ax_apical.set_xlim(0, apical_xmax)
    ax_apical_cdf.set_xlim(0, apical_xmax)
    
    # Plot Dendritic Properties
    if df_dend_props is not None:
        # Create 2x2 grid for F and G
        gs_F = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[2, 1], wspace=0.6)
        gs_G = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[2, 2], wspace=0.6)
        
        # Panel F: Branch Sum (Total Branch Length)
        # F1: Basal Branch Sum
        ax_f_basal = fig.add_subplot(gs_F[0])
        add_subplot_label(ax_f_basal, "F")
        df_basal_props = df_dend_props[df_dend_props['Dendrite_Type'] == 'Basal']
        max_h = plot_bar_scatter(ax_f_basal, df_basal_props, 'Genotype', 'branch_sum', 'Genotype', order=['WT', 'I80T/+'])
        annotate_from_stats(ax_f_basal, df_stats, 'Fig 3F (Left)', 'Basal Total Branch Length', 0, 1, max_h)
        
        ax_f_basal.set_ylabel('Total Branch Length ($\mu$m)', fontsize=8)
        ax_f_basal.set_title('Basal', fontsize=8)
        ax_f_basal.set_xlabel('')
        
        # F2: Apical Branch Sum
        ax_f_apical = fig.add_subplot(gs_F[1])
        df_apical_props = df_dend_props[df_dend_props['Dendrite_Type'] == 'Apical']
        max_h = plot_bar_scatter(ax_f_apical, df_apical_props, 'Genotype', 'branch_sum', 'Genotype', order=['WT', 'I80T/+'])
        annotate_from_stats(ax_f_apical, df_stats, 'Fig 3F (Right)', 'Apical Total Branch Length', 0, 1, max_h)
        
        ax_f_apical.set_ylabel('')
        ax_f_apical.set_title('Apical', fontsize=8)
        ax_f_apical.set_xlabel('')
        
        # Panel G: Number of Terminal Branches
        # G1: Basal Terminal Branches
        ax_g_basal = fig.add_subplot(gs_G[0])
        add_subplot_label(ax_g_basal, "G")
        max_h = plot_bar_scatter(ax_g_basal, df_basal_props, 'Genotype', 'N_terminal_branches', 'Genotype', order=['WT', 'I80T/+'])
        annotate_from_stats(ax_g_basal, df_stats, 'Fig 3G (Left)', 'Basal Terminal Branches', 0, 1, max_h)
        
        ax_g_basal.set_ylabel('Number of Terminal Branches', fontsize=8)
        ax_g_basal.set_title('Basal', fontsize=8)
        ax_g_basal.set_xlabel('')
        
        # G2: Apical Terminal Branches
        ax_g_apical = fig.add_subplot(gs_G[1])
        max_h = plot_bar_scatter(ax_g_apical, df_apical_props, 'Genotype', 'N_terminal_branches', 'Genotype', order=['WT', 'I80T/+'])
        annotate_from_stats(ax_g_apical, df_stats, 'Fig 3G (Right)', 'Apical Terminal Branches', 0, 1, max_h)
        
        ax_g_apical.set_ylabel('')
        ax_g_apical.set_title('Apical', fontsize=8)
        ax_g_apical.set_xlabel('')
    else:
        # Add placeholders if dendrite properties data is not available
        # These placeholders need to be created on actual axes, not just variables
        # For simplicity, let's assume ax_branch_sum and ax_terminal_branches are defined
        # or create new placeholder axes if they are not.
        # Given the structure, it's likely these were meant to be on the subplots F and G.
        # Let's create placeholder axes for F and G if df_dend_props is None.
        
        # Create placeholder axes for F and G
        ax_f_placeholder = fig.add_subplot(gs[2, 1])
        ax_g_placeholder = fig.add_subplot(gs[2, 2])

        ax_f_placeholder.text(0.5, 0.5, 'Dendrite Properties\nData Unavailable', 
                          ha='center', va='center', color='gray')
        ax_f_placeholder.set_xticks([])
        ax_f_placeholder.set_yticks([])
        ax_f_placeholder.set_title('Branch Sum', fontsize=8)
        
        ax_g_placeholder.text(0.5, 0.5, 'Dendrite Properties\nData Unavailable', 
                                 ha='center', va='center', color='gray')
        ax_g_placeholder.set_xticks([])
        ax_g_placeholder.set_yticks([])
        ax_g_placeholder.set_title('Terminal Branches', fontsize=8)
    
    save_current_fig('Figure_3_Morphology')

# ==================================================================================================
# FIGURE 4: E:I BALANCE
# ==================================================================================================

def plot_figure_4_EI():
    """
    Figure 4: E:I Balance Across Three Pathways (ECIII/Perforant, CA3/Schaffer, SO/Basal)
    
    Layout: 5 rows x 3 columns
    - Row 1: WT example traces (ECIII | CA3 | SO)
    - Row 2: I80T/+ example traces (legend centered below)
    - Row 3 (C): Gabazine - WT vs I80T/+ overlaid
    - Row 4 (D): Gabazine Supralinearity (WT black vs I80T/+ red)
    - Row 5 (E): E:I imbalance (WT black vs I80T/+ red)
    """
    print("\\n--- Generating Figure 4: E:I Balance (3 Pathways) ---")
    setup_publication_style()
    
    # Load E:I traces and amplitudes
    df_traces = pd.read_pickle('paper_data/E_I_data/E_I_traces_for_plotting.pkl')
    df_amplitudes = pd.read_csv('paper_data/E_I_data/E_I_amplitudes.csv')

    # Rename GNB1 -> I80T/+ for display
    df_traces = rename_genotype(df_traces)
    df_amplitudes = rename_genotype(df_amplitudes)
    
    # Load statistics from corrected R scripts (condition-specific corrections)
    
    # # EI_Amplitudes Genotype comparison (corrected within condition)
    # stats_genotype_anova = 'paper_data/E_I_data/Figure_4_EI_Amplitudes_Genotype_ANOVA.csv'
    # stats_genotype_posthoc = 'paper_data/E_I_data/Figure_4_EI_Amplitudes_Genotype_PostHoc.csv'
    # df_stats_genotype_anova = None
    # df_stats_genotype_posthoc = None
    # if os.path.exists(stats_genotype_anova):
    #     df_stats_genotype_anova = pd.read_csv(stats_genotype_anova)
    #     print(f"✓ Loaded EI_Amplitudes Genotype ANOVA")
    # if os.path.exists(stats_genotype_posthoc):
    #     df_stats_genotype_posthoc = pd.read_csv(stats_genotype_posthoc)
    #     print(f"✓ Loaded EI_Amplitudes Genotype Post-Hoc")
    
    # # EI_Amplitudes Drug effect (Control vs Gabazine, corrected within genotype)
    # stats_drug_anova = 'paper_data/E_I_data/Figure_4_EI_Amplitudes_Drug_ANOVA.csv'
    # stats_drug_posthoc = 'paper_data/E_I_data/Figure_4_EI_Amplitudes_Drug_PostHoc.csv'
    # df_stats_drug_anova = None
    # df_stats_drug_posthoc = None
    # if os.path.exists(stats_drug_anova):
    #     df_stats_drug_anova = pd.read_csv(stats_drug_anova)
    #     print(f"✓ Loaded EI_Amplitudes Drug ANOVA")
    # if os.path.exists(stats_drug_posthoc):
    #     df_stats_drug_posthoc = pd.read_csv(stats_drug_posthoc)
    #     print(f"✓ Loaded EI_Amplitudes Drug Post-Hoc")
    
    # # EI_Amplitudes Gabazine vs Expected comparison
    # stats_gab_expected_anova = 'paper_data/E_I_data/Figure_4_EI_Amplitudes_GabazineVsExpected_ANOVA.csv'
    # stats_gab_expected_posthoc = 'paper_data/E_I_data/Figure_4_EI_Amplitudes_GabazineVsExpected_PostHoc.csv'
    # df_stats_gab_expected_anova = None
    # df_stats_gab_expected_posthoc = None
    # if os.path.exists(stats_gab_expected_anova):
    #     df_stats_gab_expected_anova = pd.read_csv(stats_gab_expected_anova)
    #     print(f"✓ Loaded EI_Amplitudes GabazineVsExpected ANOVA")
    # if os.path.exists(stats_gab_expected_posthoc):
    #     df_stats_gab_expected_posthoc = pd.read_csv(stats_gab_expected_posthoc)
    #     print(f"✓ Loaded EI_Amplitudes GabazineVsExpected Post-Hoc")
    
    # # EI_Supralinearity stats
    # stats_supralin_anova = 'paper_data/E_I_data/Figure_4_EI_Supralinearity_ANOVA.csv'
    # stats_supralin_posthoc = 'paper_data/E_I_data/Figure_4_EI_Supralinearity_PostHoc.csv'
    # df_stats_supralin_anova = None
    # df_stats_supralin_posthoc = None
    # if os.path.exists(stats_supralin_anova):
    #     df_stats_supralin_anova = pd.read_csv(stats_supralin_anova)
    #     print(f"✓ Loaded EI_Supralinearity ANOVA")
    # if os.path.exists(stats_supralin_posthoc):
    #     df_stats_supralin_posthoc = pd.read_csv(stats_supralin_posthoc)
    #     print(f"✓ Loaded EI_Supralinearity Post-Hoc")
    
    # # EI_Imbalance stats
    # stats_imbalance_anova = 'paper_data/E_I_data/Figure_4_EI_Imbalance_ANOVA.csv'
    # stats_imbalance_posthoc = 'paper_data/E_I_data/Figure_4_EI_Imbalance_PostHoc.csv'
    # df_stats_imbalance_anova = None
    # df_stats_imbalance_posthoc = None
    # if os.path.exists(stats_imbalance_anova):
    #     df_stats_imbalance_anova = pd.read_csv(stats_imbalance_anova)
    #     print(f"✓ Loaded EI_Imbalance ANOVA")
    # if os.path.exists(stats_imbalance_posthoc):
    #     df_stats_imbalance_posthoc = pd.read_csv(stats_imbalance_posthoc)
    #     print(f"✓ Loaded EI_Imbalance Post-Hoc")
    
    # if df_traces is None or df_traces.empty:
    #     print("❌ Error: Could not load E:I traces")
    #     return
    
    # if df_amplitudes is None or df_amplitudes.empty:
    #     print("❌ Error: Could not load E:I amplitudes")
    #     return
    
    # Convert 17.5 cm to inches
    fig_width = 17.5 / 2.54  # = 6.89 inches
    fig_height = 7.5  # inches (reduced)
    
    fig = plt.figure(figsize=(fig_width, fig_height))
    
    # Create 5×3 grid
    gs = fig.add_gridspec(5, 3,
                         wspace=0.35, hspace=0.65,
                         left=0.08, right=0.98,
                         top=0.96, bottom=0.05,
                         height_ratios=[1, 1, 1.2, 1, 1])
    
    # Define pathways
    pathways = [
        ('ECIII', 'perforant', 'channel_1'),
        ('CA3', 'schaffer', 'channel_2'),
        ('SO', 'basal', 'Basal_Stratum_Oriens')
    ]
    
    # =========================================================================
    # ROW 1: WT Example Traces
    # =========================================================================
    for col, (label, pathway, channel) in enumerate(pathways):
        ax = fig.add_subplot(gs[0, col])
        if col == 0:
            add_subplot_label(ax, "A", fontsize=8, fontweight='bold')
        
        if pathway == 'basal':
            plot_ei_averages(ax, df_traces, 'WT', 10, f'{label} - WT', 
                           add_legend=False, pathway=pathway)
        else:
            plot_ei_averages(ax, df_traces, 'WT', 10, f'{label} - WT',
                           add_legend=False, pathway=pathway)
        
        if col > 0:
            ax.set_title(label, fontsize=8, fontweight='bold', loc='center')
    
    # =========================================================================
    # ROW 2: GNB1 Example Traces
    # =========================================================================
    for col, (label, pathway, channel) in enumerate(pathways):
        ax = fig.add_subplot(gs[1, col])
        if col == 0:
            add_subplot_label(ax, "B", fontsize=8, fontweight='bold')
        
        if pathway == 'basal':
            plot_ei_averages(ax, df_traces, 'GNB1', 10, f'{label} - I80T/+', pathway=pathway, add_legend=False)
        else:
            plot_ei_averages(ax, df_traces, 'GNB1', 10, f'{label} - I80T/+', pathway=pathway, add_legend=False)
    
    # Add custom legend CENTERED below row 2
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='dimgray', lw=1.5, label='Control: Measured - With Inhibition'),
        Line2D([0], [0], color='black', lw=1.5, label='Gabazine: Measured - No Inhibition'),
        Line2D([0], [0], color='gray', lw=1.5, linestyle='--', label='Expected - Linear Summation')
    ]
    
    fig.legend(handles=legend_elements, loc='upper center',  
              bbox_to_anchor=(0.5, 0.62), ncol=3, frameon=False, fontsize=7)
    
    # =========================================================================
    # ROW 3 (C): Gabazine - WT vs GNB1 Overlaid
    # =========================================================================
    ymin_r3, ymax_r3 = 0, 0
    
    if 'Gabazine_Amplitude' in df_amplitudes.columns:
        grouped = df_amplitudes.groupby(['Genotype', 'Pathway', 'ISI'])['Gabazine_Amplitude'].agg(['mean', 'sem'])
        upper = (grouped['mean'] + grouped['sem']).max()
        ymax_r3 = upper * 1.1
        ymin_r3 = 0
    
    for col, (label, pathway, channel) in enumerate(pathways):
        ax = fig.add_subplot(gs[2, col])
        if col == 0:
            add_subplot_label(ax, "C", fontsize=8, fontweight='bold')
        
        pathway_name = {'perforant': 'Perforant', 'schaffer': 'Schaffer', 
                       'basal': 'Basal_Stratum_Oriens'}[pathway]
        
        # Plot both WT and GNB1 overlaid - use Between-Genotype ANOVA stats
        plot_gabazine_genotype_comparison(ax, df_amplitudes, pathway_name)
        
        ax.set_ylim(ymin_r3, ymax_r3)
        if col > 0:
            ax.set_ylabel('')

    # =========================================================================
    # ROW 4 (D): Gabazine Supralinearity
    # =========================================================================
    ymin_r4, ymax_r4 = -2, 2
    if 'Gabazine_Supralinearity' in df_amplitudes.columns:
        grouped = df_amplitudes.groupby(['Genotype', 'Pathway', 'ISI'])['Gabazine_Supralinearity'].agg(['mean', 'sem'])
        upper = (grouped['mean'] + grouped['sem']).max()
        lower = (grouped['mean'] - grouped['sem']).min()
        
        span = upper - lower
        ymax_r4 = upper + span * 0.1
        ymin_r4 = lower - span * 0.1
        
    for col, (label, pathway, channel) in enumerate(pathways):
        ax = fig.add_subplot(gs[3, col])
        if col == 0:
            add_subplot_label(ax, "D", fontsize=8, fontweight='bold')
        
        pathway_name = {'perforant': 'Perforant', 'schaffer': 'Schaffer', 
                       'basal': 'Basal_Stratum_Oriens'}[pathway]
        # Use Supralinearity/EI stats
        plot_metric_comparison(ax, df_amplitudes, pathway_name, 
                             'Gabazine_Supralinearity', 'Supralinearity (mV)',
                             add_legend=False)
        
        ax.set_ylim(ymin_r4, ymax_r4)
        if col > 0:
            ax.set_ylabel('')
    
    # =========================================================================
    # ROW 5 (E): E:I Imbalance
    # =========================================================================
    ymin_r5, ymax_r5 = 0, 1.2
    if 'E_I_Imbalance' in df_amplitudes.columns:
        grouped = df_amplitudes.groupby(['Genotype', 'Pathway', 'ISI'])['E_I_Imbalance'].agg(['mean', 'sem'])
        upper = (grouped['mean'] + grouped['sem']).max()
        lower = (grouped['mean'] - grouped['sem']).min()
        
        span = upper - lower
        ymax_r5 = upper + span * 0.1
        ymin_r5 = max(0, lower - span * 0.1)

    for col, (label, pathway, channel) in enumerate(pathways):
        ax = fig.add_subplot(gs[4, col])
        if col == 0:
            add_subplot_label(ax, "E", fontsize=8, fontweight='bold')
        
        pathway_name = {'perforant': 'Perforant', 'schaffer': 'Schaffer', 
                       'basal': 'Basal_Stratum_Oriens'}[pathway]
        # Use Supralinearity/EI stats
        plot_metric_comparison(ax, df_amplitudes, pathway_name,
                             'E_I_Imbalance', 'E:I Imbalance',
                             add_legend=False)
                             
        ax.set_ylim(ymin_r5, ymax_r5)
        if col > 0:
            ax.set_ylabel('')
    
    save_current_fig('Figure_4_EI')

# ==================================================================================================

# ==================================================================================================
# FIGURE 6: DENDRITIC EXCITABILITY
# ==================================================================================================

def plot_figure_6_dendritic():
    """
    Figure 6: Dendritic Excitability (Restructured)
    
    Panel A (Rows 1-2): Raw Theta Burst traces (WT vs I80T/+) for ECIII, CA3, Both.
    Panel B (Rows 3-4): Processed (from Pickle) + Expected traces (WT vs I80T/+).
    Panel C (Row 5): Plateau Area bar plots
    Panel D (Row 6): Single WT Example - Difference Traces
    Panel E (Row 7): Averaged Supralinearity Traces
    Panel F (Row 8): Spike Rate Across Theta Cycles
    """
    
    print("\\n--- Generating Figure 6: Dendritic Excitability ---")
    setup_publication_style()
    
    # Load AUC Data for Panel F (needed for filtering inside helper)
    auc_data_path = os.path.join('paper_data', 'supralinearity', 'Supralinear_AUC_Total.csv')
    df_auc_total = None
    if os.path.exists(auc_data_path):
        df_auc_total = pd.read_csv(auc_data_path)
    
    # PREPARE DATA (Logic moved to plotting_utils)
    raw_data, processed_stats, plateau_df, df_auc_total, supralin_traces = prepare_figure_6_data(df_auc_total)

    # Rename GNB1 -> I80T/+ for display
    plateau_df = rename_genotype(plateau_df)
    df_auc_total = rename_genotype(df_auc_total)
    
    # Re-define config variables for plotting calls (matching helper)
    acq_freq = 20000
    start_ms = 400
    end_ms = 1500
    start_idx = int(start_ms * acq_freq / 1000)
    end_idx = int(end_ms * acq_freq / 1000)

    # 4. Create Figure
    # -------------------------------------------------------------------------
    fig = plt.figure(figsize=(6.89, 11))
    gs = fig.add_gridspec(8, 3, hspace=0.6, wspace=0.3)
    
    cols = ['Perforant', 'Schaffer', 'Both']
    col_titles = ['ECIII (Perforant)', 'CA3 (Schaffer)', 'Both Pathways']
    
    # 5. Plot Panels Using Modular Functions
    # -------------------------------------------------------------------------
    
    # Panel A: Raw Traces
    plot_theta_raw_traces(fig, gs, raw_data, cols, col_titles, acq_freq, start_idx, end_idx)
    
    # Panel B: Averaged + Expected Traces
    plot_theta_averaged_traces(fig, gs, processed_stats, cols, acq_freq, start_idx, end_idx)

    # Panel C: Plateau Area Bar Plots
    # Panel C: Plateau Area Bar Plots
    stats_path = os.path.join('paper_data', 'Plateau_data', 'Stats_Results_Figure_6.csv')
    df_stats = pd.read_csv(stats_path) if os.path.exists(stats_path) else None
    
    if plateau_df is not None and not plateau_df.empty:
        plot_plateau_area_bars_fig6(fig, gs, plateau_df, df_stats)
    else:
        ax_c = fig.add_subplot(gs[4, :])
        add_subplot_label(ax_c, "C")
        ax_c.text(0.5, 0.5, "Plateau data not found (or empty)", ha='center', va='center')
        ax_c.axis('off')

    # Panel D: Single WT Example
    start_idx_d = int(400 * acq_freq / 1000)
    end_idx_d = int(1500 * acq_freq / 1000)
    
    # supralin_traces is already loaded via prepare_figure_6_data
    
    master_df_temp = pd.read_csv('master_df.csv', low_memory=False)
    plot_example_difference_traces(fig, gs, supralin_traces, master_df_temp, start_idx_d, end_idx_d)

    # Panel E: Averaged Supralinearity Traces
    if supralin_traces:
        plot_averaged_difference_traces(fig, gs, supralin_traces, master_df_temp, start_idx_d, end_idx_d)
    else:
        ax_e = fig.add_subplot(gs[6, :])
        add_subplot_label(ax_e, "E")
        ax_e.text(0.5, 0.5, "Supralinearity traces not found", ha='center', va='center')
        ax_e.axis('off')

    # Panel F: Supralinear Total AUC (Bar plot like Panel C)
    # df_auc_total is already loaded and filtered above to match Panel C cells
    supralin_stats_path = os.path.join('paper_data', 'Plateau_data', 'Stats_Results_Figure_6.csv')
    
    if df_auc_total is not None and not df_auc_total.empty:
        df_stats_full = pd.read_csv(supralin_stats_path) if os.path.exists(supralin_stats_path) else None
        plot_supralinear_auc_bars_fig6(fig, gs, df_auc_total, df_stats_full)
    else:
        ax_f = fig.add_subplot(gs[7, :])
        add_subplot_label(ax_f, "F")
        ax_f.text(0.5, 0.5, "Supralinearity AUC data not found", ha='center', va='center')
        ax_f.axis('off')

    save_current_fig('Figure_6_Dendritic')

# ==================================================================================================
# FIGURE 6: GIRK Channel Analysis (ML297 / ETX Effects)
# ==================================================================================================

def plot_figure_7_GIRK():
    """
    Figure 7: GIRK Channel Analysis (ML297 / ETX Effects)
    New Layout (2 Rows):
    Panel A: ML297 Traces (WT/I80T/+) + Delta Quantification
    Panel B: ETX Traces (WT/I80T/+) + Delta Quantification
    """
    print("\n--- Generating Figure 7: GIRK Channel Analysis ---")
    setup_publication_style()
    
    # Load Delta Data
    delta_csv_path = 'paper_data/Plateau_data/Plateau_Delta_GIRK.csv'
    df_delta = None
    if os.path.exists(delta_csv_path):
        df_delta = pd.read_csv(delta_csv_path)
        # Rename GNB1 -> I80T/+ for display
        df_delta = rename_genotype(df_delta)
    else:
        print("⚠ Warning: Plateau_Delta_GIRK.csv not found. Run analysis script.")

    # Load Stats
    stats_path = 'paper_data/Plateau_data/Stats_Results_Figure_7.csv'
    df_stats = None
    if os.path.exists(stats_path):
        df_stats = pd.read_csv(stats_path)
    
    # Load Traces
    trace_path = 'paper_data/Plateau_data/All_Plateau_Traces.pkl'
    plateau_traces = {}
    if os.path.exists(trace_path):
        plateau_traces = pd.read_pickle(trace_path)
    
    # Setup Figure
    fig = plt.figure(figsize=(6.89, 5)) # Shorter height makes plots wider aspect
    gs = fig.add_gridspec(2, 3, hspace=0.4, wspace=0.4, width_ratios=[1.3, 1.3, 1])
    
    # =========================================================================
    # ROW 1 - PANEL A: ML297 (After = Yellow)
    # =========================================================================
    if 'Before_ML297' in plateau_traces and 'After_ML297' in plateau_traces:
        # Col 0: WT
        ax_a1 = fig.add_subplot(gs[0, 0])
        add_subplot_label(ax_a1, "A")
        plot_traces_GIRK_v2(ax_a1, plateau_traces['Before_ML297'], plateau_traces['After_ML297'], 
                            'WT', 'ML297', after_color='gold', add_legend=True, add_scale=True)
        
        # Col 1: I80T/+
        ax_a2 = fig.add_subplot(gs[0, 1])
        plot_traces_GIRK_v2(ax_a2, plateau_traces['Before_ML297'], plateau_traces['After_ML297'], 
                            'I80T/+', 'ML297', after_color='gold', add_legend=False, add_scale=False)
    
    # Col 2: Delta Quantification
    ax_a3 = fig.add_subplot(gs[0, 2])
    if df_delta is not None:
        plot_girk_delta_bars(ax_a3, df_delta, 'ML297', df_stats)
    
    # =========================================================================
    # ROW 2 - PANEL B: ETX (After = Red)
    # =========================================================================
    if 'Before_ETX' in plateau_traces and 'After_ETX' in plateau_traces:
        # Col 0: WT
        ax_b1 = fig.add_subplot(gs[1, 0])
        add_subplot_label(ax_b1, "B")
        plot_traces_GIRK_v2(ax_b1, plateau_traces['Before_ETX'], plateau_traces['After_ETX'], 
                            'WT', 'ETX', after_color='blue', add_legend=True, add_scale=True)
        
        # Col 1: I80T/+
        ax_b2 = fig.add_subplot(gs[1, 1])
        plot_traces_GIRK_v2(ax_b2, plateau_traces['Before_ETX'], plateau_traces['After_ETX'], 
                            'I80T/+', 'ETX', after_color='blue', add_legend=False, add_scale=False)

    # Col 2: Delta Quantification
    ax_b3 = fig.add_subplot(gs[1, 2])
    if df_delta is not None:
        plot_girk_delta_bars(ax_b3, df_delta, 'ETX', df_stats)

    save_current_fig('Figure_7_GIRK')


# ==================================================================================================
# FIGURE 5: GABAb Analysis (Schaffer / Perforant Pathways)
# ==================================================================================================

def plot_figure_5_GABAb():
    """
    Figure 5: GABAb Analysis
    Row 1: Traces - ECIII (Perforant, A) | CA3 Apical (Schaffer, B) | CA3 Basal (C)
    Row 2: Quantifications - ECIII (D) | CA3 Apical (E) | CA3 Basal (F)
    Row 3: Baclofen - Diagram (G) | WT Vm Traces (H left) | GNB1 Vm Traces (H right)
    Row 4: Vm Change Quantification (I)
    """
    print("\n--- Generating Figure 5: GABAb Analysis ---")
    setup_publication_style()
    
    gabab_df = pd.read_csv('paper_data/gabab_analysis/GABAb_Analysis_Metrics.csv')
    
    # Load statistical results if available
    df_stats = None
    stats_file = 'paper_data/gabab_analysis/Stats_Results_Figure_5.csv'
    if os.path.exists(stats_file):
        df_stats = pd.read_csv(stats_file)
        print(f"✓ Loaded Figure 5 statistics: {len(df_stats)} comparisons")
    else:
        print("⚠ Note: Run Analyze_Stats_Python.py to generate statistical annotations")
    
    trace_path = 'paper_data/gabab_analysis/GABAb_Individual_Traces_Hierarchical.pkl'
    gabab_traces = {}
    if os.path.exists(trace_path):
        gabab_traces = pd.read_pickle(trace_path)
    
    # Rename GNB1 -> I80T/+ for display
    gabab_df = rename_genotype(gabab_df)

    # Filter for Gabazine condition only (case-insensitive)
    gabab_gab = gabab_df[gabab_df['Condition'].str.lower() == 'gabazine'].copy()
    gabab_gab = gabab_gab.drop_duplicates(subset=['Cell_ID', 'Channel_Name'])
    
    # Figure dimensions: 3 rows × 3 columns
    fig = plt.figure(figsize=(6.5, 8.5))
    gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.3, 
                         left=0.08, right=0.98, top=0.96, bottom=0.05,
                         height_ratios=[1, 1, 1.2])
    
    # Only Slow IPSP Area metric
    metric_col = 'Integral_mV_ms'
    metric_ylabel = 'Slow IPSP Area\n(mV·ms)'
    
    # ---------------------------------------------------------
    # ROW 1: Traces for all 3 pathways
    # ---------------------------------------------------------
    pathways_traces = [
        ('Perforant Path', 'ECIII (Perforant)', 0, 'A', 'Perforant'),
        ('Schaffer Collateral', 'CA3 Apical (Schaffer)', 1, 'B', 'Schaffer'),
        ('Stratum Oriens', 'CA3 Basal', 2, 'C', 'Basal'),
    ]
    
    for pathway_key, pathway_label, col, trace_label, pathway_match in pathways_traces:
        ax_trace = fig.add_subplot(gs[0, col])
        title = f'Unitary EPSP (300ms ISI)\n{pathway_label}'
        plot_gabab_traces(ax_trace, gabab_traces, pathway_key, title, trace_label, gabab_gab)
    
    # ---------------------------------------------------------
    # ROW 2: Quantifications for all 3 pathways
    # ---------------------------------------------------------
    pathways_metrics = [
        ('Perforant Path', 0, 'D', 'Perforant'),
        ('Schaffer Collateral', 1, 'E', 'Schaffer'),
        ('Stratum Oriens', 2, 'F', 'Basal'),
    ]
    
    for pathway_key, col, metric_label, pathway_match in pathways_metrics:
        ax_metric = fig.add_subplot(gs[1, col])
        plot_gabab_metric_bar(ax_metric, gabab_gab, pathway_key, metric_col, metric_ylabel, 
                             metric_label, df_stats, pathway_match)
    
    # ---------------------------------------------------------
    # ---------------------------------------------------------
    # ROW 3: Baclofen Effects (Diagram G | Traces H | Stats I)
    # ---------------------------------------------------------
    
    # --- Panel G: Diagram Placeholder ---
    ax_diagram = fig.add_subplot(gs[2, 0])
    add_subplot_label(ax_diagram, 'G')
    ax_diagram.text(0.5, 0.5, 'Diagram', ha='center', va='center', fontsize=10, 
                    color='gray', style='italic')
    ax_diagram.set_xlim(0, 1)
    ax_diagram.set_ylim(0, 1)
    ax_diagram.axis('off')
    
    # --- Panel H: Baclofen Vm Example Traces (Stacked) ---
    vm_traces_path = 'paper_data/gabab_analysis/Baclofen_Vm_Example_Traces.pkl'
    ax_traces = fig.add_subplot(gs[2, 1])
    plot_baclofen_vm_traces(ax_traces, vm_traces_path, label='H')
    
    # --- Panel I: Vm Change Quantification ---
    vm_csv = 'paper_data/gabab_analysis/Baclofen_Vm_Change.csv'
    ax_vm = fig.add_subplot(gs[2, 2])
    plot_gabab_vm_change(ax_vm, vm_csv, "I", df_stats=df_stats)
    
    save_current_fig('Figure_5_GABAb')


def plot_supplemental_figure_1():
    """
    Supplemental Figure 1: Full E:I Breakdown (Control, Gabazine, Expected, Est. Inhibition)
    Separate panels for WT and I80T/+ across 3 pathways.
    """
    print("\n--- Generating Supplemental Figure 1: Full E:I Breakdown ---")
    
    # Load data
    ei_amp_path = 'paper_data/E_I_data/E_I_amplitudes.csv'
    if not os.path.exists(ei_amp_path):
        print(f"❌ Error: {ei_amp_path} not found")
        return
        
    df = pd.read_csv(ei_amp_path)

    # Rename GNB1 -> I80T/+ for display
    df = rename_genotype(df)
    
    # WIDTH constraint: max 17.5 cm, explicitly made smaller as requested
    fig_width = 15 / 2.54 
    fig_height = 5  # inches - adjusted for square aspect ratio
    
    # Increase bottom margin for legend
    fig, axes = plt.subplots(2, 3, figsize=(fig_width, fig_height), sharex=False, sharey=True)
    fig.subplots_adjust(wspace=0.35, hspace=0.35, left=0.08, right=0.98, top=0.9, bottom=0.15)
    
    # Remove top/right spines for all
    for ax in axes.flat:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(labelleft=True)  # Force y-labels to be visible with sharey=True
    
    pathways = [
        ('Perforant', 'Perforant'),
        ('Schaffer', 'Schaffer'),
        ('Basal SO', 'Basal_Stratum_Oriens')
    ]
    
    genotypes = ['WT', 'I80T/+']
    
    metrics = [
        ('Control_Amplitude', 'Measured (With Inhibition)', 'black'),
        ('Gabazine_Amplitude', 'Measured (No Inhibition)', 'magenta'),
        ('Expected_EPSP_Amplitude', 'Expected (Linear Summation)', 'grey')]
    
    isi_order = [300, 100, 50, 25, 10]

    plot_supplemental_figure_1_helper(isi_order, metrics, pathways, genotypes, df, axes)    
    # --- Custom Legend at Bottom ---
    legend_elements = [
        Line2D([0], [0], color='black', marker='o', lw=1.5, label='Measured - With Inhibition'),
        Line2D([0], [0], color='magenta', marker='o', lw=1.5, label='Measured - No Inhibition'),
        Line2D([0], [0], color='skyblue', marker='o', lw=1.5, label='Inhibition (No Inhibition - With Inhibition)'),
        Line2D([0], [0], color='grey', marker='o', lw=1.5, linestyle='--', label='Expected - Linear Summation')
    ]
    
    fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, 0.001),
              ncol=2, frameon=False, fontsize=8)

    save_current_fig('Supplemental_Figure_1_EI_Breakdown')


def plot_supplemental_figure_2():
    """
    Supplemental Figure 2: Pie Chart of number of births by genotype and sex
    """

    print("\n--- Generating Supplemental Figure 2: Births by Genotype ---")


    master_birth_df = pd.read_csv('Master_DF_littermate_Sex.csv')

    df = pd.DataFrame(master_birth_df)
    # Rename GNB1 -> I80T/+ for display
    df = rename_genotype(df)

    # WIDTH constraint: max 17.5 cm, explicitly made smaller as requested
    fig_width = 15 / 2.54
    fig_height = 5  # inches

    fig, axes = plt.subplots(1, 2, figsize=(fig_width, fig_height), sharex=False, sharey=True)
    fig.subplots_adjust(wspace=0.35, hspace=0.35, left=0.08, right=0.98, top=0.9, bottom=0.15)


    group_counts = df.groupby("Genotype").size()

    labels = [f"{genotype}" for genotype in group_counts.index]
    sizes = group_counts.values

    axes[0].pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=plt.cm.Paired.colors)
    axes[0].set_title("Distribution of Genotypes (WT vs I80T/+) ") 

    group_counts = df.groupby(["Genotype", "Sex"]).size()

    labels = [f"{genotype} {sex}" for genotype, sex in group_counts.index]
    sizes = group_counts.values

    axes[1].pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=plt.cm.Paired.colors)
    axes[1].set_title("Distribution of WT and I80T/+ Litters by Sex") 

    save_current_fig('Supplemental_Figure_2_Births_By_Genotype')

def plot_supplemental_figure_3():
    """
    Supplemental Figure 3: Protein Levels of WT and I80T/+ in Hippocampus
    """

    print("\n--- Generating Supplemental Figure 3: Protein Levels ---")

    protein_df = pd.read_csv('paper_data/GNB1_Protein_Levels_Hippocampus.csv')
    
    # Load Stats
    stats_path = 'paper_data/Stats_Results_Supplemental_Figure_3.csv'
    df_stats = None
    if os.path.exists(stats_path):
        df_stats = pd.read_csv(stats_path)
    
    fig_width = 8.5 / 2.54 
    fig_height = 2  # Small panels

    fig, axes = plt.subplots(1, 2, figsize=(fig_width, fig_height), sharex=False, sharey=False)
    fig.subplots_adjust(wspace=0.5, hspace=0.35, left=0.15, right=0.95, top=0.8, bottom=0.2)

    # Left Panel: Absolute
    plot_protein_expression(axes[0], protein_df, 'Absolute Protein Signal', 
                          'GNB1 Hippocampal Protein Levels', 
                          'GNB1/Vinculin', 
                          df_stats, 'Supp Fig 3 (Top)')
                          
                          
    # Right Panel: Relative (Manual Plot from Summary Columns)
    ax_rel = axes[1]
    
    if 'WT_Relative' in protein_df.columns and 'I80T/+_Relative' in protein_df.columns:
        wt_mean = protein_df['WT_Relative'].values[0]
        wt_sem = protein_df['WT_Relative_SEM'].values[0]
        gnb1_mean = protein_df['I80T/+_Relative'].values[0]
        gnb1_sem = protein_df['I80T/+_Relative_SEM'].values[0]
        
        # Data for loop
        groups = ['WT', 'I80T/+']
        means = [wt_mean, gnb1_mean]
        sems = [wt_sem, gnb1_sem]
        bar_colors = ['black', 'red'] # Explicitly matching COLORS from plotting_utils
        
        for i, (group, mean, sem, color) in enumerate(zip(groups, means, sems, bar_colors)):
             # Bar with alpha, no edge line to prevent 'below zero' artifacts in SVG
             ax_rel.bar(i, mean, width=0.6, color=color, alpha=0.5, label=group, linewidth=0, clip_on=False)
             # Error Bar matching plot_bar_scatter style (fmt='o', specific size)
             ax_rel.errorbar(i, mean, yerr=sem, fmt='o', color=color, capsize=1, elinewidth=1, markersize=2)
        
        ax_rel.set_xticks([0, 1])
        ax_rel.set_xticklabels(['WT', 'I80T/+'])
        ax_rel.set_ylabel('GNB1/Vinculin Relative to Ctrl.')
        ax_rel.set_title('GNB1 Hippocampal Protein Levels (Relative)', fontsize=9)
        
        # Stats
        y_max = max(mean + sem for mean, sem in zip(means, sems))
        annotate_from_stats(ax_rel, df_stats, 'Supp Fig 3 (Bottom)', 'Relative Protein Levels', 0, 1, y_max + 0.1)
        
        # Smart Y-Lim
        ax_rel.set_ylim(0, y_max * 1.25)
        
    else:
        ax_rel.text(0.5, 0.5, 'Summary Data Not Found', ha='center', va='center')

    save_current_fig('Supplemental_Figure_3_Protein_Levels')
    



if __name__ == "__main__":
    setup_publication_style()
    plot_figure_1_behavior()
    plot_figure_2_physiology()
    plot_figure_3_morphology()
    plot_figure_4_EI()
    plot_figure_5_GABAb()
    plot_figure_6_dendritic()
    plot_figure_7_GIRK()
    plot_supplemental_figure_1()
    plot_supplemental_figure_2()
    plot_supplemental_figure_3()