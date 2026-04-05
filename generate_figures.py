import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import os
import sys
import numpy as np
import pickle
import box_utils
from matplotlib.patches import Patch

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
        plot_bar_scatter(ax_c, df_of_loco, 'Genotype', 'Distance (m)', 'Genotype', order=geno_order, ymax=50)
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
        apply_clean_yticks(ax_d)
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
        apply_clean_yticks(ax_f)
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
        plot_bar_scatter(ax_h, df_tmaze_entries, 'Genotype', 'Distance (m)', 'Genotype', order=geno_order, ymax=50)
        ax_h.set_title('Distance Traveled', fontsize=8)
        ax_h.set_ylabel('Distance (m)')
        annotate_from_stats(ax_h, df_stats, "Fig 1I", "Distance", x1=0, x2=1, y_pos=get_safe_y(df_tmaze_entries['Distance (m)']))
    
    # Panel I: Total Port Entries
    ax_i = fig.add_subplot(gs_row4[2])
    add_subplot_label(ax_i, "I")
    if df_tmaze_entries is not None:
        plot_bar_scatter(ax_i, df_tmaze_entries, 'Genotype', 'Total_Entries', 'Genotype', order=geno_order, ymax=100)
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
        # Percent alternation is bounded 0-100%; use fixed scale
        ax_j.set_ylim(0, 100)
        ax_j.set_yticks([0, 25, 50, 75, 100])
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

    fig = plt.figure(figsize=(6.89, 9.5))  # Slightly taller for larger traces
    outer_grid = gridspec.GridSpec(3, 1, height_ratios=[1.5, 0.6, 0.6], hspace=0.35)

    # ==========================================================
    # ROW 1: A | B | C
    # A = Input Resistance bar (narrow)
    # B = Vsag traces stacked (top) + Vsag bar (bottom)
    # C = AP traces stacked   (top) + AHP Decay bar (bottom)
    # ==========================================================
    gs_row1 = gridspec.GridSpecFromSubplotSpec(
        1, 3, subplot_spec=outer_grid[0],
        width_ratios=[0.34, 0.33, 0.33], wspace=0.4)

    # --- Panel A: Input Resistance traces (stacked) + bar ---
    gs_A = gridspec.GridSpecFromSubplotSpec(
        2, 2, subplot_spec=gs_row1[0],
        height_ratios=[0.60, 0.40], hspace=0.15, wspace=0.25)
    
    # A-top: Input Resistance traces
    ax_a_wt  = fig.add_subplot(gs_A[0, 0])
    ax_a_gnb = fig.add_subplot(gs_A[0, 1])
    add_subplot_label(ax_a_wt, "A")
    
    if raw_traces_path and master_df is not None:
        from plotting_utils import plot_input_resistance_comparison
        plot_input_resistance_comparison(ax_a_wt, ax_a_gnb, raw_traces_path, master_df,
                                    target_wt='04232024_c2', target_gnb1='05142025_c2')
        ax_a_wt.set_ylim(-85, -60)
        ax_a_gnb.set_ylim(-85, -60)
    else:
        plot_trace_placeholder(ax_a_wt, "Unavailable")
        plot_trace_placeholder(ax_a_gnb, "Unavailable")

    # A-bottom: Input Resistance bar
    ax_a = fig.add_subplot(gs_A[1, :])
    if df_intrinsic is not None and 'Input_Resistance_MOhm' in df_intrinsic.columns:
        plot_bar_scatter(ax_a, df_intrinsic, 'Genotype', 'Input_Resistance_MOhm',
                         'Genotype', order=['WT', 'I80T/+'], ymin=-2, ymax=300)
        ax_a.set_ylabel('Input Resistance (MΩ)', fontsize=7)
        ax_a.set_title('Input Resistance', fontsize=8)
        ax_a.set_box_aspect(1)
        if df_stats is not None:
            annotate_from_stats(ax_a, df_stats, "Fig 2A", "Input Resistance",
                                x1=0, x2=1,
                                y_pos=get_safe_y(df_intrinsic['Input_Resistance_MOhm']))
    else:
        ax_a.text(0.5, 0.5, 'Rin Missing', ha='center', color='red')

    # --- Panel B: Voltage Sag traces (stacked) + Voltage Sag bar ---
    gs_B = gridspec.GridSpecFromSubplotSpec(
        2, 2, subplot_spec=gs_row1[1],
        height_ratios=[0.60, 0.40], hspace=0.15, wspace=0.25)

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
                         'Genotype', order=['WT', 'I80T/+'], ymin=-2, ymax=30)
        ax_b_bar.set_ylabel('Voltage Sag (%)', fontsize=7)
        ax_b_bar.set_title('Voltage Sag', fontsize=8)
        ax_b_bar.set_box_aspect(1)
        if df_stats is not None:
            annotate_from_stats(ax_b_bar, df_stats, "Fig 2A", "Voltage Sag",
                                x1=0, x2=1,
                                y_pos=get_safe_y(df_intrinsic['Voltage_sag']))
    else:
        ax_b_bar.text(0.5, 0.5, 'Sag Missing', ha='center', color='red')

    # --- Panel C: AP traces (stacked) + AHP Decay bar ---
    gs_C = gridspec.GridSpecFromSubplotSpec(
        2, 2, subplot_spec=gs_row1[2],
        height_ratios=[0.60, 0.40], hspace=0.15, wspace=0.25)

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
                                             analysis_df=df_ap_ahp, show_values=False,
                                             show_annotations=False, color='red')
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
                         'Genotype', order=['WT', 'I80T/+'], ymin=-2, ymax=60)
        ax_c_bar.set_ylabel('AHP Area\n(mV·ms)', fontsize=7)
        ax_c_bar.set_title('AHP Decay', fontsize=8)
        ax_c_bar.set_box_aspect(1)
        if df_stats is not None:
            annotate_from_stats(ax_c_bar, df_stats, "Fig 2E", "AHP Decay",
                                x1=0, x2=1,
                                y_pos=get_safe_y(df_ap_ahp['decay_area']))
    else:
        ax_c_bar.text(0.5, 0.5, 'AHP Missing', ha='center', color='red')

    # ==========================================================
    # ROW 2: D (FI example traces stacked) | E (F-I curve) | F (FI Midpoint)
    # ==========================================================
    gs_row2 = gridspec.GridSpecFromSubplotSpec(
        1, 3, subplot_spec=outer_grid[1],
        width_ratios=[0.34, 0.33, 0.33], wspace=0.4)

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

        preferred_wt   = ['03262024_c2', '02012024_c1', '04042024_c1', '02262024_c2', '03142024_c1']
        preferred_gnb1 = ['07232024_c6', '10312025_c3', '07302024_c3', '02132024_c2', '10312025_c1']
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

    # --- Panel E: F-I Curve ---
    ax_e = fig.add_subplot(gs_row2[1])
    add_subplot_label(ax_e, "E")
    ax_e.set_box_aspect(1)
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
    # ROW 3: G (Rheobase) | H (AP-ISI traces) | I (AP-ISI adaptation)
    # ==========================================================
    gs_row3 = gridspec.GridSpecFromSubplotSpec(
        1, 3, subplot_spec=outer_grid[2],
        width_ratios=[0.33, 0.34, 0.33], wspace=0.4)

    # Panel F: F-I Midpoint
    ax_f = fig.add_subplot(gs_row2[2])
    add_subplot_label(ax_f, "F")
    if fi_midpoints_df is not None:
        plot_bar_scatter(ax_f, fi_midpoints_df, 'Genotype', 'FI_Midpoint',
                         'Genotype', order=['WT', 'I80T/+'], ymin=-2, ymax=600)
        ax_f.set_ylabel('F-I Curve Midpoint (pA)')
        ax_f.set_title('F-I Midpoint', fontsize=8)
        ax_f.set_box_aspect(1)
        if df_stats is not None:
            annotate_from_stats(ax_f, df_stats, 'Fig 2F', 'F-I Curve Midpoint',
                                x1=0, x2=1, y_pos=get_safe_y(fi_midpoints_df['FI_Midpoint']))
    else:
        ax_f.text(0.5, 0.5, 'Midpoint Data Missing', ha='center', color='red')

    # Panel G: Rheobase
    ax_rheo = fig.add_subplot(gs_row3[0])
    add_subplot_label(ax_rheo, "G")
    if df_ap_ahp is not None and 'Rheobase_Current' in df_ap_ahp.columns:
        plot_data = df_ap_ahp.dropna(subset=['Rheobase_Current'])
        plot_bar_scatter(ax_rheo, plot_data, 'Genotype', 'Rheobase_Current',
                         'Genotype', order=['WT', 'I80T/+'], ymin=-2, ymax=600)
        ax_rheo.set_ylabel('Rheobase (pA)')
        ax_rheo.set_title('Rheobase', fontsize=8)
        ax_rheo.set_box_aspect(1)
        if df_stats is not None:
            annotate_from_stats(ax_rheo, df_stats, "Fig 2C", "Rheobase",
                                x1=0, x2=1, y_pos=get_safe_y(plot_data['Rheobase_Current']))
    else:
        ax_rheo.text(0.5, 0.5, 'Rheobase Data Missing', ha='center', color='red')

    # Panel H: ISI example traces
    gs_h = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_row3[1], hspace=0.1)
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
    ax_i = fig.add_subplot(gs_row3[2])
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
        apply_clean_yticks(ax_i)
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
    ax_basal.set_ylim(bottom=0)
    ax_basal.yaxis.set_major_locator(plt.MaxNLocator(nbins=5, integer=True))
    ax_basal.spines['top'].set_visible(False)
    ax_basal.spines['right'].set_visible(False)
    ax_basal.legend(frameon=False, loc='upper right')
    
    # Plot Apical Dendrites
    plot_sholl_data(ax_apical, df_wt, 'WT', 'Apical', COLORS['WT'])
    plot_sholl_data(ax_apical, df_gnb1, 'I80T/+', 'Apical', COLORS['GNB1'])
    ax_apical.set_title('Apical Dendrites', fontsize=9)
    ax_apical.set_xlabel('Distance from Soma (μm)', fontsize=8)
    ax_apical.set_ylabel('Number of Intersections', fontsize=8)
    ax_apical.set_ylim(bottom=0)
    ax_apical.yaxis.set_major_locator(plt.MaxNLocator(nbins=5, integer=True))
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
        ax_f_basal.set_ylim(bottom=0)  # branch lengths cannot be negative
        
        # F2: Apical Branch Sum
        ax_f_apical = fig.add_subplot(gs_F[1])
        df_apical_props = df_dend_props[df_dend_props['Dendrite_Type'] == 'Apical']
        max_h = plot_bar_scatter(ax_f_apical, df_apical_props, 'Genotype', 'branch_sum', 'Genotype', order=['WT', 'I80T/+'])
        annotate_from_stats(ax_f_apical, df_stats, 'Fig 3F (Right)', 'Apical Total Branch Length', 0, 1, max_h)
        ax_f_apical.set_ylabel('')
        ax_f_apical.set_title('Apical', fontsize=8)
        ax_f_apical.set_xlabel('')
        ax_f_apical.set_ylim(bottom=0)
        
        # Panel G: Number of Terminal Branches
        # G1: Basal Terminal Branches
        ax_g_basal = fig.add_subplot(gs_G[0])
        add_subplot_label(ax_g_basal, "G")
        max_h = plot_bar_scatter(ax_g_basal, df_basal_props, 'Genotype', 'N_terminal_branches', 'Genotype', order=['WT', 'I80T/+'])
        annotate_from_stats(ax_g_basal, df_stats, 'Fig 3G (Left)', 'Basal Terminal Branches', 0, 1, max_h)
        
        ax_g_basal.set_ylabel('Number of Terminal Branches', fontsize=8)
        ax_g_basal.set_title('Basal', fontsize=8)
        ax_g_basal.set_xlabel('')
        ax_g_basal.set_ylim(bottom=0)
        
        # G2: Apical Terminal Branches
        ax_g_apical = fig.add_subplot(gs_G[1])
        max_h = plot_bar_scatter(ax_g_apical, df_apical_props, 'Genotype', 'N_terminal_branches', 'Genotype', order=['WT', 'I80T/+'])
        annotate_from_stats(ax_g_apical, df_stats, 'Fig 3G (Right)', 'Apical Terminal Branches', 0, 1, max_h)
        ax_g_apical.set_ylabel('')
        ax_g_apical.set_title('Apical', fontsize=8)
        ax_g_apical.set_xlabel('')
        ax_g_apical.set_ylim(bottom=0)
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
# FIGURE 4-6: E:I BALANCE
# ==================================================================================================

def plot_figure_4_Unitary_E_I_Breakdown():
    """
    Figure 4: Unitary E:I Breakdown
    
    Layout: 5 rows x 3 columns
    - Row 1: WT Unitary Breakdown (ECIII | CA3 | Basal)
    - Row 2: I80T/+ Unitary Breakdown
    - Row 3: Excitation Amplitudes (Gabazine) - only unitary
    - Row 4: Inh (GABAA) Amplitudes (Estimated) - only unitary
    - Row 5: Inh (GABAB) Area - only unitary

    """
    print("\n--- Generating Figure 4: Unitary E:I Breakdown ---")
    # Convert 17.5 cm to inches
    fig_width = 17.5 / 2.54 
    fig_height = 8.5 # Refined height for 4 rows
    
    fig = plt.figure(figsize=(fig_width, fig_height))
    
    gs = fig.add_gridspec(5, 3,
                         wspace=0.45, hspace=0.8,
                         left=0.08, right=0.98,
                         top=0.96, bottom=0.08,
                         height_ratios=[1.2, 1, 1, 1, 1])
    
    # Load E:I traces and amplitudes
    df_traces = pd.read_pickle('paper_data/E_I_data/E_I_traces_for_plotting.pkl')
    df_amplitudes = pd.read_csv('paper_data/E_I_data/E_I_amplitudes.csv')

    # Rename GNB1 -> I80T/+ for display
    df_traces = rename_genotype(df_traces)
    df_amplitudes = rename_genotype(df_amplitudes)

    #Adding stats to specific subplots
    panel_to_analysis = {
        'B': ('Gabazine_Amplitude', 'WT_vs_GNB1'),
        'C': ('Inhibition_Amplitude', 'WT_vs_GNB1'),
        'D': ('GABAB_Area', 'WT_vs_GNB1')
    }

    #Pathways to do analysis on
    pathways = [
        ('ECIII (Perforant)', 'perforant', 'channel_1'),
        ('CA3 Apical (Schaffer)', 'schaffer', 'channel_2'),
        ('CA3 Basal', 'basal', 'Basal_Stratum_Oriens')
    ]

    # Load Significance Markers
    #....TODO


    ## ROW 1: Example Unitary Breakdown (Methodology)
    # Only display ONE example (Perforant) and put labels to the right
    ax_wt_perf = fig.add_subplot(gs[0, 0]) # Wider subplot for the trace + labels
    add_subplot_label(ax_wt_perf, "A", fontsize=10, fontweight='bold')

    #Add a row label for WT
    ax_wt_perf.text(-0.2, 1, 'WT', rotation='vertical', va='center', ha='center', fontsize=12, fontweight='bold')

    # Use WT as the example for methodology
    plot_unitary_breakdown(ax_wt_perf, df_traces, 'WT', 'ECIII Input', annotate=True)
    
    # Hide the empty axes in Row 1 if any (we used 0:2, so only 2 is empty)
    ax_wt_apical = fig.add_subplot(gs[0, 1])
    #plot unitary traces without measurements 
    plot_unitary_breakdown(ax_wt_apical, df_traces, 'WT', 'CA3 Apical Input', annotate=False)

    ax_wt_basal = fig.add_subplot(gs[0, 2])
    #plot unitary traces without measurements 
    plot_unitary_breakdown(ax_wt_basal, df_traces, 'WT', 'CA3 Basal Input', annotate=False)

    #ADD A SCALEBAR TO THE FIRST PLOT
    add_scale_bar(ax_wt_perf, x_scale_ms=50, y_scale_mv=2, x_pos=0.05, y_pos=0.05) 

    ## ROW 2: I80T/+ Unitary Breakdown
    ax_mut_perf = fig.add_subplot(gs[1, 0]) # Wider subplot for the trace + labels
    add_subplot_label(ax_mut_perf, "B", fontsize=10, fontweight='bold')
    #Add a row label for I80T/+
    ax_mut_perf.text(-0.2, 1, 'I80T/+', rotation='vertical', va='center', ha='center', fontsize=12, fontweight='bold')
    # Use WT as the example for methodology
    plot_unitary_breakdown(ax_mut_perf, df_traces, 'I80T/+', 'ECIII Input', annotate=False)

    # Hide the empty axes in Row 1 if any (we used 0:2, so only 2 is empty)
    ax_mut_apical = fig.add_subplot(gs[1, 1])
    #plot unitary traces without measurements 
    plot_unitary_breakdown(ax_mut_apical, df_traces, 'I80T/+', 'CA3 Apical Input', annotate=False)

    ax_mut_basal = fig.add_subplot(gs[1, 2])
    #plot unitary traces without measurements 
    plot_unitary_breakdown(ax_mut_basal, df_traces, 'I80T/+', 'CA3 Basal Input', annotate=False)

    #ADD A SCALEBAR TO THE SECOND ROW (I80T/+)
    add_scale_bar(ax_mut_perf, x_scale_ms=50, y_scale_mv=2, x_pos=0.05, y_pos=0.05)

    # -----------------------------------------------------------------------
    # Helper: filter amplitudes to ISI=300, one pathway, rename genotype
    # -----------------------------------------------------------------------
    pathway_map_amp = {
        'ECIII Input':      'Perforant',
        'CA3 Apical Input': 'Schaffer',
        'CA3 Basal Input':  'Basal_Stratum_Oriens',
    }
    pathway_labels = ['ECIII Input', 'CA3 Apical Input', 'CA3 Basal Input']

    def get_unitary(df_amp, pathway_label, metric):
        path_key = pathway_map_amp[pathway_label]
        sub = df_amp[
            (df_amp['ISI'] == 300) &
            (df_amp['Pathway'] == path_key)
        ][['Cell_ID', 'Genotype', metric]].dropna(subset=[metric]).copy()
        sub = rename_genotype(sub)
        return sub

    ## ROW 3: Gabazine Amplitude (Excitation) at ISI=300
    axs_row3 = []
    for col, plabel in enumerate(pathway_labels):
        ax = fig.add_subplot(gs[2, col])
        axs_row3.append(ax)
        sub = get_unitary(df_amplitudes, plabel, 'Gabazine_Amplitude')
        plot_bar_scatter(ax, sub, 'Genotype', 'Gabazine_Amplitude', 'Genotype',
                         order=['WT', 'I80T/+'], unique_col='Cell_ID')
        if col == 0:
            add_subplot_label(ax, 'C', fontsize=10, fontweight='bold')
            ax.set_ylabel('EPSP Amplitude (mV)\n(No Inhibition)', fontsize=8)
        else:
            ax.set_ylabel('')
        ax.set_xlabel('')
        ax.set_title(plabel, fontsize=8)
        ax.tick_params(axis='x', labelsize=7)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_ylim(bottom=0)

    # Sync all three y-axes for Row 3 (Excitation)
    ymax_r3 = max(ax.get_ylim()[1] for ax in axs_row3)
    for ax in axs_row3:
        ax.set_ylim(0, ymax_r3)
        apply_clean_yticks(ax)

    ## ROW 4: Estimated GABAA Inhibition Amplitude at ISI=300
    axs_row4 = []
    for col, plabel in enumerate(pathway_labels):
        ax = fig.add_subplot(gs[3, col])
        axs_row4.append(ax)
        sub = get_unitary(df_amplitudes, plabel, 'Estimated_Inhibition_Amplitude')
        plot_bar_scatter(ax, sub, 'Genotype', 'Estimated_Inhibition_Amplitude', 'Genotype',
                         order=['WT', 'I80T/+'], unique_col='Cell_ID')
        if col == 0:
            add_subplot_label(ax, 'D', fontsize=10, fontweight='bold')
            ax.set_ylabel('Estimated Inhibition\nAmplitude (mV)', fontsize=8)
        else:
            ax.set_ylabel('')
        ax.set_xlabel('')
        ax.set_title('')
        ax.tick_params(axis='x', labelsize=7)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_ylim(top=0)  # values are negative; cap at 0

    # Sync all three y-axes for Row 4 (GABAA - negative)
    ymin_r4 = min(ax.get_ylim()[0] for ax in axs_row4)
    for ax in axs_row4:
        ax.set_ylim(ymin_r4, 0)
        apply_clean_yticks(ax)

    ## ROW 5: GABAB Decay Area at ISI=300 (stored as negative, plot as positive)
    axs_row5 = []
    for col, plabel in enumerate(pathway_labels):
        ax = fig.add_subplot(gs[4, col])
        axs_row5.append(ax)
        sub = get_unitary(df_amplitudes, plabel, 'GABAB_Area')
        plot_bar_scatter(ax, sub, 'Genotype', 'GABAB_Area', 'Genotype',
                         order=['WT', 'I80T/+'], unique_col='Cell_ID')
        if col == 0:
            add_subplot_label(ax, 'E', fontsize=10, fontweight='bold')
            ax.set_ylabel('Slow IPSP Area\n(mV·ms)', fontsize=8)
        else:
            ax.set_ylabel('')
        ax.set_xlabel('')
        ax.set_title('')
        ax.tick_params(axis='x', labelsize=7)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_ylim(top=0)  # values are negative; cap at 0

    # Sync only CA3 Apical (1) and CA3 Basal (2) y-axes for Row 5 (GABAB - negative)
    ymin_r5_ca3 = min(axs_row5[1].get_ylim()[0], axs_row5[2].get_ylim()[0])
    for ax in [axs_row5[1], axs_row5[2]]:
        ax.set_ylim(ymin_r5_ca3, 0)   # cap top at 0 — no blank space above baseline
        apply_clean_yticks(ax)
    # ECIII scales independently but same top-cap rule
    eciii_ax = axs_row5[0]
    eciii_ax.set_ylim(eciii_ax.get_ylim()[0], 0)
    apply_clean_yticks(eciii_ax)

    # -----------------------------------------------------------------------
    # Load stats and annotate significance brackets (drawn after ylims are set)
    # -----------------------------------------------------------------------
    stats_path = 'paper_data/E_I_data/Stats_Results_Figure_4.csv'
    df_stats4 = pd.read_csv(stats_path) if os.path.exists(stats_path) else None

    def annotate_fig4(ax, df_s, metric, pathway_key):
        if df_s is None: return
        row = df_s[(df_s['Metric'] == metric) & (df_s['Pathway'] == pathway_key)]
        if row.empty: return
        p = row.iloc[0]['P_Value']
        ylo, yhi = ax.get_ylim()
        # For positive values bracket above, for negative bracket below baseline
        y_bracket = yhi * 0.9 if yhi > 0 else ylo * 0.9
        draw_significance(ax, 0, 1, p, y_bracket, bracket=True)

    pathway_keys = ['Perforant', 'Schaffer', 'Basal_Stratum_Oriens']
    for col, (ax3, ax4, ax5, pk) in enumerate(zip(axs_row3, axs_row4, axs_row5, pathway_keys)):
        annotate_fig4(ax3, df_stats4, 'Gabazine_Amplitude', pk)
        annotate_fig4(ax4, df_stats4, 'Estimated_Inhibition_Amplitude', pk)
        annotate_fig4(ax5, df_stats4, 'GABAB_Area', pk)

    output_path = 'paper_figures/Figure_4_Unitary_E_I_Breakdown.png'
    for ext in ['.png', '.pdf', '.svg']:
        fig.savefig(output_path.replace('.png', ext), dpi=300, bbox_inches='tight')
    print(f"✓ Saved Figure 4 to: {output_path}")
    plt.close()



# ==================================================================================================
# FIGURE 4: E:I BALANCE
# ==============================================================================

def plot_figure_5_EI_frequency_dependence(output_path='paper_figures/Figure_5_EI_frequency_dependence.png'):
    """
    Figure 5: Redigned E:I balance now across frequencies (ISIs)
    
    Layout: 5 rows x 3 columns
    - Row 1: ISI 10 for WT 
    - Row 2: ISI 10 for I80T/+
    - Row 3: Excitation Amplitudes (Gabazine) - All ISIs
    - Row 4: Inh (GABAA) Amplitudes (Estimated) - All ISIs
    - Row 5: Inh (GABAB) Area - All ISIs
    """
    print("\n--- Generating Figure 5: E:I Balance Redesign ---")
    setup_publication_style()
    
    # Load E:I traces and amplitudes
    df_traces = pd.read_pickle('paper_data/E_I_data/E_I_traces_for_plotting.pkl')
    df_amplitudes = pd.read_csv('paper_data/E_I_data/E_I_amplitudes.csv')

    # Rename GNB1 -> I80T/+ for display
    df_traces = rename_genotype(df_traces)
    df_amplitudes = rename_genotype(df_amplitudes)
    
    # Convert 17.5 cm to inches
    fig_width = 17.5 / 2.54 
    fig_height = 10.5  # 4 rows: 1 example + 3 quantification
    
    fig = plt.figure(figsize=(fig_width, fig_height))
    
    gs = fig.add_gridspec(4, 3,
                         wspace=0.45, hspace=0.9,
                         left=0.08, right=0.98,
                         top=0.96, bottom=0.08,
                         height_ratios=[1.1, 1, 1, 1])
    
    # Load Significance Markers
    base_stats_dir = 'paper_data/E_I_data/'
    markers_path = os.path.join(base_stats_dir, 'Figure_5_6_Significance_Markers.csv')
    df_markers = pd.read_csv(markers_path) if os.path.exists(markers_path) else None
    if df_markers is not None:
        print(f"✓ Loaded significance markers from: {markers_path}")
    else:
        print(f"⚠ Markers file not found: {markers_path}")

    panel_to_analysis = {
        'C': ('Gabazine_Amplitude', 'WT_vs_GNB1'),
        'D': ('Inhibition_Amplitude', 'WT_vs_GNB1'),
        'E': ('GABAB_Area', 'WT_vs_GNB1')
    }

    pathways = [
        ('ECIII (Perforant)', 'perforant', 'channel_1'),
        ('CA3 Apical (Schaffer)', 'schaffer', 'channel_2'),
        ('CA3 Basal', 'basal', 'Basal_Stratum_Oriens')
    ]
    
    # ROW 1 (A): Two ECIII example traces side-by-side
    # Left: ISI=50ms with annotations  |  Right: ISI=10ms, no annotations
    print("  Row 1: WT example traces (ECIII, 50ms | 10ms)")
    from matplotlib.gridspec import GridSpecFromSubplotSpec
    gs_row1 = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0, :],
                                      wspace=0.05, hspace=0)

    ax_50 = fig.add_subplot(gs_row1[0, 0])
    plot_example_ISI_trace(ax_50, df_traces, df_amplitudes, 50, 'ECIII Input', annotate=True)
    add_subplot_label(ax_50, 'A', x=-0.05)
    ax_50.set_title('ECIII — ISI 50 ms', fontsize=9, fontweight='bold')

    ax_10 = fig.add_subplot(gs_row1[0, 1])
    plot_example_ISI_trace(ax_10, df_traces, df_amplitudes, 10, 'ECIII Input', annotate=False)
    ax_10.set_title('ECIII — ISI 10 ms', fontsize=9, fontweight='bold')
    # Legend on the right panel
    ax_10.legend(frameon=False, fontsize=7, loc='lower right',
                 bbox_to_anchor=(1.02, -0.08), ncol=1)

    # ROW 2 (B): Excitation Amplitudes — shared Y-axis anchored to ECIII (col 0)
    ylims_exc = []
    axs_exc = []
    for col, (label, pathway_key, channel) in enumerate(pathways):
        ax = fig.add_subplot(gs[1, col])
        axs_exc.append(ax)
        if col == 0:
            add_subplot_label(ax, "B", fontsize=10, fontweight='bold')
        pathway_name = {'perforant': 'Perforant', 'schaffer': 'Schaffer',
                        'basal': 'Basal_Stratum_Oriens'}[pathway_key]
        ylim = plot_metric_comparison(ax, df_amplitudes, pathway_name,
                                      'Gabazine_Amplitude', 'Gabazine Amplitude (mV)',
                                      add_legend=False)
        # Add Markers
        analysis, comp = panel_to_analysis['C']
        annotate_with_sig_markers(ax, df_markers, analysis, pathway_name, comp, range(5))
        ylims_exc.append(ylim)
        if col > 0:
            ax.set_ylabel('')

    # Enforce shared Y-axis: 0 to 25 mV for all panels
    for ax in axs_exc:
        ax.set_ylim(0, 25)

    # ROW 4: Inh (GABAA) Amplitudes
    # ROW 3 (C): Inh (GABAA) Amplitudes
    ylims_inh_a = []
    axs_inh_a = []
    for col, (label, pathway_key, channel) in enumerate(pathways):
        ax = fig.add_subplot(gs[2, col])
        axs_inh_a.append(ax)
        if col == 0:
            add_subplot_label(ax, "C", fontsize=10, fontweight='bold')
        pathway_name = {'perforant': 'Perforant', 'schaffer': 'Schaffer', 
                        'basal': 'Basal_Stratum_Oriens'}[pathway_key]
        ylim = plot_metric_comparison(ax, df_amplitudes, pathway_name, 
                                     'Estimated_Inhibition_Amplitude', 'Inh (GABAA) (mV)')
        
        # Add Markers
        analysis, comp = panel_to_analysis['D']
        annotate_with_sig_markers(ax, df_markers, analysis, pathway_name, comp, range(5))
        
        ylims_inh_a.append(ylim)
        if col > 0: ax.set_ylabel('')
        
    # Sync Y-axis for ALL pathways in Row 3 (Panel C)
    max_y = max([yl[1] for yl in ylims_inh_a if yl is not None])
    min_y = min([yl[0] for yl in ylims_inh_a if yl is not None])
    for ax in axs_inh_a:
        ax.set_ylim(min_y, max_y)

    # ROW 4 (D): Inh (GABAB) Area
    ylims_inh_b = []
    axs_inh_b = []
    for col, (label, pathway_key, channel) in enumerate(pathways):
        ax = fig.add_subplot(gs[3, col])
        axs_inh_b.append(ax)
        if col == 0:
            add_subplot_label(ax, "D", fontsize=10, fontweight='bold')
        pathway_name = {'perforant': 'Perforant', 'schaffer': 'Schaffer', 
                        'basal': 'Basal_Stratum_Oriens'}[pathway_key]
        ylim = plot_metric_comparison(ax, df_amplitudes, pathway_name, 
                                     'GABAB_Area', 'Slow IPSP Area (mV·ms)')
        
        # Add Markers
        analysis, comp = panel_to_analysis['E']
        annotate_with_sig_markers(ax, df_markers, analysis, pathway_name, comp, range(5))
        
        ylims_inh_b.append(ylim)
        if col > 0: ax.set_ylabel('')
        
    # Sync Y-axis for ALL pathways in Row 4 (Panel D) — cap top at 0
    max_y = 0  # GABAB area is always negative; force top tick at 0
    min_y = min([yl[0] for yl in ylims_inh_b if yl is not None])
    for ax in axs_inh_b:
        ax.set_ylim(min_y, max_y)
        apply_clean_yticks(ax)

    # Save multiple formats
    for ext in ['.png', '.pdf', '.svg']:
        fig.savefig(output_path.replace('.png', ext), dpi=300, bbox_inches='tight')
    
    print(f"✓ Figure 5 saved to: {output_path}")
    plt.close()


def plot_figure_6_Supralinear_E_I():
    """
    Figure 6: Supralinearity Across ISIs
    Row A: WT 10ms ISI Traces (Measured, Gabazine, Expected)
    Row B: I80T/+ 10ms ISI Traces (Measured, Gabazine, Expected)
    Row C: GABAB Area Quantification (Unitary 300ms)
    Row D: Baclofen Effects (Diagram, Traces, Quantification)
    Row E: Gabazine Supralinearity Quantification
    """
    print("\n--- Generating Figure 6: Supralinearity ---")
    setup_publication_style()
    
    # Load Primary Datasets
    # Unified dataset ensures N-counts are consistent with Figure 4
    df_amplitudes = pd.read_csv('paper_data/E_I_data/E_I_amplitudes.csv')
    df_amplitudes = rename_genotype(df_amplitudes)
    
    # Statistics Markers (Unified Figure 4/5 Stats)
    markers_path = 'paper_data/E_I_data/Figure_5_6_Significance_Markers.csv'
    df_markers = pd.read_csv(markers_path) if os.path.exists(markers_path) else None
    
    # Traces Dataframe (Calculated during E-I export)
    df_traces = pd.read_pickle('paper_data/E_I_data/E_I_traces_for_plotting.pkl')
    df_traces = rename_genotype(df_traces)
    
    # Layout: 3 rows
    fig = plt.figure(figsize=(7.5, 8))
    gs = fig.add_gridspec(3, 3, hspace=0.8, wspace=0.35, 
                         left=0.08, right=0.98, top=0.95, bottom=0.1,
                         height_ratios=[1, 1, 1])
    
    pathways = [
        ('ECIII Input', 0, 'Perforant'),
        ('CA3 Apical Input', 1, 'Schaffer'),
        ('CA3 Basal Input', 2, 'Basal_Stratum_Oriens'),
    ]

    # ROW 1 (A): WT 10ms traces
    print("  Row 1: WT traces")
    for col, (pathway_label, _, _) in enumerate(pathways):
        ax = fig.add_subplot(gs[0, col])
        plot_10ms_ISI_breakdown(ax, df_traces, 'WT', pathway_label)
        if col == 0:
            add_subplot_label(ax, 'A', x=-0.2)
            ax.text(-0.35, 0.5, 'WT', transform=ax.transAxes, ha='right', va='center', fontweight='bold', fontsize=10)

    # ROW 2 (B): I80T/+ 10ms traces
    print("  Row 2: Mutant traces")
    for col, (pathway_label, _, _) in enumerate(pathways):
        ax = fig.add_subplot(gs[1, col])
        plot_10ms_ISI_breakdown(ax, df_traces, 'I80T/+', pathway_label)
        if col == 0:
            add_subplot_label(ax, 'B', x=-0.2)
            ax.text(-0.35, 0.5, 'I80T/+', transform=ax.transAxes, ha='right', va='center', fontweight='bold', fontsize=10)
        if col == 2:
            ax.legend(frameon=False, fontsize=6.5, loc='lower center',
                      bbox_to_anchor=(0.5, -0.55), ncol=2)

    # ROW 3: Supralinearity (C, D, E)
    print("  Row 3: Supralinearity")
    ylims_sup = []
    axs_sup = []
    for col, (pathway_label, _, pathway_match) in enumerate(pathways):
        ax = fig.add_subplot(gs[2, col])
        axs_sup.append(ax)
        label = chr(ord('C') + col)
        ylim = plot_metric_comparison(ax, df_amplitudes, pathway_match, 'Gabazine_Supralinearity', 'Supralinearity (mV)', add_legend=(col==2))
        add_subplot_label(ax, label)
        ylims_sup.append(ylim)
        if df_markers is not None:
            annotate_with_sig_markers(ax, df_markers, 'Gabazine_Supralinearity', pathway_match, 'WT_vs_GNB1', range(5))
        if col > 0: ax.set_ylabel('')

    # Sync Y-axis for ALL Supralinearity pathways
    max_y = max(yl[1] for yl in ylims_sup if yl is not None)
    min_y = min(yl[0] for yl in ylims_sup if yl is not None)
    for ax in axs_sup:
        ax.set_ylim(min_y, max_y)
        apply_clean_yticks(ax)

    save_current_fig('Figure_6_Supralinear_E_I')

# ==============================================================================
# FIGURE 7: DENDRITIC EXCITABILITY
# ==============================================================================

def plot_figure_7_dendritic():
    """
    Figure 7: Dendritic Excitability (Restructured)
    
    Panel A (Rows 1-2): Raw Theta Burst traces (WT vs I80T/+) for ECIII, CA3, Both.
    Panel B (Rows 3-4): Processed (from Pickle) + Expected traces (WT vs I80T/+).
    Panel C (Row 5): Averaged Supralinearity Traces
    Panel D (Row 6): Supralinear Total AUC bar plots
    Panel E (Row 7): Plateau Area bar plots
    """
    
    print("\n--- Generating Figure 7: Dendritic Excitability ---")
    setup_publication_style()
    
    # Load AUC Data for Panel D (needed for filtering inside helper)
    auc_data_path = os.path.join('paper_data', 'supralinearity', 'Supralinear_AUC_Total.csv')
    df_auc_total = None
    if os.path.exists(auc_data_path):
        df_auc_total = pd.read_csv(auc_data_path)
    
    # PREPARE DATA (Logic moved to plotting_utils)
    raw_data, processed_stats, plateau_df, df_auc_total, supralin_traces = prepare_figure_7_data(df_auc_total)

    # Rename GNB1 -> I80T/+ for display
    plateau_df = rename_genotype(plateau_df)
    df_auc_total = rename_genotype(df_auc_total)
    
    # Re-define config variables for plotting calls
    acq_freq = 20000
    start_ms = 400
    end_ms = 1500
    # Panel A: Raw traces are NOT pre-cropped, need absolute indices
    raw_start_idx = int(start_ms * acq_freq / 1000)   # 8000
    raw_end_idx = int(end_ms * acq_freq / 1000)        # 30000
    # Panel B: processed_stats already sliced in prepare_figure_6_data (pre-cropped traces)
    panel_b_start = 0
    panel_b_end = int((end_ms - start_ms) * acq_freq / 1000)  # 22000

    # 4. Create Figure
    # -------------------------------------------------------------------------
    fig = plt.figure(figsize=(6.93, 10)) # Exact 17.6 cm width
    gs = fig.add_gridspec(7, 3, hspace=0.6, wspace=0.35, 
                         height_ratios=[1, 1, 1, 1, 1.4, 1.1, 1.4])
    
    cols = ['Perforant', 'Schaffer', 'Both']
    col_titles = ['ECIII (Perforant)', 'CA3 (Schaffer)', 'Both Pathways']
    
    # 5. Plot Panels Using Modular Functions (New Order: A, B, E -> C, C -> D, D -> E)
    # -------------------------------------------------------------------------
    
    # Panel A: Raw Traces (Rows 0-1)
    plot_theta_raw_traces(fig, gs, raw_data, cols, col_titles, acq_freq, raw_start_idx, raw_end_idx, start_row=0, label="A")
    
    # Panel B: Averaged + Expected Traces (Rows 2-3)
    plot_theta_averaged_traces(fig, gs, processed_stats, cols, acq_freq, panel_b_start, panel_b_end, start_row=2, label="B")
    
    # Panel C: Plateau Area Bar Plots (Row 4) - MOVED FROM E TO C
    stats_path = os.path.join('paper_data', 'Plateau_data', 'Stats_Results_Figure_7.csv')
    df_stats = pd.read_csv(stats_path) if os.path.exists(stats_path) else None
    
    if plateau_df is not None and not plateau_df.empty:
        plot_plateau_area_bars_fig6(fig, gs, plateau_df, df_stats, start_row=4, label="C")
    else:
        ax_c_bar = fig.add_subplot(gs[4, :])
        add_subplot_label(ax_c_bar, "C")
        ax_c_bar.text(0.5, 0.5, "Plateau data missing", ha='center', va='center')
        ax_c_bar.axis('off')

    # Panel D: Averaged Supralinearity Traces (Row 5) - MOVED FROM C TO D
    master_df_temp = pd.read_csv('master_df.csv', low_memory=False)
    if supralin_traces:
        plot_averaged_difference_traces(fig, gs, supralin_traces, master_df_temp, 
                                        panel_b_start, panel_b_end, start_row=5, label="D")
    else:
        ax_d_trace = fig.add_subplot(gs[5, :])
        add_subplot_label(ax_d_trace, "D")
        ax_d_trace.text(0.5, 0.5, "Difference traces not found", ha='center', va='center')
        ax_d_trace.axis('off')

    # Panel E: Supralinear Total AUC (Row 6) - MOVED FROM D TO E
    supralin_stats_path = os.path.join('paper_data', 'Plateau_data', 'Stats_Results_Figure_7.csv')
    if df_auc_total is not None and not df_auc_total.empty:
        df_stats_full = pd.read_csv(supralin_stats_path) if os.path.exists(supralin_stats_path) else None
        plot_supralinear_auc_bars_fig6(fig, gs, df_auc_total, df_stats_full, start_row=6, label="E")
        # Clamp all three AUC panels to ymin=-20 so scale is compact
        for col in range(3):
            _ax = fig.axes[-(3 - col)]   # last 3 axes added by the function
            _, _top = _ax.get_ylim()
            _ax.set_ylim(-20, _top)
            apply_clean_yticks(_ax)
    else:
        ax_e_auc = fig.add_subplot(gs[6, :])
        add_subplot_label(ax_e_auc, "E")
        ax_e_auc.text(0.5, 0.5, "AUC data not found", ha='center', va='center')
        ax_e_auc.axis('off')

    save_current_fig('Figure_7_Dendritic')

# ==================================================================================================
# FIGURE 8: GIRK Channel Analysis (ML297 / ETX Effects)
# ==================================================================================================

def plot_figure_8_GIRK():
    """
    Figure 8: GIRK Channel Analysis & GABAB Integration
    Row 1: A (Diagram placeholder), B/C (Baclofen Vm traces & Delta Vm)
    Row 2: Unitary ML297 (D/E Traces, F Delta Area)
    Row 3: GIRK ML297 Plateau Suppression (G/H Traces, I Delta)
    Row 4: GIRK ETX Plateau Suppression (J/K Traces, L Delta)
    """
    print("\n--- Generating Figure 8: GIRK and GABAB Analysis (Restructured) ---")
    setup_publication_style()
    
    # Data Paths
    unitary_traces_path = 'paper_data/Plateau_data/Figure8_Unitary_Traces.pkl'
    unitary_delta_path = 'paper_data/Plateau_data/GIRK_Unitary_GABAB_Deltas.csv'
    
    baclofen_traces_path = 'paper_data/gabab_analysis/Baclofen_Vm_Example_Traces.pkl'
    baclofen_vm_path = 'paper_data/gabab_analysis/Baclofen_Vm_Change.csv'
    
    plateau_delta_path = 'paper_data/Plateau_data/Plateau_Delta_GIRK.csv'
    plateau_traces_path = 'paper_data/Plateau_data/All_Plateau_Traces.pkl'
    
    stats_path = 'paper_data/Plateau_data/Stats_Results_Figure_8.csv'
    df_stats = pd.read_csv(stats_path) if os.path.exists(stats_path) else None

    # Figure Setup — 4 rows
    fig = plt.figure(figsize=(7.5, 11))
    gs = fig.add_gridspec(4, 1, hspace=0.45, left=0.08, right=0.97, top=0.98, bottom=0.04)

    # ------------------------------------------------------------------
    # ROW 1: GIRK Basics (Baclofen)
    # ------------------------------------------------------------------
    print("  Row 1: GIRK Basics (Baclofen)")
    # Panel A: GIRK Basics (Diagram)
    gs_row1 = gs[0].subgridspec(1, 3, width_ratios=[1.2, 1.2, 0.85], wspace=0.4)
    
    # Panel A: Blank for Diagram
    ax1_a = fig.add_subplot(gs_row1[0])
    add_subplot_label(ax1_a, 'A')
    ax1_a.axis('off')
    ax1_a.text(0.5, 0.5, '[ GIRK Diagram ]', ha='center', va='center', color='grey', alpha=0.5)

    # Panel B: Baclofen Traces
    ax1_b = fig.add_subplot(gs_row1[1])
    plot_baclofen_vm_traces(ax1_b, baclofen_traces_path, label='B')

    # Panel C: Baclofen Delta Vm — clamped to -10 at bottom
    ax1_c = fig.add_subplot(gs_row1[2])
    plot_gabab_vm_change(ax1_c, baclofen_vm_path, 'C', df_stats)
    ax1_c.set_ylim(-10, 0)
    apply_clean_yticks(ax1_c, n=6)

    # ------------------------------------------------------------------
    # ROW 2: Unitary ML297
    # ------------------------------------------------------------------
    print("  Row 2: Unitary ML297")
    gs_row2 = gs[1].subgridspec(1, 3, width_ratios=[1.2, 1.2, 0.85], wspace=0.4)
    
    ax2_d = fig.add_subplot(gs_row2[0])
    add_subplot_label(ax2_d, 'D')
    plot_unitary_gabab_traces_by_pathway(ax2_d, unitary_traces_path, 'WT', 'Perforant', '', drugs=['Gabazine', 'ML297'])

    ax2_e = fig.add_subplot(gs_row2[1])
    plot_unitary_gabab_traces_by_pathway(ax2_e, unitary_traces_path, 'GNB1', 'Perforant', '', drugs=['Gabazine', 'ML297'])

    ax2_f = fig.add_subplot(gs_row2[2])
    plot_unitary_gabab_area_delta_single(ax2_f, unitary_delta_path, 'ML297', 'Perforant', 'F', df_stats)

    # ------------------------------------------------------------------
    # ROW 3: ML297 Theta Burst (Plateau)
    # ------------------------------------------------------------------
    print("  Row 3: ML297 Plateau")
    gs_row3 = gs[2].subgridspec(1, 3, width_ratios=[1.2, 1.2, 0.85], wspace=0.4)
    plateau_traces = pd.read_pickle(plateau_traces_path) if os.path.exists(plateau_traces_path) else {}

    if 'Before_ML297' in plateau_traces and 'After_ML297' in plateau_traces:
        ax3_g = fig.add_subplot(gs_row3[0])
        add_subplot_label(ax3_g, 'G')
        plot_traces_GIRK_v2(ax3_g, plateau_traces['Before_ML297'], plateau_traces['After_ML297'],
                            'WT', 'ML297', after_color='gold', add_legend=True, add_scale=True)

        ax3_h = fig.add_subplot(gs_row3[1])
        plot_traces_GIRK_v2(ax3_h, plateau_traces['Before_ML297'], plateau_traces['After_ML297'],
                            'I80T/+', 'ML297', after_color='gold', add_legend=False)

        ax3_i = fig.add_subplot(gs_row3[2])
        plot_plateau_girk_delta(ax3_i, plateau_delta_path, 'ML297', 'I', df_stats)
        _bot, _ = ax3_i.get_ylim()
        ax3_i.set_ylim(_bot, 5)   # cap ymax at +5
        apply_clean_yticks(ax3_i, n=6)

    # ------------------------------------------------------------------
    # ROW 4: ETX Theta Burst (Plateau)
    # ------------------------------------------------------------------
    print("  Row 4: ETX Plateau")
    gs_row4 = gs[3].subgridspec(1, 3, width_ratios=[1.2, 1.2, 0.85], wspace=0.4)
    if 'Before_ETX' in plateau_traces and 'After_ETX' in plateau_traces:
        ax4_j = fig.add_subplot(gs_row4[0])
        add_subplot_label(ax4_j, 'J')
        plot_traces_GIRK_v2(ax4_j, plateau_traces['Before_ETX'], plateau_traces['After_ETX'],
                            'WT', 'ETX', after_color='cyan', add_legend=False, add_scale=True)

        ax4_k = fig.add_subplot(gs_row4[1])
        plot_traces_GIRK_v2(ax4_k, plateau_traces['Before_ETX'], plateau_traces['After_ETX'],
                            'I80T/+', 'ETX', after_color='cyan', add_legend=False)

        ax4_l = fig.add_subplot(gs_row4[2])
        plot_plateau_girk_delta(ax4_l, plateau_delta_path, 'ETX', 'L', df_stats)
        apply_clean_yticks(ax4_l, n=6)  # tighter spacing for small delta range

    save_current_fig('Figure_8_GIRK_Restructured')


def plot_supplemental_figure_1():
    """
    Supplemental Figure 1: Full E/I Imbalance Index Summary
    Plots genotype comparisons and statistical significance across all ISIs and pathways.
    """
    print("\n--- Generating Supplemental Figure 1: E/I Imbalance ---")
    setup_publication_style()
    
    # Load data
    ei_amp_path = 'paper_data/E_I_data/E_I_amplitudes.csv'
    if not os.path.exists(ei_amp_path):
        print(f"❌ Error: {ei_amp_path} not found")
        return
        
    df = pd.read_csv(ei_amp_path)
    df = rename_genotype(df)
    
    # Statistics Markers
    markers_path = 'paper_data/E_I_data/Figure_5_6_Significance_Markers.csv'
    df_markers = pd.read_csv(markers_path) if os.path.exists(markers_path) else None
    
    # Match dimensions of Figure 6 subplots
    fig_width = 7.5
    fig_height = 2.5 
    
    fig, axes = plt.subplots(1, 3, figsize=(fig_width, fig_height), sharex=False, sharey=False)
    fig.subplots_adjust(wspace=0.35, left=0.08, right=0.98, top=0.8, bottom=0.2)
    
    pathways = [
        ('Perforant', 'Perforant'),
        ('Schaffer', 'Schaffer'),
        ('Basal SO', 'Basal_Stratum_Oriens')
    ]
    
    ylims = []
    for col_idx, (path_label, path_code) in enumerate(pathways):
        ax = axes[col_idx]
        ylim = plot_metric_comparison(ax, df, path_code, 'E_I_Imbalance', 
                                     'E/I Imbalance Index' if col_idx==0 else '', 
                                     add_legend=(col_idx==2))
        ax.set_title(path_label, fontsize=10, fontweight='bold')
        ax.set_xlabel('ISI (ms)')
        
        # Add labels and significance markers
        if df_markers is not None:
            annotate_with_sig_markers(ax, df_markers, 'E_I_Imbalance', path_code, 'WT_vs_GNB1', range(5))
        
        ylims.append(ylim)
        
    # Sync Y axes
    ymin = min(y[0] for y in ylims if y is not None)
    ymax = max(y[1] for y in ylims if y is not None)
    for ax in axes:
        ax.set_ylim(ymin, ymax)
        apply_clean_yticks(ax)

    save_current_fig('Supplemental_Figure_1_EI_Imbalance')

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
    plot_figure_4_Unitary_E_I_Breakdown()
    plot_figure_5_EI_frequency_dependence()
    plot_figure_6_Supralinear_E_I()
    plot_figure_7_dendritic()
    plot_figure_8_GIRK()
    plot_supplemental_figure_1()
    plot_supplemental_figure_2()
    plot_supplemental_figure_3()