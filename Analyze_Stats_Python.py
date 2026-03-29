import pandas as pd
import numpy as np
import os
import sys

# Import custom utils
from stats_utils import *

# IMPORT BOX UTILITIES
try:
    import box_utils
except ImportError:
    print("Error: box_utils.py not found. Please ensure it is in the same directory.")
    sys.exit(1)

# ==================================================================================================
# CONFIGURATION
# ==================================================================================================

DATA_ROOT = 'paper_data'

# Helper to load data
def load_data(subfolder, filename):
    path = os.path.join(DATA_ROOT, subfolder, filename)
    if os.path.exists(path):
        return pd.read_csv(path)
    else:
        print(f"⚠ Warning: Data not found at {path}")
        return None

# ==================================================================================================
# FIGURE 1: BEHAVIOR STATS
# ==================================================================================================

def run_stats_figure_1():
    print("\n" + "="*80)
    print("FIGURE 1: BEHAVIOR STATISTICS")
    print("="*80)
    
    # List to collect all results for export
    all_stats_export = []

    # Helper to print and store
    def record_stat(fig_panel, comparison_name, res):
        print_stat_result(fig_panel, comparison_name, res)
        all_stats_export.append({
            'Figure_Panel': fig_panel,
            'Comparison': comparison_name,
            'Test_Used': res['Test'],
            'Statistic': res['Statistic'],
            'P_Value': res['p'],
            'Significance': res['Significance'],
            'Normality_Check': res['Normality']
        })

    # --- Load Data ---
    df_weights = load_data('Behavior_Analysis', 'Mouse_Weights_Processed.csv')
    df_of_loco = load_data('Behavior_Analysis', 'Open_Field_Locomotion_Trial1.csv')
    df_of_anx = load_data('Behavior_Analysis', 'Open_Field_Anxiety_Processed.csv')
    df_dvc_cages = load_data('DVC_Analysis', 'Cage_Specific_Hours_Summary.csv')
    df_olm_delta = load_data('Behavior_Analysis', 'OLM_Summary_Deltas.csv')
    df_tmaze = load_data('Behavior_Analysis', 'T_Maze_Alternations.csv')
    
    # --- 1B: Weights (Snapshots at P8-P10, P28, and Adult) ---
    if df_weights is not None:
        for tp in ['P8-P10', 'P28', 'Adult']:
            sub = df_weights[df_weights['Timepoint_Label'] == tp]
            wt = sub[sub['Genotype'] == 'WT']['Weight_g']
            gnb1 = sub[sub['Genotype'] == 'GNB1']['Weight_g']
            
            res = compare_two_groups(wt, gnb1)
            record_stat("Fig 1B", f"Weights ({tp}): WT vs GNB1", res)

    # --- 1C: Open Field Locomotion ---
    if df_of_loco is not None:
        wt = df_of_loco[df_of_loco['Genotype'] == 'WT']['Distance (m)']
        gnb1 = df_of_loco[df_of_loco['Genotype'] == 'GNB1']['Distance (m)']
        
        res = compare_two_groups(wt, gnb1)
        record_stat("Fig 1C", "Locomotion Total Distance: WT vs GNB1", res)

    # --- 1D: Anxiety (Ratio) ---
    if df_of_anx is not None:
        wt = df_of_anx[df_of_anx['Genotype'] == 'WT']['Center_Outer_Time_Ratio']
        gnb1 = df_of_anx[df_of_anx['Genotype'] == 'GNB1']['Center_Outer_Time_Ratio']
        
        res = compare_two_groups(wt, gnb1)
        record_stat("Fig 1D", "Anxiety Ratio: WT vs GNB1", res)

    # --- 1G: DVC Total Activity (Dark Phase) ---
    if df_dvc_cages is not None:
        wt = df_dvc_cages[df_dvc_cages['Genotype'] == 'WT']['Sum_All_Dark']
        gnb1 = df_dvc_cages[df_dvc_cages['Genotype'] == 'GNB1']['Sum_All_Dark']
        
        res = compare_two_groups(wt, gnb1)
        record_stat("Fig 1G", "DVC Dark Phase: WT vs GNB1", res)

    # --- Supplemental Fig: OLM Training vs Testing (Within Group) ---
    df_olm_raw = load_data('Behavior_Analysis', 'OLM_Data_Filtered.csv')
    
    if df_olm_raw is not None:
        training = df_olm_raw[df_olm_raw['Stage'] == 'Familiarisation Day 2'][['Animal', 'Genotype', 'Training_DI']]
        testing = df_olm_raw[df_olm_raw['Stage'] == 'Testing Stage'][['Animal', 'Testing_DI']]
        merged = training.merge(testing, on='Animal')
        
        for geno in ['WT', 'GNB1']:
            sub = merged[merged['Genotype'] == geno]
            # Paired Test
            res = compare_two_groups(sub['Training_DI'], sub['Testing_DI'], paired=True)
            record_stat("Supplemental Fig", f"OLM Learning ({geno}): Training vs Testing", res)

    # --- Supplemental Fig: OLM Delta (Between Group) ---
    if df_olm_delta is not None:
        wt = df_olm_delta[df_olm_delta['Genotype'] == 'WT']['Delta_DI']
        gnb1 = df_olm_delta[df_olm_delta['Genotype'] == 'GNB1']['Delta_DI']
        
        res = compare_two_groups(wt, gnb1)
        record_stat("Supplemental Fig", "OLM Memory Score (Delta): WT vs GNB1", res)

    # --- 1K: T-Maze Alternation ---
    if df_tmaze is not None:
        wt = df_tmaze[df_tmaze['Genotype'] == 'WT']['Percent_Alternations']
        gnb1 = df_tmaze[df_tmaze['Genotype'] == 'GNB1']['Percent_Alternations']
        
        res = compare_two_groups(wt, gnb1)
        record_stat("Fig 1K", "T-Maze Alternation: WT vs GNB1", res)

    # --- 1I: T-Maze Distance ---
    df_tmaze_entries = load_data('Behavior_Analysis', 'T_Maze_Zone_Entries.csv')
    if df_tmaze_entries is not None:
        wt = df_tmaze_entries[df_tmaze_entries['Genotype'] == 'WT']['Distance (m)']
        gnb1 = df_tmaze_entries[df_tmaze_entries['Genotype'] == 'GNB1']['Distance (m)']
        
        res = compare_two_groups(wt, gnb1)
        record_stat("Fig 1I", "T-Maze Distance: WT vs GNB1", res)

    # --- 1J: T-Maze Total Entries ---
    if df_tmaze_entries is not None:
        # Calculate total entries (sum of all zone entries)
        df_tmaze_entries['Total_Entries'] = df_tmaze_entries[['Start : entries', 'Left Arm : entries', 'Right Arm : entries']].sum(axis=1, skipna=True)
        
        wt = df_tmaze_entries[df_tmaze_entries['Genotype'] == 'WT']['Total_Entries']
        gnb1 = df_tmaze_entries[df_tmaze_entries['Genotype'] == 'GNB1']['Total_Entries']
        
        res = compare_two_groups(wt, gnb1)
        record_stat("Fig 1J", "T-Maze Total Entries: WT vs GNB1", res)

    # --- EXPORT RESULTS ---
    if all_stats_export:
        stats_df = pd.DataFrame(all_stats_export)
        save_path = os.path.join(DATA_ROOT, 'Behavior_Analysis', 'Stats_Results_Figure_1.csv')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        stats_df.to_csv(save_path, index=False)
        print(f"\n✓ Saved full statistical results to: {save_path}")

def run_stats_figure_2():
    print("\n" + "="*80)
    print("FIGURE 2: PHYSIOLOGY STATISTICS")
    print("="*80)
    
    all_stats_export = []

    def record_stat(fig_panel, comparison_name, res):
        print(f"  > {comparison_name}: p={res['p']:.4f} ({res['Significance']})")
        all_stats_export.append({
            'Figure_Panel': fig_panel,
            'Comparison': comparison_name,
            'Test_Used': res['Test'],
            'Statistic': res['Statistic'],
            'P_Value': res['p'],
            'Significance': res['Significance'],
            'Normality_Check': res['Normality']
        })

    # --- Load Data ---
    # Assuming data is in 'Physiology_Analysis' subfolder. Change if needed.
    df_intrinsic = load_data('Physiology_Analysis', 'Intrinsic_properties.csv')
    df_ap_ahp = load_data('Physiology_Analysis', 'combined_AP_AHP_rheobase_analysis.csv')

    # Define groups
    def get_groups(df, col_name):
        if df is None:
            return pd.Series(dtype=float), pd.Series(dtype=float)
        
        if col_name not in df.columns:
            print(f"⚠ Warning: Column '{col_name}' not found in dataframe.")
            # FIX: Return empty Pandas Series instead of lists [], []
            # This prevents AttributeError: 'numpy.ndarray' object has no attribute 'dropna'
            return pd.Series(dtype=float), pd.Series(dtype=float)

        # Standardize Genotype
        if 'Genotype' in df.columns:
            df['Genotype'] = df['Genotype'].astype(str).str.strip()
        
        wt = df[df['Genotype'] == 'WT'][col_name]
        gnb1 = df[df['Genotype'] == 'GNB1'][col_name]
        return wt, gnb1

    # --- PANEL A: Intrinsic Properties ---
    if df_intrinsic is not None:
        # Input Resistance
        wt, gnb1 = get_groups(df_intrinsic, 'Input_Resistance_MOhm')
        record_stat("Fig 2A", "Input Resistance", compare_groups_mannwhitney(wt, gnb1))

        # Vm Rest
        wt, gnb1 = get_groups(df_intrinsic, 'Vm rest/start (mV)')
        record_stat("Fig 2A", "Vm Rest", compare_groups_mannwhitney(wt, gnb1))

        # Access Resistance
        wt, gnb1 = get_groups(df_intrinsic, 'Access Resistance (From Whole Cell V-Clamp)')
        record_stat("Fig 2A", "Access Resistance", compare_groups_mannwhitney(wt, gnb1))

        # Access Resistance
        wt, gnb1 = get_groups(df_intrinsic, 'Voltage_sag')
        record_stat("Fig 2A", "Voltage Sag", compare_groups_mannwhitney(wt, gnb1))

    # --- PANEL C: AP Properties (Panel B is trace) ---
    if df_ap_ahp is not None:
        # Rheobase
        wt, gnb1 = get_groups(df_ap_ahp, 'Rheobase_Current')
        record_stat("Fig 2C", "Rheobase", compare_groups_mannwhitney(wt, gnb1))

        # AP Threshold
        wt, gnb1 = get_groups(df_ap_ahp, 'AP_threshold')
        record_stat("Fig 2C", "AP Threshold", compare_groups_mannwhitney(wt, gnb1))

        # AP Size (Amplitude)
        wt, gnb1 = get_groups(df_ap_ahp, 'AP_size')
        record_stat("Fig 2C", "AP Size", compare_groups_mannwhitney(wt, gnb1))

        # AP Halfwidth
        wt, gnb1 = get_groups(df_ap_ahp, 'AP_halfwidth')
        record_stat("Fig 2C", "AP Halfwidth", compare_groups_mannwhitney(wt, gnb1))

    # --- PANEL E: AHP Properties (Panel D is trace) ---
    if df_ap_ahp is not None:
        # AHP Amplitude
        wt, gnb1 = get_groups(df_ap_ahp, 'AHP_size')
        record_stat("Fig 2E", "AHP Amplitude", compare_groups_mannwhitney(wt, gnb1))

        # AHP Decay
        wt, gnb1 = get_groups(df_ap_ahp, 'decay_area')
        record_stat("Fig 2E", "AHP Decay", compare_groups_mannwhitney(wt, gnb1))

    # --- PANEL F: F-I Curve Midpoint ---
    df_midpoints = load_data('Firing_Rate', 'Sigmoid_Fit_Params.csv')
    if df_midpoints is not None:
        # Standardize genotype
        df_midpoints['Genotype'] = df_midpoints['Genotype'].astype(str).str.strip()
        # Use FI_Midpoint if renamed, else Midpoint
        midpoint_col = 'FI_Midpoint' if 'FI_Midpoint' in df_midpoints.columns else 'Midpoint'
        wt, gnb1 = get_groups(df_midpoints, midpoint_col)
        record_stat("Fig 2F", "F-I Curve Midpoint: WT vs GNB1", compare_groups_mannwhitney(wt, gnb1))

    # --- EXPORT ---
    if all_stats_export:
        stats_df = pd.DataFrame(all_stats_export)
        # Saving to the same root folder as other stats
        save_path = os.path.join(DATA_ROOT, 'Physiology_Analysis', 'Stats_Results_Figure_2.csv')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        stats_df.to_csv(save_path, index=False)
        print(f"\n✓ Saved Figure 2 stats to: {save_path}")

# ==================================================================================================
# FIGURE 3: MORPHOLOGY STATISTICS
# ==================================================================================================

def run_stats_figure_3():
    print("\n" + "="*80)
    print("FIGURE 3: MORPHOLOGY STATISTICS")
    print("="*80)
    
    all_stats_export = []

    def record_stat(fig_panel, comparison_name, res):
        print_stat_result(fig_panel, comparison_name, res)
        all_stats_export.append({
            'Figure_Panel': fig_panel,
            'Comparison': comparison_name,
            'Test_Used': res['Test'],
            'Statistic': res['Statistic'],
            'P_Value': res['p'],
            'Significance': res['Significance'],
            'Normality_Check': res['Normality']
        })

    # --- Load Data ---
    df_sholl = load_data('Morphology_Analysis', 'Sholl_Intersections_Raw.csv')
    df_props = load_data('Morphology_Analysis', 'Dendrite_Properties_All.csv')

    # ===========================================================================
    # PANEL D & E: Sholl Cumulative Distributions (KS Test)
    # ===========================================================================
    if df_sholl is not None:
        print("\n--- Panels D & E: Sholl KS Tests ---")
        
        # Prepare data for KS test:
        # We need samples of radii. If Radius R has N intersections, R should appear N times in the sample.
        # Negative radius = Basal, Positive radius = Apical
        
        # Function to reconstruct samples
        def get_radius_samples(df, genotype, dendrite_type):
            if dendrite_type == 'Basal':
                subset = df[(df['Genotype'] == genotype) & (df['Radius'] < 0)].copy()
                subset['Radius'] = subset['Radius'].abs() # Use absolute distance
            else: # Apical
                subset = df[(df['Genotype'] == genotype) & (df['Radius'] > 0)].copy()
                
            samples = []
            for _, row in subset.iterrows():
                try:
                    count = int(row['Inters.'])
                    if count > 0:
                        samples.extend([row['Radius']] * count)
                except:
                    continue
            return np.array(samples)

        # Panel D: Basal
        wt_basal = get_radius_samples(df_sholl, 'WT', 'Basal')
        gnb1_basal = get_radius_samples(df_sholl, 'GNB1', 'Basal')
        
        print(f"  Basal Samples: WT n={len(wt_basal)}, GNB1 n={len(gnb1_basal)}")
        res = compare_distributions_ks(wt_basal, gnb1_basal)
        record_stat("Fig 3D", "Basal Sholl Distribution (KS): WT vs GNB1", res)
        
        # Panel E: Apical
        wt_apical = get_radius_samples(df_sholl, 'WT', 'Apical')
        gnb1_apical = get_radius_samples(df_sholl, 'GNB1', 'Apical')
        
        print(f"  Apical Samples: WT n={len(wt_apical)}, GNB1 n={len(gnb1_apical)}")
        res = compare_distributions_ks(wt_apical, gnb1_apical)
        record_stat("Fig 3E", "Apical Sholl Distribution (KS): WT vs GNB1", res)

    # ===========================================================================
    # PANEL F & G: Dendrite Properties (Branch Sum & Terminal Branches)
    # ===========================================================================
    if df_props is not None:
        print("\n--- Panels F & G: Dendrite Properties ---")
        
        # Helper to get groups
        def get_prop_groups(dend_type, metric):
            subset = df_props[df_props['Dendrite_Type'] == dend_type]
            wt = subset[subset['Genotype'] == 'WT'][metric]
            gnb1 = subset[subset['Genotype'] == 'GNB1'][metric]
            return wt, gnb1

        # F1: Basal Branch Sum
        wt, gnb1 = get_prop_groups('Basal', 'branch_sum')
        res = compare_two_groups(wt, gnb1)
        record_stat("Fig 3F (Left)", "Basal Total Branch Length", res)
        
        # F2: Apical Branch Sum
        wt, gnb1 = get_prop_groups('Apical', 'branch_sum')
        res = compare_two_groups(wt, gnb1)
        record_stat("Fig 3F (Right)", "Apical Total Branch Length", res)
        
        # G1: Basal Terminal Branches
        wt, gnb1 = get_prop_groups('Basal', 'N_terminal_branches')
        res = compare_two_groups(wt, gnb1)
        record_stat("Fig 3G (Left)", "Basal Terminal Branches", res)
        
        # G2: Apical Terminal Branches
        wt, gnb1 = get_prop_groups('Apical', 'N_terminal_branches')
        res = compare_two_groups(wt, gnb1)
        record_stat("Fig 3G (Right)", "Apical Terminal Branches", res)

    # --- EXPORT ---
    if all_stats_export:
        stats_df = pd.DataFrame(all_stats_export)
        save_path = os.path.join(DATA_ROOT, 'Morphology_Analysis', 'Stats_Results_Figure_3.csv')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        stats_df.to_csv(save_path, index=False)
        print(f"\n✓ Saved Figure 3 stats to: {save_path}")



def run_stats_figure_6():
    """
    Statistical analysis for Figure 6: Dendritic Excitability
    
    Panel C: Plateau Area - WT vs GNB1 for Schaffer and Perforant pathways
            (Only data from 07/09/2024 onwards - Hardware fix)
    """
    print("\n" + "="*80)
    print("FIGURE 6: DENDRITIC EXCITABILITY STATISTICS")
    print("="*80)
    
    all_stats_export = []

    def record_stat(fig_panel, comparison_name, res, n_wt=None, n_gnb1=None):
        print_stat_result(fig_panel, comparison_name, res)
        row = {
            'Comparison': comparison_name,
            'Mean_WT': res['Mean_WT'],
            'Mean_GNB1': res['Mean_GNB1'],
            'SEM_WT': res['SEM_WT'],
            'SEM_GNB1': res['SEM_GNB1'],
            'Figure_Panel': fig_panel,
            'Test_Used': res['Test'],
            'Statistic': res['Statistic'],
            'P_Value': res['p'],
            'Significance': res['Significance'],
            'Normality_Check': res['Normality']
        }
        if n_wt is not None:
            row['N_WT'] = n_wt
        if n_gnb1 is not None:
            row['N_GNB1'] = n_gnb1
        all_stats_export.append(row)

    # --- Load Plateau Data ---
    df_plateau = load_data('Plateau_data', 'Plateau_data.csv')
    
    # --- Load Supralinearity Data ---
    df_supralin = load_data('supralinearity', 'Supralinear_AUC_Total.csv')

    # ===========================================================================
    # PANEL C: Plateau Area - WT vs GNB1 for Schaffer and Perforant Pathways
    # Note: Filtering by 'Single Pathway Plateau Inclusion' handled during export
    # ===========================================================================
    if df_plateau is not None:
        print("\n--- Panel C: Plateau Area (Schaffer & Perforant) ---")
        
        valid_conditions = ['Gabazine_Only', 'Before_ML297', 'Before_ETX']
        
        # Both Pathways
        plateau_both = df_plateau[
            (df_plateau['Condition'].isin(valid_conditions)) & 
            (df_plateau['Pathway'] == 'Both')
        ]
        wt = plateau_both[plateau_both['Genotype'] == 'WT']['Plateau_Area'].dropna()
        gnb1 = plateau_both[plateau_both['Genotype'] == 'GNB1']['Plateau_Area'].dropna()
        mean_wt = wt.mean()
        mean_gnb1 = gnb1.mean()
        sem_wt = wt.sem()
        sem_gnb1 = gnb1.sem()

        print(f"  Both Pathway - WT: n={len(wt)}, GNB1: n={len(gnb1)}")
        res = compare_two_groups(wt, gnb1)
        res.update({'Mean_WT': mean_wt, 'Mean_GNB1': mean_gnb1, 'SEM_WT': sem_wt, 'SEM_GNB1': sem_gnb1})
        record_stat("Fig 6C", "Plateau Area (Both): WT vs GNB1", res, n_wt=len(wt), n_gnb1=len(gnb1))
        
        # Schaffer Pathway
        plateau_sch = df_plateau[
            (df_plateau['Condition'].isin(valid_conditions)) & 
            (df_plateau['Pathway'] == 'Schaffer')
        ]
        wt = plateau_sch[plateau_sch['Genotype'] == 'WT']['Plateau_Area'].dropna()
        gnb1 = plateau_sch[plateau_sch['Genotype'] == 'GNB1']['Plateau_Area'].dropna()

        print(f"  Schaffer Pathway - WT: n={len(wt)}, GNB1: n={len(gnb1)}")
        if len(wt) > 0 and len(gnb1) > 0:
            res = compare_two_groups(wt, gnb1)
            res.update({'Mean_WT': wt.mean(), 'Mean_GNB1': gnb1.mean(), 'SEM_WT': wt.sem(), 'SEM_GNB1': gnb1.sem()})
            record_stat("Fig 6C", "Plateau Area (Schaffer): WT vs GNB1", res, n_wt=len(wt), n_gnb1=len(gnb1))
        
        # Perforant Pathway
        plateau_perf = df_plateau[
            (df_plateau['Condition'].isin(valid_conditions)) & 
            (df_plateau['Pathway'] == 'Perforant')
        ]
        wt = plateau_perf[plateau_perf['Genotype'] == 'WT']['Plateau_Area'].dropna()
        gnb1 = plateau_perf[plateau_perf['Genotype'] == 'GNB1']['Plateau_Area'].dropna()

        print(f"  Perforant Pathway - WT: n={len(wt)}, GNB1: n={len(gnb1)}")
        if len(wt) > 0 and len(gnb1) > 0:
            res = compare_two_groups(wt, gnb1)
            res.update({'Mean_WT': wt.mean(), 'Mean_GNB1': gnb1.mean(), 'SEM_WT': wt.sem(), 'SEM_GNB1': gnb1.sem()})
            record_stat("Fig 6C", "Plateau Area (Perforant): WT vs GNB1", res, n_wt=len(wt), n_gnb1=len(gnb1))

    # ===========================================================================
    # PANEL F: Supralinear Total AUC - WT vs GNB1 by Pathway
    # N values MUST match Panel C per pathway (filter to Panel C cell set)
    # ===========================================================================
    if df_supralin is not None and df_plateau is not None:
        print("\n--- Panel F: Supralinear Total AUC ---")
        
        valid_conditions = ['Gabazine_Only', 'Before_ML297', 'Before_ETX']
        
        # Map supralinearity pathway names -> plateau pathway names -> display labels
        pathway_info = {
            'Both Pathways': ('Both', 'Both'),
            'Schaffer':      ('Schaffer', 'CA3'),
            'Perforant':     ('Perforant', 'ECIII'),
        }
        
        for supra_name, (plat_name, display_label) in pathway_info.items():
            # Get Panel C cell IDs for this pathway
            panel_c_cells = df_plateau[
                (df_plateau['Condition'].isin(valid_conditions)) & 
                (df_plateau['Pathway'] == plat_name)
            ].dropna(subset=['Plateau_Area'])['Cell_ID'].unique()
            
            # Filter supralinearity to ONLY cells present in Panel C
            pathway_data = df_supralin[
                (df_supralin['Pathway'] == supra_name) &
                (df_supralin['Cell_ID'].isin(panel_c_cells))
            ]
            
            wt = pathway_data[pathway_data['Genotype'] == 'WT']['Total_AUC'].dropna()
            gnb1 = pathway_data[pathway_data['Genotype'] == 'GNB1']['Total_AUC'].dropna()
            
            print(f"  {display_label}: WT n={len(wt)}, GNB1 n={len(gnb1)}")
            if len(wt) > 0 and len(gnb1) > 0:
                res = compare_two_groups(wt, gnb1)
                res.update({'Mean_WT': wt.mean(), 'Mean_GNB1': gnb1.mean(), 'SEM_WT': wt.sem(), 'SEM_GNB1': gnb1.sem()})
                record_stat("Fig 6F", f"Supralinear Total AUC ({display_label}): WT vs GNB1", res, n_wt=len(wt), n_gnb1=len(gnb1))

    # ===========================================================================
    # PANEL G: Supralinear Peak Across Theta Cycles
    # Test Genotype effect per pathway across cycles
    # ANOVA first, then post-hoc only if significant
    # ===========================================================================
    # Load per-cycle peak data (separate file from Total AUC)
    df_supralin_peaks = load_data('supralinearity', 'Supralinear_Peaks_Wide.csv')
    
    if df_supralin_peaks is not None:
        print("\n--- Panel G: Supralinear Peak Across Cycles (ANOVA Check) ---")
        
        # Try to import statsmodels for ANOVA
        try:
            import statsmodels.formula.api as smf
            import statsmodels.api as sm
            has_statsmodels = True
        except ImportError:
            print("⚠ Warning: statsmodels not installed. Skipping ANOVA check and running all comparisons.")
            has_statsmodels = False
        
        pathway_map = {
            'Schaffer': 'CA3',
            'Perforant': 'ECIII', 
            'Both Pathways': 'Both'
        }
        
        # Prepare Long Format Data for ANOVA
        df_long = pd.melt(df_supralin_peaks, 
                          id_vars=['Cell_ID', 'Genotype', 'Pathway'],
                          value_vars=[f'Cycle_{i}' for i in range(1, 6)],
                          var_name='Cycle', value_name='Peak_Amplitude')
        
        for pathway_orig, pathway_label in pathway_map.items():
            print(f"\nAnalyzing {pathway_label} ({pathway_orig})...")
            
            # Subset data
            path_long = df_long[df_long['Pathway'] == pathway_orig].copy()
            path_wide = df_supralin_peaks[df_supralin_peaks['Pathway'] == pathway_orig]
            
            run_posthoc = True # Default to true if no statsmodels
            
            if has_statsmodels:
                try:
                    # Run Mixed Linear Model (Genotype * Cycle, random intercept per Cell_ID)
                    # Treat Cycle as categorical
                    model = smf.mixedlm("Peak_Amplitude ~ Genotype * C(Cycle)", 
                                      path_long, 
                                      groups=path_long["Cell_ID"])
                    fit = model.fit()
                    print(fit.summary().tables[1])
                    
                    # Extract p-values
                    p_vals = fit.pvalues
                    
                    # Check for significance (Genotype or Interaction)
                    # Note: Keys might vary depending on reference level, check for string containment
                    genotype_sig = any(p < 0.05 for k, p in p_vals.items() if 'Genotype' in k)
                    
                    if genotype_sig:
                        print(f"  ✓ Significant Genotype Effect/Interaction found for {pathway_label}. Running post-hocs.")
                        run_posthoc = True
                    else:
                        print(f"  x No significant Genotype effect for {pathway_label}. Skipping post-hocs.")
                        run_posthoc = False
                        
                        # Add a dummy 'ns' record so plots know it was tested but not significant?
                        # Or just don't record anything. Existing code in generate_figures checks for records.
                        
                except Exception as e:
                    print(f"  ⚠ ANOVA failed: {e}. Defaulting to running comparisons.")
                    run_posthoc = True
            
            if run_posthoc:
                # Compare each cycle
                for cycle_num in range(1, 6):
                    col_name = f'Cycle_{cycle_num}'
                    wt = path_wide[path_wide['Genotype'] == 'WT'][col_name].dropna()
                    gnb1 = path_wide[path_wide['Genotype'] == 'GNB1'][col_name].dropna()
                    
                    if len(wt) > 0 and len(gnb1) > 0:
                        res = compare_two_groups(wt, gnb1)
                        res.update({'Mean_WT': wt.mean(), 'Mean_GNB1': gnb1.mean(), 'SEM_WT': wt.sem(), 'SEM_GNB1': gnb1.sem()})
                        record_stat("Fig 6G", f"Cycle {cycle_num} ({pathway_label}): WT vs GNB1", res)
            else:
                 # Record a non-significant result effectively for the whole group?
                 # Actually, if we skip recording, generate_figures won't draw asterisks, which is what we want.
                 pass

    # --- EXPORT RESULTS ---
    if all_stats_export:
        stats_df = pd.DataFrame(all_stats_export)
        
        save_path = os.path.join(DATA_ROOT, 'Plateau_data', 'Stats_Results_Figure_6.csv')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        stats_df.to_csv(save_path, index=False)
        print(f"\n✓ Saved Figure 6 stats to: {save_path}")


# ==================================================================================================
# FIGURE 5: GABAb ANALYSIS STATS
# ==================================================================================================

def run_stats_figure_5():
    print("\n" + "="*80)
    print("FIGURE 5: GABAb ANALYSIS STATISTICS")
    print("="*80)
    
    # Load GABAb metrics
    gabab_df = load_data('gabab_analysis', 'GABAb_Analysis_Metrics.csv')
    
    if gabab_df is None:
        print("❌ Error: GABAb data not found")
        return
    
    # Filter for Gabazine condition only (case-insensitive)
    gabab_gab = gabab_df[gabab_df['Condition'].str.lower() == 'gabazine'].copy()
    gabab_gab = gabab_gab.drop_duplicates(subset=['Cell_ID', 'Channel_Name'])
    
    all_stats_export = []
    
    def record_stat_fig5(fig_panel, comparison_name, res):
        print_stat_result(fig_panel, comparison_name, res)
        all_stats_export.append({
            'Comparison': comparison_name,
            'Mean_WT': res['Mean_WT'],
            'Mean_GNB1': res['Mean_GNB1'],
            'SEM_WT': res['SEM_WT'],
            'SEM_GNB1': res['SEM_GNB1'],
            'Figure_Panel': fig_panel,
            'Test_Used': res['Test'],
            'Statistic': res['Statistic'],
            'P_Value': res['p'],
            'Significance': res['Significance'],
            'Normality_Info': res['Normality']
        })
    
    # Define metrics and panels
    pathways = [
        ('Perforant Path', 'Perforant'),
        ('Schaffer Collateral', 'Schaffer'),
        ('Stratum Oriens', 'Basal')
    ]
    
    # Only test the metric that's actually plotted (Slow IPSP Area)
    metric_col = 'Integral_mV_ms'
    metric_name = 'Slow IPSP Area'
    
    # Panel mapping for Figure 5 Row 2 (Metrics):
    # D = Perforant, E = Schaffer, F = Basal
    panel_letters = {
        'Perforant': 'D',
        'Schaffer': 'E',
        'Basal': 'F'
    }
    
    print()
    for pathway_key, pathway_short in pathways:
        pathway_data = gabab_gab[gabab_gab['Channel_Name'] == pathway_key]
        
        print(f"--- {pathway_short} Pathway ---")
        
        panel = f"Fig 5{panel_letters[pathway_short]}"
        comparison = f"{pathway_short} - {metric_name}: WT vs GNB1"
        
        wt = pathway_data[pathway_data['Genotype'] == 'WT'][metric_col].dropna()
        gnb1 = pathway_data[pathway_data['Genotype'] == 'GNB1'][metric_col].dropna()

        wt_mean = wt.mean()
        wt_sem = wt.sem()
        gnb1_mean = gnb1.mean()
        gnb1_sem = gnb1.sem()
                
        if len(wt) > 0 and len(gnb1) > 0:
            res = compare_two_groups(wt, gnb1)
            res.update({'Mean_WT': wt_mean, 'Mean_GNB1': gnb1_mean, 'SEM_WT': wt_sem, 'SEM_GNB1': gnb1_sem})
            record_stat_fig5(panel, comparison, res)
        else:
            print(f"[{panel}] {comparison}: Insufficient data")
        
        print()
    
    # Add Baclofen Vm Change analysis
    print("--- Baclofen-Induced Vm Change ---")
    vm_change_df = load_data('gabab_analysis', 'Baclofen_Vm_Change.csv')
    
    if vm_change_df is not None:
        panel = "Fig 5I"
        comparison = "Baclofen Vm Change: WT vs GNB1"
        
        wt_vm = vm_change_df[vm_change_df['Genotype'] == 'WT']['Voltage Change'].dropna()
        gnb1_vm = vm_change_df[vm_change_df['Genotype'] == 'GNB1']['Voltage Change'].dropna()

        wt_mean = wt_vm.mean()
        wt_sem = wt_vm.sem()
        gnb1_mean = gnb1_vm.mean()
        gnb1_sem = gnb1_vm.sem()
        
        if len(wt_vm) > 0 and len(gnb1_vm) > 0:
            res = compare_two_groups(wt_vm, gnb1_vm)
            res.update({'Mean_WT': wt_mean, 'Mean_GNB1': gnb1_mean, 'SEM_WT': wt_sem, 'SEM_GNB1': gnb1_sem})
            record_stat_fig5(panel, comparison, res)
        else:
            print(f"[{panel}] {comparison}: Insufficient data")
        
        print()
    else:
        print("⚠ Baclofen Vm Change data not found")
        print()
    
    # Save results
    stats_df = pd.DataFrame(all_stats_export)
    save_path = os.path.join(DATA_ROOT, 'gabab_analysis', 'Stats_Results_Figure_5.csv')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    stats_df.to_csv(save_path, index=False)
    print(f"✓ Saved Figure 5 stats to: {save_path}")


# ==================================================================================================
# SUPPLEMENTAL FIGURE 3: PROTEIN LEVELS
# ==================================================================================================

def run_stats_supplemental_figure_3():
    print("\n" + "="*80)
    print("SUPPLEMENTAL FIGURE 3: PROTEIN LEVELS")
    print("="*80)
    
    all_stats_export = []

    def record_stat(fig_panel, comparison_name, res):
        print_stat_result(fig_panel, comparison_name, res)
        all_stats_export.append({
            'Figure_Panel': fig_panel,
            'Comparison': comparison_name,
            'Test_Used': res['Test'],
            'Statistic': res['Statistic'],
            'P_Value': res['p'],
            'Significance': res['Significance'],
            'Normality_Check': res['Normality']
        })

    # Load Data
    df_protein = load_data('', 'GNB1_Protein_Levels_Hippocampus.csv')
    
    if df_protein is not None and not df_protein.empty:
        # Define Columns for Absolute
        wt_abs_cols = ['WT GNB1 Absolute Protein Signal Rep 1', 'WT GNB1 Absolute Protein Signal Rep 2', 'WT GNB1 Absolute Protein Signal Rep 3']
        i80t_abs_cols = ['I80T/+ GNB1 Absolute Protein Signal Rep 1', 'I80T/+ GNB1 Absolute Protein Signal Rep 2', 'I80T/+ GNB1 Absolute Protein Signal Rep 3']
        
        # Extract Absolute values (flatten to 1D array and remove NaNs)
        wt_abs = df_protein[wt_abs_cols].values.flatten()
        wt_abs = wt_abs[~np.isnan(wt_abs)]
        i80t_abs = df_protein[i80t_abs_cols].values.flatten()
        i80t_abs = i80t_abs[~np.isnan(i80t_abs)]
        
        # Calculate Relative (Normalized to WT Mean)
        wt_mean = np.mean(wt_abs)
        wt_rel = wt_abs / wt_mean
        i80t_rel = i80t_abs / wt_mean
        
        print(f"Absolute Levels: WT n={len(wt_abs)}, I80T n={len(i80t_abs)}")
        
        # Compare Absolute
        res = compare_two_groups(pd.Series(wt_abs), pd.Series(i80t_abs))
        record_stat("Supp Fig 3 (Top)", "Absolute Protein Levels: WT vs I80T", res)
        
        # Compare Relative
        res = compare_two_groups(pd.Series(wt_rel), pd.Series(i80t_rel))
        record_stat("Supp Fig 3 (Bottom)", "Relative Protein Levels: WT vs I80T", res)

    # --- EXPORT ---
    if all_stats_export:
        stats_df = pd.DataFrame(all_stats_export)
        save_path = os.path.join(DATA_ROOT, 'Stats_Results_Supplemental_Figure_3.csv')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        stats_df.to_csv(save_path, index=False)
        print(f"\n✓ Saved Supplemental Figure 3 stats to: {save_path}")

if __name__ == "__main__":
    run_stats_figure_1()
    run_stats_figure_2()
    run_stats_figure_3()
    run_stats_figure_5()  # GABAb
    run_stats_figure_6()  # Dendritic Excitability
    run_stats_supplemental_figure_3() # Protein Levels