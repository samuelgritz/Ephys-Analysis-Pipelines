from analysis_utils import *
import pandas as pd
import os
import sys
import yaml
import numpy as np

# IMPORT BOX UTILITIES
try:
    import box_utils
except ImportError:
    print("\n❌ Error: box_utils.py not found.")
    print("Please ensure box_utils.py is in the same directory as this script.")
    sys.exit(1)

# ==================================================================================================
# CONFIGURATION
# ==================================================================================================

PROJECT_DATA_FOLDER = 'All_Combined_Data'

# Helper for YAML loading (if not in utils)
def read_yaml_file_local(filepath):
    with open(filepath, 'r') as file: return yaml.safe_load(file)

# ==================================================================================================
# EXECUTION
# ==================================================================================================

if __name__ == "__main__":

    # 1. SETUP & LOAD DATA
    # --------------------------------------------------------------------------------------------------
    
    # --- Load Master Dataframe ---
    if os.path.exists('master_df.csv'):
        master_df_path = 'master_df.csv'
        print(f"✓ Found Master DF in current directory.")
    elif os.path.exists('../master_df.csv'):
        master_df_path = '../master_df.csv'
        print(f"✓ Found Master DF in parent directory.")
    else:
        print(f"\n⚠ Could not find 'master_df.csv'.")
        while True:
            user_input = input("Please drag and drop 'master_df.csv' here: ").strip()
            clean_path = user_input.replace('"', '').replace("'", "")
            if os.path.exists(clean_path):
                master_df_path = clean_path
                break
            print("❌ File not found. Try again.")

    print("--- 1. LOADING METADATA ---")
    master_df_raw = pd.read_csv(master_df_path, low_memory=False)
    
    # --- FILTER: Only cells with 'plateau' in the Inclusion column ---
    # This is the primary gate: master_df 'Inclusion' must contain the word 'plateau'
    # (e.g. 'Yes: [Coarse-FI, E/I, plateau]'). Cells without it are excluded.
    # The 'Single Pathway Plateau Inclusion' column is checked separately downstream 
    # in analyze_supralinearity_peaks() for Schaffer/Perforant pathways only.
    # 'Both Pathways' analysis only requires this 'plateau' Inclusion gate.
    plateau_mask = master_df_raw['Inclusion'].astype(str).str.contains('plateau', case=False, na=False)
    master_df = master_df_raw[plateau_mask].copy()
    print(f"  Plateau Inclusion Filter: {plateau_mask.sum()} / {len(master_df_raw)} cells retained.")
    
    # --- NO HARDCODED DATE FILTER ---
    # Date-based exclusion for individual pathways is handled in analysis_utils.py
    # via _extract_single_plateau_condition() and the 'Single Pathway Plateau Inclusion' column.


    # --- Get Data Directory (Box Compatible) ---
    DATA_DIR = box_utils.get_data_path(target_folder_name=PROJECT_DATA_FOLDER)
    
    if not DATA_DIR:
        print("Exiting...")
        sys.exit(1)
        
    OUTPUT_DIR = 'paper_data/supralinearity'
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

    # 2. LOAD CONFIGURATION FILES (YAML & STIM PROTOCOLS)
    # --------------------------------------------------------------------------------------------------
    
    # --- Load YAMLs ---
    # Attempt to find them relative to script
    yaml_new_path = 'Yaml_files/Default_Metadata_new_stims_dev_080724.yaml'
    yaml_old_path = 'Yaml_files/Default_Metadata_old_stims_dev_080724.yaml'
    
    if not os.path.exists(yaml_new_path):
        print(f"\n⚠ Could not find YAML: {yaml_new_path}")
        yaml_new_path = input("Drag and drop 'Default_Metadata_new_stims...yaml': ").strip().replace('"', '').replace("'", "")

    if not os.path.exists(yaml_old_path):
        print(f"\n⚠ Could not find YAML: {yaml_old_path}")
        yaml_old_path = input("Drag and drop 'Default_Metadata_old_stims...yaml': ").strip().replace('"', '').replace("'", "")

    yaml_config_new = read_yaml_file_local(yaml_new_path)
    yaml_config_old = read_yaml_file_local(yaml_old_path)

    # --- 3. Construct Unitary Settings ---
    unitary_stim_starts_newer = {
        'channel_1': yaml_config_new['defaults']['analysis']['E_I_pulse']['channel_1']['ISI_stim_times'][300][0], 
        'channel_2': yaml_config_new['defaults']['analysis']['E_I_pulse']['channel_2']['ISI_stim_times'][300][0]
    } 
    unitary_stim_starts_older = {
        'channel_1': yaml_config_old['defaults']['analysis']['E_I_pulse']['channel_1']['ISI_stim_times'][300][0], 
        'channel_2': yaml_config_old['defaults']['analysis']['E_I_pulse']['channel_2']['ISI_stim_times'][300][0]
    } 

    unitary_stim_starts_dict = {
        'older': unitary_stim_starts_older,
        'newer': unitary_stim_starts_newer
    }

    ISI_times_dict_newer = {
        'channel_1': yaml_config_new['defaults']['analysis']['E_I_pulse']['channel_1']['ISI_stim_times'], 
        'channel_2': yaml_config_new['defaults']['analysis']['E_I_pulse']['channel_2']['ISI_stim_times']
    } 
    ISI_times_dict_older = {
        'channel_1': yaml_config_old['defaults']['analysis']['E_I_pulse']['channel_1']['ISI_stim_times'], 
        'channel_2': yaml_config_old['defaults']['analysis']['E_I_pulse']['channel_2']['ISI_stim_times']
    } 

    ISI_times_dict_mapping = {
        'older': ISI_times_dict_older,
        'newer': ISI_times_dict_newer
    }

    # --- 4. Construct Theta Protocols ---
    theta_stim_protocols = {}
    
    # A. Legacy Files
    # Try to find folder relative to script
    old_stims_path = 'Stim_Protocal_Files/MCII(Before_05_16_2024)_dat_files'
    
    if not os.path.exists(old_stims_path):
        # Check if user has it one level up
        if os.path.exists(f'../{old_stims_path}'):
            old_stims_path = f'../{old_stims_path}'
        else:
            print(f"\n⚠ Warning: Could not find Legacy Stim folder: {old_stims_path}")
            print("Legacy cells might not be analyzed correctly.")
            # Optional: Ask user for path if strictly required
            # old_stims_path = input("Drag and drop 'MCII...dat_files' folder: ").strip().replace('"', '').replace("'", "")

    # B. Legacy Protocols 
    # (Previously tried to parse .dat files, but they contain waveform instructions, not just timestamps)
    # We now hardcode the standard timing for the legacy protocol.
    
    # Timing matches the structure of Theta_Burst_MCII_old.dat (500ms delay, 10ms ISI, 110ms inter-burst)
    theta_old_std = [
        500.0, 510.0, 520.0, 530.0, 540.0, 
        649.7, 659.7, 669.7, 679.7, 689.7,
        799.4, 809.4, 819.4, 829.4, 839.4,
        949.1, 959.1, 969.1, 979.1, 989.1,
        1098.8, 1108.8, 1118.8, 1128.8, 1138.8
    ]
    
    theta_stim_protocols['Theta_Burst_MCII_old'] = theta_old_std
    
    # Legacy: Also support looking up by file key if needed, but manual definition is safer.
    # (Removed dynamic .dat parsing)

    # B. New Protocols
    # B. New Protocols (Hardcoded per User Correction) (Step 7676)
    # Theta_Burst_MCIII_new_edited
    theta_new_std = [
        500.0, 510.0, 520.0, 530.0, 540.0, 
        649.7, 659.7, 669.7, 679.7, 689.7,
        799.4, 809.4, 819.4, 829.4, 839.4,
        949.1, 959.1, 969.1, 979.1, 989.1,
        1098.8, 1108.8, 1118.8, 1128.8, 1138.8
    ]

    # Theta_Burst_MCIII_new_variant_2
    theta_new_var2 = [
        500.0, 510.0, 520.0, 530.0, 540.0, 
        650.0, 660.0, 670.0, 680.0, 690.0, 
        800.0, 810.0, 820.0, 830.0, 840.0, 
        950.0, 960.0, 970.0, 980.0, 990.0, 
        1100.0, 1110.0, 1120.0, 1130.0, 1140.0
    ]

    theta_stim_protocols['Theta_Burst_MCIII_new_edited'] = theta_new_std
    theta_stim_protocols['Theta_Burst_MCIII_new_variant_2'] = theta_new_var2


    # 5. RUN ANALYSIS
    # --------------------------------------------------------------------------------------------------

    print("\n--- 2. LOADING TRACES ---")
    plateau_traces = load_plateau_traces_from_dir(DATA_DIR, master_df=master_df)
    print(f"Loaded Plateau Traces for {len(plateau_traces)} cells.")

    print("Loading Unitary (E-I) Traces...")
    E_I_traces = get_E_I_traces(DATA_DIR, unitary_stim_starts_dict, ISI_times_dict_mapping, master_df)
    print(f"Loaded E-I Traces for {len(E_I_traces)} cells.")

    print("\n--- 3. ANALYZING SUPRALINEARITY ---")
    results_list, trace_export = analyze_supralinearity_peaks(
        plateau_traces, 
        E_I_traces, 
        theta_stim_protocols, 
        master_df
    )

    print(f"Generated results for {len(results_list)} cycles.")

    print("\n--- 4. EXPORTING RESULTS ---")
    csv_path_peaks = os.path.join(OUTPUT_DIR, 'Supralinear_Peaks_Wide.csv')
    wide_df_peaks = export_supralinearity_wide_format(results_list, csv_path_peaks, value_column='Difference_Peak')

    if not wide_df_peaks.empty:
        print("\nPeak Data Preview:")
        print(wide_df_peaks.head())
    
    # Export AUC data
    csv_path_auc = os.path.join(OUTPUT_DIR, 'Supralinear_AUC_Wide.csv')
    wide_df_auc = export_supralinearity_wide_format(results_list, csv_path_auc, value_column='Difference_AUC')
    
    if not wide_df_auc.empty:
        print("\nAUC Data Preview:")
        print(wide_df_auc.head())
    
    # Export Total Summed AUC (sum across all 5 cycles for comparison with Plateau Area)
    csv_path_auc_total = os.path.join(OUTPUT_DIR, 'Supralinear_AUC_Total.csv')
    if not wide_df_auc.empty:
        # Sum across Cycle_1 through Cycle_5
        cycle_cols = [f'Cycle_{i}' for i in range(1, 6)]
        wide_df_auc['Total_AUC'] = wide_df_auc[cycle_cols].sum(axis=1)
        
        # Keep only Cell_ID, Genotype, Sex, Pathway, Total_AUC
        auc_total_df = wide_df_auc[['Cell_ID', 'Genotype', 'Sex', 'Pathway', 'Total_AUC']].copy()
        auc_total_df.to_csv(csv_path_auc_total, index=False)
        
        print("\nTotal AUC Data Preview:")
        print(auc_total_df.head())

    pkl_path = os.path.join(OUTPUT_DIR, 'Supralinear_Traces_Plotting.pkl')
    pd.to_pickle(trace_export, pkl_path)
    print(f"✓ Traces saved to: {pkl_path}")