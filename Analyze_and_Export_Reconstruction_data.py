from analysis_utils import *
import pandas as pd
import numpy as np
import os
import sys
import re
from scipy.stats import sem

# IMPORT BOX UTILITIES
try:
    import box_utils
except ImportError:
    print("Error: box_utils.py not found. Please ensure it is in the same directory.")
    sys.exit(1)

# ==================================================================================================
# CONFIGURATION
# ==================================================================================================

# Folders to look for on Box
SHOLL_FOLDER_NAME = "Sholl_analysis_data"
DENDRITE_PROP_FOLDER_NAME = "Dendrite_Analysis"
APICAL_SUBTYPES_FILE = "Dendrite_counts.csv" # Assumed to be in Dendrite_Analysis or parent

OUTPUT_DIR = 'paper_data/Morphology_Analysis'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# ==================================================================================================
# EXECUTION
# ==================================================================================================

if __name__ == "__main__":

    # 1. LOAD MASTER DF
    # -----------------
    if os.path.exists('master_df.csv'):
        master_df_path = 'master_df.csv'
    elif os.path.exists('../master_df.csv'):
        master_df_path = '../master_df.csv'
    else:
        print("⚠ Could not find 'master_df.csv'. Using fallback logic.")
        master_df_path = box_utils.get_data_path(target_folder_name="master_df.csv") # Attempt to find via Box utils if not local
        if not master_df_path: sys.exit(1)

    print("Loading Master DataFrame...")
    master_df = pd.read_csv(master_df_path, low_memory=False)

    # 2. LOCATE DATA FOLDERS ON BOX
    # -----------------------------
    print("\nLocating Data Folders...")
    
    # Locate Sholl Data
    sholl_root_path = box_utils.get_data_path(target_folder_name=SHOLL_FOLDER_NAME)
    if not sholl_root_path: sys.exit(1)
    
    WT_dir_path = os.path.join(sholl_root_path, 'WT')
    GNB1_dir_path = os.path.join(sholl_root_path, 'GNB1')

    # Locate Dendrite Properties Data
    # (Try looking inside the Sholl root parent, or search specifically)
    dendrite_root_path = box_utils.get_data_path(target_folder_name=DENDRITE_PROP_FOLDER_NAME)
    # If not found separately, check if it's inside the Sholl root parent
    if not dendrite_root_path:
        dendrite_root_path = os.path.dirname(sholl_root_path) # Assuming sibling folders

    # 3. PROCESS SHOLL DATA
    # ---------------------
    print("\n--- Processing Sholl Analysis ---")
    
    # Pull Data
    print(f"Pulling WT data from {WT_dir_path}...")
    _, _, WT_sholl_df = pull_and_process_all_data_cells(WT_dir_path)
    print(f"Pulling GNB1 data from {GNB1_dir_path}...")
    _, _, GNB1_sholl_df = pull_and_process_all_data_cells(GNB1_dir_path)

    # Merge with Master DF to get Sex
    print("Merging with Metadata...")
    WT_sholl_df_merged = WT_sholl_df.merge(master_df[['Cell_ID', 'Sex']], on='Cell_ID', how='left')
    GNB1_sholl_df_merged = GNB1_sholl_df.merge(master_df[['Cell_ID', 'Sex']], on='Cell_ID', how='left')

    # --- Analysis A0: Export Raw Data (Added per request) ---
    export_raw_sholl_data(WT_sholl_df_merged, GNB1_sholl_df_merged, OUTPUT_DIR)

    # --- Analysis A: Intersections per Radius (Mean/SEM) ---
    print("Calculating Mean/SEM Intersections by Sex...")
    
    # Helper to calculate and format for export
    def process_intersections(df, genotype, sex):
        subset = df[df['Sex'] == sex]
        if subset.empty: return pd.DataFrame()
        radii, means, sems = calculate_mean_sem_sholl(subset['Radius'], subset['Inters.'])
        return pd.DataFrame({
            'Genotype': genotype, 'Sex': sex, 'Radius': radii, 
            'Intersection_Mean': means, 'Intersection_SEM': sems
        })

    inters_dfs = [
        process_intersections(WT_sholl_df_merged, 'WT', 'Male'),
        process_intersections(WT_sholl_df_merged, 'WT', 'Female'),
        process_intersections(GNB1_sholl_df_merged, 'GNB1', 'Male'),
        process_intersections(GNB1_sholl_df_merged, 'GNB1', 'Female')
    ]
    
    sholl_inters_export = pd.concat(inters_dfs, ignore_index=True)
    sholl_inters_export.to_csv(os.path.join(OUTPUT_DIR, 'Sholl_Intersections_Mean_SEM.csv'), index=False)
    print(f"✓ Saved: Sholl_Intersections_Mean_SEM.csv")

    # --- Analysis B: Cumulative Distributions (Apical vs Basal) ---
    print("Calculating Cumulative Distributions...")
    
    cdf_export_list = []

    # Define groups to process
    groups = [
        ('WT', 'Male', WT_sholl_df_merged[WT_sholl_df_merged['Sex'] == 'Male']),
        ('WT', 'Female', WT_sholl_df_merged[WT_sholl_df_merged['Sex'] == 'Female']),
        ('GNB1', 'Male', GNB1_sholl_df_merged[GNB1_sholl_df_merged['Sex'] == 'Male']),
        ('GNB1', 'Female', GNB1_sholl_df_merged[GNB1_sholl_df_merged['Sex'] == 'Female'])
    ]

    for geno, sex, df_subset in groups:
        if df_subset.empty: continue
        
        # Apical (Radius > 0)
        apical_data = df_subset[df_subset['Radius'] > 0]
        if not apical_data.empty:
            qs, bins, sems, cum_prob, _ = compute_radius_distribution(apical_data, n_bins=20)
            cdf_export_list.append(export_cdf_data(geno, sex, 'Apical', bins, qs, sems, cum_prob))
            
        # Basal (Radius < 0)
        basal_data = df_subset[df_subset['Radius'] < 0]
        if not basal_data.empty:
            qs, bins, sems, cum_prob, _ = compute_radius_distribution(basal_data, n_bins=20)
            cdf_export_list.append(export_cdf_data(geno, sex, 'Basal', bins, qs, sems, cum_prob))

    if cdf_export_list:
        cdf_final_df = pd.concat(cdf_export_list, ignore_index=True)
        cdf_final_df.to_csv(os.path.join(OUTPUT_DIR, 'Sholl_Cumulative_Distributions.csv'), index=False)
        print(f"✓ Saved: Sholl_Cumulative_Distributions.csv")

    # 4. DENDRITE PROPERTIES (From CSVs in folders)
    # ---------------------------------------------
    print("\n--- Processing Dendrite Properties ---")
    if dendrite_root_path:
        print(f"Looking for property files in: {dendrite_root_path}")
        # Note: You might need to point to specific WT/GNB1 subfolders if they exist inside Dendrite_Analysis
        # Just passing the root should work if collect_dendrite_properties walks recursively
        
        apical_WT, basal_WT = collect_dendrite_properties(os.path.join(dendrite_root_path, 'WT'))
        apical_GNB1, basal_GNB1 = collect_dendrite_properties(os.path.join(dendrite_root_path, 'GNB1'))

        # Helper to convert dicts to DF
        def props_to_df(data_dict, genotype, dend_type):
            df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in data_dict.items()]))
            df['Genotype'] = genotype
            df['Dendrite_Type'] = dend_type
            return df

        props_dfs = [
            props_to_df(apical_WT, 'WT', 'Apical'),
            props_to_df(basal_WT, 'WT', 'Basal'),
            props_to_df(apical_GNB1, 'GNB1', 'Apical'),
            props_to_df(basal_GNB1, 'GNB1', 'Basal')
        ]
        
        dendrite_props_export = pd.concat(props_dfs, ignore_index=True)
        dendrite_props_export.to_csv(os.path.join(OUTPUT_DIR, 'Dendrite_Properties_All.csv'), index=False)
        print(f"✓ Saved: Dendrite_Properties_All.csv")
    else:
        print("⚠ Dendrite Analysis folder not found.")


    # 5. DENDRITE COUNTS (Apical Subtypes)
    # ------------------------------------
    print("\n--- Processing Apical Subtypes (Counts) ---")
    # Search for Dendrite_counts.csv in the Dendrite Analysis folder
    counts_file_path = os.path.join(dendrite_root_path, APICAL_SUBTYPES_FILE)
    
    if os.path.exists(counts_file_path):
        counts_df = pd.read_csv(counts_file_path)
        # Filter for WT/GNB1
        counts_filtered = counts_df[counts_df['Genotype'].isin(['WT', 'GNB1'])].copy()
        
        # Export
        counts_filtered.to_csv(os.path.join(OUTPUT_DIR, 'Apical_Subtypes_Counts.csv'), index=False)
        print(f"✓ Saved: Apical_Subtypes_Counts.csv")
    else:
        print(f"⚠ File not found: {APICAL_SUBTYPES_FILE} in {dendrite_root_path}")

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)