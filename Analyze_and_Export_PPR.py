from analysis_utils import *
import os
import sys
import pandas as pd
import pickle

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

# The specific folder name inside Box containing the experiment data
PROJECT_DATA_FOLDER = 'All_Combined_Data'

# ==================================================================================================
# Analyze E-I Balance and Export Results and Traces for Plotting
# ==================================================================================================

if __name__ == "__main__":

    # --- 1. Load Master Dataframe ---
    # Priority 1: Check current directory (Repository Root)
    if os.path.exists('master_df.csv'):
        master_df_path = 'master_df.csv'
        print(f"✓ Found Master DF in current directory.")
        
    # Priority 2: Check parent directory (In case script is in a subfolder)
    elif os.path.exists('../master_df.csv'):
        master_df_path = '../master_df.csv'
        print(f"✓ Found Master DF in parent directory.")
        
    # Priority 3: Fallback to manual input
    else:
        print(f"\n⚠ Could not find 'master_df.csv' in the local repository.")
        while True:
            user_input = input("Please drag and drop 'master_df.csv' here: ").strip()
            clean_path = user_input.replace('"', '').replace("'", "")
            if os.path.exists(clean_path):
                master_df_path = clean_path
                break
            print("❌ File not found. Try again.")

    master_df = pd.read_csv(master_df_path, low_memory=False)

    # --- 2. Get Data Directory (Box Compatible) ---
    # Hardcoded path to avoid interactive prompts
    data_path = "/Users/samgritz/Library/CloudStorage/Box-Box/Milstein-Shared/Sam/GNB1_Paper_Data/Electrophysiology_Experiments/All_Combined_Data"
    
    if not os.path.exists(data_path):
        print(f"❌ Error: Data path not found: {data_path}")
        # data_path = box_utils.get_data_path(target_folder_name=PROJECT_DATA_FOLDER)
        sys.exit(1)
    
    print(f"✓ Using data path: {data_path}")

    # --- 3. Load YAML Configuration ---
    # Assuming these are relative to the script location
    default_metadata_yaml_new = read_yaml_file('Yaml_files/Default_Metadata_new_stims_dev_080724.yaml')
    default_metadata_yaml_older = read_yaml_file('Yaml_files/Default_Metadata_old_stims_dev_080724.yaml')

    # --- 5. Run Analysis ---
    print(f"\nStarting PPR Analysis...")

    #analyze_and_export_PPR(master_df, data_dir, output_path=None, stim_config=None)

    #generate new folder for PPR data
    if not os.path.exists('paper_data/PPR_data'):
        os.makedirs('paper_data/PPR_data')

    results = analyze_and_export_PPR(
        master_df=master_df,
        data_dir=data_path,
        output_path='paper_data/PPR_data/PPR_amplitudes.csv',
        stim_config=None) 

    
    #plot and export PPR data
    import plotting_utils
    if not results.empty:
        plot_path = 'paper_data/PPR_data/PPR_by_Genotype.png'
        plotting_utils.plot_PPR_by_genotype_and_channel(results, plot_path)
        plotting_utils.plot_PPR_examples(data_path, master_df, 'paper_data/PPR_data/')

    print("\n✓ All analyses complete.")
    