from analysis_utils import *
import os
import sys
import pandas as pd

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

def filter_master_df_FI_data(master_df):
    """
    Filters the Master DataFrame to only include rows where
    the 'Inclusion' column CONTAINS the text 'Coarse-FI' (case-insensitive).
    """
    if 'Inclusion' not in master_df.columns:
        print("WARNING: 'Inclusion' column not found in Master DF. Returning full dataframe.")
        return master_df

    if 'Cell_ID' in master_df.columns:
        master_df['Cell_ID'] = master_df['Cell_ID'].astype(str)

    # Filter logic: Check if 'Inclusion' contains 'Coarse-FI' (case-insensitive)
    # This handles "Yes: [Coarse-FI]", "Yes: [Coarse-FI, E/I]", etc.
    mask = master_df['Inclusion'].astype(str).str.contains('Coarse-FI', case=False, na=False)
    
    filtered_df = master_df[mask].copy()
    
    dropped_count = len(master_df) - len(filtered_df)
    print(f"Data Filtering: Kept {len(filtered_df)} cells. Dropped {dropped_count} cells.")
    print(f"   (Criteria: Inclusion column contains 'Coarse-FI')")
    
    return filtered_df

# ==================================================================================================
# Analyze Firing Rates, ISI, and Export in Multiple Formats
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

    master_df_raw = pd.read_csv(master_df_path, low_memory=False)

    # --- CRITICAL STEP: FILTER FOR INCLUSION ---
    master_df = filter_master_df_FI_data(master_df_raw)

    # --- 2. Get Data Directory (Box Compatible) ---
    # This handles auto-detection AND fixes the "Operation Canceled" error
    data_path = box_utils.get_data_path(target_folder_name=PROJECT_DATA_FOLDER)
    
    if not data_path:
        print("Exiting...")
        sys.exit(1)

    # ==================================================================================================
    # Run Complete Analysis
    # ==================================================================================================

    print(f"\nStarting FI and ISI Analysis...")

    # Note: analyze_and_export_FI_and_ISI_data uses the filtered master_df passed here
    all_results = analyze_and_export_FI_and_ISI_data(
        master_df=master_df,
        data_dir=data_path,
        output_path='paper_data/Firing_Rate/Firing_Rates'
    )

    # ==================================================================================================
    # Summary Statistics
    # ==================================================================================================

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)