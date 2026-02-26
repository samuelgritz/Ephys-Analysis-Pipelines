from analysis_utils import *
import os
import sys
import pandas as pd
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

# ==================================================================================================
# Analyze and Export Intrinsic Properties
# ==================================================================================================

def filter_master_df_general_inclusion(master_df):
    """
    Filters the Master DataFrame to only include rows where
    the 'Inclusion' column starts with 'Yes' (case-insensitive).
    """
    if 'Inclusion' not in master_df.columns:
        print("WARNING: 'Inclusion' column not found in Master DF. Returning full dataframe.")
        return master_df

    if 'Cell_ID' in master_df.columns:
        master_df['Cell_ID'] = master_df['Cell_ID'].astype(str)

    # Filter logic: starts with 'yes' (case-insensitive)
    mask = master_df['Inclusion'].astype(str).str.strip().str.lower().str.startswith('yes')
    
    filtered_df = master_df[mask].copy()
    
    dropped_count = len(master_df) - len(filtered_df)
    print(f"Data Filtering: Kept {len(filtered_df)} cells. Dropped {dropped_count} cells.")
    print(f"   (Criteria: Inclusion starts with 'Yes')")
    
    return filtered_df

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

    # --- CRITICAL STEP: FILTER FOR GENERAL INCLUSION ("YES") ---
    # We filter only for "Yes" here so we get Input Res, Vm, Access for ALL valid cells.
    # We will handle Voltage Sag exclusion specifically later.
    master_df = filter_master_df_general_inclusion(master_df_raw)

    # --- 2. Get Data Directory (Box Compatible) ---
    # This handles auto-detection AND fixes the "Operation Canceled" error
    data_path = box_utils.get_data_path(target_folder_name=PROJECT_DATA_FOLDER)
    
    if not data_path:
        print("Exiting...")
        sys.exit(1)

    # --- 3. Define Analysis Parameters ---
    properties = ['steady_state_input_resistance', 'Voltage_sag']

    # --- 4. Run Analysis (BUT DO NOT SAVE YET) ---
    print(f"\nStarting Intrinsic Properties Analysis...")
    
    # We pass output_path=None so we can modify the dataframe before saving
    intrinsic_df = analyze_and_export_intrinsic_properties(
        master_df=master_df,
        data_dir=data_path,
        output_path=None, 
        properties_to_extract=properties,
        vm_rest_threshold=-40 
    )

    # --- 5. Apply Specific Inclusion Criteria for Voltage Sag ---
    print("\nApplying strict inclusion criteria for Voltage Sag...")
    
    # We need to map Cell IDs back to their specific inclusion string to check for "Voltage-Sag"
    # Create lookup dictionary from the master_df we used
    inclusion_map = dict(zip(master_df['Cell_ID'].astype(str), master_df['Inclusion'].astype(str)))
    
    sag_dropped_count = 0
    
    if 'Voltage_sag' in intrinsic_df.columns:
        for index, row in intrinsic_df.iterrows():
            cell_id = str(row['Cell_ID'])
            inclusion_str = inclusion_map.get(cell_id, "")
            
            # Check if 'Voltage-Sag' is in the inclusion criteria (case-insensitive)
            # If NOT present, set the calculated Voltage Sag to NaN
            if 'voltage-sag' not in inclusion_str.lower():
                if pd.notna(row['Voltage_sag']):
                    intrinsic_df.at[index, 'Voltage_sag'] = np.nan
                    sag_dropped_count += 1
    
    print(f"-> Set Voltage Sag to NaN for {sag_dropped_count} cells (Missing 'Voltage-Sag' in Inclusion column).")

    # --- 6. Save Final Dataframe ---
    final_output_path = 'paper_data/Physiology_Analysis/intrinsic_properties.csv'
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(final_output_path), exist_ok=True)
    
    intrinsic_df.to_csv(final_output_path, index=False)
    print(f"\n✓ Analysis complete. Data exported to: {final_output_path}")