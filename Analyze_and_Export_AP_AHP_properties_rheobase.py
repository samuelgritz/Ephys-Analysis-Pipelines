from analysis_utils import *
import pandas as pd
import os
import sys

# IMPORT THE NEW UTILITY
try:
    import box_utils
except ImportError:
    print("Error: box_utils.py not found. Please ensure it is in the same directory.")
    sys.exit(1)

# ==================================================================================================
# CONFIGURATION
# ==================================================================================================

# The name of the folder inside Box that holds the data
PROJECT_DATA_FOLDER = "All_Combined_Data" 

# Define output directory
OUTPUT_DIR = 'paper_data/Physiology_Analysis'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# ==================================================================================================
# MAIN ANALYSIS
# ==================================================================================================

if __name__ == "__main__":

    # 1. Load Master DF
    # Adjust path if necessary
    master_df_path = '/Users/samgritz/Desktop/Rutgers/Milstein_Lab/Code/Rutgers-Neuroscience-PhD/GNB1_Paper_Analysis/master_df.csv'
    
    if not os.path.exists(master_df_path):
        print(f"Master DF not found at hardcoded path: {master_df_path}")
        master_df_path = input("Drag and drop 'master_df.csv' here: ").strip().replace('"', '').replace("'", "")

    if not os.path.exists(master_df_path):
        print("Error: Master DF file not found.")
        sys.exit(1)

    master_df_raw = pd.read_csv(master_df_path, low_memory=False)
    master_df = filter_master_df_by_inclusion(master_df_raw)

    # 2. GET DATA DIRECTORY (Box Compatible)
    data_path = box_utils.get_data_path(target_folder_name=PROJECT_DATA_FOLDER)
    
    if not data_path:
        print("Error: Could not determine data path via Box Utils.")
        sys.exit(1)

    # --- DIAGNOSTICS START ---
    print("\n--- DIAGNOSTICS ---")
    
    # Check Master DF
    print(f"Master DF Rows: {len(master_df)}")
    if len(master_df) > 0:
        print(f"First 3 Cell IDs in Master DF: {master_df['Cell_ID'].head(3).tolist()}")
    else:
        print("❌ CRITICAL WARNING: Master DF is empty after filtering! Check your 'Inclusion' column.")

    # Check File Matching
    pkl_files = [f for f in os.listdir(data_path) if f.endswith('.pkl')]
    print(f"Found {len(pkl_files)} .pkl files in data folder.")
    
    if len(pkl_files) > 0:
        sample_file = pkl_files[0]
        converted_id = convert_pkl_filename_to_cell_id(sample_file)
        print(f"Sample File: '{sample_file}' -> Converted ID: '{converted_id}'")
        
        # Check if this ID exists in master_df
        if len(master_df) > 0:
            match = master_df['Cell_ID'].astype(str).isin([converted_id]).any()
            if match:
                print(f"✅ Match confirmed: '{converted_id}' found in Master DF.")
            else:
                print(f"❌ Match FAILED: '{converted_id}' NOT found in Master DF.")
                print("   (Check for typos or formatting differences between filename and csv)")
    else:
        print(f"❌ CRITICAL WARNING: No .pkl files found in {data_path}")

    print("-------------------\n")
    # --- DIAGNOSTICS END ---

    # 3. Define Properties
    # These match the keys found in the debug step
    properties = [
        'AP_threshold', 
        'AP_halfwidth', 
        'AP_size', 
        'AHP_size', 
        'decay_area', 
        'Rheobase_Current'
    ]

    # 4. Run Analysis
    print(f"Starting analysis using data from: {data_path}")
    
    output_file = os.path.join(OUTPUT_DIR, 'combined_AP_AHP_rheobase_analysis.csv')
    
    combined_df = analyze_and_export_rheobase_properties(
        master_df=master_df,
        data_dir=data_path,
        output_path=output_file,
        AP_properties_to_plot=properties,
        AP_trace_end=200 # Increased window for AHP recovery
    )
    
    if combined_df is not None and not combined_df.empty:
        print(f"\n✓ Analysis complete. Data saved to: {output_file}")
        print(f"  Total Rows: {len(combined_df)}")
    else:
        print("\n❌ Analysis produced an empty result. See diagnostics above.")