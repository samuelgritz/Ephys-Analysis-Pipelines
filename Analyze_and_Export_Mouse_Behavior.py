import sys
import os

# Add specific path to system to ensure utils are found
sys.path.append('/Users/samgritz/Desktop/Rutgers/Milstein_Lab/Code/Rutgers-Neuroscience-PhD/GNB1_Paper_Analysis/')

from analysis_utils import *
import pandas as pd
import numpy as np

# IMPORT BOX UTILITIES
try:
    import box_utils
except ImportError:
    print("Error: box_utils.py not found. Please ensure it is in the same directory.")
    sys.exit(1)

# ==================================================================================================
# CONFIGURATION
# ==================================================================================================

# The main project root folder in Box to search for
PROJECT_ROOT_FOLDER = "GNB1_Paper_Data"
BEHAVIOR_SUBFOLDER = "Mouse_Behavior_Data"

OUTPUT_DIR = 'paper_data/Behavior_Analysis'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# --- FILE PATHS (Relative to PROJECT_DATA_FOLDER / BEHAVIOR_SUBFOLDER) ---

# 1. Weights
WEIGHTS_FILE = "Mouse Weights.csv"

# 2. Metadata (Sex/Litter)
METADATA_FILE = "Master_DF_littermate_Sex.csv"

# 3. Open Field Locomotion Files
OF_LOCOMOTION_FILES = [
    'Open_Field_data/11152024_Test_1_OF/Open_Field_Data_11152024.csv',
    'Open_Field_data/11272024_Test_2_OF/Open_Field_Day_1_Data.csv',
    'Open_Field_data/10312024_NOR_OF_data/NOR_data_w_Distances.csv',
    'Open_Field_data/12052024_Test_3_OF/Open_Field_Day_1_Data_12052024.csv',
    'Open_Field_data/01082025_Test_4_OF/All_Data_Day_1_01082025_OLM.csv',
    'Open_Field_data/02132025_Test_5_OF/All_Data_Day_1_02132025_OLM_new.csv',
    'Open_Field_data/3062025_Tst_6_OF/All_Data_Day_1_03062025_OLM.csv'
]

# 4. Open Field Anxiety File
OF_ANXIETY_FILE = 'Open_Field_data/Open_Field_Anxiety_Data.csv'

# 5. Object Location Memory (OLM) Files
OLM_FILES = [
    'Object_Location_Memory_Experiments/OLM_data/11272204_Test_2/OLM_Data_Test_2_11272024.csv',
    'Object_Location_Memory_Experiments/OLM_data/12062024_Test_3/OLM_Data_Test_3_12062024.csv',
    'Object_Location_Memory_Experiments/OLM_data/01082025_Test_4/All_Data_Day_2_01082025_OLM.csv',
    'Object_Location_Memory_Experiments/OLM_data/02132025_Test_5/All_Data_Day_2_02132025_OLM.csv',
    'Object_Location_Memory_Experiments/OLM_data/03072025_Test_6/All_Data_Day_2_03062025_OLM.csv'
]

# 6. T-Maze Files
TMAZE_POSITIONS_FILE = 'T-Maze/Position_Strings_T_Maze.csv'
TMAZE_RAW_FILES = [
    'T-Maze/T_Maze_05202025_Test_1_data.csv',
    'T-Maze/T_Maze_05212025_Test_2_data.csv',
    'T-Maze/T_Maze_06102025_Test_3_data.csv'
]

# ==================================================================================================
# EXECUTION
# ==================================================================================================

if __name__ == "__main__":

    # 1. GET DATA DIRECTORY
    # ---------------------
    print(f"Searching for project root: {PROJECT_ROOT_FOLDER}...")
    project_root = box_utils.get_data_path(target_folder_name=PROJECT_ROOT_FOLDER)
    
    if not project_root: 
        print("❌ Could not find project root folder.")
        sys.exit(1)

    data_path = os.path.join(project_root, BEHAVIOR_SUBFOLDER)
    
    if not os.path.exists(data_path):
        print(f"❌ Error: Could not find '{BEHAVIOR_SUBFOLDER}' inside '{PROJECT_ROOT_FOLDER}'")
        sys.exit(1)
        
    print(f"✓ Data Directory: {data_path}")

    # 2. LOAD METADATA
    # ----------------
    print("\n--- Loading Metadata ---")
    
    # Define search paths based on where the script actually lives
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    
    potential_paths = [
        os.path.join(script_dir, METADATA_FILE),       # Same folder as script
        os.path.join(parent_dir, METADATA_FILE),       # Parent folder
        os.path.join(project_root, METADATA_FILE),     # Box Root
        os.path.join(data_path, METADATA_FILE),        # Box Behavior folder
        "Master_DF_littermate_Sex.csv"                 # Current working dir
    ]
    
    meta_path = None
    for p in potential_paths:
        if os.path.exists(p):
            meta_path = p
            print(f"✓ Found Metadata at: {p}")
            break
            
    if not meta_path:
        print(f"⚠ Could not find {METADATA_FILE}.")
        print("   Checked locations:")
        for p in potential_paths:
            print(f"   - {p}")
            
        # Fallback
        user_input = input(f"Please drag and drop '{METADATA_FILE}' here: ").strip()
        if user_input:
            meta_path = user_input.replace("'", "").replace('"', "")
    
    if meta_path and os.path.exists(meta_path):
        master_sex_df = pd.read_csv(meta_path)
        master_sex_df['Animal ID'] = master_sex_df['Animal ID'].astype(str)
        print(f"✓ Loaded Metadata successfully.")
    else:
        print("❌ Metadata missing. Cannot proceed with Sex merging.")
        sys.exit(1)

    # ==============================================================================================
    # ANALYSIS 1: MOUSE WEIGHTS
    # ==============================================================================================
    print("\n--- Processing Mouse Weights ---")
    weights_path = os.path.join(data_path, WEIGHTS_FILE)
    
    if os.path.exists(weights_path):
        weights_df = pd.read_csv(weights_path)
        if 'Mouse ID' in weights_df.columns: weights_df.rename(columns={'Mouse ID': 'Animal ID'}, inplace=True)
        if 'After DVC Weigh' in weights_df.columns: weights_df.rename(columns={'After DVC Weigh': 'After DVC Weight'}, inplace=True)

        id_vars = [c for c in ['Animal ID', 'Genotype', 'Sex', 'Litter'] if c in weights_df.columns]
        val_vars = [c for c in ['Genotyping Weight (P8-P10)', 'P28 (weaning weight)', 'After DVC Weight'] if c in weights_df.columns]

        if id_vars and val_vars:
            weights_long = weights_df.melt(id_vars=id_vars, value_vars=val_vars, var_name='Timepoint', value_name='Weight_g')
            tp_map = {'Genotyping Weight (P8-P10)': 'P8-P10', 'P28 (weaning weight)': 'P28', 'After DVC Weight': 'Adult'}
            weights_long['Timepoint_Label'] = weights_long['Timepoint'].map(tp_map)
            weights_long = weights_long.dropna(subset=['Weight_g'])
            
            save_path = os.path.join(OUTPUT_DIR, 'Mouse_Weights_Processed.csv')
            weights_long.to_csv(save_path, index=False)
            print(f"✓ Exported weights: {save_path}")
    else:
        print(f"⚠ Weights file not found: {WEIGHTS_FILE}")

    # ==============================================================================================
    # ANALYSIS 2: OPEN FIELD LOCOMOTION
    # ==============================================================================================
    print("\n--- Processing Open Field Locomotion ---")
    OF_master_df = load_and_concat_behavior_files(data_path, OF_LOCOMOTION_FILES)
    
    if not OF_master_df.empty:
        OF_master_df['Animal'] = OF_master_df['Animal'].astype(str)
        OF_master_df = OF_master_df.merge(master_sex_df[['Animal ID', 'Sex']], left_on='Animal', right_on='Animal ID', how='left')
        
        OF_trial1 = OF_master_df[(OF_master_df['Stage'] == 'Habituation Day 1') & (OF_master_df['Trial'] == 1)].copy()
        
        cols_to_keep = [c for c in ['Animal', 'Genotype', 'Sex', 'Distance (m)', 'Velocity (cm/s)', 'Trial', 'Stage'] if c in OF_trial1.columns]
        OF_trial1[cols_to_keep].to_csv(os.path.join(OUTPUT_DIR, 'Open_Field_Locomotion_Trial1.csv'), index=False)
        print(f"✓ Exported OF Locomotion")
    else:
        print("⚠ No Open Field Locomotion files found.")

    # ==============================================================================================
    # ANALYSIS 3: OPEN FIELD ANXIETY
    # ==============================================================================================
    print("\n--- Processing Open Field Anxiety ---")
    anxiety_path = os.path.join(data_path, OF_ANXIETY_FILE)
    
    if os.path.exists(anxiety_path):
        OF_anxiety_df = pd.read_csv(anxiety_path)
        if 'Treatment' in OF_anxiety_df.columns: OF_anxiety_df.rename(columns={'Treatment': 'Genotype'}, inplace=True)

        OF_anxiety_df['Animal'] = OF_anxiety_df['Animal'].astype(str)
        OF_anxiety_df = OF_anxiety_df.merge(master_sex_df[['Animal ID', 'Sex']], left_on='Animal', right_on='Animal ID', how='left')
        OF_anxiety_df = process_anxiety_ratios(OF_anxiety_df)
        
        OF_anxiety_filtered = OF_anxiety_df[(OF_anxiety_df['Stage'] == 'Open_Field_Test') & (OF_anxiety_df['Trial'] == 1)].copy()
        
        cols_anxiety = [c for c in ['Animal', 'Genotype', 'Sex', 'Center Zone : time (s)', 'Outer Zone : time (s)', 'Center_Outer_Time_Ratio', 'Center Zone : distance (m)', 'Outer Zone : distance (m)'] if c in OF_anxiety_filtered.columns]
        OF_anxiety_filtered[cols_anxiety].to_csv(os.path.join(OUTPUT_DIR, 'Open_Field_Anxiety_Processed.csv'), index=False)
        print(f"✓ Exported OF Anxiety")
    else:
        print(f"⚠ Anxiety file not found: {OF_ANXIETY_FILE}")

    # ==============================================================================================
    # ANALYSIS 4: OBJECT LOCATION MEMORY (OLM)
    # ==============================================================================================
    print("\n--- Processing Object Location Memory (OLM) ---")
    OLM_master_df = load_and_concat_behavior_files(data_path, OLM_FILES)

    if not OLM_master_df.empty:
        OLM_master_df['Animal'] = OLM_master_df['Animal'].astype(str)
        OLM_master_df = OLM_master_df.merge(master_sex_df[['Animal ID', 'Sex']], left_on='Animal', right_on='Animal ID', how='left')
        
        # 1. Calculate Metrics
        OLM_master_df = process_olm_metrics(OLM_master_df)
        
        # 2. Filter 
        OLM_filtered, excluded_animals = filter_olm_by_exploration(OLM_master_df, threshold=20)
        print(f"   Excluded {len(excluded_animals)} animals (<20s): {list(excluded_animals)}")
        
        # 3. Create Summary DF
        training_df = OLM_filtered[OLM_filtered['Stage'] == 'Familiarisation Day 2'][['Animal', 'Training_DI']].set_index('Animal')
        
        testing_cols = ['Animal', 'Testing_DI', 'Genotype', 'Sex']
        testing_cols = [c for c in testing_cols if c in OLM_filtered.columns]
        testing_df = OLM_filtered[OLM_filtered['Stage'] == 'Testing Stage'][testing_cols].set_index('Animal')
        
        delta_df = testing_df.join(training_df, how='inner')
        delta_df['Delta_DI'] = delta_df['Testing_DI'] - delta_df['Training_DI']
        
        # 4. Exports
        OLM_filtered.to_csv(os.path.join(OUTPUT_DIR, 'OLM_Data_Filtered.csv'), index=False)
        delta_df.to_csv(os.path.join(OUTPUT_DIR, 'OLM_Summary_Deltas.csv'))
        print(f"✓ Exported OLM Data")
        
    else:
        print("⚠ No OLM files found.")

    # ==============================================================================================
    # ANALYSIS 5: T-MAZE
    # ==============================================================================================
    print("\n--- Processing T-Maze ---")
    
    # A. Process Alternations (Position Strings)
    tmaze_pos_path = os.path.join(data_path, TMAZE_POSITIONS_FILE)
    if os.path.exists(tmaze_pos_path):
        pos_df = pd.read_csv(tmaze_pos_path)
        
        # Calculate Metrics using utils function
        results_df = calculate_t_maze_alternations(pos_df)
        
        # Merge Metadata (Animal ID matches)
        if 'Animal' in results_df.columns and 'Animal ID' not in results_df.columns:
            results_df.rename(columns={'Animal': 'Animal ID'}, inplace=True)
            
        results_df['Animal ID'] = results_df['Animal ID'].astype(str)
        results_df = results_df.merge(master_sex_df[['Animal ID', 'Sex', 'Genotype']], on='Animal ID', how='left')
        
        # Export
        results_df.to_csv(os.path.join(OUTPUT_DIR, 'T_Maze_Alternations.csv'), index=False)
        print(f"✓ Exported T-Maze Alternations")
    else:
        print(f"⚠ T-Maze Position file not found: {TMAZE_POSITIONS_FILE}")

    # B. Process Zone Entries and Distance (Raw Data Files)
    tmaze_raw_df = load_and_concat_behavior_files(data_path, TMAZE_RAW_FILES)
    if not tmaze_raw_df.empty:
        if 'Animal' in tmaze_raw_df.columns:
            tmaze_raw_df['Animal'] = tmaze_raw_df['Animal'].astype(str)
            tmaze_raw_df = tmaze_raw_df.merge(master_sex_df[['Animal ID', 'Sex']], left_on='Animal', right_on='Animal ID', how='left')
        
        # Include Distance column if available
        zone_cols = ['Animal', 'Genotype', 'Sex', 'Distance (m)', 'Start : entries', 'Left Arm : entries', 'Right Arm : entries']
        zone_cols = [c for c in zone_cols if c in tmaze_raw_df.columns]
        
        tmaze_raw_df[zone_cols].to_csv(os.path.join(OUTPUT_DIR, 'T_Maze_Zone_Entries.csv'), index=False)
        print(f"✓ Exported T-Maze Zone Entries and Distance")
    else:
        print("⚠ No Raw T-Maze files found.")

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)