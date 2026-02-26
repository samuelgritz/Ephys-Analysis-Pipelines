from analysis_utils import *
import pandas as pd
import os
import sys

# IMPORT BOX UTILITIES
try:
    import box_utils
except ImportError:
    print("Error: box_utils.py not found. Please ensure it is in the same directory.")
    sys.exit(1)

# ==================================================================================================
# CONFIGURATION
# ==================================================================================================

PROJECT_DATA_FOLDER = "All_Aggregated_data_single_housed" 
OUTPUT_DIR = 'paper_data/DVC_Analysis'

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# List of files to load (filenames only, no full paths)
dvc_files_list = [
    'all_inputs_up_to_09232024_tracking_distance_cleaned.csv',
    '10162024_inputs_tracking_distance_cleaned.csv',
    '10292024_tracking_distance_cleaned_new.csv',
    '12162024_tracking_distance_cleaned_updated.csv',
    '1172025_tracking_distance_cleaned_updated.csv',
    '01312025_tracking_distance_cleaned.csv',
    '02102025_tracking_distance_cleaned_1.csv'
]

sex_id_filename = 'All_single_housed_DVC_sex_per_cage.csv'

# ==================================================================================================
# EXECUTION
# ==================================================================================================

if __name__ == "__main__":

    # 1. GET DATA DIRECTORY (Box Compatible)
    data_path = box_utils.get_data_path(target_folder_name=PROJECT_DATA_FOLDER)
    
    if not data_path:
        print("Exiting...")
        sys.exit(1)

    print("\n" + "="*70)
    print("LOADING DVC DATA")
    print("="*70)

    # 2. LOAD & CONCATENATE DVC FILES
    dfs_to_concat = []
    
    for fname in dvc_files_list:
        full_path = os.path.join(data_path, fname)
        if os.path.exists(full_path):
            print(f"Loading: {fname}...")
            df = pd.read_csv(full_path)
            dfs_to_concat.append(df)
        else:
            print(f"⚠ Warning: File not found: {fname}")

    if not dfs_to_concat:
        print("❌ No data files loaded. Exiting.")
        sys.exit(1)

    DVC_data = pd.concat(dfs_to_concat, ignore_index=True)
    
    # 3. LOAD METADATA
    sex_id_path = os.path.join(data_path, sex_id_filename)
    if os.path.exists(sex_id_path):
        sex_identification_df = pd.read_csv(sex_id_path)
        print(f"Loaded Metadata: {sex_id_filename}")
    else:
        print(f"❌ Error: Metadata file not found at {sex_id_path}")
        sys.exit(1)

    # 4. PROCESS DATA
    print("\nProcessing Data...")
    
    # Group by Hours
    DVC_data_grouped = DVC_data.groupby(['hour']).mean()

    # Drop day column if it exists
    if 'day' in DVC_data_grouped.columns:
        DVC_data_grouped = DVC_data_grouped.drop(columns=['day'])

    # Convert to dataframe using utils function
    # NOTE: Assuming 'convert_DVC_data_to_df_with_cage' is in analysis_utils
    DVC_data_df = convert_DVC_data_to_df_with_cage(DVC_data_grouped) 

    # Ensure columns are strings for matching
    DVC_data_df['Cage'] = DVC_data_df['Cage'].astype(str)
    sex_identification_df['Cage_ID'] = sex_identification_df['Cage_ID'].astype(str)

    # FIX: Drop 'Genotype' from DVC_data_df if it exists to avoid collision (_x, _y columns)
    # when merging with the metadata file which also contains 'Genotype'.
    if 'Genotype' in DVC_data_df.columns:
        DVC_data_df = DVC_data_df.drop(columns=['Genotype'])

    # Merge DVC_data_df with sex_identification_df based on Cage and Cage_ID
    # IMPORTANT: Added 'Genotype' to the merge list so we can split by it later
    DVC_data_df = DVC_data_df.merge(
        sex_identification_df[['Cage_ID', 'Sex', 'Genotype']], 
        left_on='Cage', 
        right_on='Cage_ID', 
        how='left'
    )

    print(f"Organized dataframe structure and concatenated data.")
    
    # Save the Master Aggregated File (Hourly Data)
    master_csv_path = os.path.join(OUTPUT_DIR, 'Aggregated_DVC_Data_Master.csv')
    DVC_data_df.to_csv(master_csv_path, index=False)
    print(f"✓ Master data (Hourly) saved to: {master_csv_path}")

    # ==============================================================================================
    # 5. GENERATE SPECIFIC HOUR SUMMARIES (Per Cage)
    # ==============================================================================================
    print("\n" + "="*70)
    print("CALCULATING SUMMED ACTIVITY FOR SPECIFIC HOURS")
    print("="*70)

    # Define Time Periods based on your request
    # Dark Morning: 0-6 (exclusive of 6)
    # Light: 6-17 (exclusive of 17)
    # Dark Evening: 17-23 (inclusive of 23)
    
    periods = {
        'Dark_Morning': (DVC_data_df['Hour'] >= 0) & (DVC_data_df['Hour'] < 6),
        'Light': (DVC_data_df['Hour'] >= 6) & (DVC_data_df['Hour'] < 17),
        'Dark_Evening': (DVC_data_df['Hour'] >= 17) & (DVC_data_df['Hour'] <= 23),
        # All Dark is the union of Morning and Evening masks
        'All_Dark': ((DVC_data_df['Hour'] >= 0) & (DVC_data_df['Hour'] < 6)) | ((DVC_data_df['Hour'] >= 17) & (DVC_data_df['Hour'] <= 23))
    }

    # Initialize a summary dataframe with unique cages/metadata
    # We drop duplicates to get one row per cage
    cage_summary_df = DVC_data_df[['Cage', 'Sex', 'Genotype']].drop_duplicates().reset_index(drop=True)

    # Loop through periods, calculate sums, and merge into summary df
    for period_name, mask in periods.items():
        print(f"Calculating sums for: {period_name}...")
        
        # Filter data for this period
        period_data = DVC_data_df[mask]
        
        # Sum by Cage
        period_sum = period_data.groupby('Cage')['Activity_Value'].sum().reset_index()
        period_sum.rename(columns={'Activity_Value': f'Sum_{period_name}'}, inplace=True)
        
        # Merge into main summary dataframe
        cage_summary_df = cage_summary_df.merge(period_sum, on='Cage', how='left')

    # Fill NaN with 0 (in case a cage had NO activity recorded during a specific window)
    cage_summary_df = cage_summary_df.fillna(0)

    # Export the Cage Summary
    summary_csv_path = os.path.join(OUTPUT_DIR, 'Cage_Specific_Hours_Summary.csv')
    cage_summary_df.to_csv(summary_csv_path, index=False)
    print(f"✓ Summed Activity Summary saved to: {summary_csv_path}")
    print(cage_summary_df.head())

    # ==============================================================================================
    # 6. GENERATE HOURLY STATS (Aggregated Mean/SEM for Plotting)
    # ==============================================================================================
    print("\n" + "="*70)
    print("GENERATING HOURLY GROUP STATISTICS")
    print("="*70)

    # 1. By Genotype Only
    print("Calculating Hourly Stats by Genotype...")
    hourly_genotype = analyze_hourly_DVC_activity(DVC_data_df, group_by_cols=['Genotype'])
    hourly_genotype.to_csv(os.path.join(OUTPUT_DIR, 'Hourly_Stats_By_Genotype.csv'), index=False)

    # 2. By Genotype + Sex
    print("Calculating Hourly Stats by Genotype + Sex...")
    hourly_sex = analyze_hourly_DVC_activity(DVC_data_df, group_by_cols=['Genotype', 'Sex'])
    hourly_sex.to_csv(os.path.join(OUTPUT_DIR, 'Hourly_Stats_By_Sex.csv'), index=False)

    print(f"\n✓ All analysis files exported to: {OUTPUT_DIR}")
    print("  1. Aggregated_DVC_Data_Master.csv (All hours, raw)")
    print("  2. Cage_Specific_Hours_Summary.csv (Summed activity for Dark/Light periods)")
    print("  3. Hourly_Stats_By_Genotype.csv (Mean/SEM per hour)")
    print("  4. Hourly_Stats_By_Sex.csv (Mean/SEM per hour)")