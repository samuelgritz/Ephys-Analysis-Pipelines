from analysis_utils import *
import pandas as pd
import os
import sys

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
# CRITICAL HARDWARE FIX NOTE
# ==================================================================================================
# 
# For Schaffer and Perforant pathways ONLY (not Both), data is ONLY analyzed from 
# 07/09/2024 onwards due to:
#
#     'Correct Hardware fix from this date on for individual pathways'
#
# This filter is applied automatically in analysis_utils._extract_single_plateau_condition()
# Cells recorded BEFORE 07/09/2024 will have 'Both' pathway data but NOT individual pathways.
#
# ==================================================================================================

# ==================================================================================================
# UNIFIED PLATEAU ANALYSIS & EXPORT
# ==================================================================================================

if __name__ == "__main__":

    # 1. SETUP & LOAD DATA
    # --------------------------------------------------------------------------------------------------
    
    # --- Load Master Dataframe ---
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

    output_dir = 'paper_data/Plateau_data'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("="*70)
    print("LOADING DATA")
    print("="*70)
    master_df_raw = pd.read_csv(master_df_path, low_memory=False)

    # --- CRITICAL STEP: FILTER FOR INCLUSION ---
    master_df = filter_master_df_by_inclusion(master_df_raw)

    # --- Get Data Directory (Box Compatible) ---
    # This handles auto-detection AND fixes the "Operation Canceled" error
    data_path = box_utils.get_data_path(target_folder_name=PROJECT_DATA_FOLDER)
    
    if not data_path:
        print("Exiting...")
        sys.exit(1)

    # Pass the filtered master_df to the loader so it skips excluded files
    plateau_traces = load_plateau_traces_from_dir(data_path, master_df=master_df)

    # 3. RUN ANALYSIS
    # --------------------------------------------------------------------------------------------------
    print("\n" + "="*70)
    print("EXTRACTING & CATEGORIZING CONDITIONS")
    print("="*70)
    # --- Plateau Area Threshold ---
    # Traces are offset (not baseline-subtracted to RMP)
    # Sweeps with max voltage below this threshold are excluded
    default_threshold = 20  # mV
    threshold_input = input(f"\nPlateau area threshold (mV, default={default_threshold}): ").strip()
    if threshold_input:
        try:
            plateau_threshold_mv = float(threshold_input)
        except ValueError:
            print(f"  Invalid input, using default: {default_threshold} mV")
            plateau_threshold_mv = default_threshold
    else:
        plateau_threshold_mv = default_threshold
    print(f"  Using plateau threshold: {plateau_threshold_mv} mV")

    # This function does the heavy lifting: splits into groups (Gabazine, Before_ML297, etc.)
    data_list, traces_dict = categorize_and_extract_plateau_data(
        plateau_traces, master_df, plateau_threshold_mv=plateau_threshold_mv
    )

    # 4. EXPORT MASTER CSV (Stats & Codes)
    # --------------------------------------------------------------------------------------------------
    print("\n" + "="*70)
    print("EXPORTING MASTER CSV (R-FORMAT)")
    print("="*70)

    csv_path = os.path.join(output_dir, 'Plateau_data.csv')
    master_df_results = export_plateau_master_dataframe(data_list, csv_path)

    if not master_df_results.empty:
        print("\nPreview of Exported Data:")
        print(master_df_results[['Cell_ID', 'Condition', 'Drug_Code', 'Condition_Code', 'Plateau_Area']].head())

    # 5. EXPORT MASTER TRACES (PKL)
    # --------------------------------------------------------------------------------------------------
    print("\n" + "="*70)
    print("EXPORTING MASTER TRACES (PLOTTING)")
    print("="*70)

    pkl_path = os.path.join(output_dir, 'All_Plateau_Traces.pkl')
    export_plateau_traces_for_plotting(traces_dict, master_df, pkl_path)

    # 6. ANALYZE & EXPORT SPIKE RATE PER THETA CYCLE
    # --------------------------------------------------------------------------------------------------
    print("\n" + "="*70)
    print("ANALYZING SPIKE RATE PER THETA CYCLE")
    print("="*70)
    
    spike_rate_results, spike_rates_per_cycle = analyze_spike_rate_per_theta_cycle(
        data_path, master_df, plateau_traces=plateau_traces
    )
    
    spike_rate_csv_path = os.path.join(output_dir, 'Spike_Rate_Per_Cycle.csv')
    spike_rate_df = export_spike_rate_wide_format(spike_rate_results, spike_rate_csv_path)
    
    if not spike_rate_df.empty:
        print("\nPreview of Spike Rate Data:")
        print(spike_rate_df.head())

    # 6.5 ANALYZE & EXPORT PLATEAU AREA PER THETA CYCLE
    # --------------------------------------------------------------------------------------------------
    print("\n" + "="*70)
    print("ANALYZING PLATEAU AREA PER THETA CYCLE")
    print("="*70)
    
    # Use traces_dict which contains categorized condition data (Gabazine, Before, After)
    # The analysis will filter for baseline conditions automatically
    plateau_area_results = analyze_plateau_area_per_theta_cycle(
        master_df, categorized_traces=traces_dict, threshold_mv=plateau_threshold_mv
    )
    
    if plateau_area_results:
        p_area_df = pd.DataFrame(plateau_area_results)
        p_area_csv_path = os.path.join(output_dir, 'Plateau_Area_Per_Cycle.csv')
        
        # Pivot to wide format (Cycles as columns)
        p_area_wide = p_area_df.pivot_table(
            index=['Cell_ID', 'Genotype', 'Sex', 'Pathway'],
            columns='Cycle_Index',
            values='Plateau_Area'
        ).reset_index()
        
        # Rename columns: 1 -> Cycle_1_Area, etc.
        p_area_wide.columns = [
            f'Cycle_{c}_Area' if isinstance(c, (int, np.integer)) else c 
            for c in p_area_wide.columns
        ]
        
        p_area_wide.to_csv(p_area_csv_path, index=False)
        print(f"✓ Exported Plateau Area data to: {p_area_csv_path}")
        
        # Calculate Mean and SEM per group for the user
        print("\nMean Plateau Area per Cycle (Genotype x Pathway) [mV-s]:")
        summary = p_area_df.groupby(['Genotype', 'Pathway', 'Cycle_Index'])['Plateau_Area'].agg(['mean', 'sem']).reset_index()
        print(summary.head(10))
        
        # Save summary too
        summary_csv_path = os.path.join(output_dir, 'Plateau_Area_Per_Cycle_Summary.csv')
        summary.to_csv(summary_csv_path, index=False)
        print(f"✓ Exported Summary to: {summary_csv_path}")

    # 7. GENERATE FIGURE 6 EXAMPLE TRACES (Pre-computed for fast figure generation)
    # --------------------------------------------------------------------------------------------------
    example_traces_path = os.path.join(output_dir, 'Figure6_Example_Traces.pkl')
    
    # Configuration - EXACT files and row indices from user
    # WT cell 03252025_c2: Matches WT_Candidates_Comparison (Set 2)
    # Row 10: Perforant, Row 11: Schaffer, Row 12: Both
    selected_cells_fig6 = {
        'WT': {'file': '03252025_c2_processed_data.pkl', 'rows': {'Perforant': 10, 'Schaffer': 11, 'Both': 12}, 'apply_noise_removal': True},
        'GNB1': {'file': '06062025_c1_processed_data.pkl', 'rows': {'Perforant': 7, 'Schaffer': 8, 'Both': 9}, 'apply_noise_removal': True}
    }
    
    generate_figure6_example_traces(data_path, example_traces_path, selected_cells_fig6)

    # 8. CALCULATE & EXPORT GIRK DELTAS (For Figure 7)
    # --------------------------------------------------------------------------------------------------
    print("\n" + "="*70)
    print("CALCULATING GIRK DELTAS")
    print("="*70)
    
    if os.path.exists(csv_path):
        df_long = pd.read_csv(csv_path)
        
        # Prepare list for deltas
        delta_rows = []
        
        # We need to pair Before vs After for each Cell/Pathway/Drug
        # Drug pairs: ML297 (Before_ML297 vs After_ML297), ETX (Before_ETX vs After_ETX)
        
        drugs = [('ML297', 'Before_ML297', 'After_ML297'), 
                 ('ETX', 'Before_ETX', 'After_ETX')]
        
        for drug_name, cond_pre, cond_post in drugs:
            # Filter for this drug's conditions
            df_drug = df_long[df_long['Condition'].isin([cond_pre, cond_post])]
            
            # Group by Cell and Pathway
            for (cell_id, pathway), group in df_drug.groupby(['Cell_ID', 'Pathway']):
                # Check if we have both Before and After
                row_pre = group[group['Condition'] == cond_pre]
                row_post = group[group['Condition'] == cond_post]
                
                if not row_pre.empty and not row_post.empty:
                    val_pre = row_pre.iloc[0]['Plateau_Area']
                    val_post = row_post.iloc[0]['Plateau_Area']
                    
                    # Calculate Delta (After - Before)
                    delta = val_post - val_pre
                    
                    # Get metadata from pre row
                    geno = row_pre.iloc[0]['Genotype']
                    sex = row_pre.iloc[0]['Sex'] if 'Sex' in row_pre.columns else 'Unknown'
                    
                    delta_rows.append({
                        'Cell_ID': cell_id,
                        'Genotype': geno,
                        'Sex': sex,
                        'Pathway': pathway,
                        'Drug': drug_name,
                        'Delta_Area': delta,
                        'Pre_Area': val_pre,
                        'Post_Area': val_post
                    })
        
        df_delta_path = os.path.join(output_dir, 'Plateau_Delta_GIRK.csv')
        # Check if df_delta is empty
        if delta_rows:
            df_delta = pd.DataFrame(delta_rows)
            df_delta.to_csv(df_delta_path, index=False)
            print(f"✓ Exported Delta Data to: {df_delta_path}")
            print("\nMean Delta per Group:")
            print(df_delta.groupby(['Drug', 'Pathway', 'Genotype'])['Delta_Area'].mean())
        else:
            print("⚠ No paired data found for Deltas.")

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)