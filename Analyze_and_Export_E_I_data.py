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
    
    # --- 1.5 Filter Master DF for E-I Analysis ---
    # Only include cells marked 'Yes' in 'Inclusion' 
    # AND having specific patterns in 'Experiment Notes'
    inclusion_mask = master_df['Inclusion'].astype(str).str.contains('Yes', case=False, na=False)
    
    # Patterns specified by user for Figures 4, 5, 6 AND Supplemental
    # 'E/I up' will catch both '... Gabazine' and '... Control' variants
    notes_patterns = ['E/I up', 'Full E/I', '300 ms unitary Gabazine']
    notes_mask = master_df['Experiment Notes'].str.contains('|'.join(notes_patterns), case=False, na=False)
    
    master_df = master_df[inclusion_mask & notes_mask].copy()
    print(f"✓ Filtered Master DF: {len(master_df)} cells meet E-I inclusion criteria (Inclusion='Yes' and Experiment Notes contains: {', '.join(notes_patterns)})")

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

    # --- 4. Create Dictionaries for Stim Times ---
    
    # Newer stim times
    unitary_stim_starts_newer = {
        'channel_1': default_metadata_yaml_new['defaults']['analysis']['E_I_pulse']['channel_1']['ISI_stim_times'][300][0], 
        'channel_2': default_metadata_yaml_new['defaults']['analysis']['E_I_pulse']['channel_2']['ISI_stim_times'][300][0]
    } 

    ISI_times_dict_newer = {
        'channel_1': default_metadata_yaml_new['defaults']['analysis']['E_I_pulse']['channel_1']['ISI_stim_times'], 
        'channel_2': default_metadata_yaml_new['defaults']['analysis']['E_I_pulse']['channel_2']['ISI_stim_times']
    } 

    # Older stim times
    unitary_stim_starts_older = {
        'channel_1': default_metadata_yaml_older['defaults']['analysis']['E_I_pulse']['channel_1']['ISI_stim_times'][300][0], 
        'channel_2': default_metadata_yaml_older['defaults']['analysis']['E_I_pulse']['channel_2']['ISI_stim_times'][300][0]
    } 

    ISI_times_dict_older = {
        'channel_1': default_metadata_yaml_older['defaults']['analysis']['E_I_pulse']['channel_1']['ISI_stim_times'], 
        'channel_2': default_metadata_yaml_older['defaults']['analysis']['E_I_pulse']['channel_2']['ISI_stim_times']
    } 

    # Mapping dictionaries
    unitary_stim_starts_dict = {
        'older': unitary_stim_starts_older,
        'newer': unitary_stim_starts_newer
    }

    ISI_times_dict_mapping = {
        'older': ISI_times_dict_older,
        'newer': ISI_times_dict_newer
    }

    # --- 5. Run Analysis ---
    print(f"\nStarting E-I Balance Analysis...")
    
    results = analyze_and_export_E_I_balance(
        master_df=master_df,
        data_dir=data_path,
        unitary_stim_starts_dict=unitary_stim_starts_dict,
        ISI_times_dict_mapping=ISI_times_dict_mapping,
        output_path_amplitudes='paper_data/E_I_data/E_I_amplitudes.csv',
        output_path_traces='paper_data/E_I_data/E_I_traces_for_plotting.pkl',
        export_R_formats=True,
        base_output_path_R='paper_data/E_I_data/E_I',
        interactive=False
    )
    
    # --- 6. Merge Basal Pathway (Stratum Oriens) Data ---
    print(f"\n\nProcessing Basal Pathway (Stratum Oriens) data...")
    
    # Extract basal pathway traces
    print("Extracting basal pathway traces...")
    E_I_traces_basal = get_E_I_traces_basal(
        data_dir=data_path,
        unitary_stim_starts_dict=unitary_stim_starts_dict,
        ISI_times_dict_mapping=ISI_times_dict_mapping,
        master_df=master_df
    )
    
    if E_I_traces_basal:
        print(f"✓ Extracted basal pathway traces from {len(E_I_traces_basal)} cells")
        
        # Process basal pathway data
        print("Processing basal pathway amplitudes...")
        basal_amplitudes_df, basal_traces_dict = process_basal_E_I_data(
            E_I_traces_basal=E_I_traces_basal,
            master_df=master_df,
            ISI_times_dict_mapping=ISI_times_dict_mapping
        )
        
        # Merge basal data into main amplitudes dataframe
        print(f"Merging {len(basal_amplitudes_df)} basal pathway entries into main data...")
        
        # Reload main amplitudes
        main_amplitudes = pd.read_csv('paper_data/E_I_data/E_I_amplitudes.csv')
        
        # CRITICAL: Remove any existing basal data before merging fresh basal data
        # This prevents accumulation from previous runs
        main_amplitudes_no_basal = main_amplitudes[main_amplitudes['Pathway'] != 'Basal_Stratum_Oriens']
        removed_old_basal = len(main_amplitudes) - len(main_amplitudes_no_basal)
        if removed_old_basal > 0:
            print(f"  Removed {removed_old_basal} old basal rows from previous runs")
        
        # Merge fresh basal data
        combined_amplitudes = pd.concat([main_amplitudes_no_basal, basal_amplitudes_df], ignore_index=True)
        
        combined_amplitudes.to_csv('paper_data/E_I_data/E_I_amplitudes.csv', index=False)
        print(f"✓ Merged data saved: {len(combined_amplitudes)} total entries ({len(main_amplitudes_no_basal)} apical + {len(basal_amplitudes_df)} basal)")
        
        
        # Merge trace DataFrames
        # Load main traces as DataFrame
        main_traces_df = pd.read_pickle('paper_data/E_I_data/E_I_traces_for_plotting.pkl')
        
        # Convert basal_traces_dict to DataFrame rows
        basal_rows = []
        for cell_id in basal_traces_dict:
            # Get metadata  
            cell_info = master_df[master_df['Cell_ID'] == cell_id]
            if len(cell_info) == 0:
                continue
            genotype = cell_info.iloc[0]['Genotype']
            sex = cell_info.iloc[0].get('Sex', 'Unknown')
            
            for isi in basal_traces_dict[cell_id]:
                for channel in basal_traces_dict[cell_id][isi]:
                    row = {
                        'Cell_ID': cell_id,
                        'Genotype': genotype,
                        'Sex': sex,
                        'ISI': isi,
                        'Channel': channel,
                        'Pathway': 'Basal_Stratum_Oriens'
                    }
                    
                    # Add traces
                    # Add traces
                    if 'Expected_EPSP_Trace' in basal_traces_dict[cell_id][isi][channel]:
                         row['Expected_EPSP_Trace'] = basal_traces_dict[cell_id][isi][channel]['Expected_EPSP_Trace']

                    for condition in basal_traces_dict[cell_id][isi][channel]:
                        # Skip non-condition keys
                        if condition == 'Expected_EPSP_Trace': continue
                        if condition == 'estimated_inhibition':
                             # Handle estimated inhibition separately (it follows the condition dict structure)
                             trace_key = 'unitary_average_traces' if isi == 300 else 'non_unitary_average_traces'
                             if trace_key in basal_traces_dict[cell_id][isi][channel][condition]:
                                  row['estimated_inhibition_Trace'] = basal_traces_dict[cell_id][isi][channel][condition][trace_key]
                             continue
                             
                        # Handle regular conditions (Control, Gabazine, etc)
                        # Ensure value is a dict before accessing trace_key
                        if not isinstance(basal_traces_dict[cell_id][isi][channel][condition], dict):
                             continue
                             
                        trace_key = 'unitary_average_traces' if isi == 300 else 'non_unitary_average_traces'
                        if trace_key in basal_traces_dict[cell_id][isi][channel][condition]:
                            trace = basal_traces_dict[cell_id][isi][channel][condition][trace_key]
                            row[f'{condition}_Trace'] = trace
                    
                    basal_rows.append(row)
        
        basal_traces_df = pd.DataFrame(basal_rows)
        combined_traces_df = pd.concat([main_traces_df, basal_traces_df], ignore_index=True)
        combined_traces_df.to_pickle('paper_data/E_I_data/E_I_traces_for_plotting.pkl')
        print(f"✓ Merged trace data saved ({len(main_traces_df)} apical + {len(basal_traces_df)} basal = {len(combined_traces_df)} total)")

        
        # Regenerate R format files with merged data (so stats include all 3 pathways)
        print("\nRegenerating R format files with merged data...")
        from analysis_utils import export_E_I_data_with_R_format_options
        
        # Reload the updated amplitudes with basal data
        combined_df = pd.read_csv('paper_data/E_I_data/E_I_amplitudes.csv')
        
        # Create R formats (includes all 3 pathways)
        from analysis_utils import export_E_I_data_with_R_format_options
        export_E_I_data_with_R_format_options(combined_df, base_output_path='paper_data/E_I_data/E_I', interactive=False)
        
        print(f"\n✓ Basal pathway integrated into main E:I data")
    else:
        print("⚠ No basal pathway (Stratum Oriens) data found")
    
    # --- 7. Generate Summary Files ---
    print("\n\n======================================================================")
    print("GENERATING SUMMARY FILES")
    print("======================================================================")
    
    from analysis_utils import generate_EI_summary_files
    generate_EI_summary_files(
        amplitudes_csv_path='paper_data/E_I_data/E_I_amplitudes.csv',
        output_dir='paper_data/E_I_data'
    )
    
    print("\n✓ All analyses complete.")