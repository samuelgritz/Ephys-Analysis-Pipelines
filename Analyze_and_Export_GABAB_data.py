from analysis_utils import *
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols

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
OUTPUT_DIR = 'paper_data/gabab_analysis'
PLOTS_DIR = os.path.join(OUTPUT_DIR, 'plots')

if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
if not os.path.exists(PLOTS_DIR): os.makedirs(PLOTS_DIR)

# ==================================================================================================
# MAIN EXECUTION
# ==================================================================================================

if __name__ == "__main__":

    # --- 1. Load Master Dataframe ---
    # Priority 1: Check current directory (Repository Root)
    if os.path.exists('master_df.csv'):
        master_df_path = 'master_df.csv'
        print(f"✓ Found Master DF in current directory.")
    # Priority 2: Check parent directory
    elif os.path.exists('../master_df.csv'):
        master_df_path = '../master_df.csv'
        print(f"✓ Found Master DF in parent directory.")
    # Priority 3: Fallback to manual input
    else:
        print(f"\n⚠ Could not find 'master_df.csv'.")
        while True:
            user_input = input("Please drag and drop 'master_df.csv' here: ").strip()
            clean_path = user_input.replace('"', '').replace("'", "")
            if os.path.exists(clean_path):
                master_df_path = clean_path
                break
            print("❌ File not found. Try again.")

    print("--- Loading Master DataFrame ---")
    master_df_raw = pd.read_csv(master_df_path, low_memory=False)
    
    # Ensure Cell_ID is string
    if 'Cell_ID' in master_df_raw.columns:
        master_df_raw['Cell_ID'] = master_df_raw['Cell_ID'].astype(str)
        
    # Filter (if needed, though your snippet didn't explicitly filter, it's safer to do so)
    # master_df = filter_master_df_by_inclusion(master_df_raw) 
    master_df = master_df_raw # Using raw based on your snippet, but consider filtering!

    # --- 2. Get Data Directory (Box Compatible) ---
    DATA_DIR = box_utils.get_data_path(target_folder_name=PROJECT_DATA_FOLDER)
    
    if not DATA_DIR:
        print("Exiting...")
        sys.exit(1)

    # --- 3. Load YAML Configuration ---
    # We attempt to find these in a 'Yaml_files' subdirectory first
    yaml_new_path = 'Yaml_files/Default_Metadata_new_stims_dev_080724.yaml'
    yaml_old_path = 'Yaml_files/Default_Metadata_old_stims_dev_080724.yaml'
    
    # Fallback logic for YAMLs
    if not os.path.exists(yaml_new_path):
        print(f"\n⚠ Could not find YAML config: {yaml_new_path}")
        yaml_new_path = input("Drag and drop 'Default_Metadata_new_stims...yaml' here: ").strip().replace('"', '').replace("'", "")
        
    if not os.path.exists(yaml_old_path):
        print(f"\n⚠ Could not find YAML config: {yaml_old_path}")
        yaml_old_path = input("Drag and drop 'Default_Metadata_old_stims...yaml' here: ").strip().replace('"', '').replace("'", "")

    yaml_new = read_yaml_file(yaml_new_path)
    yaml_old = read_yaml_file(yaml_old_path)

    # --- 4. Map Stim Times ---
    unitary_starts = {
        'newer': {
            'channel_1': yaml_new['defaults']['analysis']['E_I_pulse']['channel_1']['ISI_stim_times'][300][0],
            'channel_2': yaml_new['defaults']['analysis']['E_I_pulse']['channel_2']['ISI_stim_times'][300][0]
        },
        'older': {
            'channel_1': yaml_old['defaults']['analysis']['E_I_pulse']['channel_1']['ISI_stim_times'][300][0],
            'channel_2': yaml_old['defaults']['analysis']['E_I_pulse']['channel_2']['ISI_stim_times'][300][0]
        }
    }

    isi_times = {
        'newer': {
            'channel_1': yaml_new['defaults']['analysis']['E_I_pulse']['channel_1']['ISI_stim_times'],
            'channel_2': yaml_new['defaults']['analysis']['E_I_pulse']['channel_2']['ISI_stim_times']
        },
        'older': {
            'channel_1': yaml_old['defaults']['analysis']['E_I_pulse']['channel_1']['ISI_stim_times'],
            'channel_2': yaml_old['defaults']['analysis']['E_I_pulse']['channel_2']['ISI_stim_times']
        }
    }

    unitary_stim_starts_dict = {'older': unitary_starts['older'], 'newer': unitary_starts['newer']}
    ISI_times_dict_mapping = {'older': isi_times['older'], 'newer': isi_times['newer']}

    # ==================================================================================================
    # DATA LOADING - Use dedicated GABAb function to get ALL 300ms Gabazine data
    # ==================================================================================================

    print("--- Loading 300ms Gabazine Traces for GABAb Analysis ---")
    print("  This includes: Full E/I cells + Partial Gabazine cells + '300 ms unitary Gabazine' cells\n")
    
    # Load Apical pathway traces (Perforant & Schaffer)
    print("Loading Apical Pathways (Perforant & Schaffer)...")
    all_traces = get_300ms_gabazine_traces_for_gabab(
        data_dir=DATA_DIR,
        unitary_stim_starts_dict=unitary_stim_starts_dict,
        ISI_times_dict_mapping=ISI_times_dict_mapping,
        master_df=master_df,
        pathway_type='apical'
    )
    
    # Load Basal pathway traces (Stratum Oriens)
    print("\nLoading Basal Pathway (Stratum Oriens)...")
    basal_traces = get_300ms_gabazine_traces_for_gabab(
        data_dir=DATA_DIR,
        unitary_stim_starts_dict=unitary_stim_starts_dict,
        ISI_times_dict_mapping=ISI_times_dict_mapping,
        master_df=master_df,
        pathway_type='basal'
    )

    # Create Genotype and Sex Dictionaries
    genotype_dict = dict(zip(master_df['Cell_ID'], master_df['Genotype']))
    sex_dict = dict(zip(master_df['Cell_ID'], master_df['Sex']))
    
    # CRITICAL FIX: Create mapping of valid channels per cell from 'Stimulation Pathways'
    # This prevents basal cells from being processed for channel_2 (which is empty)
    cell_valid_channels = {}
    for idx, row in master_df.iterrows():
        cell_id = row['Cell_ID']
        stim_pathways = str(row['Stimulation Pathways'])
        
        # Parse the stimulation pathways string (format: "{channel_1: pathway, channel_2: pathway}")
        valid_channels = []
        if 'channel_1:' in stim_pathways:
            # Extract what's after "channel_1:"
            ch1_part = stim_pathways.split('channel_1:')[1].split(',')[0].strip()
            if ch1_part and ch1_part != '}' and ch1_part != '':
                valid_channels.append('channel_1')
        
        if 'channel_2:' in stim_pathways:
            # Extract what's after "channel_2:"
            ch2_part = stim_pathways.split('channel_2:')[1].split('}')[0].strip()
            if ch2_part and ch2_part != '':
                valid_channels.append('channel_2')
        
        cell_valid_channels[cell_id] = valid_channels
    
    print(f"\n✓ Parsed valid channels for {len(cell_valid_channels)} cells from 'Stimulation Pathways'")

    # Split Apical Traces by Genotype
    traces_WT = {k: v for k, v in all_traces.items() if genotype_dict.get(k) == 'WT'}
    traces_GNB1 = {k: v for k, v in all_traces.items() if genotype_dict.get(k) == 'GNB1'}

    # Split Basal Traces by Genotype
    basal_traces_WT = {k: v for k, v in basal_traces.items() if genotype_dict.get(k) == 'WT'}
    basal_traces_GNB1 = {k: v for k, v in basal_traces.items() if genotype_dict.get(k) == 'GNB1'}

    print(f"WT Cells (Apical): {len(traces_WT)}")
    print(f"GNB1 Cells (Apical): {len(traces_GNB1)}")
    print(f"WT Cells (Basal/Stratum Oriens): {len(basal_traces_WT)}")
    print(f"GNB1 Cells (Basal/Stratum Oriens): {len(basal_traces_GNB1)}")

    # ==================================================================================================
    # GABAB ANALYSIS & PLOTTING
    # ==================================================================================================

    # Channel configuration - now includes Stratum Oriens
    channels_to_analyze = ['channel_1', 'channel_2', 'stratum_oriens']
    channel_names = {
        'channel_1': 'Perforant Path', 
        'channel_2': 'Schaffer Collateral',
        'stratum_oriens': 'Stratum Oriens'
    }

    # Data Collection for Exports
    metrics_export_list = []
    traces_mean_export_data = {} # For CSV (Mean/SEM)

    for ch in channels_to_analyze:
        print(f"\nAnalyzing {channel_names[ch]}...")
        
        # Determine which trace dict to use based on channel
        if ch == 'stratum_oriens':
            # Use basal traces for Stratum Oriens
            current_traces_wt = basal_traces_WT
            current_traces_gnb1 = basal_traces_GNB1
            current_all_traces = basal_traces
            analyze_channel = 'channel_1'  # Basal data is stored in channel_1
        else:
            # Use apical traces for Perforant/Schaffer
            current_traces_wt = traces_WT
            current_traces_gnb1 = traces_GNB1
            current_all_traces = all_traces
            analyze_channel = ch
        
        # CRITICAL FIX: Filter out cells that don't have this channel defined in 'Stimulation Pathways'
        # This prevents basal cells (which have empty channel_2) from contaminating Schaffer data
        if ch != 'stratum_oriens':  # For apical channels (channel_1, channel_2)
            # Filter to only keep cells that have this specific channel defined
            filtered_traces_wt = {
                cell_id: traces 
                for cell_id, traces in current_traces_wt.items() 
                if ch in cell_valid_channels.get(cell_id, [])
            }
            filtered_traces_gnb1 = {
                cell_id: traces 
                for cell_id, traces in current_traces_gnb1.items() 
                if ch in cell_valid_channels.get(cell_id, [])
            }
            
            # Report how many cells were filtered out
            num_filtered_wt = len(current_traces_wt) - len(filtered_traces_wt)
            num_filtered_gnb1 = len(current_traces_gnb1) - len(filtered_traces_gnb1)
            if num_filtered_wt > 0 or num_filtered_gnb1 > 0:
                print(f"  ✓ Filtered out cells without {ch} defined: WT={num_filtered_wt}, GNB1={num_filtered_gnb1}")
            
            current_traces_wt = filtered_traces_wt
            current_traces_gnb1 = filtered_traces_gnb1

        
        # 1. Plotting
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
        
        # Analyze and Plot WT
        res_wt_gab = analyze_gabab_component(current_traces_wt, 'black', analyze_channel, 'WT Gabazine', ax1, 'gabazine')
        res_wt_ml = analyze_gabab_component(current_traces_wt, 'blue', analyze_channel, 'WT ML297', ax1, 'gabazine + ml297')
        res_wt_etx = analyze_gabab_component(current_traces_wt, 'green', analyze_channel, 'WT ETX', ax1, 'gabazine + etx')
        
        # Analyze and Plot GNB1
        res_gnb1_gab = analyze_gabab_component(current_traces_gnb1, 'darkred', analyze_channel, 'GNB1 Gabazine', ax2, 'gabazine')
        res_gnb1_ml = analyze_gabab_component(current_traces_gnb1, 'red', analyze_channel, 'GNB1 ML297', ax2, 'gabazine + ml297')
        res_gnb1_etx = analyze_gabab_component(current_traces_gnb1, 'lightcoral', analyze_channel, 'GNB1 ETX', ax2, 'gabazine + etx')

        # Formatting
        for ax in [ax1, ax2]:
            for line in ax.get_lines():
                line.set_linestyle('-')
                line.set_linewidth(2)
        
        ax1.set_title(f'WT - {channel_names[ch]}')
        ax2.set_title(f'GNB1 - {channel_names[ch]}')
        
        plot_path = os.path.join(PLOTS_DIR, f'GABAb_Summary_{ch}.png')
        fig.savefig(plot_path)
        plt.close(fig)
        print(f"  > Plot saved to {plot_path}")

        # 2. Collect Metrics for Export
        # Combine all result dictionaries
        all_res = [res_wt_gab, res_wt_ml, res_wt_etx, res_gnb1_gab, res_gnb1_ml, res_gnb1_etx]
        
        for res_dict in all_res:
            for cell, conds in res_dict.items():
                for cond_name, metrics in conds.items():
                    row = {
                        'Cell_ID': cell,
                        'Genotype': genotype_dict.get(cell, 'Unknown'),
                        'Sex': sex_dict.get(cell, 'Unknown'),
                        'Channel': analyze_channel if ch == 'stratum_oriens' else ch,
                        'Channel_Name': channel_names[ch],
                        'Condition': cond_name,
                        'Trough_Amplitude_mV': metrics['Trough Amplitude (mV)'],
                        'Trough_Time_ms': metrics['Trough Time (ms)'],
                        'Integral_mV_ms': metrics['Integral Below Zero (mV*ms)']
                    }
                    metrics_export_list.append(row)

        # 3. Collect Mean Traces for Export (CSV)
        for cond in ['gabazine', 'gabazine + ml297', 'gabazine + etx']:
            key_base = f"{channel_names[ch]}_{cond.replace(' ', '')}"
            trace_data_mean = collect_gabab_traces_for_export(current_all_traces, analyze_channel, cond, genotype_dict)
            
            max_len = 0
            for k, v in trace_data_mean.items():
                max_len = max(max_len, len(v))
            
            df_dict = {}
            for k, v in trace_data_mean.items():
                if len(v) < max_len:
                    df_dict[k] = np.pad(v, (0, max_len - len(v)), constant_values=np.nan)
                else:
                    df_dict[k] = v
            
            if df_dict:
                traces_mean_export_data[key_base] = pd.DataFrame(df_dict)


    # ==================================================================================================
    # FILE EXPORT
    # ==================================================================================================

    # 1. Export Metrics CSV
    if metrics_export_list:
        metrics_df = pd.DataFrame(metrics_export_list)
        csv_path = os.path.join(OUTPUT_DIR, 'GABAb_Analysis_Metrics.csv')
        metrics_df.to_csv(csv_path, index=False)
        print(f"\n✓ Metrics exported to: {csv_path}")
        print(metrics_df.head())

    # 2. Export Mean Traces CSV
    if traces_mean_export_data:
        combined_traces_df = pd.concat(traces_mean_export_data.values(), keys=traces_mean_export_data.keys(), axis=1)
        mean_traces_path = os.path.join(OUTPUT_DIR, 'GABAb_Mean_Traces.csv')
        combined_traces_df.to_csv(mean_traces_path)
        print(f"✓ Mean Traces (CSV) exported to: {mean_traces_path}")

    # 3. Export Hierarchical Individual Traces Pickle (MATCHING REQUESTED STRUCTURE)
    conditions_of_interest = ['gabazine', 'gabazine + ml297', 'gabazine + etx']
    
    # Collect apical pathway traces (Perforant Path, Schaffer Collateral)
    apical_channel_names = {'channel_1': 'Perforant Path', 'channel_2': 'Schaffer Collateral'}
    hierarchical_data_apical = collect_gabab_traces_by_condition(
        all_traces, 
        apical_channel_names,
        conditions_of_interest, 
        genotype_dict, 
        sex_dict
    )
    
    # Collect basal pathway traces (Stratum Oriens)
    basal_channel_names = {'channel_1': 'Stratum Oriens'}
    hierarchical_data_basal = collect_gabab_traces_by_condition(
        basal_traces, 
        basal_channel_names,
        conditions_of_interest, 
        genotype_dict, 
        sex_dict
    )
    
    # Merge apical and basal data
    # The structure is: { Condition: { Cell_ID: { 'genotype': ..., 'sex': ..., 'traces': { Pathway: Trace } } } }
    hierarchical_data = hierarchical_data_apical.copy() if hierarchical_data_apical else {}
    
    if hierarchical_data_basal:
        for condition, cells in hierarchical_data_basal.items():
            if condition not in hierarchical_data:
                hierarchical_data[condition] = {}
            for cell_id, cell_data in cells.items():
                if cell_id not in hierarchical_data[condition]:
                    hierarchical_data[condition][cell_id] = cell_data
                else:
                    # Merge traces from basal into existing cell data
                    if 'traces' in cell_data:
                        hierarchical_data[condition][cell_id]['traces'].update(cell_data['traces'])

    if hierarchical_data:
        pkl_path = os.path.join(OUTPUT_DIR, 'GABAb_Individual_Traces_Hierarchical.pkl')
        pd.to_pickle(hierarchical_data, pkl_path)
        print(f"✓ Hierarchical Traces (.pkl) exported to: {pkl_path}")
        print("  Structure: { Condition: { Cell_ID: { 'genotype': ..., 'sex': ..., 'traces': { Pathway: Trace } } } }")
        print(f"  Pathways included: Perforant Path, Schaffer Collateral, Stratum Oriens")
    
    # ==================================================================================================
    # 5. Delta Vm Analysis (Baclofen Wash-in)
    # ==================================================================================================
    print("\n--- Analyzing Baclofen Vm Change ---")
    if 'Voltage Change' in master_df.columns:
        vm_change_df = master_df[['Cell_ID', 'Genotype', 'Voltage Change']].copy()
        # Ensure numeric
        vm_change_df['Voltage Change'] = pd.to_numeric(vm_change_df['Voltage Change'], errors='coerce')
        vm_change_df = vm_change_df.dropna(subset=['Voltage Change'])
        
        vm_csv_path = os.path.join(OUTPUT_DIR, 'Baclofen_Vm_Change.csv')
        vm_change_df.to_csv(vm_csv_path, index=False)
        print(f"✓ Baclofen Vm Change exported to: {vm_csv_path}")
        print(f"  Count: {len(vm_change_df)} cells")
    else:
        print("⚠ 'Voltage Change' column not found in master_df.")

    # ==================================================================================================
    # 6. Firing Rate Analysis (Gabazine vs Gabazine + Baclofen)
    # ==================================================================================================
    print("\n--- Analyzing Firing Rates (Coarse FI) ---")
    
    # Identify cells with the specific Gabazine → Baclofen + Gabazine wash-in sequence
    # Filter by 'Drugs' column pattern
    if 'Drugs' in master_df.columns:
        # Pattern: "10_M Gabazine washed in first; then 10_M Baclofen + 10_M Gabazine washed in next"
        drug_pattern = r'10_M Gabazine washed in first.*10_M Baclofen.*10_M Gabazine'
        fi_cells_mask = master_df['Drugs'].str.contains(drug_pattern, na=False, case=False, regex=True)
        
        print(f"  Found {fi_cells_mask.sum()} cells with Gabazine → Baclofen+Gabazine wash-in pattern.")
        
        baclofen_cells = set(master_df[fi_cells_mask]['Cell_ID'].astype(str))
        
        # Show breakdown by genotype
        wt_count = len(master_df[fi_cells_mask & (master_df['Genotype'] == 'WT')])
        gnb1_count = len(master_df[fi_cells_mask & (master_df['Genotype'] == 'GNB1')])
        print(f"    - WT cells: {wt_count}")
        print(f"    - GNB1 cells: {gnb1_count}")
    else:
        baclofen_cells = set()
        print("⚠ 'Drugs' column missing, skipping FI analysis.")

    fi_rows = []
    fi_rim_rows = [] # Store Input Resistance Data
    vm_trace_data = {}  # Store first 300ms of lowest-current sweep for Vm visualization
    
    if baclofen_cells:
        # Custom extraction logic
        files = [f for f in os.listdir(DATA_DIR) if f.endswith('.pkl')]
        
        for name in files:
            cell_id = convert_pkl_filename_to_cell_id(name)
            if cell_id not in baclofen_cells: continue
            
            try:
                # Load pkl
                with open(os.path.join(DATA_DIR, name), 'rb') as f:
                    data = pd.read_pickle(f)
                    
                # Handle DataFrame vs Dict format
                if isinstance(data, pd.DataFrame):
                    iterable = range(len(data))
                    get_stim = lambda i: data['stimulus_metadata_dict'].iloc[i]
                    get_analysis = lambda i: data['analysis_dict'].iloc[i]
                    get_sweep = lambda i: data['sweep'].iloc[i] # Get raw trace
                else:
                    if 'analysis_dict' not in data: continue
                    iterable = range(len(data['analysis_dict']))
                    get_stim = lambda i: data['stimulus_metadata_dict'][i]
                    get_analysis = lambda i: data['analysis_dict'][i]
                    # Attempt to get sweep from data dictionary. Structure varies.
                    # Usually 'sweep' is a key if it was converted from a DataFrame-like structure
                    get_sweep = lambda i: data['sweep'][i] if 'sweep' in data else None

                for i in iterable:
                    stim_meta = get_stim(i)
                    if not stim_meta: continue
                    
                    analysis = get_analysis(i)
                    if not analysis: continue
                    
                    coarse_f_I = analysis.get('Coarse_FI') or analysis.get('IV_stim')
                    if not coarse_f_I: continue
                    
                    condition_str = stim_meta.get('condition', '').lower()
                    
                    if 'baclofen' in condition_str:
                        final_cond = 'Gabazine + Baclofen'
                    else:
                        final_cond = 'Gabazine'
                    
                    # Extract Data
                    current_amplitudes = np.atleast_1d(coarse_f_I['current_amplitudes'])
                    current_amplitudes = np.round(current_amplitudes, 1)
                    firing_rates = np.atleast_1d(coarse_f_I['firing_rates'])
                    
                    # Extract Trace for Input Resistance
                    # Only calculate Rim ONCE per sweep.
                    # We can associate it with the first current step (or just store it separately)
                    current_trace = get_sweep(i)
                    rim_val = np.nan
                    
                    if current_trace is not None and len(current_trace) > 0:
                        # Use our new helper function
                        rim_val = calculate_input_resistance_from_test_pulse(
                            current_trace, 
                            sampling_rate=20000,
                            pulse_start_ms=50,
                            pulse_duration_ms=100,
                            pulse_amp_pA=-50
                        )

                    # Store FI Data
                    for amp, rate in zip(current_amplitudes, firing_rates):
                        fi_rows.append({
                            'Cell_ID': cell_id,
                            'Genotype': genotype_dict.get(cell_id, 'Unknown'),
                            'Condition': final_cond,
                            'Original_Condition': condition_str, 
                            'Current_pA': amp,
                            'Firing_Rate_Hz': rate
                        })
                    
                    # Store Rim Data (One value per sweep)
                    if not np.isnan(rim_val):
                         fi_rim_rows.append({
                            'Cell_ID': cell_id,
                            'Genotype': genotype_dict.get(cell_id, 'Unknown'),
                            'Condition': final_cond,
                            'Input_Resistance_MOhm': rim_val
                        })
                    
                    # Store Vm trace (first 300ms) for the lowest current step only
                    # This captures the test pulse and resting Vm
                    if current_trace is not None and len(current_trace) >= 6000:
                        min_current = current_amplitudes.min()
                        key = (cell_id, final_cond)
                        if key not in vm_trace_data or min_current < vm_trace_data[key]['current']:
                            vm_trace_data[key] = {
                                'trace': current_trace[:6000].copy(),  # First 300ms at 20kHz
                                'current': min_current,
                                'genotype': genotype_dict.get(cell_id, 'Unknown'),
                            }
                        
            except Exception as e:
                print(f"Error processing {name}: {e}")
                continue

    # --- EXPORT Vm EXAMPLE TRACES (Figure 5H) ---
    if vm_trace_data:
        # Reorganize: { Cell_ID: { 'genotype': ..., 'Gabazine': trace, 'Gabazine + Baclofen': trace } }
        vm_export = {}
        for (cell_id, condition), info in vm_trace_data.items():
            if cell_id not in vm_export:
                vm_export[cell_id] = {'genotype': info['genotype']}
            vm_export[cell_id][condition] = info['trace']
        
        # Keep only cells that have BOTH conditions
        vm_export = {k: v for k, v in vm_export.items() 
                     if 'Gabazine' in v and 'Gabazine + Baclofen' in v}
        
        vm_traces_path = os.path.join(OUTPUT_DIR, 'Baclofen_Vm_Example_Traces.pkl')
        pd.to_pickle(vm_export, vm_traces_path)
        print(f"\n✓ Baclofen Vm Example Traces exported to: {vm_traces_path}")
        print(f"  Cells with both conditions: {len(vm_export)}")
        for cid, cdata in vm_export.items():
            print(f"    {cid} ({cdata['genotype']})")
    
    # --- PROCESS FI DATA ---
    if fi_rows:
        fi_df = pd.DataFrame(fi_rows)
        
        # Aggregate duplicates (multiple sweeps per current step)
        fi_df = fi_df.groupby(['Cell_ID', 'Genotype', 'Condition', 'Current_pA'])['Firing_Rate_Hz'].mean().reset_index()
        
        # FILTER: Experimental protocol only goes up to 350 pA
        fi_df = fi_df[fi_df['Current_pA'] <= 350].copy()
        
        fi_csv_path = os.path.join(OUTPUT_DIR, 'Baclofen_FI_Analysis.csv')
        fi_df.to_csv(fi_csv_path, index=False)
        print(f"✓ Baclofen FI Analysis exported to: {fi_csv_path}")
        print(f"  Count: {len(fi_df['Cell_ID'].unique())} cells")

        # ==================================================================================================
        # 7. Baclofen-Induced Reduction (Difference Analysis)
        # ==================================================================================================
        print("\n--- Analyzing Baclofen-Induced Reduction (Stats) ---")
        
        gabazine_cells = set(fi_df[fi_df['Condition'] == 'Gabazine']['Cell_ID'])
        baclofen_cells = set(fi_df[fi_df['Condition'] == 'Gabazine + Baclofen']['Cell_ID'])
        both_conditions = gabazine_cells & baclofen_cells
        
        diff_rows = []
        for cell_id in both_conditions:
            genotype = fi_df[fi_df['Cell_ID'] == cell_id]['Genotype'].values[0]
            gab_data = fi_df[(fi_df['Cell_ID'] == cell_id) & (fi_df['Condition'] == 'Gabazine')]
            bac_data = fi_df[(fi_df['Cell_ID'] == cell_id) & (fi_df['Condition'] == 'Gabazine + Baclofen')]
            
            for _, gab_row in gab_data.iterrows():
                current = gab_row['Current_pA']
                gab_fr = gab_row['Firing_Rate_Hz']
                bac_row = bac_data[bac_data['Current_pA'] == current]
                
                if not bac_row.empty:
                    bac_fr = bac_row['Firing_Rate_Hz'].values[0]
                    diff_fr = gab_fr - bac_fr 
                    
                    diff_rows.append({
                        'Cell_ID': cell_id,
                        'Genotype': genotype,
                        'Current_pA': current,
                        'Gabazine_FR': gab_fr,
                        'Baclofen_FR': bac_fr,
                        'Difference_FR': diff_fr
                    })
        
        if diff_rows:
            diff_df = pd.DataFrame(diff_rows)
            diff_csv_path = os.path.join(OUTPUT_DIR, 'Baclofen_FI_Difference.csv')
            diff_df.to_csv(diff_csv_path, index=False)
            print(f"✓ Difference data exported to: {diff_csv_path}")
            
            try:
                model = ols('Difference_FR ~ C(Genotype) * C(Current_pA)', data=diff_df).fit()
                anova_table = sm.stats.anova_lm(model, typ=2)
                
                stats_path = os.path.join(OUTPUT_DIR, 'Figure_5_FI_Stats.csv')
                anova_table.to_csv(stats_path)
                print(f"✓ ANOVA Stats saved to: {stats_path}")
                print(anova_table)

                # Plot Difference
                plt.figure(figsize=(5.5, 4.5))
                for geno, color in [('WT', 'black'), ('GNB1', 'red')]:
                    subset = diff_df[diff_df['Genotype'] == geno]
                    if subset.empty: continue
                    means = subset.groupby('Current_pA')['Difference_FR'].mean()
                    sems = subset.groupby('Current_pA')['Difference_FR'].sem()
                    x_vals = means.index
                    plt.errorbar(x_vals, means.values, yerr=sems.values, 
                                 fmt='-o', color=color, label=f"{geno} (n={len(subset['Cell_ID'].unique())})", 
                                 capsize=3, markersize=5)
                
                plt.title('Baclofen-Induced Firing Reduction\n(Gabazine - Gabazine+Baclofen)')
                plt.ylabel('Δ Firing Rate (Hz)')
                plt.xlabel('Current (pA)')
                plt.legend(frameon=False)
                plt.tight_layout()
                ax = plt.gca()
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                
                diff_plot_path = os.path.join(PLOTS_DIR, 'Figure_5_FI_Difference.png')
                plt.savefig(diff_plot_path, dpi=300)
                plt.close()
                print(f"✓ Difference plot saved to: {diff_plot_path}")
                
            except Exception as e:
                print(f"⚠ Stats/Plotting failed: {e}")
                import traceback
                traceback.print_exc()

    else:
        print("⚠ No FI data extracted for Voltage Change cells.")

    # --- PROCESS INPUT RESISTANCE DATA ---
    print("\n--- Analyzing Input Resistance (From Test Pulse) ---")
    if fi_rim_rows:
        rim_df = pd.DataFrame(fi_rim_rows)
        
        # Average Rim per cell per condition (averaging across sweeps)
        rim_summary = rim_df.groupby(['Cell_ID', 'Genotype', 'Condition'])['Input_Resistance_MOhm'].mean().reset_index()
        
        rim_csv_path = os.path.join(OUTPUT_DIR, 'Baclofen_Input_Resistance.csv')
        rim_summary.to_csv(rim_csv_path, index=False)
        print(f"✓ Input Resistance data exported to: {rim_csv_path}")
        print(f"  Count: {len(rim_summary)} measurements")
        
        # Calculate Stats (WT vs GNB1)
        # We can do a t-test for Gabazine condition, and Gabazine+Baclofen condition
        
        stats_results = []
        for condition in ['Gabazine', 'Gabazine + Baclofen']:
            wt_vals = rim_summary[(rim_summary['Genotype'] == 'WT') & (rim_summary['Condition'] == condition)]['Input_Resistance_MOhm']
            gnb1_vals = rim_summary[(rim_summary['Genotype'] == 'GNB1') & (rim_summary['Condition'] == condition)]['Input_Resistance_MOhm']
            
            if len(wt_vals) > 1 and len(gnb1_vals) > 1:
                t_stat, p_val = scipy.stats.ttest_ind(wt_vals, gnb1_vals, nan_policy='omit')
                stats_results.append({
                    'Condition': condition,
                    'WT_N': len(wt_vals),
                    'GNB1_N': len(gnb1_vals),
                    'WT_Mean_MOhm': wt_vals.mean(),
                    'GNB1_Mean_MOhm': gnb1_vals.mean(),
                    'p_value': p_val
                })
                print(f"  {condition}: WT={wt_vals.mean():.1f} vs GNB1={gnb1_vals.mean():.1f} MΩ, p={p_val:.4f}")
        
        if stats_results:
            rim_stats_df = pd.DataFrame(stats_results)
            rim_stats_path = os.path.join(OUTPUT_DIR, 'Baclofen_Input_Resistance_Stats.csv')
            rim_stats_df.to_csv(rim_stats_path, index=False)
            print(f"✓ Input Resistance Stats exported to: {rim_stats_path}")

        # --- Plot Input Resistance (Paired) ---
        plt.figure(figsize=(6, 5))
        
        # Helper to plot paired data
        def plot_paired_rim(ax, genotype, color_base):
            df_g = rim_summary[rim_summary['Genotype'] == genotype]
            cells = df_g['Cell_ID'].unique()
            
            x_pos = [1, 2] # X-axis at 1 and 2
            labels = ['Gabazine', 'Gab+Baclofen']
            
            # Collect paired values
            before_vals = []
            after_vals = []
            
            for cell in cells:
                val_gab = df_g[(df_g['Cell_ID'] == cell) & (df_g['Condition'] == 'Gabazine')]['Input_Resistance_MOhm'].values
                val_bac = df_g[(df_g['Cell_ID'] == cell) & (df_g['Condition'] == 'Gabazine + Baclofen')]['Input_Resistance_MOhm'].values
                
                if len(val_gab) > 0 and len(val_bac) > 0:
                    v1 = val_gab[0]
                    v2 = val_bac[0]
                    # Plot lines connecting paired points (NO jitter)
                    ax.plot(x_pos, [v1, v2], color='gray', alpha=0.4, linewidth=1, zorder=1)
                    # Plot individual points
                    ax.scatter(x_pos[0], v1, color='gray', s=20, alpha=0.6, zorder=2)
                    ax.scatter(x_pos[1], v2, color='gray', s=20, alpha=0.6, zorder=2)
                    
                    before_vals.append(v1)
                    after_vals.append(v2)
            
            # Plot Means (with Error Bars)
            if before_vals:
                mean_before = np.mean(before_vals)
                sem_before = scipy.stats.sem(before_vals)
                mean_after = np.mean(after_vals)
                sem_after = scipy.stats.sem(after_vals)
                
                # Plot Mean points bigger and colored
                ax.errorbar(x_pos[0], mean_before, yerr=sem_before, fmt='o', color=color_base, 
                            capsize=5, markersize=10, elinewidth=2, zorder=3)
                ax.errorbar(x_pos[1], mean_after, yerr=sem_after, fmt='o', color=color_base, 
                            capsize=5, markersize=10, elinewidth=2, zorder=3)
                
            ax.set_xticks(x_pos)
            ax.set_xticklabels(labels, rotation=45, ha='right')
            ax.set_title(f"{genotype} (n={len(before_vals)})")
            ax.set_ylabel("Input Resistance (MΩ)")
            ax.set_xlim(0.5, 2.5) 
            
            # T-test Paired
            if len(before_vals) > 1:
                t, p = scipy.stats.ttest_rel(before_vals, after_vals)
                y_max = max(max(before_vals), max(after_vals))
                ax.text(1.5, y_max * 1.05, f"p={p:.4f}", ha='center')
                ax.set_ylim(top=y_max * 1.2)
                
            # Clean spines
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        ax1 = plt.subplot(1, 2, 1)
        plot_paired_rim(ax1, 'WT', 'black')
        
        ax2 = plt.subplot(1, 2, 2, sharey=ax1)
        plot_paired_rim(ax2, 'GNB1', 'red')
        
        plt.tight_layout()
        rim_plot_path = os.path.join(PLOTS_DIR, 'Baclofen_Input_Resistance_Paired.png')
        plt.savefig(rim_plot_path, dpi=300)
        plt.close()
        print(f"✓ Input Resistance Plot saved to: {rim_plot_path}")

        # --- Analyze Delta Rim (Stats on Change) ---
        delta_rim_data = []
        for cell in rim_summary['Cell_ID'].unique():
            df_c = rim_summary[rim_summary['Cell_ID'] == cell]
            if len(df_c) == 2: # Has both conditions
                val_gab = df_c[df_c['Condition'] == 'Gabazine']['Input_Resistance_MOhm'].values[0]
                val_bac = df_c[df_c['Condition'] == 'Gabazine + Baclofen']['Input_Resistance_MOhm'].values[0]
                genotype = df_c['Genotype'].values[0]
                
                # Percent Change or Absolute? Let's do Absolute Drop
                delta = val_gab - val_bac # Positive means drop
                pct_change = (val_bac - val_gab) / val_gab * 100
                
                delta_rim_data.append({
                    'Cell_ID': cell,
                    'Genotype': genotype,
                    'Delta_Rim_MOhm': delta,
                    'Pct_Change': pct_change
                })
        
        if delta_rim_data:
            df_delta = pd.DataFrame(delta_rim_data)
            delta_csv_path = os.path.join(OUTPUT_DIR, 'Baclofen_Input_Resistance_Delta.csv')
            df_delta.to_csv(delta_csv_path, index=False)
            
            # T-test on Delta
            wt_delta = df_delta[df_delta['Genotype'] == 'WT']['Delta_Rim_MOhm']
            gnb1_delta = df_delta[df_delta['Genotype'] == 'GNB1']['Delta_Rim_MOhm']
            
            p_val = np.nan
            if len(wt_delta) > 1 and len(gnb1_delta) > 1:
                t, p_val = scipy.stats.ttest_ind(wt_delta, gnb1_delta)
                print(f"  Delta Rim (Gab - Bac): WT={wt_delta.mean():.1f} vs GNB1={gnb1_delta.mean():.1f} MΩ, p={p_val:.4f}")
            
            # --- Plot Delta Rim (WT vs GNB1) ---
            plt.figure(figsize=(4, 5))
            
            x_pos = [1, 2]
            
            # Means and SEMs
            means = [wt_delta.mean(), gnb1_delta.mean()]
            sems = [scipy.stats.sem(wt_delta), scipy.stats.sem(gnb1_delta)]
            labels = [f"WT\n(n={len(wt_delta)})", f"GNB1\n(n={len(gnb1_delta)})"]
            colors = ['black', 'red']
            
            # Bar Plot
            bars = plt.bar(x_pos, means, yerr=sems, color=colors, capsize=5, alpha=0.5, width=0.6, zorder=1)
            
            # Individual Points (NO Jitter, centered at 1 and 2)
            plt.scatter(np.full(len(wt_delta), 1), wt_delta, color='black', alpha=0.7, zorder=2, s=30)
            plt.scatter(np.full(len(gnb1_delta), 2), gnb1_delta, color='maroon', alpha=0.7, zorder=2, s=30)
            
            plt.xticks(x_pos, labels)
            plt.ylabel('Δ Input Resistance (MΩ)\n(Gabazine - Gabazine+Baclofen)')
            plt.title('Baclofen-Induced\nResistance Drop')
            plt.xlim(0.2, 2.8)
            
            # Add p-value
            if not np.isnan(p_val):
                y_max = max(wt_delta.max(), gnb1_delta.max())
                plt.text(1.5, y_max * 1.1, f"p={p_val:.4f}", ha='center', fontsize=12)
                plt.ylim(top=y_max * 1.25)
            
            # Clean spines
            ax = plt.gca()
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            delta_plot_path = os.path.join(PLOTS_DIR, 'Baclofen_Input_Resistance_Delta.png')
            plt.savefig(delta_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✓ Delta Input Resistance Plot saved to: {delta_plot_path}")


    else:
        print("⚠ No Input Resistance data extracted.")

    # ==================================================================================================
    # 8. ML297 Input Resistance Analysis (From Unitary Traces)
    # ==================================================================================================
    print("\n--- Analyzing ML297 Input Resistance (From Unitary Traces) ---")
    
    ml297_rim_data = []

    # Iterate through APICAL traces (Perforant & Schaffer)
    # all_traces structure: { Cell_ID: { 'channel_1': { 'gabazine': trace, ... }, 'channel_2': ... } }
    
    for cell_id, data_by_isi in all_traces.items():
        genotype = genotype_dict.get(cell_id, 'Unknown')
        
        # Data is nested: cell_id -> ISI (300) -> channel -> condition -> trace
        if 300 not in data_by_isi: continue
        cell_data = data_by_isi[300]
        
        # Check both channels
        for channel in ['channel_1', 'channel_2']:
            if channel not in cell_data: continue
            
            chan_data = cell_data[channel]
            
            # Robust key matching (lowercase check)
            chan_keys_map = {k.lower().strip(): k for k in chan_data.keys()}
            
            # DEBUG: Print keys to debug missing matches
            if len(ml297_rim_data) < 2:
                 print(f"DEBUG: {cell_id} {channel} keys: {list(chan_data.keys())}")
            
            # Check for generic 'gabazine' and 'gabazine + ml297'
            cond_gab_key = chan_keys_map.get('gabazine')
            # Try variations for ML297
            cond_ml_key = chan_keys_map.get('gabazine + ml297')
            if not cond_ml_key: cond_ml_key = chan_keys_map.get('ml297')
            if not cond_ml_key: cond_ml_key = chan_keys_map.get('gabazine+ml297')
            
            # DEBUG: first unique cell check
            if not ml297_rim_data and len(chan_data) > 0 and cond_ml_key is None:
                 # Print keys if we expected ML297 but didn't match it (or just debugging)
                 pass
                 # print(f"DEBUG: Keys for {cell_id} {channel}: {list(chan_data.keys())}")
            
            if cond_gab_key and cond_ml_key:
                trace_gab = chan_data[cond_gab_key]
                trace_ml = chan_data[cond_ml_key]
                
                # Extract traces if they are dictionaries (e.g. from E/I analysis)
                actual_trace_gab = None
                actual_trace_ml = None
                
                # Helper to extract trace array from dict or array
                def extract_trace(tr):
                    if isinstance(tr, dict):
                         # Check for average trace or numeric keys
                         # Heuristic: if values are arrays, average them
                         sweeps = [v for v in tr.values() if hasattr(v, 'shape')]
                         if sweeps:
                             try:
                                 return np.mean(sweeps, axis=0)
                             except:
                                 return sweeps[0]
                    elif hasattr(tr, 'shape'):
                         return tr
                    return None

                actual_trace_gab = extract_trace(trace_gab)
                actual_trace_ml = extract_trace(trace_ml)
                
                if actual_trace_gab is not None and actual_trace_ml is not None:
                    rim_gab = calculate_input_resistance_from_test_pulse(actual_trace_gab)
                    rim_ml = calculate_input_resistance_from_test_pulse(actual_trace_ml)
                else:
                    rim_gab = np.nan
                    rim_ml = np.nan
                
                if not np.isnan(rim_gab) and not np.isnan(rim_ml):
                    # Store (one row per cell/channel comparison)
                    ml297_rim_data.append({
                        'Cell_ID': cell_id,
                        'Genotype': genotype,
                        'Channel': channel,
                        'Rim_Gabazine': rim_gab,
                        'Rim_ML297': rim_ml,
                        'Delta_Rim': rim_gab - rim_ml, # Positive = Drop
                        'Pct_Change': (rim_ml - rim_gab) / rim_gab * 100
                    })

    if ml297_rim_data:
        ml_df = pd.DataFrame(ml297_rim_data)
        
        # Aggregate by Cell (average across channels if a cell has both, to avoid pseudo-replication)
        ml_df_cell = ml_df.groupby(['Cell_ID', 'Genotype']).mean(numeric_only=True).reset_index()
        
        ml_csv_path = os.path.join(OUTPUT_DIR, 'ML297_Input_Resistance.csv')
        ml_df_cell.to_csv(ml_csv_path, index=False)
        print(f"✓ ML297 Input Resistance data exported to: {ml_csv_path}")
        print(f"  Count: {len(ml_df_cell)} cells")
        
        # Stats: Paired T-test (Gab vs ML) per Genotype
        print("  Stats (Paired T-test Gab vs ML297):")
        for geno in ['WT', 'GNB1']:
            subset = ml_df_cell[ml_df_cell['Genotype'] == geno]
            if len(subset) > 1:
                t, p = scipy.stats.ttest_rel(subset['Rim_Gabazine'], subset['Rim_ML297'])
                print(f"    {geno} (n={len(subset)}): p={p:.4f} (Drop: {subset['Delta_Rim'].mean():.1f} MΩ)")
        
        # Stats: Delta comparison (WT vs GNB1)
        wt_delta = ml_df_cell[ml_df_cell['Genotype'] == 'WT']['Delta_Rim']
        gnb1_delta = ml_df_cell[ml_df_cell['Genotype'] == 'GNB1']['Delta_Rim']
        
        p_val_delta = np.nan
        if len(wt_delta) > 1 and len(gnb1_delta) > 1:
            t, p_val_delta = scipy.stats.ttest_ind(wt_delta, gnb1_delta)
            print(f"  Delta Rim (WT vs GNB1): p={p_val_delta:.4f}")

        # --- Plot 1: Paired Plot (ML297) ---
        plt.figure(figsize=(6, 5))
        
        ax1 = plt.subplot(1, 2, 1)
        ax2 = plt.subplot(1, 2, 2, sharey=ax1)
        
        axes = {'WT': ax1, 'GNB1': ax2}
        colors = {'WT': 'black', 'GNB1': 'red'}
        
        for geno in ['WT', 'GNB1']:
            ax = axes[geno]
            subset = ml_df_cell[ml_df_cell['Genotype'] == geno]
            before = subset['Rim_Gabazine'].values
            after = subset['Rim_ML297'].values
            
            if len(before) == 0: continue
            
            # Lines
            x_pos = [1, 2]
            for b, a in zip(before, after):
                ax.plot(x_pos, [b, a], color='gray', alpha=0.4, linewidth=1, zorder=1)
                ax.scatter(x_pos[0], b, color='gray', s=20, alpha=0.6, zorder=2)
                ax.scatter(x_pos[1], a, color='gray', s=20, alpha=0.6, zorder=2)
            
            # Means
            m_b, s_b = np.mean(before), scipy.stats.sem(before)
            m_a, s_a = np.mean(after), scipy.stats.sem(after)
            
            ax.errorbar(x_pos[0], m_b, yerr=s_b, fmt='o', color=colors[geno], capsize=5, markersize=10, elinewidth=2, zorder=3)
            ax.errorbar(x_pos[1], m_a, yerr=s_a, fmt='o', color=colors[geno], capsize=5, markersize=10, elinewidth=2, zorder=3)
            
            # Style
            ax.set_xticks(x_pos)
            ax.set_xticklabels(['Gabazine', 'Gab+ML297'], rotation=45, ha='right')
            ax.set_title(f"{geno} (n={len(before)})")
            ax.set_xlim(0.5, 2.5)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            # Stats text
            if len(before) > 1:
                 t, p = scipy.stats.ttest_rel(before, after)
                 y_max = max(np.max(before), np.max(after))
                 ax.text(1.5, y_max * 1.05, f"p={p:.4f}", ha='center')
                 ax.set_ylim(top=y_max * 1.2)
                 
        ax1.set_ylabel("Input Resistance (MΩ)")
        plt.tight_layout()
        ml_paired_path = os.path.join(PLOTS_DIR, 'ML297_Input_Resistance_Paired.png')
        plt.savefig(ml_paired_path, dpi=300)
        plt.close()
        
        # --- Plot 2: Delta Plot (ML297) ---
        plt.figure(figsize=(4, 5))
        
        means = [wt_delta.mean(), gnb1_delta.mean()]
        sems = [scipy.stats.sem(wt_delta), scipy.stats.sem(gnb1_delta)]
        
        x_pos = [1, 2]
        plt.bar(x_pos, means, yerr=sems, color=['black', 'red'], capsize=5, alpha=0.5, width=0.6, zorder=1)
        
        # Scatters
        plt.scatter(np.full(len(wt_delta), 1), wt_delta, color='black', alpha=0.7, zorder=2, s=30)
        plt.scatter(np.full(len(gnb1_delta), 2), gnb1_delta, color='maroon', alpha=0.7, zorder=2, s=30)
        
        plt.xticks([1, 2], [f"WT\n(n={len(wt_delta)})", f"GNB1\n(n={len(gnb1_delta)})"])
        plt.ylabel('Δ Input Resistance (MΩ)\n(Gabazine - Gabazine+ML297)')
        plt.title('ML297-Induced\nResistance Drop')
        plt.xlim(0.2, 2.8)
        
        if not np.isnan(p_val_delta):
            y_max_d = max(wt_delta.max(), gnb1_delta.max())
            plt.text(1.5, y_max_d * 1.1, f"p={p_val_delta:.4f}", ha='center', fontsize=12)
            plt.ylim(top=y_max_d * 1.25)
            
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        ml_delta_path = os.path.join(PLOTS_DIR, 'ML297_Input_Resistance_Delta.png')
        plt.savefig(ml_delta_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ ML297 Delta Plot saved to: {ml_delta_path}")

    else:
        print("⚠ No ML297 Input Resistance data found. Check if traces exist for 'gabazine + ml297'.")

    print("\n✓ Analysis Complete.")