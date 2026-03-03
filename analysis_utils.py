import pandas as pd
import numpy as np
import os
import ast
import scipy
import scipy.signal as signal
from scipy.signal import find_peaks, savgol_filter
from scipy.signal import find_peaks
from scipy import integrate
from scipy.stats import sem
from scipy.optimize import curve_fit
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import yaml
import re
import dill
from pathlib import Path
from scipy.stats import linregress
import warnings
# Suppress specific pandas warnings if necessary
warnings.filterwarnings('ignore', category=FutureWarning)

#TODO: More analysis to add:
#Basal Analysis
#Baclofen - FI plotting

#----------------------------------------------------------------------------------
#Helper Functions
#----------------------------------------------------------------------------------

def get_clean_data(prop_key, data_source):
    """Returns cleaned data array (NaNs removed)."""
    data = np.array(data_source.get(prop_key, []))
    # Check if the array is non-empty before filtering NaNs
    return data[~np.isnan(data)] if data.size > 0 else np.array([])

def filter_master_df_by_inclusion(master_df):
    """
    Filters the Master DataFrame to only include rows where the 
    'Inclusion' column starts with 'Yes' (case-insensitive).
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
    print(f"Data Filtering: Kept {len(filtered_df)} cells. Dropped {dropped_count} cells based on 'Inclusion'.")
    
    return filtered_df

def read_yaml_file(filepath):
    with open(filepath, 'r') as file:
        try:
            yaml_data = yaml.safe_load(file)
            return yaml_data
        except yaml.YAMLError as e:
            print(f"Error reading YAML file: {e}")
            return None

def convert_pkl_filename_to_cell_id(pkl_filename):
    """
    Convert .pkl filename from monthdayyear format to Cell_ID format.
    
    Parameters:
        pkl_filename: Filename like '01012024_c1_processed_data.pkl' or '01012024_c1_processed_new.pkl'
    
    Returns:
        Cell_ID in format '20240101_c1'
    """
    # Remove .pkl extension and get basename
    basename = Path(pkl_filename).stem
    
    # Remove common suffixes
    basename = basename.replace('_processed_data', '').replace('_processed_new', '')
    
    # Extract date and cell parts
    # Handles formats like: 01012024_c1, 01012024_c2, etc.
    match = re.match(r'(\d{2})(\d{2})(\d{4})_(c\d+)', basename)
    
    if match:
        month, day, year, cell_id = match.groups()
        # Convert to Cell_ID format: YYYYMMDD_cX
        formatted_date = f"{year}{month}{day}"
        cell_id_final = f"{formatted_date}_{cell_id}"
        return cell_id_final
    else:
        print(f"Warning: Could not parse filename: {pkl_filename}")
        return None

def convert_filename_to_standard_id(filename):
    """
    Input: '01042024_c1_processed_data.pkl' (mmddyyyy_c#)
    Output: '20240104_c1' (yyyymmdd_c#) - MATCHES MASTER_DF
    """
    try:
        # Regex to pull out mm, dd, yyyy, and c#
        # Looks for 8 digits at start, followed by underscore, then c + numbers
        match = re.search(r'^(\d{2})(\d{2})(\d{4})_(c\d+)', filename)
        
        if match:
            mm, dd, yyyy, cell = match.groups()
            standard_id = f"{yyyy}{mm}{dd}_{cell}"
            return standard_id
        else:
            return None
    except Exception:
        return None

def zero_clip_and_interpolate(trace):
    """Clip all values below 0 to exactly 0. No negative values allowed."""
    trace = trace.copy()
    trace[trace < 0] = 0
    return trace

#--------------------------------------------------------------------------------------------
#Analysis of Action Potential Properties and Afterhyperpolarization Properties
#--------------------------------------------------------------------------------------------

def calculate_AHP_duration(trace, relative_AHP_trough_idx, AP_threshold_value, sampling_rate):
    """
    Calculates AHP duration: time from AHP trough until trace crosses AP Threshold again.
    """
    dt_ms = 1000 / sampling_rate 
    
    # Clip trace starting from AHP trough
    trace_after_trough = trace[relative_AHP_trough_idx:]
    
    # Find first index >= threshold
    recovery_indices = np.where(trace_after_trough >= AP_threshold_value)[0]

    if len(recovery_indices) > 0:
        recovery_idx = recovery_indices[0]
        return recovery_idx * dt_ms
    else:
        return np.nan 

def AHP_time_to_peak(peak_idx, relative_AHP_trough_idx, sampling_rate):
    """Calculates time interval between AP peak and AHP trough."""
    dt_ms = 1000 / sampling_rate
    return (relative_AHP_trough_idx - peak_idx) * dt_ms

def get_AP_and_AHP_rheobase_properties_data_and_traces(master_df, data_dir, AP_properties_to_plot, 
                                                       AP_trace_end=200, duration_ms=50, 
                                                       sampling_rate=20000):
    """
    Extracts AP properties. 
    NOTE: AP_trace_end default increased to 200ms to ensure capture of full AHP recovery.
    """
    fine_FI_stim_properties = {}
    rheobase_traces = {}
    decay_area_properties = {}
    
    dt = 1 / sampling_rate
    duration_samples = int(duration_ms / 1000 / dt)
    
    # Create lookup dict: Cell_ID -> Rheobase Sweep
    valid_df = master_df.dropna(subset=['Cell_ID', 'Rheobase Sweep'])
    master_lookup = dict(zip(
        valid_df['Cell_ID'].astype(str),
        valid_df['Rheobase Sweep'].astype(int)
    ))
    
    data_files = [f for f in os.listdir(data_dir) if f.endswith('.pkl')]
    
    for data_file in data_files:
        try:
            cell_id = convert_pkl_filename_to_cell_id(data_file)
            if cell_id is None or cell_id not in master_lookup:
                continue
                
            parsed_sweep = master_lookup[cell_id]
            data_df = pd.read_pickle(os.path.join(data_dir, data_file))
            
            if parsed_sweep < 0 or parsed_sweep >= len(data_df): continue

            # Initialize dicts
            if cell_id not in fine_FI_stim_properties: fine_FI_stim_properties[cell_id] = {}
            if cell_id not in decay_area_properties: decay_area_properties[cell_id] = {}

            # Access Analysis Data
            if 'analysis_dict' not in data_df.iloc[parsed_sweep]: continue
            current_analysis_data = data_df.iloc[parsed_sweep]['analysis_dict']
            
            if 'AP' not in current_analysis_data: continue
            current_analysis_data_AP = current_analysis_data['AP']
            AP_threshold_indices = current_analysis_data_AP.get('AP_threshold_indices')
            
            # --- Valid Cell Check ---
            if AP_threshold_indices is None or len(AP_threshold_indices) == 0:
                continue 

            # --- Check for Multiple APs ---
            # Calculate number of APs for exclusion criteria
            num_aps = len(AP_threshold_indices) if isinstance(AP_threshold_indices, (list, tuple, np.ndarray)) else 1
            has_multiple_aps = num_aps > 1

            # Handle AP Threshold Index extraction
            if isinstance(AP_threshold_indices, (list, tuple, np.ndarray)):
                if isinstance(AP_threshold_indices[0], (list, tuple, np.ndarray)):
                    AP_threshold_idx = int(AP_threshold_indices[0][0])
                else:
                    AP_threshold_idx = int(AP_threshold_indices[0])
            else:
                AP_threshold_idx = int(AP_threshold_indices)

            # Extract Trace
            current_rheobase_trace = data_df.iloc[parsed_sweep]['sweep']
            trace_end_idx_full = AP_threshold_idx + int(AP_trace_end * sampling_rate / 1000)
            if trace_end_idx_full > len(current_rheobase_trace):
                 trace_end_idx_full = len(current_rheobase_trace)
                 
            clipped_trace = current_rheobase_trace[AP_threshold_idx:trace_end_idx_full]
            
            if cell_id not in rheobase_traces:
                rheobase_traces[cell_id] = {}
            rheobase_traces[cell_id]['Rheobase_Trace'] = [clipped_trace]

            # Extract Rheobase Current
            current_lookup_keys = ['Fine_FI', 'IV_stim', 'Coarse_FI']
            for key in current_lookup_keys:
                if key in current_analysis_data and 'current_amplitudes' in current_analysis_data[key]:
                    current_val = current_analysis_data[key]['current_amplitudes']
                    rheobase_traces[cell_id]['Rheobase_Current'] = current_val
                    
                    if 'Rheobase_Current' not in fine_FI_stim_properties[cell_id]:
                        fine_FI_stim_properties[cell_id]['Rheobase_Current'] = []
                    
                    val_to_store = current_val[0] if isinstance(current_val, (list, np.ndarray)) else current_val
                    
                    # Exclude cells with >3 APs for consistency with AHP analysis
                    if num_aps > 3:
                        val_to_store = np.nan
                    
                    fine_FI_stim_properties[cell_id]['Rheobase_Current'].append(val_to_store)
                    break
            
            # --- AHP Analysis ---
            # Initialize with NaNs
            AHP_trough_voltage = np.nan
            AHP_trough_amplitude = np.nan
            time_to_peak_ms = np.nan
            duration_to_threshold_ms = np.nan
            decay_area = np.nan
            
            # Only calculate AHP if 3 or fewer APs (more than 3 APs contaminate AHP measurement)
            
            if num_aps <= 3:
                # AHP Analysis (analyzes first AP)
                # Get Threshold Value
                AP_threshold_value = -40
                if 'AP_threshold' in current_analysis_data_AP:
                    val = current_analysis_data_AP['AP_threshold']
                    if isinstance(val, (list, tuple, np.ndarray)) and len(val) > 0:
                        AP_threshold_value = val[0][0] if isinstance(val[0], (list, tuple, np.ndarray)) else val[0]
                    else:
                        AP_threshold_value = val
                
                trace_after_threshold = clipped_trace.copy()
                
                try:
                    peak_idx = np.argmax(trace_after_threshold)
                    # Simple peak check
                    if peak_idx > 0:
                        # Find Trough
                        min_val_after_peak_idx = np.argmin(trace_after_threshold[peak_idx:])
                        relative_AHP_trough_idx = peak_idx + min_val_after_peak_idx
                        
                        if relative_AHP_trough_idx != peak_idx:
                            # Metrics
                            _AHP_vol = trace_after_threshold[relative_AHP_trough_idx]
                            _AHP_amp = AP_threshold_value - _AHP_vol # Calculated amplitude
                            
                            # Valid AHP check: Must have positive amplitude
                            if not np.isnan(_AHP_amp) and _AHP_amp > 0:
                                AHP_trough_voltage = _AHP_vol
                                AHP_trough_amplitude = _AHP_amp

                                time_to_peak_ms = AHP_time_to_peak(peak_idx, relative_AHP_trough_idx, sampling_rate)
                                duration_to_threshold_ms = calculate_AHP_duration(
                                    trace_after_threshold, relative_AHP_trough_idx, AP_threshold_value, sampling_rate
                                )
                                
                                # Decay Area
                                end_idx = int(relative_AHP_trough_idx) + duration_samples
                                if end_idx >= len(trace_after_threshold): end_idx = len(trace_after_threshold) - 1
                                
                                clipped_area = AP_threshold_value - trace_after_threshold[int(relative_AHP_trough_idx):end_idx + 1]
                                norm_area = clipped_area / AHP_trough_amplitude
                                decay_area = np.trapz(norm_area, dx=dt) * 1000

                except Exception:
                    pass

            # Store AHP Props (Will be NaNs if multiple APs or Calc failed)
            decay_area_properties[cell_id]['AHP_Trough_Voltage'] = AHP_trough_voltage
            decay_area_properties[cell_id]['AHP_size'] = AHP_trough_amplitude 
            decay_area_properties[cell_id]['AHP_Time_to_Peak_ms'] = time_to_peak_ms
            decay_area_properties[cell_id]['AHP_Duration_to_Threshold_ms'] = duration_to_threshold_ms
            decay_area_properties[cell_id]['decay_area'] = decay_area

            # Store Plotting Indices
            decay_area_properties[cell_id]['relative_AHP_trough_idx'] = relative_AHP_trough_idx
            decay_area_properties[cell_id]['AHP_decay_end_idx'] = end_idx
            decay_area_properties[cell_id]['relative_peak_idx'] = peak_idx
            decay_area_properties[cell_id]['AP_threshold_value'] = AP_threshold_value

            # --- Extract Requested AP Keys from Pickle ---
            for key in AP_properties_to_plot:
                # If key is already calculated in AHP section (e.g. decay_area), skip lookup
                if key in decay_area_properties[cell_id]:
                    continue
                
                if key not in fine_FI_stim_properties[cell_id]:
                    fine_FI_stim_properties[cell_id][key] = []

                if key in current_analysis_data_AP:
                    value = current_analysis_data_AP[key]
                    if isinstance(value, (list, tuple, np.ndarray)) and len(value) > 0:
                        val_to_store = value[0][0] if isinstance(value[0], (list, tuple, np.ndarray)) else value[0]
                    else:
                        val_to_store = value
                    fine_FI_stim_properties[cell_id][key].append(val_to_store)
        
        except Exception as e:
            print(f"Skipping {data_file}: {e}")
            continue

    return fine_FI_stim_properties, decay_area_properties, rheobase_traces

def combine_AP_and_AHP_properties(AP_dict, AHP_dict):
    """Merges AP dict and AHP dict."""
    combined_dict = {}
    
    # Keys that come from the AHP dict calculation
    AHP_KEYS = ['AHP_size', 'decay_area', 'AHP_Time_to_Peak_ms', 
                'AHP_Duration_to_Threshold_ms', 'AHP_Trough_Voltage']

    for cell in AP_dict:
        cell_props = AP_dict[cell].copy()
        
        if cell in AHP_dict:
            for key in AHP_KEYS:
                val = AHP_dict[cell].get(key)
                cell_props[key] = [val] if val is not None else [np.nan]

        combined_dict[cell] = cell_props
        
    return combined_dict

def analyze_and_export_rheobase_properties(master_df, data_dir, output_path=None, 
                                          AP_properties_to_plot=None,
                                          AP_trace_end=200, duration_ms=50, 
                                          sampling_rate=20000):
    
    if AP_properties_to_plot is None:
        AP_properties_to_plot = [
            'AP_threshold', 'AP_halfwidth', 'AP_size', 'AHP_size', 
            'decay_area', 
            'AHP_Time_to_Peak_ms', 
            'AHP_Duration_to_Threshold_ms'
        ]
    
    # 1. Extract
    AP_dict, AHP_dict, traces_dict = get_AP_and_AHP_rheobase_properties_data_and_traces(
        master_df, data_dir, AP_properties_to_plot,
        AP_trace_end=AP_trace_end, duration_ms=duration_ms, 
        sampling_rate=sampling_rate
    )
    
    # 2. Combine
    combined_dict = combine_AP_and_AHP_properties(AP_dict, AHP_dict)
    
    # 3. Export to DF
    export_df = export_rheobase_properties_to_dataframe(
        combined_dict, master_df, properties_to_export=AP_properties_to_plot + ['Rheobase_Current']
    )
    
    if output_path:
        export_df.to_csv(output_path, index=False)
        print(f"Data exported to: {output_path}")
    
    return export_df

def export_rheobase_properties_to_dataframe(combined_dict, master_df, properties_to_export=None):
    master_df_copy = master_df.copy()
    master_df_copy['Cell_ID'] = master_df_copy['Cell_ID'].astype(str)
    
    genotype_lookup = dict(zip(master_df_copy['Cell_ID'], master_df_copy['Genotype']))
    sex_lookup = dict(zip(master_df_copy['Cell_ID'], master_df_copy['Sex']))
    
    rows = []
    for cell_id, properties in combined_dict.items():
        row = {
            'Cell_ID': cell_id,
            'Genotype': genotype_lookup.get(cell_id, 'Unknown'),
            'Sex': sex_lookup.get(cell_id, 'Unknown')
        }
        
        target_props = properties_to_export if properties_to_export else properties.keys()
        
        for prop in target_props:
            if prop in properties:
                val = properties[prop]
                row[prop] = val[0] if isinstance(val, (list, np.ndarray)) and len(val) > 0 else np.nan
            else:
                row[prop] = np.nan
        rows.append(row)
    
    if not rows: return pd.DataFrame()
    
    df = pd.DataFrame(rows)
    # Order columns
    meta = ['Cell_ID', 'Genotype', 'Sex']
    data_cols = [c for c in df.columns if c not in meta]
    return df[meta + data_cols]

#-------------------------------------------------------------------------------------------
#Analysis of Firing Rates and Spike Rate Adaptation
#-------------------------------------------------------------------------------------------

def get_firing_rate_data(dir_path, master_df=None):
    """
    Extract firing rate data from .pkl files.
    """
    FI_data = {}
    
    # Create valid_ids set for filtering
    valid_ids = None
    if master_df is not None and 'Cell_ID' in master_df.columns:
        valid_ids = set(master_df['Cell_ID'].astype(str))

    # Use os.walk to be consistent with ISI function
    for root, dirs, files in os.walk(dir_path):
        for name in files:
            if not name.endswith('.pkl'): 
                continue

            cell_id = convert_pkl_filename_to_cell_id(name)
            if cell_id is None: 
                continue
            
            # FILTER: Skip if not in master_df (e.g., filtered by Inclusion criteria)
            if valid_ids is not None and cell_id not in valid_ids:
                continue

            try:
                with open(os.path.join(root, name), 'rb') as f:
                    data = dill.load(f)

                if 'analysis_dict' not in data: 
                    continue

                FI_plot_data_unique = {}
                
                for analysis in data['analysis_dict']:
                    coarse_f_I = analysis.get('Coarse_FI') or analysis.get('IV_stim')
                    if not coarse_f_I: 
                        continue
                    
                    # Round currents
                    current_amplitudes = np.round(np.unique(coarse_f_I['current_amplitudes']), 1)
                    firing_rates = np.unique(coarse_f_I['firing_rates'])

                    # Average rates if multiple sweeps exist for same current
                    for amp, rate in zip(current_amplitudes, firing_rates):
                        if amp in FI_plot_data_unique:
                            FI_plot_data_unique[amp] = (FI_plot_data_unique[amp] + rate) / 2
                        else:
                            FI_plot_data_unique[amp] = rate

                if FI_plot_data_unique:
                    FI_data[cell_id] = FI_plot_data_unique
            except Exception as e:
                print(f"Error processing {name}: {e}")
                continue

    return FI_data

def process_and_merge_FI_data(FI_data, experiment_master_df, genotype_label=None):
    """
    Convert FI_data dictionary to DataFrame and merge with master_df.
    """
    data = []
    for cell_id, currents in FI_data.items():
        for current, firing_rate in currents.items():
            data.append({
                'Cell_ID': cell_id,  
                'Current_Amplitude': current,  
                'Firing_Rate': firing_rate
            })
    
    FI_df = pd.DataFrame(data)
    
    if FI_df.empty:
        print("WARNING: No firing rate data extracted.")
        return pd.DataFrame(columns=['Cell_ID', 'Current_Amplitude', 'Firing_Rate', 
                                    'Genotype', 'Sex', 'Holding Voltage', 'Slice Solution'])
    
    FI_df['Cell_ID'] = FI_df['Cell_ID'].astype(str)
    experiment_master_df_copy = experiment_master_df.copy()
    experiment_master_df_copy['Cell_ID'] = experiment_master_df_copy['Cell_ID'].astype(str)
    
    # Merge available columns
    merge_cols = ['Cell_ID']
    potential_cols = ['Sex', 'Genotype', 'Holding Voltage', 'Slice Solution', 'Animal_ID']
    
    for col in potential_cols:
        if col in experiment_master_df_copy.columns:
            merge_cols.append(col)
    
    merged_df = FI_df.merge(experiment_master_df_copy[merge_cols], on='Cell_ID', how='left')
    
    if genotype_label and 'Genotype' not in merged_df.columns:
        merged_df['Genotype'] = genotype_label
    
    return merged_df

def calculate_FI_slopes(FI_data):
    """
    Calculate the slope of F-I curves (rates > 0).
    """
    slopes_dict = {}
    skipped_cells = 0
    
    for cell_id, currents_dict in FI_data.items():
        firing_rates = []
        current_amplitudes = []
        
        for current_amplitude, rate in currents_dict.items():
            amp = float(current_amplitude)
            if rate > 0:
                firing_rates.append(rate)
                current_amplitudes.append(amp)
        
        if len(firing_rates) >= 2:
            slope, intercept, r_value, p_value, std_err = linregress(current_amplitudes, firing_rates)
            slopes_dict[cell_id] = slope
        else:
            slopes_dict[cell_id] = np.nan
            skipped_cells += 1
    
    print(f"FI Slopes calculated for {len(slopes_dict) - skipped_cells} cells. Skipped {skipped_cells} cells.")
    return slopes_dict

# ISI FUNCTIONS

def get_FI_ISI_times(dir_path, master_df=None):
    """
    Extracts ISI times with specific filtering for Adaptation Analysis (Panel I).
    
    Logic:
    1. Filter by Master DF Inclusion.
    2. Filter cells that NEVER fire > 20Hz (Global Exclusion).
    3. Filter Current Amplitudes: Only consider [50, 100, ... 400] pA.
    4. "Step 2" Sweep Selection: For each cell, select ONLY the sweep corresponding 
       to the SMALLEST Current Amplitude that elicited at least 6 AP Intervals (7 Spikes).
    
    Returns:
        ISI_data: Dict {cell_id: {current: {ap_num: isi_val}}} (Contains only the winning sweep)
        excluded_cells: List of cells excluded based on firing rate criteria.
    """
    # 1. Configuration
    TARGET_AMPLITUDES = [50.0, 100.0, 150.0, 200.0, 250.0, 300.0, 350.0, 400.0]
    REQUIRED_AP_INTERVAL = 6 # We need the 6th ISI to exist (i.e., at least 7 spikes)
    
    candidate_data = {} # Temp storage: {cell: {amp: {ap_key: isi}}}
    ISI_data = {}       # Final storage: Only the winning sweep per cell
    excluded_cells = []

    # 2. Setup Valid IDs
    valid_ids = None
    if master_df is not None and 'Cell_ID' in master_df.columns:
        valid_ids = set(master_df['Cell_ID'].astype(str))

    # 3. Extraction Loop
    for path, subdirs, files in os.walk(dir_path):
        for name in files:
            if not name.endswith('.pkl'):
                continue
            
            full_cell_name = convert_pkl_filename_to_cell_id(name)
            if full_cell_name is None:
                continue

            if valid_ids is not None and full_cell_name not in valid_ids:
                continue

            try:
                file_path = os.path.join(path, name)
                data_file = pd.read_pickle(file_path)
                analysis_data = data_file.get('analysis_dict', [])

                cell_firing_rates_above_20 = False
                
                # We collect ALL valid target sweeps first, then select the best one later
                if full_cell_name not in candidate_data:
                    candidate_data[full_cell_name] = {}

                for analysis in analysis_data:
                    # Protocol Check
                    if 'Coarse_FI' in analysis.keys() and 'AP' in analysis.keys():
                        stim = analysis['Coarse_FI']
                    elif 'IV_stim' in analysis.keys() and 'AP' in analysis.keys():
                        stim = analysis['IV_stim']
                    else:
                        continue

                    currents = np.unique(np.round(stim['current_amplitudes'], 1))
                    firing_rates = stim['firing_rates']
                    ISI_times = analysis['AP']['AP_ISI_time']

                    # Check >20Hz Exclusion Criteria
                    if isinstance(firing_rates, (list, np.ndarray)):
                        if any(rate > 20 for rate in firing_rates):
                            cell_firing_rates_above_20 = True
                    elif firing_rates > 20:
                        cell_firing_rates_above_20 = True

                    # Extract Data
                    for amplitude in currents:
                        # --- FILTER 1: Target Amplitudes Only ---
                        if amplitude not in TARGET_AMPLITUDES:
                            continue

                        # Initialize dicts
                        if amplitude not in candidate_data[full_cell_name]:
                            candidate_data[full_cell_name][amplitude] = {}

                        for i, isi_val in enumerate(ISI_times):
                            ap_key = i + 1
                            # Skip 1st ISI (often artifactual/bursty start)
                            if ap_key >= 2:
                                if ap_key not in candidate_data[full_cell_name][amplitude]:
                                    candidate_data[full_cell_name][amplitude][ap_key] = []
                                candidate_data[full_cell_name][amplitude][ap_key].append(isi_val)

                if not cell_firing_rates_above_20:
                    excluded_cells.append(full_cell_name)
                    del candidate_data[full_cell_name]

            except Exception as e:
                print(f"Error processing ISI for {name}: {e}")
                continue

    # 4. "Step 2" Logic: Select Smallest Current with 6 APs per Cell
    for cell_name, amps_dict in candidate_data.items():
        # Average the sweeps first (if duplicates exist)
        avg_amps_dict = {}
        for amp, ap_data in amps_dict.items():
            avg_amps_dict[amp] = {}
            for ap_key, val_list in ap_data.items():
                avg_amps_dict[amp][ap_key] = np.nanmean(val_list)
        
        # Sort amplitudes low -> high
        sorted_amps = sorted(avg_amps_dict.keys())
        
        winning_amp = None
        
        # Find smallest amp where AP #6 exists
        for amp in sorted_amps:
            if REQUIRED_AP_INTERVAL in avg_amps_dict[amp]:
                winning_amp = amp
                break
        
        # If we found a suitable sweep, store ONLY that one
        if winning_amp is not None:
            ISI_data[cell_name] = {winning_amp: avg_amps_dict[winning_amp]}
    
    print(f"\nCells excluded (never fired > 20Hz): {len(excluded_cells)}")
    print(f"Cells retained for ISI analysis (reached {REQUIRED_AP_INTERVAL} intervals): {len(ISI_data)}")
    
    return ISI_data, excluded_cells

def convert_ISI_to_list_format(ISI_data):
    """
    Flatten ISI data into a single sorted list of all ISIs per cell.
    """
    ISI_lists = {}
    for cell_id in ISI_data:
        all_isi_times = []
        for amp in ISI_data[cell_id]:
            for isi_num, val in ISI_data[cell_id][amp].items():
                if not pd.isna(val):
                    all_isi_times.append(val)
        ISI_lists[cell_id] = sorted(all_isi_times)
    return ISI_lists

def calculate_ISI_adaptation_slopes(ISI_data):
    """
    Calculate adaptation slope based on the first 7 ISIs found across all sweeps.
    """
    ISI_slopes = {}
    for cell_id in ISI_data:
        all_isi_times = []
        for amp in ISI_data[cell_id]:
            for isi_num, val in ISI_data[cell_id][amp].items():
                if not pd.isna(val):
                    all_isi_times.append(val)
        
        sorted_times = sorted(all_isi_times)
        
        if len(sorted_times) >= 7:
            slope = np.nanmean(np.diff(sorted_times[:7]))
            ISI_slopes[cell_id] = slope
        else:
            ISI_slopes[cell_id] = np.nan
    return ISI_slopes

def format_FI_data_for_plotting(FI_df, ISI_lists=None, ISI_slopes=None):
    """
    Aggregates data into one row per cell with list-columns for plotting.
    """
    if FI_df.empty: return pd.DataFrame()
    
    agg_dict = {
        'Current_Amplitude': lambda x: list(x),
        'Firing_Rate': lambda x: list(x),
        'Genotype': 'first',
        'Sex': 'first',
    }
    for col in ['Holding Voltage', 'Slice Solution']:
        if col in FI_df.columns: agg_dict[col] = 'first'
    
    plotting_df = FI_df.groupby('Cell_ID').agg(agg_dict).reset_index()
    plotting_df = plotting_df.rename(columns={'Current_Amplitude': 'Currents_List', 'Firing_Rate': 'Firing_Rates_List'})
    
    def sort_pairs(row):
        c, r = row['Currents_List'], row['Firing_Rates_List']
        pairs = sorted(zip(c, r), key=lambda x: x[0])
        row['Currents_List'] = [p[0] for p in pairs]
        row['Firing_Rates_List'] = [p[1] for p in pairs]
        return row
        
    plotting_df = plotting_df.apply(sort_pairs, axis=1)
    
    # if FI_slopes: plotting_df['FI_Slope'] = plotting_df['Cell_ID'].map(FI_slopes)
    if ISI_lists: plotting_df['ISI_Times_List'] = plotting_df['Cell_ID'].map(ISI_lists)
    if ISI_slopes: plotting_df['ISI_Adaptation_Slope'] = plotting_df['Cell_ID'].map(ISI_slopes)
        
    return plotting_df

def FI_sigmoid(x, L, x0, k, b):
    # L: max value (amplitude/saturation rate)
    # x0: midpoint (x value where y = L/2 + b)
    # k: steepness (slope factor)
    # b: baseline (starting rate)
    return L / (1 + np.exp(-k * (x - x0))) + b

def analyze_fi_midpoint(FI_plotting_format):

    script_dir = os.path.dirname(os.path.abspath(__file__))
    if os.path.exists(os.path.join(script_dir, '../paper_data')):
        project_root = os.path.abspath(os.path.join(script_dir, '..'))
    elif os.path.exists(os.path.join(script_dir, 'paper_data')):
        project_root = script_dir
    else:
        project_root = os.getcwd()
    # Robust path handling
    FI_df = FI_plotting_format
    # Create directory for fit plots
    fit_plot_dir = os.path.join(project_root, 'paper_data/Firing_Rate/FI_Fits')
    if not os.path.exists(fit_plot_dir):
        os.makedirs(fit_plot_dir)
    
    # --- PART 1: Sigmoid Fitting & Midpoint Analysis ---
    sigmoid_results = []
    
    # We will pick a few random examples to check fit quality
    examples_to_plot = {'WT': 0, 'GNB1': 0}
    max_examples = 5
    
    for index, row in FI_df.iterrows():
        try:
            # Handle both string and list input
            raw_currents = row['Currents_List']
            raw_rates = row['Firing_Rates_List']
            if isinstance(raw_currents, str):
                currents = np.array(ast.literal_eval(raw_currents))
            else:
                currents = np.array(raw_currents)
            if isinstance(raw_rates, str):
                rates = np.array(ast.literal_eval(raw_rates))
            else:
                rates = np.array(raw_rates)
            
            currents = currents.astype(float)
            rates = rates.astype(float)
            mask = ~np.isnan(currents) & ~np.isnan(rates)
            currents = currents[mask]
            rates = rates[mask]
            
            if len(currents) < 4: continue

            # Initial guess
            L_guess = np.max(rates) - np.min(rates)
            x0_guess = np.mean(currents)
            k_guess = 0.01
            b_guess = np.min(rates)
            p0 = [L_guess, x0_guess, k_guess, b_guess]
            bounds = ([0, -np.inf, 0, 0], [np.inf, np.inf, np.inf, np.inf])

            try:
                popt, pcov = curve_fit(FI_sigmoid, currents, rates, p0=p0, bounds=bounds, maxfev=5000)
                genotype = row['Genotype'].strip()
                
                # Check R-squared
                residuals = rates - FI_sigmoid(currents, *popt)
                ss_res = np.sum(residuals**2)
                ss_tot = np.sum((rates - np.mean(rates))**2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                
                sigmoid_results.append({
                    'Cell_ID': row['Cell_ID'],
                    'Genotype': genotype,
                    'Midpoint': popt[1],
                    'Max_Rate': popt[0],
                    'Slope_k': popt[2],
                    'Baseline': popt[3],
                    'R_squared': r_squared
                })
                
                # Plot example fits
                if examples_to_plot[genotype] < max_examples:
                    plt.figure(figsize=(5, 4))
                    plt.scatter(currents, rates, color='black', label='Data')
                    
                    x_fit = np.linspace(min(currents), max(currents), 100)
                    y_fit = FI_sigmoid(x_fit, *popt)
                    
                    color = 'black' if genotype == 'WT' else 'red'
                    plt.plot(x_fit, y_fit, color=color, label=f'Fit (R2={r_squared:.2f})')
                    
                    # Mark midpoint
                    midpoint = popt[1]
                    half_max = popt[0]/2 + popt[3]
                    if min(currents) <= midpoint <= max(currents):
                        plt.scatter([midpoint], [half_max], color='blue', zorder=5, marker='x', s=100, label=f'Midpoint: {midpoint:.1f}')
                    
                    plt.title(f"{genotype} Cell: {row['Cell_ID']}")
                    plt.xlabel('Current (pA)')
                    plt.ylabel('Rate (Hz)')
                    plt.legend()
                    plt.tight_layout()
                    plt.savefig(os.path.join(fit_plot_dir, f"Fit_{genotype}_{row['Cell_ID']}.png"))
                    plt.close()
                    examples_to_plot[genotype] += 1
                    
            except Exception as e:
                continue
        except Exception as e:
            continue

    sigmoid_df = pd.DataFrame(sigmoid_results)
    
    if sigmoid_df.empty:
        print("WARNING: No sigmoid fits were successful.")
        return pd.DataFrame(columns=['Cell_ID', 'Genotype', 'Midpoint', 'Max_Rate', 'Slope_k', 'Baseline', 'R_squared'])
    
    # Save params to CSV for user inspection
    sigmoid_df.to_csv(os.path.join(project_root, 'paper_data/Firing_Rate/Sigmoid_Fit_Params.csv'), index=False)
    print(f"\nSaved all fit parameters to paper_data/Firing_Rate/Sigmoid_Fit_Params.csv")
    print(f"Saved example fit plots to paper_data/Firing_Rate/FI_Fits/")
    
    # Filter outliers
    sigmoid_df = sigmoid_df[np.abs(sigmoid_df['Midpoint']) < 5000]

    wt_sigmoid = sigmoid_df[sigmoid_df['Genotype'] == 'WT']
    gnb1_sigmoid = sigmoid_df[sigmoid_df['Genotype'] == 'GNB1']

    if len(wt_sigmoid) > 0 and len(gnb1_sigmoid) > 0:
        midpoint_stat, midpoint_p = stats.ttest_ind(wt_sigmoid['Midpoint'], gnb1_sigmoid['Midpoint'], nan_policy='omit')
    else:
        midpoint_p = np.nan
    
    print("\n--- Sigmoid Fit Parameters Summary ---")
    print(f"WT Midpoint: {wt_sigmoid['Midpoint'].mean():.2f} +/- {wt_sigmoid['Midpoint'].sem():.2f} (n={len(wt_sigmoid)})")
    print(f"GNB1 Midpoint: {gnb1_sigmoid['Midpoint'].mean():.2f} +/- {gnb1_sigmoid['Midpoint'].sem():.2f} (n={len(gnb1_sigmoid)})")
    print(f"Comparison p-value: {midpoint_p:.4f}")
    
    print("\nFit Quality (R-squared):")
    print(f"WT Mean R2: {wt_sigmoid['R_squared'].mean():.3f}")
    print(f"GNB1 Mean R2: {gnb1_sigmoid['R_squared'].mean():.3f}")

    # Rename Midpoint column to FI_Midpoint for consistency with generate_figures.py
    sigmoid_df = sigmoid_df.rename(columns={'Midpoint': 'FI_Midpoint'})

    return sigmoid_df

def analyze_and_export_FI_and_ISI_data(master_df, data_dir, output_path=None):
    """
    Master workflow function.
    """
    if output_path is None: output_path = 'paper_data/firing_rates'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    results = {}
    
    print(">>> Extracting Firing Rates...")
    FI_data = get_firing_rate_data(data_dir, master_df=master_df)
    
    if not FI_data:
        print("No FI data found.")
        return results

    print(">>> Merging Metadata...")
    FI_df = process_and_merge_FI_data(FI_data, experiment_master_df=master_df)
    results['FI_long_format'] = FI_df
    FI_df.to_csv(f"{output_path}_long_format.csv", index=False)

    print(">>> Creating Plotting Format...")
    plotting_df = format_FI_data_for_plotting(FI_df)
    plotting_df.to_csv(f"{output_path}_plotting_format.csv", index=False)
    results['plotting_format'] = plotting_df
    
    print(">>> Calculating Slopes...")
    FI_slopes = calculate_FI_slopes(FI_data)
    results['FI_slopes'] = FI_slopes

    #get FI midpoint data
    #need to get data in plotting format first
    print(">>> Getting FI Midpoints...")
    FI_midpoints = analyze_fi_midpoint(plotting_df)
    results['FI_midpoints'] = FI_midpoints
    FI_midpoints.to_csv(f"{output_path}_midpoints.csv", index=False)
    
    print(">>> Extracting ISI Data...")
    ISI_data, excluded = get_FI_ISI_times(data_dir, master_df=master_df)
    ISI_lists = convert_ISI_to_list_format(ISI_data)
    ISI_slopes = calculate_ISI_adaptation_slopes(ISI_data)
    results['ISI_lists'] = ISI_lists
    results['ISI_slopes'] = ISI_slopes

    # Add ISI data to plotting_df and re-save
    print(">>> Adding ISI data to plotting format...")
    plotting_df_with_isi = format_FI_data_for_plotting(FI_df, ISI_lists=ISI_lists, ISI_slopes=ISI_slopes)
    plotting_df_with_isi.to_csv(f"{output_path}_plotting_format.csv", index=False)
    results['plotting_format'] = plotting_df_with_isi
    
    print(f"\nDone! Files saved to {output_path}*")
    return results

#-------------------------------------------------------------------------------------------------------------------
#Analysis of Intrinsic Properties: Input Resistance, Voltage Sag, Vm Rest, Access Resistance
#--------------------------------------------------------------------------------------------------------------------

def get_intrinsic_properties_by_cell(data_dir, cell_properties_to_plot, master_df=None):
    """
    Extract intrinsic cell properties.
    Args:
        master_df: (Optional) If provided, only loads cells present in this DF.
    """
    data_files = [file for file in os.listdir(data_dir) if file.endswith('.pkl')]
    
    intrinsic_cell_properties = {}
    
    # Filter Setup
    valid_ids = None
    if master_df is not None and 'Cell_ID' in master_df.columns:
        valid_ids = set(master_df['Cell_ID'].astype(str))

    for data_file in data_files:
        cell_id = convert_pkl_filename_to_cell_id(data_file)
        if cell_id is None: continue
        
        # FILTER CHECK
        if valid_ids is not None and cell_id not in valid_ids:
            continue
        
        try:
            current_data_df = pd.read_pickle(os.path.join(data_dir, data_file))
            
            intrinsic_cell_properties[cell_id] = {}
            
            for i in range(len(current_data_df)):
                if 'Intrinsic_cell' in current_data_df['analysis_dict'][i]:
                    if current_data_df['analysis_dict'][i]['Intrinsic_cell'] is not None:
                        current_intrinsics = current_data_df['analysis_dict'][i]['Intrinsic_cell']
                        
                        for key in cell_properties_to_plot:
                            if key not in intrinsic_cell_properties[cell_id]:
                                intrinsic_cell_properties[cell_id][key] = []
                            
                            if key in current_intrinsics and current_intrinsics[key] is not None:
                                intrinsic_cell_properties[cell_id][key].append(current_intrinsics[key])
                            else:
                                intrinsic_cell_properties[cell_id][key].append(np.nan)
        except Exception as e:
            continue 

    # Average across properties per cell
    try:
        average_intrinsic_cell_properties = {}
        for cell_id in intrinsic_cell_properties:
            average_intrinsic_cell_properties[cell_id] = {}
            for key in intrinsic_cell_properties[cell_id]:
                average_intrinsic_cell_properties[cell_id][key] = np.nanmean(intrinsic_cell_properties[cell_id][key])
        return average_intrinsic_cell_properties
    except Exception as e:
        print('Error in extracting intrinsic properties:', str(e))
        return None


def get_vm_and_rin_from_test_pulses(data_dir, master_df=None,
                                    vm_rest_threshold=-40,
                                    pulse_amp_pA=-50,
                                    pulse_amp_tolerance_pA=10,
                                    min_pulse_samples=500):
    """
    Compute Resting Membrane Potential (Vm_rest) and Input Resistance (Rin)
    from the SAME test-pulse sweeps in each cell's pkl file.

    Both metrics are derived from the identical set of sweeps, guaranteeing
    that Vm and Rin describe the same recording epoch.

    Strategy
    --------
    For every sweep in the pkl, inspect `stim_command` for a sustained
    negative current step (test pulse) near `pulse_amp_pA`.  If found:
      - Vm_rest  = mean voltage of the 50 ms pre-pulse baseline  (mV)
      - Rin      = (delta_Vm_steady / |pulse_amp|) * 1000         (MOhm)
    Then average both across all valid test-pulse sweeps for the cell.

    Parameters
    ----------
    data_dir              : str   - directory with per-cell .pkl files
    master_df             : DataFrame or None - if given, restrict to these cells
    vm_rest_threshold     : float - exclude sweeps where Vm_rest >= this (mV)
    pulse_amp_pA          : float - expected test-pulse amplitude (default -50 pA)
    pulse_amp_tolerance_pA: float - tolerance around expected amplitude (pA)
    min_pulse_samples     : int   - min consecutive samples at pulse level

    Returns
    -------
    dict  { cell_id: {'Vm_rest_mV': float, 'Input_Resistance_MOhm': float,
                       'n_sweeps': int} }
    """
    valid_ids = None
    if master_df is not None and 'Cell_ID' in master_df.columns:
        valid_ids = set(master_df['Cell_ID'].astype(str))

    results = {}
    pkl_files = [f for f in os.listdir(data_dir) if f.endswith('.pkl')]

    for pkl_file in pkl_files:
        cell_id = convert_pkl_filename_to_cell_id(pkl_file)
        if cell_id is None:
            continue
        if valid_ids is not None and cell_id not in valid_ids:
            continue

        try:
            df = pd.read_pickle(os.path.join(data_dir, pkl_file))
        except Exception:
            continue

        vm_vals  = []
        rin_vals = []

        for _, row in df.iterrows():
            sweep_raw = row.get('sweep', None)
            sc_raw    = row.get('stim_command', None)
            acq_freq  = float(row.get('acquisition_frequency', 20000))
            dt_ms     = 1000.0 / acq_freq

            if sweep_raw is None or sc_raw is None:
                continue

            try:
                sweep = np.array(sweep_raw, dtype=float)
                sc    = np.array(
                    sc_raw[0] if isinstance(sc_raw, (list, tuple)) else sc_raw,
                    dtype=float)
            except Exception:
                continue

            if sweep.ndim != 1 or sc.ndim != 1 or len(sweep) != len(sc):
                continue

            # --- Detect test pulse: longest consecutive run at expected amplitude ---
            at_pulse = np.abs(sc - pulse_amp_pA) < pulse_amp_tolerance_pA
            runs, in_run, run_start = [], False, 0
            for idx in range(len(at_pulse)):
                if at_pulse[idx] and not in_run:
                    in_run, run_start = True, idx
                elif not at_pulse[idx] and in_run:
                    in_run = False
                    runs.append((run_start, idx - 1))
            if in_run:
                runs.append((run_start, len(at_pulse) - 1))

            if not runs:
                continue

            p_start, p_end = max(runs, key=lambda r: r[1] - r[0])
            if (p_end - p_start + 1) < min_pulse_samples:
                continue

            detected_amp = float(np.median(sc[p_start:p_end + 1]))

            # --- Vm_rest: mean of pre-pulse baseline (start to 5 ms before pulse) ---
            baseline_end = max(0, p_start - int(5 / dt_ms))
            if baseline_end < 5:
                continue
            vm_rest = float(np.mean(sweep[0:baseline_end]))

            if vm_rest >= vm_rest_threshold:   # exclude depolarised / unhealthy cells
                continue

            # --- Rin: steady-state deflection (last 50% of pulse, -2 ms guard) ---
            pulse_samples = p_end - p_start + 1
            ss_start = p_start + int(pulse_samples * 0.5)
            ss_end   = max(ss_start + 1, p_end - int(2 / dt_ms))
            if ss_end <= ss_start:
                continue

            steady_v = float(np.mean(sweep[ss_start:ss_end]))
            delta_v  = steady_v - vm_rest

            if detected_amp == 0:
                continue
            rin = (delta_v / detected_amp) * 1000.0   # MOhm

            if rin <= 0:   # sanity check (negative pulse -> negative delta -> positive Rin)
                continue

            vm_vals.append(vm_rest)
            rin_vals.append(rin)

        if vm_vals:
            results[cell_id] = {
                'Vm_rest_mV':            float(np.nanmean(vm_vals)),
                'Input_Resistance_MOhm': float(np.nanmean(rin_vals)),
                'n_sweeps':              len(vm_vals)
            }

    n_sweeps_total = sum(v['n_sweeps'] for v in results.values())
    print(f"  [Test-pulse Vm/Rin] {len(results)} cells, {n_sweeps_total} sweeps used")
    return results

def calculate_input_resistance_from_test_pulse(trace, sampling_rate=20000, 
                                               pulse_start_ms=50, 
                                               pulse_duration_ms=100, 
                                               pulse_amp_pA=-50):
    """
    Calculates Input Resistance (Rim) from a test pulse at the start of a trace.
    
    Args:
        trace: Voltage trace (mV) - numpy array
        sampling_rate: Hz (default 20000)
        pulse_start_ms: Start time of the pulse (default 50ms)
        pulse_duration_ms: Duration of the pulse (default 100ms)
        pulse_amp_pA: Amplitude of current injection (default -50pA)
    
    Returns:
        rim_mohm: Input resistance in MΩ
    """
    # Define timing windows
    dt_ms = 1000 / sampling_rate
    
    # Baseline: 0ms to 5ms before pulse start
    base_start_idx = 0
    base_end_idx = int((pulse_start_ms - 5) / dt_ms)
    
    # Steady State: Last 50% of the pulse (avoid onset transient)
    pulse_end_ms = pulse_start_ms + pulse_duration_ms
    steady_start_ms = pulse_start_ms + (pulse_duration_ms * 0.5)
    
    steady_start_idx = int(steady_start_ms / dt_ms)
    steady_end_idx = int((pulse_end_ms - 2) / dt_ms) # End slighty before pulse off
    
    # Safety checks
    if len(trace) < steady_end_idx:
        return np.nan
    if base_end_idx <= base_start_idx:
        return np.nan
        
    baseline_v = np.mean(trace[base_start_idx:base_end_idx])
    steady_v = np.mean(trace[steady_start_idx:steady_end_idx])
    
    delta_v_mV = steady_v - baseline_v
    
    # Calculate Resistance (V=IR -> R=V/I)
    # Unit Analysis:
    # R (MΩ) = (V (mV) / I (pA)) * 1000
    # Proof: (10^-3 V) / (10^-12 A) = 10^9 Ω = 1000 MΩ
    
    if pulse_amp_pA == 0:
        return np.nan
        
    rim_mohm = (delta_v_mV / pulse_amp_pA) * 1000
    
    # Validation: Resistance must be positive. 
    # If pulse is negative (-50), delta V should be negative. Ratio is positive.
    # If pulse is positive, delta V is positive. Ratio is positive.
    # If calculated Rim is negative, something is wrong (e.g., rebound spike, huge noise).
    
    return rim_mohm

def export_intrinsic_properties_to_dataframe(intrinsic_props_dict, master_df, 
                                            properties_to_export=None,
                                            vm_rest_threshold=-40):
    """Export intrinsic properties to a DataFrame with metadata."""
    if intrinsic_props_dict is None or len(intrinsic_props_dict) == 0:
        print("WARNING: No intrinsic properties data provided")
        return pd.DataFrame()
    
    master_df_copy = master_df.copy()
    master_df_copy['Cell_ID'] = master_df_copy['Cell_ID'].astype(str)
    
    genotype_lookup = dict(zip(master_df_copy['Cell_ID'], master_df_copy['Genotype']))
    sex_lookup = dict(zip(master_df_copy['Cell_ID'], master_df_copy['Sex']))
    
    # Prepare Vm rest from master_df
    vm_rest_lookup = {}
    if 'Vm rest/start (mV)' in master_df_copy.columns:
        master_df_copy['Vm rest/start (mV)'] = pd.to_numeric(master_df_copy['Vm rest/start (mV)'], errors='coerce')
        vm_rest_series = master_df_copy['Vm rest/start (mV)'].copy()
        vm_rest_series[vm_rest_series > vm_rest_threshold] = np.nan
        vm_rest_lookup = dict(zip(master_df_copy['Cell_ID'], vm_rest_series))
    
    # Prepare Access Resistance from master_df
    access_resistance_lookup = {}
    if 'Access Resistance (From Whole Cell V-Clamp)' in master_df_copy.columns:
        master_df_copy['Access Resistance (From Whole Cell V-Clamp)'] = pd.to_numeric(
            master_df_copy['Access Resistance (From Whole Cell V-Clamp)'], errors='coerce'
        )
        access_resistance_lookup = dict(zip(
            master_df_copy['Cell_ID'], 
            master_df_copy['Access Resistance (From Whole Cell V-Clamp)']
        ))
    
    rows = []
    for cell_id, properties in intrinsic_props_dict.items():
        genotype = genotype_lookup.get(cell_id, 'Unknown')
        sex = sex_lookup.get(cell_id, 'Unknown')
        
        row = {'Cell_ID': cell_id, 'Genotype': genotype, 'Sex': sex}
        
        properties_to_add = properties_to_export if properties_to_export is not None else properties.keys()
        
        for prop in properties_to_add:
            if prop in properties:
                value = properties[prop]
                row[prop] = value[0] if isinstance(value, (list, np.ndarray)) and len(value) > 0 else value
            else:
                row[prop] = np.nan
        
        row['Vm rest/start (mV)'] = vm_rest_lookup.get(cell_id, np.nan)
        row['Access Resistance (From Whole Cell V-Clamp)'] = access_resistance_lookup.get(cell_id, np.nan)
        
        rows.append(row)
    
    if len(rows) == 0: return pd.DataFrame()
    
    df = pd.DataFrame(rows)
    metadata_cols = ['Cell_ID', 'Genotype', 'Sex']
    intrinsic_property_cols = [col for col in df.columns if col not in metadata_cols and col not in ['Vm rest/start (mV)', 'Access Resistance (From Whole Cell V-Clamp)']]
    master_df_cols = ['Vm rest/start (mV)', 'Access Resistance (From Whole Cell V-Clamp)']
    
    # Check if cols exist before selecting
    final_cols = metadata_cols + intrinsic_property_cols
    for c in master_df_cols:
        if c in df.columns: final_cols.append(c)
        
    return df[final_cols]

def analyze_and_export_intrinsic_properties(master_df, data_dir, output_path=None,
                                            properties_to_extract=None,
                                            vm_rest_threshold=-40):
    """
    Extracts, converts, and exports intrinsic properties.
    """
    if properties_to_extract is None:
        properties_to_extract = ['Input_Resistance', 'Voltage_Sag']
    
    print("\n--- INTRINSIC PROPERTIES ANALYSIS ---")
    print(f"Extracting: {properties_to_extract}")
    
    # 1. Extract
    intrinsic_props = get_intrinsic_properties_by_cell(data_dir, properties_to_extract, master_df)
    
    if intrinsic_props is None or len(intrinsic_props) == 0:
        print("WARNING: No intrinsic properties extracted.")
        return pd.DataFrame()
    
    # 2. Convert to DataFrame
    combined_df = export_intrinsic_properties_to_dataframe(
        intrinsic_props, 
        master_df, 
        properties_to_export=properties_to_extract,
        vm_rest_threshold=vm_rest_threshold
    )

    # 4. Export
    if output_path:
        combined_df.to_csv(output_path, index=False)
        print(f"✓ Data exported to: {output_path}")
    
    return combined_df

#----------------------------------------------------------------------------------
# Analysis of E-I data and EPSP Properties
#----------------------------------------------------------------------------------

# =============================================================================
# TRACE QUALITY FILTER: Minimum trace length for E/I analysis
# =============================================================================
# Traces must be at least 4000 samples (200ms at 20kHz) to be valid.
# 
# Rationale:
# - Very short traces (e.g., 200 samples = 10ms) cannot capture full EPSP waveform
# - Short traces may indicate failed recordings, acquisition errors, or incomplete data
# - Standard E/I traces are 6200 samples (310ms) or 4200 samples (210ms)
# - Anything below 4000 samples is likely problematic and excluded
# =============================================================================
MIN_EI_TRACE_LENGTH = 4000  # samples (200ms at 20kHz)

def get_E_I_traces(data_dir, unitary_stim_starts_dict, ISI_times_dict_mapping, master_df):
    """
    Extract E-I traces from .pkl files with detailed debugging.
    
    NOTE: Traces shorter than MIN_EI_TRACE_LENGTH (4000 samples) are excluded
    as they cannot reliably capture EPSP waveforms.
    
    CRITICAL: Only cells with 'E/I' explicitly in Inclusion column are processed.
    Cells with only 'Unitary Gabazine' experiments are excluded.
    """

    # Create lookup for which ISI_times_dict to use per cell
    master_df_copy = master_df.copy()
    master_df_copy['Cell_ID'] = master_df_copy['Cell_ID'].astype(str)
    
    # BUILD E/I INCLUSION SET - Trust the passed master_df (caller filters it)
    # Previously, we strictly required 'E/I' in the Inclusion column. 
    # However, this excluded valid cells that were marked 'Yes' but not 'E/I'.
    # Now we rely on the caller's filtering (e.g., Inclusion='Yes').
    
    ei_inclusion_set = set(master_df_copy['Cell_ID'])
    
    print(f"  E/I Inclusion Filter: {len(ei_inclusion_set)} cells (Trusting passed master_df)")
    
    stim_file_lookup = {}
    genotype_lookup = {}
    
    for _, row in master_df_copy.iterrows():
        cell_id = row['Cell_ID']
        genotype_lookup[cell_id] = row.get('Genotype', 'Unknown')
        stim_file = row.get('ESPS Stim Time File Name', None)
        
        if pd.notna(stim_file):
            if 'newer' in str(stim_file).lower() or 'new' in str(stim_file).lower():
                stim_file_lookup[cell_id] = 'newer'
            elif 'older' in str(stim_file).lower() or 'old' in str(stim_file).lower():
                stim_file_lookup[cell_id] = 'older'
            else:
                stim_file_lookup[cell_id] = 'older'
        else:
            stim_file_lookup[cell_id] = 'older'
    
    data_files = [file for file in os.listdir(data_dir) if file.endswith('.pkl')]
    E_I_data_traces = {}
    
    skipped_no_ei_inclusion = 0
    
    for data_file in data_files:
        cell_id = convert_pkl_filename_to_cell_id(data_file)
        if cell_id is None:
            continue
        
        # CRITICAL CHECK: Only process cells with 'E/I' in Inclusion column
        if cell_id not in ei_inclusion_set:
            skipped_no_ei_inclusion += 1
            continue
        
        genotype = genotype_lookup.get(cell_id, 'Unknown')
        stim_version = stim_file_lookup.get(cell_id, 'older')
        unitary_stim_starts = unitary_stim_starts_dict[stim_version]
        
        # --- PATCH: Check for Theta_Burst_MCIII_new_variant_2 (Special E:I Timings) ---
        # User specified: Ch1=[500, 600, 700], Ch2=[1500, 1600, 1700] for 300ms ISI
        # This overrides standard older/newer lookups
        row_meta = master_df[master_df['Cell_ID'].astype(str) == str(cell_id)]
        if not row_meta.empty:
             theta_col = str(row_meta.iloc[0].get('Theta Burst Stim Time File Name', ''))
             if 'Theta_Burst_MCIII_new_variant_2' in theta_col:
                 # print(f"DEBUG: Detected Variant 2 for {cell_id}. Using special E:I timings.")
                 unitary_stim_starts = {
                     'channel_1': [500.0, 600.0, 700.0],
                     'channel_2': [1500.0, 1600.0, 1700.0]
                 }
        
        current_data_df = pd.read_pickle(os.path.join(data_dir, data_file))
        
        # Count traces extracted for this cell - use defaultdict to handle any condition
        from collections import defaultdict
        traces_extracted = defaultdict(int)

        for i in range(len(current_data_df)):
            stimulus_metadata = current_data_df['stimulus_metadata_dict'].iloc[i]
            if not stimulus_metadata or 'ISI' not in stimulus_metadata:
                continue
                
            ISI_value = stimulus_metadata['ISI']
            if not ISI_value or ISI_value == 'nan' or pd.isna(ISI_value):
                continue
                
            ISI_time = int(float(ISI_value))
            condition = stimulus_metadata['condition']
            
            # Initialize E_I_data_traces
            if cell_id not in E_I_data_traces:
                E_I_data_traces[cell_id] = {}
            if ISI_time not in E_I_data_traces[cell_id]:
                E_I_data_traces[cell_id][ISI_time] = {}

            entry = current_data_df.iloc[i]
            
            # Check for E_I_pulse in analysis_dict OR offset_trace in intermediate_traces
            has_ei_in_analysis = 'E_I_pulse' in entry['analysis_dict']
            has_offset_trace = ('intermediate_traces' in entry and 
                              isinstance(entry['intermediate_traces'], dict) and
                              'offset_trace' in entry['intermediate_traces'])
            
            if not has_ei_in_analysis and not has_offset_trace:
                continue
            
            # Get channels from analysis_dict if available, otherwise from offset_trace
            if has_ei_in_analysis:
                channels = entry['analysis_dict']['E_I_pulse'].keys()
            elif has_offset_trace:
                # Fallback: get channels from offset_trace structure
                offset_trace_data = entry['intermediate_traces']['offset_trace']
                if isinstance(offset_trace_data, dict):
                    channels = [k for k in offset_trace_data.keys() if k.startswith('channel_')]
                else:
                    continue
            else:
                continue

            # Parse Stimulation Pathways from master_df for robust filtering
            pathway_info = row_meta.iloc[0].get('Stimulation Pathways', '')
            is_channel_apical = True # Default to True unless known to be basal
            
            # Helper to check if a channel is basal based on master_df
            def check_is_basal(channel_key, pathway_str):
                return f"{channel_key}: stratum oriens" in str(pathway_str).lower()

            for channel in channels:
                # EXCLUSION LOGIC: 
                # Priority 1: Trust master_df 'Stimulation Pathways'
                # Priority 2: Trust data file label if master_df is ambiguous
                
                is_basal_in_master = check_is_basal(channel, pathway_info)
                
                channel_label = stimulus_metadata.get(f"{channel}_label", "")
                is_basal_label = (channel_label == 'Stratum Oriens')
                
                # If master_df says it's basal (Stratum Oriens), skip it
                if is_basal_in_master:
                    continue
                    
                # If master_df does NOT say it's basal, but label says it IS:
                # This suggests a labeling error (like 20240905_c1).
                # TRUST MASTER_DF: If master_df defines pathways but doesn't say this is basal, treat as apical.
                # Only trust the label if master_df has NO pathway info at all.
                has_pathway_info = pd.notna(pathway_info) and 'channel' in str(pathway_info).lower()
                
                if not is_basal_in_master and is_basal_label:
                    if has_pathway_info:
                        # Master DF defines pathways and didn't call this basal -> Treat as Apical (ignore label)
                        pass 
                    else:
                        # No master info -> Trust the label -> Skip
                        continue

                if channel not in E_I_data_traces[cell_id][ISI_time]:
                    E_I_data_traces[cell_id][ISI_time][channel] = {}

                if condition not in E_I_data_traces[cell_id][ISI_time][channel]:
                    trace_dict = {
                        'unitary_average_traces': [] if ISI_time == 300 else None,
                        'unitary_all_traces': [] if ISI_time == 300 else None,
                        'non_unitary_average_traces': None if ISI_time == 300 else [],
                        'non_unitary_all_traces': None if ISI_time == 300 else [],
                        'holding_potential': []
                    }
                    E_I_data_traces[cell_id][ISI_time][channel][condition] = trace_dict
                
                partitioned_traces = entry['intermediate_traces'].get('partitioned_trace', {})
                offset_trace = entry['intermediate_traces'].get('offset_trace', {})

                if not offset_trace:
                    continue
                    
                holding_potentials = []
                all_traces = []

                if ISI_time == 300:  # Unitary traces
                    # offset_trace[channel] should be a dict with stim_start as keys
                    if channel in offset_trace:
                        channel_data = offset_trace[channel]
                        
                        # --- Auto-Detect Protocol Mismatch (Fuzzy) ---
                        current_starts = unitary_stim_starts.get(channel, [])
                        
                        def has_fuzzy_match(targets, data_keys):
                            for d in data_keys:
                                if any(abs(t - d) < 2.0 for t in targets):
                                    return True
                            return False
                            
                        has_match = has_fuzzy_match(current_starts, channel_data)
                        
                        if not has_match:
                            # Try alternate protocol
                            alt_version = 'newer' if stim_version == 'older' else 'older'
                            alt_starts = unitary_stim_starts_dict[alt_version].get(channel, [])
                            
                            if has_fuzzy_match(alt_starts, channel_data):
                                # print(f"DEBUG: Auto-switching {cell_id} to '{alt_version}' protocol")
                                unitary_stim_starts = unitary_stim_starts_dict[alt_version]
                                stim_version = alt_version # Persist for this cell

                        if isinstance(channel_data, dict):
                            for stim_start in channel_data:
                                # Fuzzy match check against valid starts
                                valid_starts = unitary_stim_starts.get(channel, [])
                                is_valid_start = any(abs(s - stim_start) < 2.0 for s in valid_starts)
                                
                                if is_valid_start:
                                    trace = channel_data[stim_start][0]
                                    # TRACE LENGTH FILTER: Skip traces that are too short
                                    # Short traces (< MIN_EI_TRACE_LENGTH) cannot capture full EPSP
                                    if isinstance(trace, np.ndarray) and trace.size >= MIN_EI_TRACE_LENGTH:
                                        all_traces.append(trace)
                                    elif isinstance(trace, np.ndarray) and trace.size > 0:
                                        print(f"  WARNING: {cell_id} {channel} ISI {ISI_time} - trace too short ({trace.size} samples), skipped")

                    # Get holding potentials from partitioned_traces
                    if channel in partitioned_traces:
                        channel_data = partitioned_traces[channel]
                        if isinstance(channel_data, dict):
                            for stim_start in channel_data:
                                if stim_start in unitary_stim_starts.get(channel, []):
                                    trace = channel_data[stim_start][0]
                                    if isinstance(trace, np.ndarray) and trace.size > 0:
                                        baseline_voltage = np.mean(trace[:int(10 * 20000 / 1000)])
                                        holding_potentials.append(baseline_voltage)

                    if all_traces:
                        min_length = min(trace.shape[0] for trace in all_traces)
                        all_traces = [trace[:min_length] for trace in all_traces]
                        average_baseline = np.mean(holding_potentials) if holding_potentials else 0
                        
                        unitary_average = np.mean(all_traces, axis=0)
                        traces_extracted[condition] += 1
                        
                        E_I_data_traces[cell_id][ISI_time][channel][condition]['unitary_all_traces'].extend(all_traces)
                        E_I_data_traces[cell_id][ISI_time][channel][condition]['holding_potential'].append(average_baseline)
                        E_I_data_traces[cell_id][ISI_time][channel][condition]['unitary_average_traces'] = unitary_average

                else:  # Non-unitary traces
                    if channel in offset_trace:
                        trace = offset_trace[channel]
                        # TRACE LENGTH FILTER: Skip traces that are too short
                        # Short traces (< MIN_EI_TRACE_LENGTH) cannot capture full EPSP
                        if isinstance(trace, np.ndarray) and trace.size >= MIN_EI_TRACE_LENGTH:
                            # Append trace to the accumulator without truncation
                            E_I_data_traces[cell_id][ISI_time][channel][condition]['non_unitary_all_traces'].append(trace)
                            traces_extracted[condition] += 1
                        elif isinstance(trace, np.ndarray) and trace.size > 0:
                            print(f"  WARNING: {cell_id} {channel} ISI {ISI_time} - trace too short ({trace.size} samples), skipped")

                    if channel in partitioned_traces:
                        trace = partitioned_traces[channel]
                        if isinstance(trace, np.ndarray) and trace.size > 0:
                            baseline_voltage = np.mean(trace[:int(10 * 20000 / 1000)])
                            E_I_data_traces[cell_id][ISI_time][channel][condition]['holding_potential'].append(baseline_voltage)
        
        # Print summary for this cell
        if traces_extracted:
            condition_counts = ', '.join([f"{cond}={count}" for cond, count in traces_extracted.items()])
            print(f"{cell_id} ({genotype}): {condition_counts}")
        else:
            print(f"{cell_id} ({genotype}): No traces extracted")

    # Post-process: Calculate non-unitary averages after all traces are collected
    for cell_id in E_I_data_traces:
        for ISI_time in E_I_data_traces[cell_id]:
            if ISI_time == 300:  # Skip unitary traces, already processed
                continue
            for channel in E_I_data_traces[cell_id][ISI_time]:
                for condition in E_I_data_traces[cell_id][ISI_time][channel]:
                    all_traces = E_I_data_traces[cell_id][ISI_time][channel][condition].get('non_unitary_all_traces', [])
                    holding_potentials = E_I_data_traces[cell_id][ISI_time][channel][condition].get('holding_potential', [])
                    
                    if all_traces and len(all_traces) > 0:
                        # Trim all traces to the same minimum length
                        min_length = min(trace.shape[0] for trace in all_traces if isinstance(trace, np.ndarray))
                        trimmed_traces = [trace[:min_length] for trace in all_traces if isinstance(trace, np.ndarray)]
                        
                        if trimmed_traces:
                            # Store trimmed traces back
                            E_I_data_traces[cell_id][ISI_time][channel][condition]['non_unitary_all_traces'] = trimmed_traces
                            # Calculate and store the average
                            non_unitary_average = np.mean(trimmed_traces, axis=0)
                            E_I_data_traces[cell_id][ISI_time][channel][condition]['non_unitary_average_traces'] = non_unitary_average

    return E_I_data_traces

def get_E_I_traces_basal(data_dir, unitary_stim_starts_dict, ISI_times_dict_mapping, master_df):
    """
    Extract E-I traces from basal pathway (Stratum Oriens) experiments.
    
    Args:
        data_dir: Directory containing .pkl files
        unitary_stim_starts_dict: Dict with 'older' and 'newer' stim starts for unitary (ISI 300) traces
        ISI_times_dict_mapping: Dict mapping 'older'/'newer' to their respective ISI times dictionaries
        master_df: Master dataframe for genotype lookup and stim time configuration
    
    Returns:
        E_I_data_traces: Dict with structure {cell_id: {ISI_time: {'channel_1': {condition: trace_dict}}}}
    
    CRITICAL: Only cells with 'E/I' explicitly in Inclusion column are processed.
    Uses master_df 'ESPS Stim Time File Name' column to determine correct stim times per cell.
    """
    # Create genotype lookup, stim time lookup, and identify stratum oriens cells from master_df
    genotype_lookup = {}
    stim_time_lookup = {}  # Maps cell_id to 'older' or 'newer'
    stratum_oriens_cells = set()
    ei_inclusion_set = set()  # Only cells with E/I in Inclusion
    master_df_copy = master_df.copy()
    master_df_copy['Cell_ID'] = master_df_copy['Cell_ID'].astype(str)
    
    for _, row in master_df_copy.iterrows():
        cell_id = row['Cell_ID']
        genotype_lookup[cell_id] = row.get('Genotype', 'Unknown')
        
        # Determine stim time configuration from master_df
        stim_file_name = str(row.get('ESPS Stim Time File Name', '')).strip()
        if 'newer' in stim_file_name:
            stim_time_lookup[cell_id] = 'newer'
        else:
            stim_time_lookup[cell_id] = 'older'
        
        # Check if 'E/I' is in Inclusion column
        inclusion = str(row.get('Inclusion', '')).strip()
        if 'E/I' in inclusion:
            ei_inclusion_set.add(cell_id)
        
        # Check if this cell has stratum oriens stimulation
        stim_pathways = str(row.get('Stimulation Pathways', '')).lower()
        if 'stratum oriens' in stim_pathways:
            stratum_oriens_cells.add(cell_id)
    
    # Only process cells that are BOTH stratum oriens AND have E/I in Inclusion
    valid_basal_cells = stratum_oriens_cells & ei_inclusion_set
    
    print(f"  Found {len(stratum_oriens_cells)} Stratum Oriens cells, {len(valid_basal_cells)} with E/I in Inclusion")
    print(f"  Valid Basal E:I cells: {sorted(valid_basal_cells)}")
    
    # Show stim time configuration for each valid cell
    for cell_id in sorted(valid_basal_cells):
        config = stim_time_lookup.get(cell_id, 'unknown')
        print(f"    {cell_id}: using {config} stim times")
    
    data_files = [file for file in os.listdir(data_dir) if file.endswith('.pkl')]
    E_I_data_traces = {}

    for data_file in data_files:
        # Convert filename to cell_id for consistency
        cell_id = convert_pkl_filename_to_cell_id(data_file)
        if cell_id is None:
            continue
        
        # Skip if not a valid basal E:I cell (must be stratum oriens AND have E/I in Inclusion)
        if cell_id not in valid_basal_cells:
            continue
            
        current_data_df = pd.read_pickle(os.path.join(data_dir, data_file))

        for i in range(len(current_data_df)):
            stimulus_metadata = current_data_df['stimulus_metadata_dict'].iloc[i]

            # Check if 'Stratum Oriens' is in either channel_1_label or channel_2_label
            if not stimulus_metadata:
                continue
                
            channel_1_label = stimulus_metadata.get('channel_1_label', '')
            channel_2_label = stimulus_metadata.get('channel_2_label', '')
            
            # Accept if either:
            # 1. Channel labels explicitly say 'Stratum Oriens', OR
            # 2. This cell is in our stratum_oriens_cells set from master_df
            has_stratum_oriens_label = (channel_1_label == 'Stratum Oriens' or channel_2_label == 'Stratum Oriens')
            is_stratum_oriens_cell = cell_id in stratum_oriens_cells
            
            if not has_stratum_oriens_label and not is_stratum_oriens_cell:
                continue
            
            # Data is always in channel_1 regardless of which label says 'Stratum Oriens'
            source_channel = 'channel_1'
                        
            if 'ISI' in stimulus_metadata:
                ISI_value = stimulus_metadata['ISI']
                if ISI_value and ISI_value != 'nan' and not pd.isna(ISI_value):
                    ISI_time = int(float(ISI_value))
                    condition = stimulus_metadata['condition']
                    
                    # Initialize E_I_data_traces
                    if cell_id not in E_I_data_traces:
                        E_I_data_traces[cell_id] = {}
                    if ISI_time not in E_I_data_traces[cell_id]:
                        E_I_data_traces[cell_id][ISI_time] = {}

                    entry = current_data_df.iloc[i]
                    
                    # Get trace data
                    partitioned_traces = entry['intermediate_traces'].get('partitioned_trace', {})
                    offset_trace = entry['intermediate_traces'].get('offset_trace', {})
                    
                    # Check if source_channel has any trace data
                    has_data = (source_channel in offset_trace) or (source_channel in partitioned_traces)
                    
                    if has_data:
                        # Always use channel_1 for storage
                        storage_channel = 'channel_1'
                        
                        if storage_channel not in E_I_data_traces[cell_id][ISI_time]:
                            E_I_data_traces[cell_id][ISI_time][storage_channel] = {}

                        if condition not in E_I_data_traces[cell_id][ISI_time][storage_channel]:
                            trace_dict = {
                                'unitary_average_traces': [] if ISI_time == 300 else None,
                                'unitary_all_traces': [] if ISI_time == 300 else None,
                                'non_unitary_average_traces': None if ISI_time == 300 else [],
                                'non_unitary_all_traces': None if ISI_time == 300 else [],
                                'holding_potential': []
                            }
                            E_I_data_traces[cell_id][ISI_time][storage_channel][condition] = trace_dict

                        holding_potentials = []
                        all_traces = []
                        
                        # Get correct stim times for this cell
                        stim_config = stim_time_lookup.get(cell_id, 'older')
                        unitary_stim_starts = unitary_stim_starts_dict[stim_config]

                        if ISI_time == 300:  # Handle unitary traces for ISI 300
                            for stim_start in offset_trace.get(source_channel, {}):
                                if stim_start in unitary_stim_starts.get(source_channel, []):
                                    trace = offset_trace[source_channel][stim_start][0]
                                    if isinstance(trace, np.ndarray) and trace.size >= MIN_EI_TRACE_LENGTH:
                                        all_traces.append(trace)

                            for stim_start in partitioned_traces.get(source_channel, {}):
                                if stim_start in unitary_stim_starts.get(source_channel, []):
                                    trace = partitioned_traces[source_channel][stim_start][0]
                                    if isinstance(trace, np.ndarray) and trace.size >= MIN_EI_TRACE_LENGTH:
                                        baseline_voltage = np.mean(trace[:int(10 * 20000 / 1000)])
                                        holding_potentials.append(baseline_voltage)

                            if all_traces:
                                min_length = min(trace.shape[0] for trace in all_traces)
                                all_traces = [trace[:min_length] for trace in all_traces]
                                average_baseline = np.mean(holding_potentials) if holding_potentials else 0
                                
                                E_I_data_traces[cell_id][ISI_time][storage_channel][condition]['unitary_all_traces'].extend(all_traces)
                                E_I_data_traces[cell_id][ISI_time][storage_channel][condition]['holding_potential'].append(average_baseline)
                                unitary_average = np.mean(all_traces, axis=0)
                                E_I_data_traces[cell_id][ISI_time][storage_channel][condition]['unitary_average_traces'] = unitary_average

                        else:  # Handle non-unitary traces for non-300 ISI
                            if source_channel in offset_trace:
                                trace = offset_trace[source_channel]
                                if isinstance(trace, np.ndarray) and trace.size >= MIN_EI_TRACE_LENGTH:
                                    all_traces.append(trace)

                            if source_channel in partitioned_traces:
                                trace = partitioned_traces[source_channel]
                                if isinstance(trace, np.ndarray) and trace.size >= MIN_EI_TRACE_LENGTH:
                                    baseline_voltage = np.mean(trace[:int(10 * 20000 / 1000)])
                                    holding_potentials.append(baseline_voltage)

                            if all_traces:
                                min_length = min(trace.shape[0] for trace in all_traces)
                                all_traces = [trace[:min_length] for trace in all_traces]
                                average_baseline = np.mean(holding_potentials) if holding_potentials else 0

                                E_I_data_traces[cell_id][ISI_time][storage_channel][condition]['non_unitary_all_traces'].extend(all_traces)
                                E_I_data_traces[cell_id][ISI_time][storage_channel][condition]['holding_potential'].append(average_baseline)
                                non_unitary_average = np.mean(all_traces, axis=0)
                                E_I_data_traces[cell_id][ISI_time][storage_channel][condition]['non_unitary_average_traces'] = non_unitary_average

    return E_I_data_traces

def get_300ms_gabazine_traces_for_gabab(data_dir, unitary_stim_starts_dict, ISI_times_dict_mapping, master_df, pathway_type='apical'):
    """
    Extract 300ms ISI Gabazine traces specifically for GABAb analysis.
    Includes cells with EITHER:
      - 'E/I' in their Inclusion column, OR
      - '300 ms unitary Gabazine' in their Experiment Notes
    
    Args:
        data_dir: Path to data directory
        unitary_stim_starts_dict: Dict with 'older' and 'newer' unitary stim times
        ISI_times_dict_mapping: Dict mapping 'older'/'newer' to ISI_times_dict  
        master_df: Master dataframe
        pathway_type: 'apical' for Perforant/Schaffer or 'basal' for Stratum Oriens
        
    Returns:
        Filtered dictionary with only 300ms Gabazine traces
    """
    import pandas as pd
    
    # Get cells with E/I in Inclusion
    ei_cells = set(master_df[master_df['Inclusion'].str.contains('E/I', na=False, case=False)]['Cell_ID'].values)
    
    # Get cells with "300 ms unitary Gabazine" in Experiment Notes
    unitary_gab_cells = set(master_df[master_df['Experiment Notes'].str.contains('300 ms unitary Gabazine', na=False, case=False)]['Cell_ID'].values)
    
    # Combine BOTH sets
    all_gabab_cells = ei_cells | unitary_gab_cells
    
    # Filter by pathway type
    if pathway_type == 'basal':
        stratum_oriens_cells = set(master_df[master_df['Stimulation Pathways'].str.contains('stratum oriens', na=False, case=False)]['Cell_ID'].values)
        target_cells = all_gabab_cells & stratum_oriens_cells
    else:  # apical
        stratum_oriens_cells = set(master_df[master_df['Stimulation Pathways'].str.contains('stratum oriens', na=False, case=False)]['Cell_ID'].values)
        target_cells = all_gabab_cells - stratum_oriens_cells
    
    print(f"Loading {pathway_type} 300ms Gabazine traces:")
    print(f"  E/I cells: {len(ei_cells)}")
    print(f"  300ms unitary Gabazine cells: {len(unitary_gab_cells)}")
    print(f"  Combined: {len(all_gabab_cells)}")
    print(f"  Target ({pathway_type}): {len(target_cells)}")
    
    # Create a modified master_df that marks these cells as having E/I for the extraction function
    master_df_modified = master_df.copy()
    for cell_id in target_cells:
        if cell_id in master_df_modified['Cell_ID'].values:
            idx = master_df_modified[master_df_modified['Cell_ID'] == cell_id].index[0]
            # Ensure they have "E/I" in Inclusion so get_E_I_traces will load them
            current_inclusion = str(master_df_modified.at[idx, 'Inclusion'])
            if 'E/I' not in current_inclusion:
                if pd.isna(current_inclusion) or current_inclusion == 'nan':
                    master_df_modified.at[idx, 'Inclusion'] = 'E/I'
                else:
                    master_df_modified.at[idx, 'Inclusion'] = current_inclusion + '; E/I'
    
    # Use existing working functions with modified master_df
    if pathway_type == 'basal':
        all_traces = get_E_I_traces_basal(
            data_dir=data_dir,
            unitary_stim_starts_dict=unitary_stim_starts_dict,
            ISI_times_dict_mapping=ISI_times_dict_mapping,
            master_df=master_df_modified
        )
    else:  # apical
        all_traces = get_E_I_traces(
            data_dir=data_dir,
            unitary_stim_starts_dict=unitary_stim_starts_dict,
            ISI_times_dict_mapping=ISI_times_dict_mapping,
            master_df=master_df_modified
        )
    
    # Filter to ONLY 300ms ISI and ONLY Gabazine condition
    filtered_traces = {}
    
    for cell_id in all_traces:
        # Only keep ISI=300
        if 300 not in all_traces[cell_id]:
            continue
        
        # Check if this cell has Gabazine data
        has_gabazine = False
        for channel in all_traces[cell_id][300]:
            for condition in all_traces[cell_id][300][channel]:
                if 'gabazine' in condition.lower():
                    has_gabazine = True
                    break
            if has_gabazine:
                break
        
        if not has_gabazine:
            continue
        
        # Create filtered entry with only 300ms and only Gabazine
        filtered_traces[cell_id] = {300: {}}
        
        for channel in all_traces[cell_id][300]:
            filtered_traces[cell_id][300][channel] = {}
            
            for condition in all_traces[cell_id][300][channel]:
                cond_lower = condition.lower()
                if 'gabazine' in cond_lower:
                    if 'ml297' in cond_lower or 'ml-297' in cond_lower:
                        filtered_traces[cell_id][300][channel]['gabazine + ml297'] = all_traces[cell_id][300][channel][condition]
                    elif 'etx' in cond_lower:
                        filtered_traces[cell_id][300][channel]['gabazine + etx'] = all_traces[cell_id][300][channel][condition]
                    else:
                        # Standard Gabazine
                        filtered_traces[cell_id][300][channel]['gabazine'] = all_traces[cell_id][300][channel][condition]
    
    print(f"✓ Loaded 300ms Gabazine data from {len(filtered_traces)} {pathway_type} cells for GABAb analysis")
    return filtered_traces

def process_basal_E_I_data(E_I_traces_basal, master_df, ISI_times_dict_mapping):
    """
    Process basal pathway E-I traces using the shared analysis pipeline.
    Renames 'channel_1' to 'Basal_Stratum_Oriens' to distinguish from Apical.
    """
    print("--- Processing Basal E:I Data with Shared Pipeline ---")
    
    # 1. Remap traces to use 'Basal_Stratum_Oriens' channel key and Normalise Keys
    E_I_traces_remapped = {}
    for cell_id, isi_dict in E_I_traces_basal.items():
        E_I_traces_remapped[cell_id] = {}
        for isi, channels in isi_dict.items():
            E_I_traces_remapped[cell_id][isi] = {}
            if 'channel_1' in channels:
                basal_data = channels['channel_1']
                
                # Normalize keys (remove leading spaces from conditions like ' Gabazine')
                # This is critical because shared functions expect clean 'Gabazine' keys
                normalized_data = {}
                for cond, data in basal_data.items():
                    clean_cond = cond.strip()
                    normalized_data[clean_cond] = data
                
                E_I_traces_remapped[cell_id][isi]['Basal_Stratum_Oriens'] = normalized_data
    
    # 2. Patch ISI_times_dict_mapping to include 'Basal_Stratum_Oriens'
    # The analysis function looks up times by channel name. 
    # Since Basal is recorded on channel_1, we alias 'Basal_Stratum_Oriens' -> 'channel_1' times.
    patched_mapping = {}
    for version, times_dict in ISI_times_dict_mapping.items():
        patched_dict = times_dict.copy()
        if 'channel_1' in patched_dict:
            patched_dict['Basal_Stratum_Oriens'] = patched_dict['channel_1']
        patched_mapping[version] = patched_dict

    # 3. Calculate Expected EPSPs
    expected_EPSPs, expected_EPSPs_peaks = calculate_expected_EPSPs_for_all_cells(
        E_I_traces_remapped, patched_mapping, master_df
    )
    
    # 4. Calculate Amplitudes and Estimated Inhibition
    # This modifies E_I_traces_remapped in-place to add inhibition traces
    E_I_amplitudes, _ = get_E_I_amplitudes_and_estimated_inhibition_traces(E_I_traces_remapped)
    
    # 5. Calculate Imbalance
    E_I_imbalances = calculate_E_I_imbalance(E_I_amplitudes)
    
    # 6. Export to DataFrame
    E_I_data_df = export_E_I_amplitudes_to_dataframe(
        E_I_amplitudes, E_I_imbalances, expected_EPSPs_peaks, master_df
    )
    
    # 7. Format Traces for Output (merge expected traces into the dict)
    # The calling function expects the dict to contain 'Expected_EPSP_Trace'
    for cell_id in expected_EPSPs:
        for channel in expected_EPSPs[cell_id]:
             if channel == 'Basal_Stratum_Oriens':
                 for isi, trace in expected_EPSPs[cell_id][channel].items():
                     if cell_id in E_I_traces_remapped and isi in E_I_traces_remapped[cell_id]:
                          if channel in E_I_traces_remapped[cell_id][isi]:
                               # Add 'Expected_EPSP_Trace' key to match original output format
                               E_I_traces_remapped[cell_id][isi][channel]['Expected_EPSP_Trace'] = trace

    return E_I_data_df, E_I_traces_remapped

def create_expected_EPSP(unitary_EPSP, stim_times):
    """Create expected compound EPSP from unitary EPSP and stimulation times."""
    baseline_len = int(10 * 20000 / 1000)
    unitary_len = len(unitary_EPSP)
    total_len = int(800 * 20000 / 1000)
    compound_EPSP = np.zeros(total_len)

    for stim in stim_times:
        stim_index = int(stim * 20000 / 1000)
        start = baseline_len + stim_index
        end = start + unitary_len
        
        if start < total_len:
            if end <= total_len:
                compound_EPSP[start:end] += unitary_EPSP
            else:
                overlap_len = total_len - start
                if overlap_len > 0:
                    compound_EPSP[start:] += unitary_EPSP[:overlap_len]

    return compound_EPSP

def calculate_peak_amplitude(trace, height_threshold=0.1):
    """Find max peak value and index in a trace."""
    peaks, _ = find_peaks(trace, height=height_threshold)
    if peaks.size > 0:
        max_peak = max(trace[peaks])
        max_peak_index = np.where(trace == max_peak)[0][0]
        return max_peak, max_peak_index
    return np.nan, np.nan

def get_E_I_amplitudes_and_estimated_inhibition_traces(E_I_traces_dict):
    """Calculate peak amplitudes and estimated inhibition traces."""
    E_I_data_amplitudes = {}
    
    for cell_id in E_I_traces_dict:
        E_I_data_amplitudes[cell_id] = {}

        for ISI in E_I_traces_dict[cell_id]:
            E_I_data_amplitudes[cell_id][ISI] = {}

            for channel in E_I_traces_dict[cell_id][ISI]:
                E_I_data_amplitudes[cell_id][ISI][channel] = {}

                # Calculate amplitudes for each condition
                for condition in E_I_traces_dict[cell_id][ISI][channel]:
                    trace_key = 'unitary_average_traces' if ISI == 300 else 'non_unitary_average_traces'
                    
                    if trace_key in E_I_traces_dict[cell_id][ISI][channel][condition]:
                        trace = E_I_traces_dict[cell_id][ISI][channel][condition][trace_key]
                        
                        # BUGFIX: Convert list to numpy array if needed
                        if isinstance(trace, list):
                            trace = np.array(trace)
                            E_I_traces_dict[cell_id][ISI][channel][condition][trace_key] = trace
                        
                        if isinstance(trace, np.ndarray) and trace.size > 0:
                            max_peak, max_peak_idx = calculate_peak_amplitude(trace.copy())
                            E_I_data_amplitudes[cell_id][ISI][channel][condition] = {
                                'max_peak_value': max_peak,
                                'max_peak_idx': max_peak_idx
                            }
                
                # Calculate estimated inhibition
                if 'Control' in E_I_traces_dict[cell_id][ISI][channel] and 'Gabazine' in E_I_traces_dict[cell_id][ISI][channel]:
                    try:
                        trace_key = 'unitary_average_traces' if ISI == 300 else 'non_unitary_average_traces'
                        
                        if trace_key in E_I_traces_dict[cell_id][ISI][channel]['Control'] and \
                           trace_key in E_I_traces_dict[cell_id][ISI][channel]['Gabazine']:
                            
                            gab_trace = E_I_traces_dict[cell_id][ISI][channel]['Gabazine'][trace_key]
                            ctrl_trace = E_I_traces_dict[cell_id][ISI][channel]['Control'][trace_key]
                            
                            # BUGFIX: Convert lists to numpy arrays if needed
                            if isinstance(gab_trace, list):
                                gab_trace = np.array(gab_trace)
                            if isinstance(ctrl_trace, list):
                                ctrl_trace = np.array(ctrl_trace)
                            
                            # BUGFIX: Handle length mismatch by trimming to shorter length
                            if len(gab_trace) != len(ctrl_trace):
                                min_len = min(len(gab_trace), len(ctrl_trace))
                                print(f"  Note: {cell_id} {channel} ISI {ISI} - trimming traces to {min_len} samples (Gab: {len(gab_trace)}, Ctrl: {len(ctrl_trace)})")
                                gab_trace = gab_trace[:min_len]
                                ctrl_trace = ctrl_trace[:min_len]
                            
                            E_I_traces_dict[cell_id][ISI][channel]['estimated_inhibition'] = {}
                            inhibition_trace = gab_trace - ctrl_trace
                            E_I_traces_dict[cell_id][ISI][channel]['estimated_inhibition'][trace_key] = inhibition_trace
                            
                            max_peak, max_peak_idx = calculate_peak_amplitude(inhibition_trace.copy())
                            E_I_data_amplitudes[cell_id][ISI][channel]['estimated_inhibition'] = {
                                'max_peak_value': max_peak,
                                'max_peak_idx': max_peak_idx
                            }
                    except Exception as e:
                        print(f"Error calculating estimated inhibition for cell {cell_id}, ISI {ISI}, channel {channel}: {e}")

    return E_I_data_amplitudes, E_I_traces_dict

def calculate_E_I_imbalance(E_I_amplitudes_dict):
    """Calculate E-I imbalance ratio from peak amplitudes."""
    E_I_imbalances = {}

    for cell_id in E_I_amplitudes_dict:
        E_I_imbalances[cell_id] = {}

        for ISI in E_I_amplitudes_dict[cell_id]:
            E_I_imbalances[cell_id][ISI] = {}

            for channel in E_I_amplitudes_dict[cell_id][ISI]:
                gabazine_peak = E_I_amplitudes_dict[cell_id][ISI][channel].get('Gabazine', {}).get('max_peak_value')
                inhibition_peak = E_I_amplitudes_dict[cell_id][ISI][channel].get('estimated_inhibition', {}).get('max_peak_value')

                if gabazine_peak is not None and inhibition_peak is not None:
                    if (gabazine_peak + inhibition_peak) != 0:
                        E_I_imbalance = gabazine_peak / (gabazine_peak + inhibition_peak)
                    else:
                        print(f'Warning: E_I_imbalance is zero for cell: {cell_id}, ISI: {ISI}, channel: {channel}')
                        E_I_imbalance = 0
                    E_I_imbalances[cell_id][ISI][channel] = E_I_imbalance

    return E_I_imbalances

def calculate_expected_EPSPs_for_all_cells(E_I_traces_dict, ISI_times_dict_mapping, master_df):
    """
    Calculate expected EPSPs using the correct ISI_times_dict for each cell.
    
    Parameters:
        E_I_traces_dict: Dictionary with E-I trace data
        ISI_times_dict_mapping: Dictionary mapping 'older'/'newer' to ISI_times_dict
        master_df: Master dataframe with 'ESPS Stim Time File Name' column
    
    Returns:
        Tuple of (expected_EPSPs_traces, expected_EPSPs_peaks)
    """
    # Create lookup for which ISI_times_dict to use per cell
    master_df_copy = master_df.copy()
    master_df_copy['Cell_ID'] = master_df_copy['Cell_ID'].astype(str)
    
    stim_file_lookup = {}
    for _, row in master_df_copy.iterrows():
        cell_id = row['Cell_ID']
        stim_file = row.get('ESPS Stim Time File Name', None)
        
        if pd.notna(stim_file):
            if 'newer' in str(stim_file).lower() or 'new' in str(stim_file).lower():
                stim_file_lookup[cell_id] = 'newer'
            elif 'older' in str(stim_file).lower() or 'old' in str(stim_file).lower():
                stim_file_lookup[cell_id] = 'older'
            else:
                stim_file_lookup[cell_id] = 'older'
        else:
            stim_file_lookup[cell_id] = 'older'
    
    expected_EPSPs = {}
    expected_EPSPs_peaks = {}
    
    for cell_id in E_I_traces_dict:
        if 300 not in E_I_traces_dict[cell_id]:
            continue
        
        # Get the correct ISI_times_dict for this cell
        stim_version = stim_file_lookup.get(cell_id, 'older')
        ISI_times_dict = ISI_times_dict_mapping[stim_version]
        
        print(f"Calculating expected EPSPs for {cell_id}: Using {stim_version} stim times")
            
        unitary_dict = E_I_traces_dict[cell_id][300]
        expected_EPSPs[cell_id] = {}
        expected_EPSPs_peaks[cell_id] = {}
            
        for channel in unitary_dict:
            expected_EPSPs[cell_id][channel] = {}
            expected_EPSPs_peaks[cell_id][channel] = {}
                
            if 'Gabazine' in unitary_dict[channel]:
                unitary_trace = unitary_dict[channel]['Gabazine']['unitary_average_traces']

                # Get the ISI times for this specific channel
                channel_ISI_times = ISI_times_dict.get(channel, {})

                for ISI in channel_ISI_times:
                    stim_times = channel_ISI_times[ISI]
                    
                    # Flatten stim_times if it's nested
                    if isinstance(stim_times, list) and len(stim_times) > 0:
                        if isinstance(stim_times[0], list):
                            # It's a list of lists, take the first one or flatten
                            stim_times_flat = stim_times[0]  # Usually want the first repetition
                        else:
                            stim_times_flat = stim_times
                    else:
                        stim_times_flat = stim_times
                    
                    # Convert absolute stim times to RELATIVE times from first stim
                    # This is needed because the extracted traces are aligned to their
                    # respective stim start, but YAML contains absolute recording times
                    # (e.g., channel_1 starts at 500ms, channel_2 at 1500ms)
                    if stim_times_flat and len(stim_times_flat) > 0:
                        first_stim = stim_times_flat[0]
                        stim_times_relative = [t - first_stim for t in stim_times_flat]
                    else:
                        stim_times_relative = stim_times_flat
                    
                    expected_EPSP = create_expected_EPSP(unitary_trace, stim_times_relative)
                    expected_EPSPs[cell_id][channel][ISI] = expected_EPSP

                    max_peak, max_peak_idx = calculate_peak_amplitude(expected_EPSP)
                    expected_EPSPs_peaks[cell_id][channel][ISI] = {
                        'max_peak_value': max_peak,
                        'max_peak_idx': max_peak_idx
                    }

    return expected_EPSPs, expected_EPSPs_peaks

def export_E_I_amplitudes_to_dataframe(E_I_amplitudes_dict, E_I_imbalances_dict, 
                                       expected_EPSPs_peaks_dict, master_df):
    """Export E-I amplitudes, imbalances, and expected EPSP peaks to DataFrame."""
    master_df_copy = master_df.copy()
    master_df_copy['Cell_ID'] = master_df_copy['Cell_ID'].astype(str)
    
    genotype_lookup = dict(zip(master_df_copy['Cell_ID'], master_df_copy['Genotype']))
    sex_lookup = dict(zip(master_df_copy['Cell_ID'], master_df_copy['Sex']))
    
    # Build a lookup for Stimulation Pathways from master_df
    stim_pathways_lookup = dict(zip(master_df_copy['Cell_ID'], master_df_copy['Stimulation Pathways']))
    
    def parse_pathway_for_channel(stim_pathways_str, channel):
        """Parse the Stimulation Pathways string to get the actual pathway for a channel."""
        # Special case for Basal Pathway (explicit key)
        if channel == 'Basal_Stratum_Oriens':
            return 'Basal_Stratum_Oriens'

        if pd.isna(stim_pathways_str):
            # Fallback to old behavior
            if channel == 'channel_1':
                return 'Perforant'
            elif channel == 'channel_2':
                return 'Schaffer'
            return 'Unknown'
        
        # Parse the string format: {channel_1: perforant, channel_2: schaffer}
        stim_str = str(stim_pathways_str).lower()
        
        # Extract the pathway for the given channel
        if channel == 'channel_1':
            if 'channel_1:' in stim_str:
                # Get text after 'channel_1:' until comma or closing brace
                start = stim_str.find('channel_1:') + len('channel_1:')
                end = stim_str.find(',', start)
                if end == -1:
                    end = stim_str.find('}', start)
                pathway_raw = stim_str[start:end].strip()
            else:
                pathway_raw = 'perforant'  # fallback
        elif channel == 'channel_2':
            if 'channel_2:' in stim_str:
                start = stim_str.find('channel_2:') + len('channel_2:')
                end = stim_str.find(',', start)
                if end == -1:
                    end = stim_str.find('}', start)
                pathway_raw = stim_str[start:end].strip()
            else:
                pathway_raw = 'schaffer'  # fallback
        else:
            pathway_raw = 'unknown'
        
        # Map to standard pathway names
        if 'perforant' in pathway_raw:
            return 'Perforant'
        elif 'schaffer' in pathway_raw:
            return 'Schaffer'
        elif 'stratum oriens' in pathway_raw or 'oriens' in pathway_raw:
            return 'Basal_Stratum_Oriens'
        elif pathway_raw == '' or pathway_raw == 'nan':
            return None  # Empty channel, skip
        else:
            return 'Unknown'
    
    rows = []
    
    for cell_id in E_I_amplitudes_dict:
        genotype = genotype_lookup.get(cell_id, 'Unknown')
        sex = sex_lookup.get(cell_id, 'Unknown')
        stim_pathways = stim_pathways_lookup.get(cell_id, None)
        
        for ISI in E_I_amplitudes_dict[cell_id]:
            for channel in E_I_amplitudes_dict[cell_id][ISI]:
                # Map channel to pathway name using actual Stimulation Pathways
                pathway = parse_pathway_for_channel(stim_pathways, channel)
                
                # Skip if pathway is None (empty channel)
                if pathway is None:
                    continue
                
                row = {
                    'Cell_ID': cell_id,
                    'Genotype': genotype,
                    'Sex': sex,
                    'ISI': ISI,
                    'Channel': channel,
                    'Pathway': pathway
                }
                
                # Add amplitudes for each condition
                # Normalize condition names to prevent duplicates (e.g., "Control" vs "control")
                for condition in E_I_amplitudes_dict[cell_id][ISI][channel]:
                    max_peak = E_I_amplitudes_dict[cell_id][ISI][channel][condition].get('max_peak_value', np.nan)
                    # Normalize condition name: title case and strip whitespace
                    normalized_condition = condition.strip().title() if condition.strip() else 'Unknown'
                    row[f'{normalized_condition}_Amplitude'] = max_peak
                
                # Add E-I imbalance
                if cell_id in E_I_imbalances_dict and ISI in E_I_imbalances_dict[cell_id]:
                    if channel in E_I_imbalances_dict[cell_id][ISI]:
                        row['E_I_Imbalance'] = E_I_imbalances_dict[cell_id][ISI][channel]
                
                # Add expected EPSP amplitude
                if cell_id in expected_EPSPs_peaks_dict and channel in expected_EPSPs_peaks_dict[cell_id]:
                    if ISI in expected_EPSPs_peaks_dict[cell_id][channel]:
                        row['Expected_EPSP_Amplitude'] = expected_EPSPs_peaks_dict[cell_id][channel][ISI].get('max_peak_value', np.nan)
                
                # Calculate Supralinearity for both Control and Gabazine
                # Supralinearity = Measured - Expected EPSP
                
                # Control Supralinearity (for analysis, typically close to zero if no inhibition)
                if 'Control_Amplitude' in row and 'Expected_EPSP_Amplitude' in row:
                    control = row['Control_Amplitude']
                    expected = row['Expected_EPSP_Amplitude']
                    if pd.notna(control) and pd.notna(expected):
                        row['Control_Supralinearity'] = control - expected
                    else:
                        row['Control_Supralinearity'] = np.nan
                else:
                    row['Control_Supralinearity'] = np.nan
                
                # Gabazine Supralinearity (primary metric - reveals nonlinear summation)
                if 'Gabazine_Amplitude' in row and 'Expected_EPSP_Amplitude' in row:
                    gabazine = row['Gabazine_Amplitude']
                    expected = row['Expected_EPSP_Amplitude']
                    if pd.notna(gabazine) and pd.notna(expected):
                        row['Gabazine_Supralinearity'] = gabazine - expected
                    else:
                        row['Gabazine_Supralinearity'] = np.nan
                else:
                    row['Gabazine_Supralinearity'] = np.nan
                
                rows.append(row)
    
    df = pd.DataFrame(rows)
    base_cols = ['Cell_ID', 'Genotype', 'Sex', 'ISI', 'Channel', 'Pathway']
    other_cols = [col for col in df.columns if col not in base_cols]
    return df[base_cols + other_cols]

def export_E_I_traces_for_plotting(E_I_traces_dict, expected_EPSPs_dict, master_df, output_path=None):
    """Export all E-I traces to DataFrame for plotting (saves as .pkl to preserve arrays)."""
    master_df_copy = master_df.copy()
    master_df_copy['Cell_ID'] = master_df_copy['Cell_ID'].astype(str)
    
    genotype_lookup = dict(zip(master_df_copy['Cell_ID'], master_df_copy['Genotype']))
    sex_lookup = dict(zip(master_df_copy['Cell_ID'], master_df_copy['Sex']))
    
    rows = []
    
    for cell_id in E_I_traces_dict:
        genotype = genotype_lookup.get(cell_id, 'Unknown')
        sex = sex_lookup.get(cell_id, 'Unknown')
        
        for ISI in E_I_traces_dict[cell_id]:
            for channel in E_I_traces_dict[cell_id][ISI]:
                # Map channel to pathway
                pathway_map = {
                    'channel_1': 'Perforant',
                    'channel_2': 'Schaffer',
                    'Basal_Stratum_Oriens': 'Basal_Stratum_Oriens'  # For basal traces
                }
                pathway = pathway_map.get(channel, channel)  # Default to channel name if not mapped
                
                row = {
                    'Cell_ID': cell_id,
                    'Genotype': genotype,
                    'Sex': sex,
                    'ISI': ISI,
                    'Channel': channel,
                    'Pathway': pathway
                }
                
                # Add traces for each condition
                trace_key = 'unitary_average_traces' if ISI == 300 else 'non_unitary_average_traces'
                
                for condition in E_I_traces_dict[cell_id][ISI][channel]:
                    if trace_key in E_I_traces_dict[cell_id][ISI][channel][condition]:
                        trace = E_I_traces_dict[cell_id][ISI][channel][condition][trace_key]
                        if isinstance(trace, np.ndarray):
                            row[f'{condition}_Trace'] = trace
                
                # Add expected EPSP trace
                if cell_id in expected_EPSPs_dict and channel in expected_EPSPs_dict[cell_id]:
                    if ISI in expected_EPSPs_dict[cell_id][channel]:
                        row['Expected_EPSP_Trace'] = expected_EPSPs_dict[cell_id][channel][ISI]
                
                rows.append(row)
    
    df = pd.DataFrame(rows)
    
    if output_path:
        df.to_pickle(output_path)
        print(f"✓ Traces saved to: {output_path}")
    
    return df

def export_EPSP_amplitudes_with_drug_for_R(amplitudes_df):
    """Export EPSP amplitudes (Control and Gabazine) in R format with Drug column."""
    if len(amplitudes_df) == 0:
        return pd.DataFrame()
    
    rows = []
    for _, row in amplitudes_df.iterrows():
        # Use Pathway column if available, otherwise use Channel
        pathway_value = row.get('Pathway', row.get('Channel', 'Unknown'))
        
        # Control row
        if pd.notna(row.get('Control_Amplitude')):
            rows.append({
                'Subject': row['Cell_ID'],
                'Drug': 0,
                'Pathway': pathway_value,
                'Genotype': row['Genotype'],
                'ISI': row['ISI'],
                'EPSP_Amplitude': row['Control_Amplitude']
            })
        
        # Gabazine row
        if pd.notna(row.get('Gabazine_Amplitude')):
            rows.append({
                'Subject': row['Cell_ID'],
                'Drug': 1,
                'Pathway': pathway_value,
                'Genotype': row['Genotype'],
                'ISI': row['ISI'],
                'EPSP_Amplitude': row['Gabazine_Amplitude']
            })
    
    df = pd.DataFrame(rows)
    
    # Map pathway names to numbers (1=Perforant, 2=Schaffer, 3=Stratum_Oriens)
    pathway_map = {'Perforant': 1, 'Schaffer': 2, 'Stratum_Oriens': 3, 'Basal_Stratum_Oriens': 3,
                   'channel_1': 1, 'channel_2': 2, 'channel_3': 3}
    df['Pathway'] = df['Pathway'].replace(pathway_map)
    
    # Ensure numeric - extract digits if still string
    if df['Pathway'].dtype == 'object':
        df['Pathway'] = df['Pathway'].str.extract(r'(\d+)')[0]
        df['Pathway'] = pd.to_numeric(df['Pathway'], errors='coerce')
        df = df.dropna(subset=['Pathway'])
        df['Pathway'] = df['Pathway'].astype(int)
    
    pivot_df = df.pivot_table(
        index=['Subject', 'Drug', 'Pathway', 'Genotype'],
        columns='ISI',
        values='EPSP_Amplitude'
    ).reset_index()
    
    pivot_df.columns.name = None
    
    pivot_df = pivot_df.rename(columns={
        col: f'ISI{int(col)}' for col in pivot_df.columns 
        if isinstance(col, (int, float))
    })
    
    non_isi_cols = ['Subject', 'Drug', 'Pathway', 'Genotype']
    isi_cols = sorted(
        [col for col in pivot_df.columns if col.startswith('ISI')],
        key=lambda x: int(x.replace('ISI', ''))
    )
    
    return pivot_df[non_isi_cols + isi_cols]

def export_E_I_imbalance_for_R(amplitudes_df):
    """Export E-I imbalance in R format."""
    if len(amplitudes_df) == 0:
        return pd.DataFrame()
    
    df_imbalance = amplitudes_df[amplitudes_df['E_I_Imbalance'].notna()].copy()
    
    if len(df_imbalance) == 0:
        print("WARNING: No E-I imbalance data found")
        return pd.DataFrame()
    
    # Only rename Cell_ID to Subject - don't rename Channel since Pathway already exists
    df = df_imbalance.rename(columns={'Cell_ID': 'Subject'})
    
    # Use existing Pathway column and map to numbers (1=Perforant, 2=Schaffer, 3=Stratum_Oriens)
    if 'Pathway' in df.columns:
        pathway_map = {'Perforant': 1, 'Schaffer': 2, 'Stratum_Oriens': 3, 'Basal_Stratum_Oriens': 3,
                       'channel_1': 1, 'channel_2': 2, 'channel_3': 3}
        df['Pathway'] = df['Pathway'].replace(pathway_map)
    elif 'Channel' in df.columns:
        # Fallback: use Channel if no Pathway column
        pathway_map = {'channel_1': 1, 'channel_2': 2, 'channel_3': 3}
        df['Pathway'] = df['Channel'].replace(pathway_map)
    
    # Ensure Pathway is numeric
    df['Pathway'] = pd.to_numeric(df['Pathway'], errors='coerce').fillna(1).astype(int)
    
    
    df['ISI'] = df['ISI'].astype(str).apply(lambda x: f'ISI{x}')
    
    pivot_df = df.pivot_table(
        index=['Subject', 'Pathway', 'Genotype'],
        columns='ISI',
        values='E_I_Imbalance',
        aggfunc='mean'
    ).reset_index()
    
    pivot_df.columns.name = None
    
    other_cols = [col for col in pivot_df.columns if not col.startswith('ISI')]
    isi_cols = sorted(
        [col for col in pivot_df.columns if col.startswith('ISI')],
        key=lambda x: int(x.replace('ISI', ''))
    )
    
    return pivot_df[other_cols + isi_cols]

def export_E_I_data_with_R_format_options(amplitudes_df, base_output_path='paper_data/E_I'):
    """Export E-I data in R-friendly formats with user prompts."""
    results = {}
    
    print("\n" + "="*70)
    print("R FORMAT EXPORT OPTIONS")
    print("="*70)
    
    export_epsp = input("\nExport EPSP amplitudes (Control & Gabazine) in R format? (y/n): ").strip().lower()
    
    if export_epsp == 'y':
        epsp_R_df = export_EPSP_amplitudes_with_drug_for_R(amplitudes_df)
        
        if len(epsp_R_df) > 0:
            print(f"\nCreated EPSP amplitude R format with {len(epsp_R_df)} rows")
            print(f"ISI columns: {[col for col in epsp_R_df.columns if col.startswith('ISI')]}")
            print("\nFirst 5 rows:")
            print(epsp_R_df.head())
            
            epsp_path = f"{base_output_path}_EPSP_amplitudes_R_format.csv"
            epsp_R_df.to_csv(epsp_path, index=False)
            print(f"\n✓ Saved EPSP amplitudes R format to: {epsp_path}")
            
            results['EPSP_R_format'] = epsp_R_df
    
    export_imbalance = input("\nExport E-I imbalance ratios in R format? (y/n): ").strip().lower()
    
    if export_imbalance == 'y':
        imbalance_R_df = export_E_I_imbalance_for_R(amplitudes_df)
        
        if len(imbalance_R_df) > 0:
            print(f"\nCreated E-I imbalance R format with {len(imbalance_R_df)} subjects")
            print(f"ISI columns: {[col for col in imbalance_R_df.columns if col.startswith('ISI')]}")
            print("\nFirst 5 rows:")
            print(imbalance_R_df.head())
            
            imbalance_path = f"{base_output_path}_EI_imbalance_R_format.csv"
            imbalance_R_df.to_csv(imbalance_path, index=False)
            print(f"\n✓ Saved E-I imbalance R format to: {imbalance_path}")
            
            results['EI_Imbalance_R_format'] = imbalance_R_df
    
    print("\n" + "="*70)
    print("R FORMAT EXPORT COMPLETE")
    print("="*70)
    
    return results

def analyze_and_export_E_I_balance(master_df, data_dir, unitary_stim_starts_dict, ISI_times_dict_mapping,
                                   output_path_amplitudes=None, output_path_traces=None,
                                   export_R_formats=True, base_output_path_R='paper_data/E_I'):
    """
    Complete workflow: analyze E-I balance and export results.
    
    Parameters:
        master_df: Master dataframe with Cell_ID and 'ESPS Stim Time File Name' column
        data_dir: Path to directory containing .pkl files
        unitary_stim_starts_dict: Dictionary mapping 'older'/'newer' to unitary_stim_starts
        ISI_times_dict_mapping: Dictionary mapping 'older'/'newer' to ISI_times_dict
        output_path_amplitudes: Path to save amplitudes CSV
        output_path_traces: Path to save traces .pkl file
        export_R_formats: Whether to prompt for R format exports (default True)
        base_output_path_R: Base path for R format output files
    
    Returns:
        Dictionary with all analysis results
    """
    results = {}
    
    print("="*70)
    print("EXTRACTING E-I TRACES")
    print("="*70)
    
    E_I_traces = get_E_I_traces(data_dir, unitary_stim_starts_dict, ISI_times_dict_mapping, master_df)
    print(f"Extracted E-I traces for {len(E_I_traces)} cells")
    results['E_I_traces'] = E_I_traces
    
    print("\n" + "="*70)
    print("CALCULATING AMPLITUDES AND ESTIMATED INHIBITION")
    print("="*70)
    
    E_I_amplitudes, E_I_traces_updated = get_E_I_amplitudes_and_estimated_inhibition_traces(E_I_traces)
    print(f"Calculated amplitudes for {len(E_I_amplitudes)} cells")
    results['E_I_amplitudes'] = E_I_amplitudes
    results['E_I_traces_with_inhibition'] = E_I_traces_updated
    
    print("\n" + "="*70)
    print("CALCULATING E-I IMBALANCE")
    print("="*70)
    
    E_I_imbalances = calculate_E_I_imbalance(E_I_amplitudes)
    print(f"Calculated E-I imbalances for {len(E_I_imbalances)} cells")
    results['E_I_imbalances'] = E_I_imbalances
    
    print("\n" + "="*70)
    print("CALCULATING EXPECTED EPSPs")
    print("="*70)
    
    expected_EPSPs, expected_EPSPs_peaks = calculate_expected_EPSPs_for_all_cells(
        E_I_traces_updated, ISI_times_dict_mapping, master_df
    )
    print(f"Calculated expected EPSPs for {len(expected_EPSPs)} cells")
    results['expected_EPSPs'] = expected_EPSPs
    results['expected_EPSPs_peaks'] = expected_EPSPs_peaks
    
    print("\n" + "="*70)
    print("EXPORTING AMPLITUDES TO DATAFRAME")
    print("="*70)
    
    amplitudes_df = export_E_I_amplitudes_to_dataframe(
        E_I_amplitudes, E_I_imbalances, expected_EPSPs_peaks, master_df
    )
    
    if output_path_amplitudes:
        amplitudes_df.to_csv(output_path_amplitudes, index=False)
        print(f"✓ Amplitudes saved to: {output_path_amplitudes}")
    
    results['amplitudes_df'] = amplitudes_df
    
    print("\n" + "="*70)
    print("EXPORTING TRACES FOR PLOTTING")
    print("="*70)
    
    traces_df = export_E_I_traces_for_plotting(
        E_I_traces_updated, expected_EPSPs, master_df, output_path_traces
    )
    
    results['traces_df'] = traces_df
    
    if export_R_formats and len(amplitudes_df) > 0:
        R_formats = export_E_I_data_with_R_format_options(amplitudes_df, base_output_path_R)
        results['R_formats'] = R_formats
    
    print("\n" + "="*70)
    print("ANALYSIS SUMMARY")
    print("="*70)
    print(f"Total cells analyzed: {len(E_I_traces)}")
    print(f"Amplitudes DataFrame shape: {amplitudes_df.shape}")
    print(f"Traces DataFrame shape: {traces_df.shape}")
    
    if 'Genotype' in amplitudes_df.columns:
        print(f"\nCells per genotype:")
        print(amplitudes_df.groupby('Genotype')['Cell_ID'].nunique())
    
    print("\n" + "="*70)
    
    return results

#----------------------------------------------------------------------------------
# Unified Analysis of Plateau Potentials (Theta Burst Stimulation): # Handles: Gabazine-only, ML297+Gabazine (GIRK Agonist), ETX+Gabazine (GIRK Antagonist Also Measure Supralinearity
#----------------------------------------------------------------------------------

def convert_filename_to_standard_id(filename):
    """Converts filename (01042024_c1...) to Cell_ID (20240104_c1)."""
    try:
        match = re.search(r'^(\d{2})(\d{2})(\d{4})_(c\d+)', filename)
        if match:
            mm, dd, yyyy, cell = match.groups()
            return f"{yyyy}{mm}{dd}_{cell}"
        return None
    except Exception:
        return None

def determine_plateau_pathway(experiment_desc):
    """
    Determines pathway from experiment description.
    FIXED: Checks for 'Both' first to prevent misclassifying 
    'Both (Perforant+Schaffer)' as just 'Perforant'.
    """
    exp_lower = experiment_desc.lower()
    
    if 'both' in exp_lower: return 'Both'
    if 'perforant' in exp_lower: return 'Perforant'
    if 'schaffer' in exp_lower: return 'Schaffer'
    
    return 'Unknown'

# SHARED DATA LOADING & PARSING (Used by Area & Supralinearity)

def parse_plateau_sweeps_column(sweeps_entry):
    """
    Robustly parses 'Plateau Sweeps' column into a dictionary.
    Returns list of sweeps per drug: {'Gabazine': ['sweep_1', 'sweep_2']}
    """
    if pd.isna(sweeps_entry) or str(sweeps_entry).lower() in ['nan', '']:
        return {}

    sweeps_str = str(sweeps_entry).strip()
    # Remove outer braces
    if sweeps_str.startswith('{') and sweeps_str.endswith('}'):
        sweeps_str = sweeps_str[1:-1]
        
    # Split by comma or semicolon
    parts = re.split(r'[,;]', sweeps_str)
    
    result = {}
    current_key = None
    
    for part in parts:
        part = part.strip()
        if not part: continue
        
        # Check for Key: Value format
        if ':' in part:
            key, val = part.split(':', 1)
            current_key = key.strip()
            
            # Normalize Drug Names
            if current_key.lower() == 'etx': current_key = 'ETX'
            elif current_key.lower() == 'ml297': current_key = 'ML297'
            elif current_key.lower() == 'gabazine': current_key = 'Gabazine'
            
            sweep_val = val.strip()
        else:
            # No colon, use previous key
            if current_key is None:
                continue 
            sweep_val = part
            
        # Clean sweep value
        if 'sweep' in sweep_val.lower():
            if current_key not in result:
                result[current_key] = []
            result[current_key].append(sweep_val)
            
    return result

def load_plateau_traces_from_dir(data_dir, master_df=None):
    """
    Loads traces specifically for Plateau experiments from .pkl files.
    Returns keys as Standard IDs (YYYYMMDD_c#).
    """
    data_files = [f for f in os.listdir(data_dir) if f.endswith('.pkl')]
    plateau_traces = {}
    
    valid_ids = None
    if master_df is not None and 'Cell_ID' in master_df.columns:
        valid_ids = set(master_df['Cell_ID'].astype(str))
    
    print(f"Scanning {len(data_files)} files for Plateau traces...")

    for f in data_files:
        std_id = convert_filename_to_standard_id(f)
        if std_id is None: continue
        
        if valid_ids is not None and std_id not in valid_ids:
            continue
            
        try:
            df = pd.read_pickle(os.path.join(data_dir, f))
            for i in range(len(df)):
                row = df.iloc[i]
                desc = row.get('experiment_description', '')
                if 'Theta Stim' in desc:
                    if std_id not in plateau_traces: plateau_traces[std_id] = {}
                    if desc not in plateau_traces[std_id]: plateau_traces[std_id][desc] = {}
                    
                    # Look for offset trace first
                    trace = None
                    if 'intermediate_traces' in row:
                        trace = row['intermediate_traces'].get('offset_trace')
                    
                    if trace is not None:
                        plateau_traces[std_id][desc][f"sweep_{i}"] = trace
        except Exception as e:
            continue

    print(f"Loaded plateau traces for {len(plateau_traces)} unique cells.")
    return plateau_traces

# PLATEAU AREA ANALYSIS FUNCTIONS

def calculate_plateau_area_under_curve(trace, sampling_rate=20000):
    """
    Calculates Area Under Curve for Plateau Potentials.
    Adaptive: Uses available data if trace ends before 2200ms target.
    """
    start_ms = 500
    target_end_ms = 2200
    
    start_idx = int(start_ms * sampling_rate / 1000)
    target_end_idx = int(target_end_ms * sampling_rate / 1000)
    
    if trace is None or len(trace) <= start_idx:
        return np.nan
    
    # Adaptive End Index
    actual_end_idx = min(len(trace), target_end_idx)
    
    if actual_end_idx < (target_end_idx - int(50 * sampling_rate / 1000)):
        # Optional: Print warning for very short traces
        pass

    segment = trace[start_idx:actual_end_idx]
    segment[segment < 0] = 0 # Rectify negative values
    
    return np.trapz(segment, dx=1/sampling_rate)

def categorize_and_extract_plateau_data(plateau_traces, master_df):
    """
    Iterates through MasterDF and categorizes plateau experiments.
    """
    final_data_list = [] 
    final_traces_dict = {} 
    
    groups = ['Gabazine_Only', 'Before_ML297', 'After_ML297', 'Before_ETX', 'After_ETX']
    for g in groups: final_traces_dict[g] = {}

    if 'Cell_ID' not in master_df.columns:
         raise ValueError("Master DF missing 'Cell_ID' (yyyymmdd_c#)")
    master_df['Cell_ID'] = master_df['Cell_ID'].astype(str).str.strip()

    print("\nCategorizing Plateau Experimental Conditions...")

    for idx, row in master_df.iterrows():
        cell_id = row['Cell_ID']
        
        if cell_id not in plateau_traces: continue
        sweeps_map = parse_plateau_sweeps_column(row['Plateau Sweeps'])
        if not sweeps_map: continue
        
        drugs_present = list(sweeps_map.keys())
        
        # --- LOGIC TREE ---
        if 'ML297' in drugs_present and 'Gabazine' in drugs_present:
            for s in sweeps_map['Gabazine']:
                _extract_single_plateau_condition(cell_id, row, s, 'Before_ML297', plateau_traces, final_data_list, final_traces_dict)
            for s in sweeps_map['ML297']:
                _extract_single_plateau_condition(cell_id, row, s, 'After_ML297', plateau_traces, final_data_list, final_traces_dict)

        elif 'ETX' in drugs_present and 'Gabazine' in drugs_present:
            for s in sweeps_map['Gabazine']:
                _extract_single_plateau_condition(cell_id, row, s, 'Before_ETX', plateau_traces, final_data_list, final_traces_dict)
            for s in sweeps_map['ETX']:
                _extract_single_plateau_condition(cell_id, row, s, 'After_ETX', plateau_traces, final_data_list, final_traces_dict)

        elif 'Gabazine' in drugs_present and len(drugs_present) == 1:
             for s in sweeps_map['Gabazine']:
                 _extract_single_plateau_condition(cell_id, row, s, 'Gabazine_Only', plateau_traces, final_data_list, final_traces_dict)
    
    return final_data_list, final_traces_dict

def _extract_single_plateau_condition(cell_id, meta_row, ref_sweep_name, condition_label, 
                                      all_traces, data_list, traces_dict):
    """
    Extracts data for 'Both', 'Schaffer', and 'Perforant' pathways.
    Logic: Both = ref, Schaffer = ref-1, Perforant = ref-2.
    
    CRITICAL NOTE: For Schaffer and Perforant pathways ONLY (not Both),
    data is only analyzed from 07/09/2024 onwards due to:
    'Correct Hardware fix from this date on for individual pathways'
    """
    # Hardware fix cutoff date: 07/09/2024 -> Cell_ID format: 20240709
    HARDWARE_FIX_CUTOFF = 20240709
    
    if cell_id not in all_traces: return
    experiments = all_traces[cell_id]
    
    # Parse Reference Sweep Number (Assumed to be 'Both')
    try:
        ref_num = int(re.search(r'sweep_(\d+)', ref_sweep_name, re.IGNORECASE).group(1))
    except (AttributeError, ValueError):
        return

    target_sweeps = {
        'Both': f"sweep_{ref_num}",
        'Schaffer': f"sweep_{ref_num - 1}",
        'Perforant': f"sweep_{ref_num - 2}"
    }

    # Extract date from Cell_ID (format: yyyymmdd_c#)
    try:
        cell_date = int(cell_id.split('_')[0])
    except (ValueError, IndexError):
        cell_date = 0

    found_any = False

    for exp_desc, sweeps in experiments.items():
        pathway = determine_plateau_pathway(exp_desc)
        if pathway == 'Unknown': continue
        
        # EXCLUSION CHECK: 'Single Pathway Plateau Inclusion'
        # If 'No', skip independent Schaffer/Perforant analysis
        if pathway in ['Schaffer', 'Perforant']:
            single_pathway_inclusion = str(meta_row.get('Single Pathway Plateau Inclusion', 'nan')).strip()
            if single_pathway_inclusion == 'No':
                continue
        
        target_sweep = target_sweeps.get(pathway)
        
        if target_sweep and target_sweep in sweeps:
            trace = sweeps[target_sweep]
            area = calculate_plateau_area_under_curve(trace)
            
            if np.isnan(area):
                print(f"Warning: {cell_id} {pathway} ({target_sweep}) - Area calc failed (NaN).")
            
            data_list.append({
                'Cell_ID': cell_id,
                'Genotype': meta_row.get('Genotype'),
                'Sex': meta_row.get('Sex'),
                'Condition': condition_label,
                'Pathway': pathway,
                'Sweep': target_sweep,
                'Plateau_Area': area
            })
            
            if cell_id not in traces_dict[condition_label]:
                traces_dict[condition_label][cell_id] = {}
            
            traces_dict[condition_label][cell_id][pathway] = trace
            found_any = True

def export_plateau_master_dataframe(data_list, output_path):
    df = pd.DataFrame(data_list)
    if df.empty: return df
    
    def get_drug_code(cond):
        if 'ML297' in cond: return 2
        if 'ETX' in cond: return 3
        return 1 
        
    df['Drug_Code'] = df['Condition'].apply(get_drug_code)
    
    def get_cond_code(cond):
        if 'After' in cond: return 2
        return 1 
    
    df['Condition_Code'] = df['Condition'].apply(get_cond_code)
    
    df.to_csv(output_path, index=False)
    print(f"✓ Master Plateau Dataframe saved to: {output_path}")
    return df

def export_plateau_traces_for_plotting(traces_dict, master_df, output_path):
    final_export = {}
    print("\nPreparing Plateau Trace Export...")
    
    for condition, cells in traces_dict.items():
        final_export[condition] = {}
        count = 0
        for cell_id, pathways in cells.items():
            meta_rows = master_df[master_df['Cell_ID'] == cell_id]
            if not meta_rows.empty:
                meta = meta_rows.iloc[0]
                geno = meta.get('Genotype')
                sex = meta.get('Sex')
            else:
                geno = 'Unknown'
                sex = 'Unknown'

            final_export[condition][cell_id] = {
                'genotype': geno,
                'sex': sex,
                'traces': pathways 
            }
            count += 1
        print(f"  > {condition}: {count} cells packaged.")
            
    pd.to_pickle(final_export, output_path)
    print(f"✓ All Plateau traces saved to: {output_path}")
    return final_export

# Supralinear AREA ANALYSIS FUNCTIONS

def filter_master_df_for_supralinearity(master_df):
    col_name = 'Supralinear Analysis Inclusion'
    if col_name not in master_df.columns:
        print(f"WARNING: '{col_name}' column not found. Returning full dataframe.")
        return master_df
    if 'Cell_ID' in master_df.columns:
        master_df['Cell_ID'] = master_df['Cell_ID'].astype(str)
    mask = master_df[col_name].astype(str).str.strip().str.lower().str.startswith('yes')
    filtered_df = master_df[mask].copy()
    print(f"Supralinear Filtering: Kept {len(filtered_df)} cells.")
    return filtered_df

def build_theta_protocol_map(master_df, theta_stim_protocols):
    theta_protocol_dict = {}
    target_col = 'Theta Burst Stim Time File Name' 
    print("Building Protocol Map...")
    for idx, row in master_df.iterrows():
        try:
            raw_date = str(row['Date'])
            if '-' in raw_date:
                 dt = pd.to_datetime(raw_date)
                 cell_date = dt.strftime('%Y%m%d')
            elif '.' in raw_date:
                 parts = raw_date.split('.')
                 cell_date = f"{parts[2]}{parts[0].zfill(2)}{parts[1].zfill(2)}"
            else:
                d_str = raw_date.split('.')[0].zfill(8)
                cell_date = f"{d_str[4:]}{d_str[:2]}{d_str[2:4]}"
            cell = str(row['Cell']).split('.')[0]
            if not cell.startswith('c'): cell = f'c{cell}'
            cell_name = f'{cell_date}_{cell}'
            
            if target_col in row:
                theta_protocol_str = str(row[target_col]).strip()
                if theta_protocol_str.endswith('.dat'): theta_protocol_str = theta_protocol_str[:-4]
                if theta_protocol_str in theta_stim_protocols:
                    theta_protocol_dict[cell_name] = theta_stim_protocols[theta_protocol_str]
        except Exception: continue
    return theta_protocol_dict

def find_matching_E_I_key(standard_id, ei_keys):
    try:
        yyyy = standard_id[:4]
        mm = standard_id[4:6]
        dd = standard_id[6:8]
        cell_suffix = standard_id.split('_')[1]
    except: return None
    candidates = [
        f"{mm}{dd}{yyyy}_{cell_suffix}_processed_data",
        f"{int(mm)}{int(dd)}{yyyy}_{cell_suffix}_processed_data"
    ]
    for c in candidates:
        if c in ei_keys: return c
    for k in ei_keys:
        if standard_id in k or f"{mm}{dd}{yyyy}" in k:
            if cell_suffix in k: return k
    return None

def analyze_supralinearity_peaks(plateau_traces, E_I_traces, theta_stim_protocols, master_df, 
                               window_ms=100, sampling_rate=20000):
    results = []
    trace_export = {} 
    
    dt = 1000 / sampling_rate
    window_pts = int(window_ms / dt)
    
    cell_protocol_map = build_theta_protocol_map(master_df, theta_stim_protocols)
    print(f"Analyzing {len(plateau_traces)} cells...")
    
    for cell_key, pathways in plateau_traces.items():
        # Get pathway-specific stim times from master_df
        meta = master_df[master_df['Cell_ID'].astype(str) == cell_key]
        if meta.empty:
            genotype, sex = 'Unknown', 'Unknown'
            stim_times_dict = {}
        else:
            genotype = meta.iloc[0]['Genotype']
            sex = meta.iloc[0]['Sex']
            
            # Extract stim times from master_df
            stim_times_dict = {}
            
            # Protocol Lookup using 'Theta Burst Stim Time File Name'
            theta_col = 'Theta Burst Stim Time File Name'
            
            protocol_str = ''
            if theta_col in meta.columns:
                protocol_str = str(meta.iloc[0][theta_col]).strip()
            
            # Fallback for specific known cell with missing metadata
            if (protocol_str == 'nan' or not protocol_str) and cell_key == '20250122_c1':
                 print(f"DEBUG: Using fallback protocol for {cell_key}")
                 protocol_str = 'Theta_Burst_MCIII_new_variant_2'

            if protocol_str and protocol_str != 'nan':
                if protocol_str.endswith('.dat'):
                    protocol_str = protocol_str[:-4]
                
                if protocol_str in theta_stim_protocols:
                    stim_times = theta_stim_protocols[protocol_str]
                    # Apply same stim times to all relevant pathways
                    stim_times_dict = {
                        'Perforant': stim_times,
                        'Schaffer': stim_times,
                        'Both Pathways': stim_times
                    }
        
        if not stim_times_dict:
            continue
        
        trace_export[cell_key] = {}
        ei_key = find_matching_E_I_key(cell_key, E_I_traces.keys())
        if not ei_key: continue

        flat_pathways = {}
        for desc, sweeps in pathways.items():
            p_type = 'Unknown'
            if 'Both' in desc: p_type = 'Both Pathways'
            elif 'Perforant' in desc: p_type = 'Perforant'
            elif 'Schaffer' in desc: p_type = 'Schaffer'
            
            valid_sweeps = [v for k,v in sweeps.items() if isinstance(v, np.ndarray)]
            if valid_sweeps:
                min_len = min(len(s) for s in valid_sweeps)
                avg = np.mean([s[:min_len] for s in valid_sweeps], axis=0)
                flat_pathways[p_type] = avg

        
        
        # --- PHASE 1: COMPUTE EXPECTED TRACES (Schaffer & Perforant) ---
        # We need these regardless of whether we have corresponding plateau traces,
        # because the "Both Pathways" expected trace is the sum of these two.
        
        saved_expected_traces = {}
        
        # Define base length for expected traces (usually same as plateau trace len)
        # We'll use the length of whatever plateau trace is available to set the size
        trace_len = 0
        first_available_trace = next(iter(flat_pathways.values())) if flat_pathways else None
        if first_available_trace is not None:
            trace_len = len(first_available_trace)
        
        if trace_len > 0:
            for pathway_name in ['Schaffer', 'Perforant']:
                # NOTE: We DO NOT apply the hardware fix date check here.
                # Even if the plateau traces for Schaffer/Perforant are excluded (pre-July 2024),
                # we still need their Unitary E:I data to build the expected trace for "Both Pathways".
                # The Unitary E:I recordings are considered valid for all dates.
                
                pathway_stim_times = stim_times_dict.get(pathway_name)
                if not pathway_stim_times: continue
                
                # Get Unitary EPSP
                unitary = None
                unitary_block = E_I_traces[ei_key].get(300, {})
                target_channel = 'channel_2' if pathway_name == 'Schaffer' else 'channel_1'
                
                if target_channel in unitary_block:
                    conds = unitary_block[target_channel]
                    u_data = conds.get('Gabazine')
                    if u_data:
                        unitary = u_data.get('unitary_average_traces')
                
                if unitary is not None and len(unitary) >= 100 and not np.all(unitary == 0):
                    # Compute expected trace
                    expected_trace = create_expected_EPSP_theta(
                        unitary, 
                        pathway_stim_times,
                        total_len=trace_len, 
                        sample_rate=sampling_rate,
                        zero_clip=True
                    )
                    saved_expected_traces[pathway_name] = expected_trace

        # --- PHASE 2: ANALYZE PLATEAU TRACES (AUC) ---
        # Now process whatever plateau traces we actually have
        
        pathway_order = ['Schaffer', 'Perforant', 'Both Pathways']
        
        for pathway_name in pathway_order:
            if pathway_name not in flat_pathways:
                continue
                
            measured_trace = flat_pathways[pathway_name]
            if measured_trace is None or len(measured_trace) == 0:
                continue
            
            # EXCLUSION CHECK: 'Single Pathway Plateau Inclusion'
            # If 'No', skip independent Schaffer/Perforant analysis
            if pathway_name in ['Schaffer', 'Perforant']:
                # 'meta' df is defined at start of loop
                single_pathway_inclusion = str(meta.iloc[0].get('Single Pathway Plateau Inclusion', 'nan')).strip().lower()
                if single_pathway_inclusion == 'no':
                    continue
            
            pathway_stim_times = stim_times_dict.get(pathway_name)
            if not pathway_stim_times: continue
                
            # Define cycles (every 5 pulses = 1 cycle)
            current_cycle_starts = [pathway_stim_times[i] for i in range(0, len(pathway_stim_times), 5)]
            
            # Define window parameters
            dt = 1000.0 / sampling_rate  # Time per sample in ms
            window_ms = 200  # 200ms per cycle
            window_pts = int(window_ms / dt)

            # Baseline correction
            pre_stim_window = int(0.020 * sampling_rate)
            if len(measured_trace) > pre_stim_window:
                baseline_val = np.nanmean(measured_trace[:pre_stim_window])
            else:
                baseline_val = 0
            measured_trace_corrected = measured_trace - baseline_val

            # Get the appropriate Expected Trace
            expected_trace = None
            
            if pathway_name in ['Schaffer', 'Perforant']:
                expected_trace = saved_expected_traces.get(pathway_name)
                
                # Resize if necessary to match measured trace
                if expected_trace is not None:
                     if len(expected_trace) < len(measured_trace_corrected):
                        expected_trace = np.pad(expected_trace, 
                                               (0, len(measured_trace_corrected) - len(expected_trace)),
                                               'constant', constant_values=0)
                     elif len(expected_trace) > len(measured_trace_corrected):
                        expected_trace = expected_trace[:len(measured_trace_corrected)]

            elif pathway_name == 'Both Pathways':
                # Sum the individual expected traces
                if 'Schaffer' in saved_expected_traces and 'Perforant' in saved_expected_traces:
                    schaffer_expected = saved_expected_traces['Schaffer']
                    perforant_expected = saved_expected_traces['Perforant']
                    
                    min_len = min(len(schaffer_expected), len(perforant_expected), len(measured_trace_corrected))
                    exp_sum = schaffer_expected[:min_len] + perforant_expected[:min_len]
                    
                    # Pad/Trimming logic (reuse from before)
                    if len(exp_sum) < len(measured_trace_corrected):
                        expected_trace = np.pad(exp_sum, (0, len(measured_trace_corrected) - len(exp_sum)), 'constant')
                    else:
                        expected_trace = exp_sum[:len(measured_trace_corrected)]
            
            # If we couldn't create an expected trace (e.g. missing E:I data), skip this pathway
            if expected_trace is None:
                continue

            # Calculate difference
            difference_trace = measured_trace_corrected - expected_trace

            trace_export[cell_key][pathway_name] = {
                'Measured': measured_trace_corrected,
                'Expected': expected_trace,
                'Difference': difference_trace
            }
            
            # Important: unitary variable is no longer needed for AUC loop logic below
            # We just set unitary=None to skip the old Expected Trace creation block if we fell through
            unitary = None 

            
            # Proceed to AUC analysis using the variables defined above
            # (expected_trace and difference_trace are already set)
            
            # 1. Apply NaN to non-positive values in Measured Trace
            measured_trace_nan = measured_trace_corrected.copy()
            # measured_trace_nan[measured_trace_nan <= 0] = np.nan
            
            # 2. Update Difference Trace (for export/plotting) to allow NaNs
            difference_trace = np.where(
                np.isnan(measured_trace_nan),
                np.nan,
                measured_trace_nan - expected_trace
            )
            
            # 3. Peak Calculation Per Cycle
            for cycle_idx, start_time in enumerate(current_cycle_starts):
                start_idx = int(start_time / dt)
                end_idx = min(start_idx + window_pts, len(measured_trace_nan))
                
                win_meas = measured_trace_nan[start_idx:end_idx]
                win_exp = expected_trace[start_idx:end_idx]
                
                if len(win_meas) == 0: continue
                
                # Calculate Peaks using np.nanpercentile (Robust Max limits noise/artifacts)
                # If window is all NaNs, nanpercentile returns NaN
                if np.all(np.isnan(win_meas)):
                    meas_peak = np.nan
                else:
                    # Using 98th percentile instead of max to ignore artifacts (top 2% outliers)
                    meas_peak = np.nanpercentile(win_meas, 98)
                    
                if np.all(np.isnan(win_exp)):
                    exp_peak = np.nan
                else:
                    exp_peak = np.nanpercentile(win_exp, 98)
                
                # Calculate Supralinear Peak
                if np.isnan(meas_peak) or np.isnan(exp_peak):
                    diff_peak = np.nan
                else:
                    diff_peak = meas_peak - exp_peak
                
                # If NaN (no valid data in window), default to 0 or handle in downstream
                if np.isnan(diff_peak): diff_peak = 0
                if np.isnan(meas_peak): meas_peak = 0
                if np.isnan(exp_peak): exp_peak = 0
                
                # --- Calculate Area Under Curve (AUC) for Difference Trace ---
                # This provides an integrated measure of supralinearity across the entire cycle
                # CRITICAL: Exclude test pulse window from integration
                # EXPANDED WINDOW: 30-250ms to fully capture onset, duration, and recovery
                win_diff = difference_trace[start_idx:end_idx]
                
                # Calculate test pulse indices relative to trace start
                # Test pulse EXPANDED window: 30ms start, 220ms duration (30-250ms)
                # This ensures we exclude all test pulse effects including recovery
                test_pulse_start_ms = 30.0
                test_pulse_end_ms = 250.0
                test_pulse_start_idx = int(test_pulse_start_ms / dt)
                test_pulse_end_idx = int(test_pulse_end_ms / dt)
                
                # Use trapezoidal integration
                # dx is the time step in ms
                dx = dt  # ms per sample
                
                # Calculate AUC (mV·ms initially, then convert to mV·s)
                # Create a mask to exclude both NaN values AND test pulse window
                valid_indices = ~np.isnan(win_diff)
                
                # Exclude test pulse window from integration
                # Get absolute indices of the window we're integrating
                for i in range(len(win_diff)):
                    absolute_idx = start_idx + i
                    # If this sample falls within test pulse window, mark as invalid
                    if test_pulse_start_idx <= absolute_idx < test_pulse_end_idx:
                        valid_indices[i] = False
                
                if np.any(valid_indices):
                    # Use only valid points (excluding NaN and test pulse) for integration
                    valid_diff = win_diff[valid_indices]
                    # Create time array for valid points
                    time_array = np.arange(len(win_diff))[valid_indices] * dx
                    
                    # Trapezoidal integration gives mV·ms
                    diff_auc_ms = np.trapz(valid_diff, time_array)
                    # Convert to mV·s (divide by 1000)
                    diff_auc = diff_auc_ms / 1000.0
                else:
                    diff_auc = 0

                results.append({
                    'Cell_ID': cell_key,
                    'Genotype': genotype,
                    'Sex': sex,
                    'Pathway': pathway_name,
                    'Cycle_Index': cycle_idx + 1,
                    'Measured_Peak': meas_peak,
                    'Expected_Peak': exp_peak,
                    'Difference_Peak': diff_peak,
                    'Difference_AUC': diff_auc  # mV·s
                })

    return results, trace_export

def create_expected_EPSP_theta(unitary_EPSP, stim_times, total_len=26000, sample_rate=20000, baseline_ms=10, zero_clip=True):
    baseline_len = int(baseline_ms * sample_rate / 1000)
    if unitary_EPSP is None or len(unitary_EPSP) <= baseline_len: return np.zeros(total_len)
    unitary_len = len(unitary_EPSP) - baseline_len
    expected_EPSP_theta = np.zeros(total_len)
    
    stims = stim_times if isinstance(stim_times, (list, np.ndarray)) else []
    
    # 1. Clean Unitary Template: Remove Stimulus Artifact (0-2ms)
    # The artifact adds up linearly and ruins the expected trace peak.
    unitary_clean = unitary_EPSP.copy()
    artifact_pts = int(2.5 * sample_rate / 1000) # 2.5 ms
    epsp_start_idx = baseline_len
    
    # Check boundaries
    if epsp_start_idx + artifact_pts < len(unitary_clean):
        # Linear interp from 0 to end of artifact
        val_start = unitary_clean[epsp_start_idx] # Often this is where artifact starts (0)
        # Actually, usually artifact starts AT baseline_len.
        # We want to interpolate from baseline_len to baseline_len + artifact_pts
        # Set to linear ramp or 0? 0 is safer if baseline is 0.
        # Ideally, interp from 'val_pre' to 'val_post'.
        val_pre = unitary_clean[epsp_start_idx - 1] if epsp_start_idx > 0 else 0
        val_post = unitary_clean[epsp_start_idx + artifact_pts]
        
        interp_vals = np.linspace(val_pre, val_post, artifact_pts + 1)
        unitary_clean[epsp_start_idx : epsp_start_idx + artifact_pts + 1] = interp_vals

    for stim in stims:
        stim_index = int(stim * sample_rate / 1000)
        # Shift start to align 'epsp_start' with 'stim_index'
        # The unitary_clean array has baseline before epsp_start.
        # We want to add unitary_clean[epsp_start:] starting at stim_index.
        
        start_fill = stim_index
        end_fill = min(total_len, start_fill + unitary_len)
        len_fill = end_fill - start_fill
        
        if len_fill > 0:
            expected_EPSP_theta[start_fill : end_fill] += unitary_clean[epsp_start_idx : epsp_start_idx + len_fill]

    # Only zero-clip if requested (for Expected traces)
    # Don't clip for Difference traces which need to show negative values
    if zero_clip:
        expected_EPSP_theta = zero_clip_and_interpolate(expected_EPSP_theta)

    return expected_EPSP_theta

def export_supralinearity_wide_format(results_list, output_path, value_column='Difference_Peak'):
    if not results_list: return pd.DataFrame()
    df = pd.DataFrame(results_list)
    pivot_df = df.pivot_table(
        index=['Cell_ID', 'Genotype', 'Sex', 'Pathway'],
        columns='Cycle_Index',
        values=value_column  # Now flexible
    ).reset_index()
    pivot_df.columns.name = None
    rename_map = {col: f"Cycle_{col}" for col in pivot_df.columns if isinstance(col, (int, float))}
    pivot_df = pivot_df.rename(columns=rename_map)
    if output_path: pivot_df.to_csv(output_path, index=False)
    return pivot_df

#----------------------------------------------------------------------------------
# Spike Rate Per Theta Cycle Analysis
#----------------------------------------------------------------------------------

def analyze_spike_rate_per_theta_cycle(data_dir, master_df, sampling_rate=20000, plateau_traces=None):
    """
    Analyze spike rate per theta cycle for Theta_Burst stimulation.
    
    Theta cycles: 5 cycles x 200ms each, starting at 500ms
    Cycle windows: 500-700ms, 700-900ms, 900-1100ms, 1100-1300ms, 1300-1500ms
    
    Parameters:
        data_dir: Path to directory containing cell pkl files (used if plateau_traces is None)
        master_df: Master dataframe with Cell_ID and Genotype
        sampling_rate: Acquisition frequency in Hz (default 20000)
        plateau_traces: Optional pre-loaded traces dict (preferred method)
        
    Returns:
        results_list: List of dicts with per-cycle spike rate data
        spike_rates_per_cycle: Nested dict structure for plotting
    """
    from scipy.signal import find_peaks
    
    results_list = []
    
    # Structure: spike_rates[pathway][genotype][cycle] = list of rates
    spike_rates_per_cycle = {
        'Schaffer': {'WT': {1:[], 2:[], 3:[], 4:[], 5:[]}, 
                     'GNB1': {1:[], 2:[], 3:[], 4:[], 5:[]}},
        'Perforant': {'WT': {1:[], 2:[], 3:[], 4:[], 5:[]}, 
                      'GNB1': {1:[], 2:[], 3:[], 4:[], 5:[]}},
        'Both Pathways': {'WT': {1:[], 2:[], 3:[], 4:[], 5:[]}, 
                          'GNB1': {1:[], 2:[], 3:[], 4:[], 5:[]}}
    }
    
    # Get genotype lookup
    genotype_lookup = dict(zip(master_df['Cell_ID'].astype(str), master_df['Genotype']))
    sex_lookup = dict(zip(master_df['Cell_ID'].astype(str), master_df['Sex']))
    
    print("  Analyzing spike rate per theta cycle...")
    
    # --------------------------------------------------------------------------
    # METHOD 1: USE PRE-LOADED TRACES (Robust to missing stim_type metadata)
    # --------------------------------------------------------------------------
    if plateau_traces is not None:
        print(f"  Using pre-loaded traces for {len(plateau_traces)} cells...")
        
        for cell_id, protocols in plateau_traces.items():
            genotype = genotype_lookup.get(cell_id, None)
            sex = sex_lookup.get(cell_id, 'Unknown')
            if genotype is None: continue
            
            for desc, sweeps in protocols.items():
                # Map description to Pathway
                pathway = None
                if 'Both' in desc or 'Both Pathway' in desc: pathway = 'Both Pathways'
                elif 'Perforant' in desc: pathway = 'Perforant'
                elif 'Schaffer' in desc: pathway = 'Schaffer'
                
                if pathway is None: continue
                
                # Count spikes per theta cycle for each sweep
                cycle_rates_per_sweep = {1:[], 2:[], 3:[], 4:[], 5:[]}
                
                # Iterate sweeps (values are arrays)
                for sweep_name, trace in sweeps.items():
                    if trace is None or len(trace) == 0: continue
                    
                    if isinstance(trace, dict):
                        print(f"Warning: Trace is dict for {cell_id}, {desc}, {sweep_name}. Keys: {trace.keys()}")
                        continue
                    
                    # Assume sampling rate if not in trace (it's raw array)
                    acq_freq = sampling_rate
                    
                    # Find all spikes
                    peaks, _ = find_peaks(trace, height=0, prominence=20)
                    peak_times_ms = peaks / (acq_freq / 1000)
                    
                    # Count spikes in each 200ms theta cycle window
                    for cycle in range(1, 6):
                        cycle_start_ms = 500 + (cycle - 1) * 200
                        cycle_end_ms = cycle_start_ms + 200
                        
                        spikes_in_cycle = np.sum((peak_times_ms >= cycle_start_ms) & (peak_times_ms < cycle_end_ms))
                        spike_rate_hz = spikes_in_cycle / 0.2
                        cycle_rates_per_sweep[cycle].append(spike_rate_hz)
                
                # Average per cycle across sweeps for this cell
                for cycle in range(1, 6):
                    if cycle_rates_per_sweep[cycle]:
                        avg_rate = np.mean(cycle_rates_per_sweep[cycle])
                        spike_rates_per_cycle[pathway][genotype][cycle].append(avg_rate)
                        
                        results_list.append({
                            'Cell_ID': cell_id,
                            'Genotype': genotype,
                            'Sex': sex,
                            'Pathway': pathway,
                            'Cycle_Index': cycle,
                            'Spike_Rate_Hz': avg_rate
                        })

    # --------------------------------------------------------------------------
    # METHOD 2: SCAN FILES (Fallback / Legacy)
    # --------------------------------------------------------------------------
    else:
        print("  Scanning files for theta traces (legacy method)...")
        # Mapping from stim_type to pathway key
        stim_pathway_map = {
            'Theta_Burst_Schaffer': 'Schaffer',
            'Theta_Burst_Perforant': 'Perforant',
            'Theta_Burst_Both_Pathway': 'Both Pathways'
        }

        import os
        for filename in os.listdir(data_dir):
            if not filename.endswith('.pkl'): continue
            
            cell_id = convert_pkl_filename_to_cell_id(filename)
            if cell_id is None: continue
            
            genotype = genotype_lookup.get(cell_id, None)
            sex = sex_lookup.get(cell_id, 'Unknown')
            if genotype is None: continue
            
            try:
                df_cell = pd.read_pickle(os.path.join(data_dir, filename))
                valid_conditions = ['Gabazine', 'Gabazine_Only', 'Before_ML297', 'Before_ETX']
                
                for stim_type, pathway in stim_pathway_map.items():
                    if 'stim_type' not in df_cell.columns: continue
                        
                    theta_sweeps = df_cell[
                        (df_cell['stim_type'] == stim_type) &
                        (df_cell['condition'].isin(valid_conditions) if 'condition' in df_cell.columns else True)
                    ]
                    
                    if theta_sweeps.empty: continue
                    
                    cycle_rates_per_sweep = {1:[], 2:[], 3:[], 4:[], 5:[]}
                    
                    for _, row in theta_sweeps.iterrows():
                        sweep = row.get('sweep', None)
                        if sweep is None: continue
                        
                        acq_freq = row.get('acquisition_frequency', sampling_rate)
                        peaks, _ = find_peaks(sweep, height=0, prominence=20)
                        peak_times_ms = peaks / (acq_freq / 1000)
                        
                        for cycle in range(1, 6):
                            cycle_start_ms = 500 + (cycle - 1) * 200
                            cycle_end_ms = cycle_start_ms + 200
                            spikes_in_cycle = np.sum((peak_times_ms >= cycle_start_ms) & (peak_times_ms < cycle_end_ms))
                            cycle_rates_per_sweep[cycle].append(spikes_in_cycle / 0.2)
                    
                    for cycle in range(1, 6):
                        if cycle_rates_per_sweep[cycle]:
                            avg_rate = np.mean(cycle_rates_per_sweep[cycle])
                            spike_rates_per_cycle[pathway][genotype][cycle].append(avg_rate)
                            
                            results_list.append({
                                'Cell_ID': cell_id,
                                'Genotype': genotype,
                                'Sex': sex,
                                'Pathway': pathway,
                                'Cycle_Index': cycle,
                                'Spike_Rate_Hz': avg_rate
                            })
                            
            except Exception as e:
                continue
    
    n_cells = len(set([r['Cell_ID'] for r in results_list])) if results_list else 0
    print(f"  Spike rates per cycle analyzed for {n_cells} cells")
    
    return results_list, spike_rates_per_cycle

def export_spike_rate_wide_format(results_list, output_path):
    """
    Export spike rate per cycle data in wide format for R statistics.
    Each row is one cell-pathway combination with columns: Cycle_1...Cycle_5
    """
    if not results_list: 
        return pd.DataFrame()
        
    df = pd.DataFrame(results_list)
    
    pivot_df = df.pivot_table(
        index=['Cell_ID', 'Genotype', 'Sex', 'Pathway'],
        columns='Cycle_Index',
        values='Spike_Rate_Hz'
    ).reset_index()
    
    pivot_df.columns.name = None
    rename_map = {col: f"Cycle_{col}" for col in pivot_df.columns if isinstance(col, (int, float))}
    pivot_df = pivot_df.rename(columns=rename_map)
    
    if output_path: 
        pivot_df.to_csv(output_path, index=False)
        print(f"  ✓ Exported spike rate data to: {output_path}")
    
    return pivot_df

def analyze_gabab_component(all_traces, color, channel_to_plot='channel_1', title='', ax=None, condition_to_plot='gabazine', save_path=None):
    """
    Analyze the GABAb component from unitary traces.
    Calculates Integral, Trough Amp, and plots Mean +/- SEM.
    """
    gabab_measurements = {}
    traces_to_plot = []
    dt = 1/20000 

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
        show_at_end = True
    else:
        show_at_end = False

    for cell in all_traces:
        for ISI in all_traces[cell]:
            if ISI == 300:
                for channel in all_traces[cell][ISI]:
                    if channel == channel_to_plot:
                        for condition in all_traces[cell][ISI][channel]:
                            valid_conds = ['gabazine', 'gabazine + ml297', 'gabazine + etx', 'gabazine + baclofen']
                            if condition.lower() in valid_conds:
                                trace = all_traces[cell][ISI][channel][condition]['unitary_average_traces']
                                
                                # SAFETY: Check if trace exists and convert to numpy array
                                if trace is None or len(trace) == 0:
                                    continue
                                if not isinstance(trace, np.ndarray):
                                    trace = np.array(trace)

                                if condition.lower() == condition_to_plot.lower():
                                    traces_to_plot.append(trace)
                                
                                # Analyze this specific trace
                                time = np.arange(0, len(trace) * dt * 1000, dt * 1000)
                                negative_trace = np.where(trace < 0, trace, 0)
                                integral_below_zero = -np.trapz(negative_trace, dx=dt)
                                trough_amplitude_neg = np.min(trace)
                                trough_amplitude_abs = np.abs(trough_amplitude_neg)
                                trough_time = time[np.argmin(trace)]

                                if cell not in gabab_measurements: gabab_measurements[cell] = {}
                                gabab_measurements[cell][condition] = {
                                    'Trough Amplitude (mV)': trough_amplitude_abs,
                                    'Trough Time (ms)': trough_time,
                                    'Integral Below Zero (mV*ms)': integral_below_zero
                                }

    if not traces_to_plot:
        print(f"No traces found for {channel_to_plot} / {condition_to_plot}.")
        return gabab_measurements

    traces_to_plot = np.array(traces_to_plot)
    max_len = max([len(trace) for trace in traces_to_plot])
    
    padded_traces = []
    for trace in traces_to_plot:
        if len(trace) < max_len:
            padded_trace = np.pad(trace, (0, max_len - len(trace)), mode='constant', constant_values=0)
        else:
            padded_trace = trace
        padded_traces.append(padded_trace)

    padded_traces = np.array(padded_traces)
    time = np.arange(0, max_len * dt * 1000, dt * 1000)

    mean_trace = np.mean(padded_traces, axis=0)
    std_trace = np.std(padded_traces, axis=0)
    sem_trace = std_trace / np.sqrt(len(padded_traces))

    ax.plot(time[:len(mean_trace)], mean_trace, label=title, color=color)  
    ax.fill_between(time[:len(mean_trace)], mean_trace - sem_trace, mean_trace + sem_trace, color='gray', alpha=0.3)

    negative_mean_trace = np.where(mean_trace < 0, mean_trace, 0)
    ax.fill_between(time[:len(mean_trace)], negative_mean_trace, 0, color='lightblue', alpha=0.3)

    trough_val = np.min(mean_trace)
    trough_t = time[np.argmin(mean_trace)]
    ax.scatter(trough_t, trough_val, color='red')

    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Voltage (mV)')
    ax.legend()
    
    if show_at_end:
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path)
            plt.close(fig)
            print(f"Figure saved to: {save_path}")
        else:
            plt.show()

    return gabab_measurements

def collect_gabab_traces_for_export(all_traces, channel, condition, genotype_dict):
    """Helper to get Mean and SEM traces for exporting to CSV."""
    collected = {'WT': [], 'GNB1': []}
    dt = 1/20000 

    for cell in all_traces:
        genotype = genotype_dict.get(cell, 'Unknown')
        if genotype not in collected: continue
        
        for ISI in all_traces[cell]:
            if ISI == 300:
                for ch in all_traces[cell][ISI]:
                    if ch == channel:
                        for cond in all_traces[cell][ISI][ch]:
                            if cond.lower() == condition.lower():
                                trace = all_traces[cell][ISI][ch][cond]['unitary_average_traces']
                                # Safety check for empty traces
                                if trace is not None and len(trace) > 0:
                                    collected[genotype].append(trace)

    export_data = {}
    
    for geno in ['WT', 'GNB1']:
        traces = collected[geno]
        if not traces: continue
        
        max_len = max(len(t) for t in traces)
        padded = [np.pad(t, (0, max_len - len(t)), mode='constant', constant_values=0) for t in traces]
        arr = np.array(padded)
        
        mean_trace = np.mean(arr, axis=0)
        sem_trace = np.std(arr, axis=0) / np.sqrt(len(arr))
        
        time = np.arange(len(mean_trace)) * dt * 1000
        export_data[f"{geno}_Time"] = time
        export_data[f"{geno}_Mean"] = mean_trace
        export_data[f"{geno}_SEM"] = sem_trace
        
    return export_data

def collect_gabab_traces_by_condition(all_traces, channel_map, conditions, genotype_dict, sex_dict):
    """
    Collects traces organized by Condition -> Cell -> Pathway.
    Matches structure of export_plateau_traces_for_plotting.
    """
    final_export = {}
    
    for condition in conditions:
        final_export[condition] = {}
        
        for cell_id, isi_data in all_traces.items():
            if 300 not in isi_data: continue
            
            # Check if this cell has data for this condition in any channel
            cell_traces = {}
            
            for channel, pathway_name in channel_map.items():
                if channel in isi_data[300]:
                    # Find specific condition key (case-insensitive)
                    actual_cond = None
                    for c in isi_data[300][channel]:
                        if c.lower() == condition.lower():
                            actual_cond = c
                            break
                    
                    if actual_cond:
                        trace = isi_data[300][channel][actual_cond].get('unitary_average_traces')
                        if trace is not None and len(trace) > 0:
                            if not isinstance(trace, np.ndarray): trace = np.array(trace)
                            cell_traces[pathway_name] = trace
            
            if cell_traces:
                final_export[condition][cell_id] = {
                    'genotype': genotype_dict.get(cell_id, 'Unknown'),
                    'sex': sex_dict.get(cell_id, 'Unknown'),
                    'traces': cell_traces
                }
                
    return final_export

#----------------------------------------------------------------------------------
# Analysis of DVC data
#----------------------------------------------------------------------------------

def convert_DVC_data_to_df_with_cage(data_df):
    new_data_df = []
    
    # Iterate through the columns of the data_df
    for column in data_df.columns:
        # Determine genotype based on column name
        genotype = 'GNB1' if 'GNB1' in column else 'WT'
        cage = column.split('_')[0]  # Extract cage ID from column name
        
        # Iterate through rows (hours) in data_df and append data with the matched Sex
        for hour, activity_value in zip(data_df.index, data_df[column]):
            new_data_df.append({
                'Cage': cage, 
                'Hour': hour, 
                'Genotype': genotype, 
                'Activity_Value': activity_value
            })
    
    # Convert the list of dictionaries to a DataFrame
    return pd.DataFrame(new_data_df)

def analyze_hourly_DVC_activity(df, group_by_cols):
    """
    Calculates hourly statistics (Mean, SEM, Std, Variance, N) for DVC data.
    Replaces logic from: 'plot_genotype_data' and 'plot_across_hours_by_sex'.

    Parameters:
    - df: The tidy DVC dataframe (must contain 'Cage', 'Hour', 'Activity_Value', and columns in group_by_cols).
    - group_by_cols: List of columns to group by (e.g., ['Genotype'] or ['Genotype', 'Sex']).

    Returns:
    - stats_df: DataFrame with one row per Hour per Group, containing calculated stats.
    """
    # 1. Ensure we have one value per Cage per Hour (Summing ensures we capture all bins within an hour)
    # We include group_by_cols here to ensure those labels persist
    cage_hourly_data = df.groupby(['Cage', 'Hour'] + group_by_cols)['Activity_Value'].sum().reset_index()

    # 2. Group by Hour and the specified categories (Genotype/Sex) to calculate stats across cages
    stats_df = cage_hourly_data.groupby(['Hour'] + group_by_cols)['Activity_Value'].agg(
        Mean='mean',
        SEM='sem',
        Std='std',
        Var='var',
        N='count'
    ).reset_index()

    return stats_df

def analyze_total_summed_DVC_activity(df, group_by_cols):
    """
    Calculates the TOTAL activity sum for each cage over the entire recording period.
    Replaces logic from: 'plot_summed_activity_per_sex' and 'plot_summed_activity_genotype'.
    
    This output is ideal for:
    1. Exporting to R for ANOVA/Mann-Whitney.
    2. Creating Bar/Scatter plots (where every dot is a cage).

    Parameters:
    - df: The tidy DVC dataframe.
    - group_by_cols: List of columns to keep associated with the cage (e.g., ['Genotype', 'Sex']).

    Returns:
    - cage_totals_df: DataFrame where each row is a Cage and its total summed activity.
    """
    # Sum activity for each cage across ALL hours available in the dataframe
    cage_totals_df = df.groupby(['Cage'] + group_by_cols)['Activity_Value'].sum().reset_index()
    
    # Rename column for clarity
    cage_totals_df.rename(columns={'Activity_Value': 'Total_Activity_Sum'}, inplace=True)
    
    return cage_totals_df

#------------------------------------------------------------------------------
# Analysis of Reconstruction data
#----------------------------------------------------------------------------------

def pull_and_process_all_data_cells(folder_path):
    ''' 
    Pull all data into a DF that contains cell names and radius/intersec data per cell.
    Returns: radii_array, intersections_array, sholl_df
    ''' 
    # Get all CSV files in the folder
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    
    # Initialize lists to hold all the data
    all_radius_data = []
    all_intersection_data = []
    all_cell_names = []
    
    for csv_file in csv_files:
        file_path = os.path.join(folder_path, csv_file)
        file_path_name_split = csv_file.split('_')
        date_pattern = re.compile(r'\b\d{8}\b')  # Matches exactly 8 digits
        cell_pattern = re.compile(r'^c\d$')      # Matches c1, c2 etc

        new_data_date = "Unknown"
        cell_number = "Unknown"

        for item in file_path_name_split:
            if date_pattern.fullmatch(item):
                file_date = item 
                cell_year = file_date[-4:]
                cell_month = file_date[0:2]
                cell_day = file_date[2:4]
                new_data_date = f"{cell_year}{cell_month}{cell_day}"

            if cell_pattern.match(item):
                cell_number = item
            
        cell_name = f'{new_data_date}_{cell_number}'
        
        try:
            data = pd.read_csv(file_path)
            # Handle column naming variations
            if 'Inters' in data.columns and 'Inters.' not in data.columns:
                data.rename(columns={'Inters': 'Inters.'}, inplace=True)

            radius_data = data['Radius']
            intersection_data = data['Inters.']
            
            # Check if 'basal' is in the filename and make radius negative
            if 'basal' in csv_file.lower():
                radius_data = -radius_data
        
            all_radius_data.extend(radius_data)
            all_intersection_data.extend(intersection_data)
            all_cell_names.extend([cell_name] * len(radius_data))
            
        except Exception as e:
            print(f"Skipping {csv_file}: {e}")
            continue
    
    sholl_df = pd.DataFrame({'Cell_ID': all_cell_names, 'Radius': all_radius_data, 'Inters.': all_intersection_data})
    
    return np.array(all_radius_data), np.array(all_intersection_data), sholl_df 

def transform_sholl_to_segments_no_drop(df_sholl_wide: pd.DataFrame) -> pd.DataFrame:
    """
    Transposes a wide Sholl analysis DataFrame, splits segments (Apical/Basal).
    """
    # 1. MELT
    df_long = df_sholl_wide.melt(id_vars=['Cell_ID'], var_name='Variable', value_name='Value')

    split_cols = df_long['Variable'].str.split(r'(\.+)', expand=True)
    df_long['Metric'] = split_cols[0].str.replace('Inters', 'Inters.', regex=False)

    try:
        df_long['Value'] = df_long['Value'].astype(float)
    except ValueError:
        df_long['Value'] = pd.to_numeric(df_long['Value'], errors='coerce')

    # Aggregation
    df_aggregated = df_long.groupby(['Cell_ID', 'Metric'])['Value'].apply(list).reset_index()

    df_final_list = df_aggregated.pivot(index='Cell_ID', columns='Metric', values='Value').reset_index()
    df_final_list.columns.name = None
    
    # 2. SPLIT SEGMENTS
    df_final_list['Radii apical'] = None
    df_final_list['Inter apical'] = None
    df_final_list['Basal radii'] = None
    df_final_list['Inter basal'] = None

    for index, row in df_final_list.iterrows():
        temp_df = pd.DataFrame({'Radius': row['Radius'], 'Inters': row['Inters.']}).dropna()
        
        radii = temp_df['Radius'].values
        inters = temp_df['Inters'].values

        if len(radii) == 0: continue

        # Soma point
        zero_index = np.where(radii == 0.0)[0]
        zero_radius = radii[zero_index] if len(zero_index) > 0 else []
        zero_inter = inters[zero_index] if len(zero_index) > 0 else []

        # Apical
        apical_indices = radii > 0
        apical_radii = np.concatenate((zero_radius, radii[apical_indices])) if len(zero_radius) > 0 else radii[apical_indices]
        apical_inters = np.concatenate((zero_inter, inters[apical_indices])) if len(zero_inter) > 0 else inters[apical_indices]
        
        # Basal (Strictly Negative)
        basal_indices = radii < 0
        basal_radii = np.abs(radii[basal_indices]) 
        basal_inters = inters[basal_indices]

        df_final_list.at[index, 'Radii apical'] = apical_radii.tolist()
        df_final_list.at[index, 'Inter apical'] = apical_inters.tolist()
        df_final_list.at[index, 'Basal radii'] = basal_radii.tolist()
        df_final_list.at[index, 'Inter basal'] = basal_inters.tolist()

    return df_final_list.drop(columns=['Radius', 'Inters.'])

def compute_radius_distribution(sholl_df, n_bins=20):
    """
    Computes radius quantiles, SEMs, and cumulative probability for a given Sholl dataframe.
    """
    radii = sholl_df["Radius"].to_numpy()
    intersec_counts = sholl_df["Inters."].to_numpy()

    # Take absolute value (handles basal negative radii)
    radii = np.abs(radii)

    # Flatten
    flattened_radii = []
    for radius, count in zip(radii, intersec_counts):
        flattened_radii.extend([radius] * int(count))

    if not flattened_radii:
        return [], [], [], [], False

    radius_data = np.array(flattened_radii)
    radius_quantiles = []
    radius_sems = []
    cdf_prob_bins = np.arange(1., n_bins + 1.) / n_bins

    for pi in cdf_prob_bins:
        quantile_value = np.percentile(radius_data, pi * 100)
        radius_quantiles.append(quantile_value)
        subset = radius_data[radius_data <= quantile_value]
        sem_value = sem(subset) if len(subset) >= 10 else np.nan
        radius_sems.append(sem_value)

    sorted_radius_data = np.sort(radius_data)
    cumulative_probability = [np.quantile(sorted_radius_data, pi) for pi in cdf_prob_bins]

    return radius_quantiles, cdf_prob_bins, radius_sems, cumulative_probability, True

def export_raw_sholl_data(wt_df, gnb1_df, output_dir):
    """
    Adds Genotype labels, combines WT and GNB1 dataframes, and exports raw data 
    (Radius vs Intersections per Cell) without aggregation.
    """
    wt_data = wt_df.copy()
    gnb1_data = gnb1_df.copy()
    
    wt_data['Genotype'] = 'WT'
    gnb1_data['Genotype'] = 'GNB1'
    
    combined_df = pd.concat([wt_data, gnb1_data], ignore_index=True)
    
    # Organize columns if possible
    cols_order = ['Cell_ID', 'Genotype', 'Sex', 'Radius', 'Inters.']
    # Filter to only columns that exist
    cols_order = [c for c in cols_order if c in combined_df.columns]
    
    save_path = os.path.join(output_dir, 'Sholl_Intersections_Raw.csv')
    combined_df[cols_order].to_csv(save_path, index=False)
    print(f"✓ Saved Raw Sholl Data (No Means): {save_path}")

def calculate_mean_sem_sholl(radii_series, inters_series):
    """
    Calculates mean and SEM intersection count per radius.
    """
    df = pd.DataFrame({'Radius': radii_series, 'Inters': inters_series})
    # Take absolute radius so Basal (-10) and Apical (10) align if comparing distance from soma
    df['Abs_Radius'] = df['Radius'].abs()
    
    grouped = df.groupby('Abs_Radius')['Inters'].agg(['mean', 'sem', 'count']).reset_index()
    return grouped['Abs_Radius'].values, grouped['mean'].values, grouped['sem'].values

def export_cdf_data(genotype, sex, dendrite_type, bins, quantiles, sems, cum_prob):
    """Helper to structure CDF data for CSV export."""
    return pd.DataFrame({
        'Genotype': genotype,
        'Sex': sex,
        'Dendrite_Type': dendrite_type,
        'CDF_Bin': bins,
        'Radius_Quantile': quantiles,
        'Radius_SEM': sems,
        'Cumulative_Probability': cum_prob
    })

def collect_dendrite_properties(data_dir):
    Apical_data = {'branch_sum': [], 'branch_mean': [], 'branch_max': [], 'N_terminal_branches': []}
    Basal_data = {'branch_sum': [], 'branch_mean': [], 'branch_max': [], 'N_terminal_branches': []} 

    for path, subdirs, files in os.walk(data_dir):
        for df in files:
            if df.endswith('.csv'):
                full_path = os.path.join(path, df)
                try:
                    current_df = pd.read_csv(full_path, encoding='latin1') 
                except Exception as e:
                    continue

                for _, row in current_df.iterrows():
                    dend_type = str(row.get('Dendrite Type', '')).strip().lower()

                    if 'apical' in dend_type:
                        Apical_data['branch_sum'].append(row.get('Branch length (µm) [Sum]'))
                        Apical_data['branch_mean'].append(row.get('Branch length (µm) [Mean]'))
                        Apical_data['branch_max'].append(row.get('Branch length (µm) [Max]'))
                        Apical_data['N_terminal_branches'].append(row.get('No. of terminal branches [Single value]'))

                    elif 'basal' in dend_type:
                        Basal_data['branch_sum'].append(row.get('Branch length (µm) [Sum]'))
                        Basal_data['branch_mean'].append(row.get('Branch length (µm) [Mean]'))
                        Basal_data['branch_max'].append(row.get('Branch length (µm) [Max]'))
                        Basal_data['N_terminal_branches'].append(row.get('No. of terminal branches [Single value]'))

    return Apical_data, Basal_data

#----------------------------------------------------------------------------------
# Analysis of Behavior Experiments
#----------------------------------------------------------------------------------

def load_and_concat_behavior_files(base_dir, file_map):
    """
    Loads a list of specific files where the key is a subfolder (optional) 
    and the value is the filename.
    
    Args:
        base_dir: Root directory (e.g., Mouse_Behavior)
        file_map: List of tuples/paths or just filenames. 
                  If the user provided paths like 'Folder/File.csv', this handles it.
    """
    dfs = []
    for relative_path in file_map:
        full_path = os.path.join(base_dir, relative_path)
        
        if os.path.exists(full_path):
            try:
                df = pd.read_csv(full_path)
                
                # Standardize 'Treatment' column to 'Genotype' if present
                if 'Treatment' in df.columns:
                    df.rename(columns={'Treatment': 'Genotype'}, inplace=True)
                
                dfs.append(df)
                # print(f"  ✓ Loaded: {os.path.basename(relative_path)}")
            except Exception as e:
                print(f"  ❌ Error loading {relative_path}: {e}")
        else:
            print(f"  ⚠ File not found: {relative_path}")
            
    if not dfs:
        return pd.DataFrame()
        
    return pd.concat(dfs, ignore_index=True)

def process_anxiety_ratios(df):
    """
    Calculates Center/Outer ratios for anxiety analysis.
    Handles division by zero by inserting NaNs.
    """
    # Create copies to avoid SettingWithCopy warnings
    center_time = df['Center Zone : time (s)'].values
    outer_time = df['Outer Zone : time (s)'].values
    
    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = center_time / outer_time
        # Replace infinity with NaN
        ratio[outer_time == 0] = np.nan
        
    df['Center_Outer_Time_Ratio'] = ratio
    return df

def process_olm_metrics(df):
    """
    Calculates OLM specific metrics: Summed times, DIs, Deltas.
    """
    # 1. Summed Times
    # Training (Familiarisation Day 2)
    # Using .fillna(0) to ensure summation works even if one column is NaN (though usually shouldn't be)
    df['Summed_Investigation_Training'] = (
        df.get('Non Moved Object 1 : time investigating (s)', 0) + 
        df.get('Non Moved Object 2 : time investigating (s)', 0)
    )
    
    # Testing (Testing Stage)
    df['Summed_Investigation_Testing'] = (
        df.get('Moved Object : time investigating (s)', 0) + 
        df.get('Familiar Object : time investigating (s)', 0)
    )
    
    # 2. Discrimination Indices (DI)
    # Training DI = (NM1 - NM2) / (NM1 + NM2)
    nm1 = df.get('Non Moved Object 1 : time investigating (s)', np.nan)
    nm2 = df.get('Non Moved Object 2 : time investigating (s)', np.nan)
    
    df['Training_DI'] = (nm1 - nm2) / df['Summed_Investigation_Training']
    
    # Testing DI = (Moved - Familiar) / (Moved + Familiar)
    moved = df.get('Moved Object : time investigating (s)', np.nan)
    familiar = df.get('Familiar Object : time investigating (s)', np.nan)
    
    df['Testing_DI'] = (moved - familiar) / df['Summed_Investigation_Testing']
    
    # 3. Delta DI (Testing - Training)
    # Note: This subtraction works row-wise. Since Training/Testing are usually in different rows 
    # (different 'Stage'), we might need to pivot or fill if analyzing Delta per animal.
    # However, for plotting Delta, we usually aggregate. 
    # BUT, if we want to calculate Delta per animal, we need both values on the same row or linked.
    # The provided snippet calculates Delta using arrays: 
    # WT_delta = WT_testing_DI.values - WT_training_DI.values (Assuming sorted order matching).
    # To do this robustly in a DF, we will handle it in the main script or creating a summary DF.
    # For now, we leave the columns as is.
    
    return df

def filter_olm_by_exploration(df, threshold=20):
    """
    Filters out animals that did not explore enough during the Testing phase.
    Returns the filtered dataframe and the list of excluded animals.
    """
    # Identify animals with insufficient exploration in Testing Stage
    # Filter to Testing Stage rows
    testing_rows = df[df['Stage'] == 'Testing Stage']
    
    # Find bad animals
    bad_animals = testing_rows[testing_rows['Summed_Investigation_Testing'] < threshold]['Animal'].unique()
    
    # Return filtered dataframe (excluding those animals entirely from ALL stages)
    filtered_df = df[~df['Animal'].isin(bad_animals)].copy()
    
    return filtered_df, bad_animals

def calculate_t_maze_alternations(positions_df):
    """
    Calculates spontaneous alternation percentages based on position strings.
    """
    # Helper to count sequence occurrences
    def count_sequence(seq, pos_list):
        return sum(pos_list[i:i+len(seq)] == seq for i in range(len(pos_list) - len(seq) + 1))

    # Define 16 alternation sequences
    alternation_patterns = {
        'SCLCSCR': ['Start', 'Center', 'Left Arm', 'Center', 'Start', 'Center', 'Right Arm'],
        'SCRCSCL': ['Start', 'Center', 'Right Arm', 'Center', 'Start', 'Center', 'Left Arm'],
        'SLCSCR':  ['Start', 'Left Arm', 'Center', 'Start', 'Center', 'Right Arm'],
        'SCLSCR':  ['Start', 'Center', 'Left Arm', 'Start', 'Center', 'Right Arm'],
        'SCLCSR':  ['Start', 'Center', 'Left Arm', 'Center', 'Start', 'Right Arm'],
        'SRCSCL':  ['Start', 'Right Arm', 'Center', 'Start', 'Center', 'Left Arm'],
        'SCRSCL':  ['Start', 'Center', 'Right Arm', 'Start', 'Center', 'Left Arm'],
        'SCRCSL':  ['Start', 'Center', 'Right Arm', 'Center', 'Start', 'Left Arm'],
        'SLSCR':   ['Start', 'Left Arm', 'Start', 'Center', 'Right Arm'],
        'SCLSR':   ['Start', 'Center', 'Left Arm', 'Start', 'Right Arm'],
        'SLCSR':   ['Start', 'Left Arm', 'Center', 'Start', 'Right Arm'],
        'SRSCL':   ['Start', 'Right Arm', 'Start', 'Center', 'Left Arm'],
        'SCRSL':   ['Start', 'Center', 'Right Arm', 'Start', 'Left Arm'],
        'SRCSL':   ['Start', 'Right Arm', 'Center', 'Start', 'Left Arm'],
        'SLSR':    ['Start', 'Left Arm', 'Start', 'Right Arm'],
        'SRSL':    ['Start', 'Right Arm', 'Start', 'Left Arm'],
    }

    # Define helper patterns for denominator
    SCL_pattern = ['Start', 'Center', 'Left Arm']
    SCR_pattern = ['Start', 'Center', 'Right Arm']
    SL_pattern =  ['Start', 'Left Arm']
    SR_pattern = ['Start', 'Right Arm']

    results = []
    
    # Iterate through rows
    for index, row in positions_df.iterrows():
        # Handle cases where position string might be missing
        if pd.isna(row.get('Positions_Strings')):
            results.append({'Num Alternations': 0, 'Percent Alternations': np.nan})
            continue
            
        pos_string = str(row['Positions_Strings'])
        pos_list = [x.strip() for x in pos_string.split(',')]

        # Count all 16 patterns
        alternation_counts = {name: count_sequence(pattern, pos_list) for name, pattern in alternation_patterns.items()}
        total_alternations = sum(alternation_counts.values())

        # Compute denominator components
        SCL = count_sequence(SCL_pattern, pos_list)
        SCR = count_sequence(SCR_pattern, pos_list)
        SL = count_sequence(SL_pattern, pos_list)
        SR = count_sequence(SR_pattern, pos_list)

        denominator = SCL + SCR + SL + SR - 1
        
        if denominator > 0:
            percent_alternations = (total_alternations / denominator) * 100
        else:
            percent_alternations = 0

        results.append({
            'Num_Alternations': total_alternations,
            'Percent_Alternations': percent_alternations,
            'Denominator': denominator
        })

    # Join results back to original dataframe
    results_df = pd.DataFrame(results)
    # Reset index of original to ensure alignment
    final_df = pd.concat([positions_df.reset_index(drop=True), results_df], axis=1)
    
    return final_df

# ==================================================================================================
# PAIRED PULSE RATIO (PPR) ANALYSIS
# ==================================================================================================

def calculate_epsp_amplitude_in_window(trace, stim_onset_idx, analysis_window_samples, 
                                        baseline_start_idx, baseline_end_idx, height_threshold=0.1):
    """
    Calculate EPSP amplitude after a stimulus.
    
    Parameters:
        trace: Voltage trace (numpy array)
        stim_onset_idx: Index of stimulus onset
        analysis_window_samples: Number of samples after stimulus to analyze
        baseline_start_idx: Start index for baseline calculation
        baseline_end_idx: End index for baseline calculation
        height_threshold: Minimum peak height for detection
    
    Returns:
        EPSP amplitude (baseline-subtracted peak) in mV
    """
    # Calculate baseline
    baseline = np.mean(trace[baseline_start_idx:baseline_end_idx])
    
    # Extract analysis window
    window_start = stim_onset_idx
    window_end = min(stim_onset_idx + analysis_window_samples, len(trace))
    analysis_trace = trace[window_start:window_end] - baseline
    
    # Find max peak
    if len(analysis_trace) > 0:
        max_val = np.max(analysis_trace)
        if max_val > height_threshold:
            return max_val
    
    return np.nan

def extract_paired_pulse_data(data_dir, master_df, stim_config=None):
    """
    Extract paired pulse traces from 'Test' experiments.
    
    Parameters:
        data_dir: Path to directory containing .pkl files
        master_df: Filtered master dataframe
        stim_config: Dictionary with stimulus timing configuration. If None, uses defaults.
    
    Returns:
        Dictionary with structure: {cell_id: {'genotype': str, 'sex': str, 
                                              'channel_1': {...}, 'channel_2': {...}}}
    """
    if stim_config is None:
        # Default configuration from YAML (EPSP_test_pulses)
        stim_config = {
            'channel_1': {
                'label': 'Perforant Path',
                'stim_times': [500.0, 550.0],  # ms
                'baseline_window': 10,  # ms before first stim
                'analysis_window': 40,  # ms after each stim
            },
            'channel_2': {
                'label': 'Schaffer Collateral',
                'stim_times': [800.0, 850.0],  # ms
                'baseline_window': 10,  # ms
                'analysis_window': 40,  # ms
            }
        }
    
    # Create lookup from master_df
    master_df_copy = master_df.copy()
    master_df_copy['Cell_ID'] = master_df_copy['Cell_ID'].astype(str)
    genotype_lookup = dict(zip(master_df_copy['Cell_ID'], master_df_copy['Genotype']))
    sex_lookup = dict(zip(master_df_copy['Cell_ID'], master_df_copy['Sex']))
    valid_ids = set(master_df_copy['Cell_ID'])
    
    PPR_data = {}
    
    # Iterate over files
    for filename in os.listdir(data_dir):
        if not filename.endswith('.pkl'):
            continue
        
        cell_id = convert_filename_to_standard_id(filename)
        if cell_id is None or cell_id not in valid_ids:
            continue
        
        try:
            filepath = os.path.join(data_dir, filename)
            df = pd.read_pickle(filepath)
            
            # Filter for Test experiments
            if 'experiment_description' not in df.columns:
                continue
            
            test_rows = df[df['experiment_description'].astype(str).str.lower() == 'test']
            
            if len(test_rows) == 0:
                continue
            
            # Get acquisition frequency
            acq_freq = test_rows.iloc[0].get('acquisition_frequency', 20000.0)
            
            # Initialize cell data
            PPR_data[cell_id] = {
                'genotype': genotype_lookup.get(cell_id, 'Unknown'),
                'sex': sex_lookup.get(cell_id, 'Unknown'),
            }
            
            # Process each channel
            for channel, config in stim_config.items():
                stim_times_ms = config['stim_times']
                baseline_window_ms = config['baseline_window']
                analysis_window_ms = config['analysis_window']
                
                # Convert to samples
                stim1_idx = int(stim_times_ms[0] * acq_freq / 1000)
                stim2_idx = int(stim_times_ms[1] * acq_freq / 1000)
                baseline_samples = int(baseline_window_ms * acq_freq / 1000)
                analysis_samples = int(analysis_window_ms * acq_freq / 1000)
                
                # Collect EPSPs from all test sweeps
                epsp1_values = []
                epsp2_values = []
                
                for _, row in test_rows.iterrows():
                    sweep = row.get('sweep', None)
                    if sweep is None or not isinstance(sweep, np.ndarray):
                        continue
                    
                    # Calculate EPSP1
                    baseline_start = stim1_idx - baseline_samples
                    baseline_end = stim1_idx
                    if baseline_start < 0:
                        baseline_start = 0
                    
                    epsp1 = calculate_epsp_amplitude_in_window(
                        sweep, stim1_idx, analysis_samples,
                        baseline_start, baseline_end
                    )
                    
                    # Calculate EPSP2
                    epsp2 = calculate_epsp_amplitude_in_window(
                        sweep, stim2_idx, analysis_samples,
                        baseline_start, baseline_end  # Use same baseline as EPSP1
                    )
                    
                    if not np.isnan(epsp1):
                        epsp1_values.append(epsp1)
                    if not np.isnan(epsp2):
                        epsp2_values.append(epsp2)
                
                # Average across sweeps
                avg_epsp1 = np.mean(epsp1_values) if epsp1_values else np.nan
                avg_epsp2 = np.mean(epsp2_values) if epsp2_values else np.nan
                
                # Calculate PPR
                ppr = avg_epsp2 / avg_epsp1 if (avg_epsp1 > 0 and not np.isnan(avg_epsp1)) else np.nan
                
                PPR_data[cell_id][channel] = {
                    'label': config.get('label', channel),
                    'EPSP1_Amplitude': avg_epsp1,
                    'EPSP2_Amplitude': avg_epsp2,
                    'PPR': ppr,
                    'n_sweeps': len(epsp1_values)
                }
                
        except Exception as e:
            print(f"Error processing {filename} for PPR: {e}")
            continue
    
    print(f"\nExtracted PPR data for {len(PPR_data)} cells")
    return PPR_data

def export_PPR_to_dataframe(PPR_data):
    """
    Export PPR data to DataFrame format.
    
    Parameters:
        PPR_data: Dictionary from extract_paired_pulse_data
    
    Returns:
        DataFrame with columns: Cell_ID, Genotype, Sex, Channel, Channel_Label, 
                                EPSP1_Amplitude, EPSP2_Amplitude, PPR, N_Sweeps
    """
    rows = []
    
    for cell_id, data in PPR_data.items():
        for channel in ['channel_1', 'channel_2']:
            if channel in data:
                ch_data = data[channel]
                rows.append({
                    'Cell_ID': cell_id,
                    'Genotype': data.get('genotype', 'Unknown'),
                    'Sex': data.get('sex', 'Unknown'),
                    'Channel': channel,
                    'Channel_Label': ch_data.get('label', channel),
                    'EPSP1_Amplitude': ch_data.get('EPSP1_Amplitude', np.nan),
                    'EPSP2_Amplitude': ch_data.get('EPSP2_Amplitude', np.nan),
                    'PPR': ch_data.get('PPR', np.nan),
                    'N_Sweeps': ch_data.get('n_sweeps', 0)
                })
    
    df = pd.DataFrame(rows)
    return df

def analyze_and_export_PPR(master_df, data_dir, output_path=None, stim_config=None):
    """
    Complete workflow: analyze paired pulse ratio and export results.
    
    Parameters:
        master_df: Master dataframe (will be filtered by Inclusion)
        data_dir: Path to directory with .pkl files
        output_path: Path to save CSV output (optional)
        stim_config: Custom stimulus configuration (optional)
    
    Returns:
        DataFrame with PPR results
    """
    print("\n" + "="*70)
    print("PAIRED PULSE RATIO (PPR) ANALYSIS")
    print("="*70)
    
    # Filter master_df
    filtered_df = filter_master_df_by_inclusion(master_df)
    
    # Extract PPR data
    print("\nExtracting paired pulse data from 'Test' experiments...")
    PPR_data = extract_paired_pulse_data(data_dir, filtered_df, stim_config)
    
    # Export to DataFrame
    print("\nExporting to DataFrame...")
    df = export_PPR_to_dataframe(PPR_data)
    
    if df.empty:
        print("WARNING: No PPR data extracted")
        return df
    
    # Summary statistics
    print("\n" + "="*70)
    print("PPR ANALYSIS SUMMARY")
    print("="*70)
    print(f"Total cells with PPR data: {df['Cell_ID'].nunique()}")
    print(f"\nCells per genotype:")
    print(df.groupby('Genotype')['Cell_ID'].nunique())
    
    print(f"\nMean PPR by Genotype and Channel:")
    summary = df.groupby(['Genotype', 'Channel_Label'])['PPR'].agg(['mean', 'sem', 'count'])
    print(summary)
    
    # Save if path provided
    if output_path:
        df.to_csv(output_path, index=False)
        print(f"\n✓ PPR data saved to: {output_path}")
    
    return df


# ==================================================================================================
# FIGURE 6 EXAMPLE TRACES GENERATION
# ==================================================================================================

def generate_figure6_example_traces(data_dir, output_path, selected_cells):
    """
    Generates stim-removed example traces for Figure 6 Panel A and saves to pickle.
    This pre-computes the traces so generate_figures.py can load them quickly.
    
    Parameters
    ----------
    data_dir : str
        Path to the Box data directory containing processed pickle files
    output_path : str
        Path where the example traces pickle file will be saved
    selected_cells : dict
        Dictionary configuring which cells/rows to use for plotting.
    """
    from scipy.signal import find_peaks
    
    # Helper functions (copied from plotting_utils to avoid circular imports)
    def remove_noise(data, noise_times, acquisition_frequency, delete_noise_duration_list):
        """Removes noise artifacts from data."""
        processed_data = np.copy(data)
        noise_times = [noise_times] if isinstance(noise_times, (int, float)) else noise_times
        delete_noise_duration_list = [delete_noise_duration_list] if isinstance(delete_noise_duration_list, (int, float)) else delete_noise_duration_list
        
        if len(noise_times) != len(delete_noise_duration_list):
            raise ValueError("noise_times and delete_noise_duration_list must have the same length.")
        
        for noise_time, delete_noise_duration in zip(noise_times, delete_noise_duration_list):
            current_noise_index = int(noise_time * acquisition_frequency / 1000)
            delete_start_index = max(0, current_noise_index - int(delete_noise_duration * acquisition_frequency / 1000))
            delete_end_index = min(len(processed_data), current_noise_index + int(delete_noise_duration * acquisition_frequency / 1000))
            processed_data[delete_start_index:delete_end_index] = np.nan
            not_nan_indices = np.arange(0, len(processed_data))[~np.isnan(processed_data)]
            processed_data = np.interp(np.arange(0, len(processed_data)), not_nan_indices, processed_data[~np.isnan(processed_data)])
        return processed_data

    def normalize_plateau_trace(trace, baseline_window=1000):
        """Normalize a trace by subtracting the baseline (first baseline_window samples)."""
        baseline = np.mean(trace[:baseline_window])
        return trace - baseline

    def remove_artifacts_automated(data, acquisition_frequency, delete_start_stim, delete_end_stim):
        """Removes stimulation artifacts from data automatically."""
        processed_data = np.copy(data)
        derivative = np.diff(processed_data, n=1)
        AP_peaks = find_peaks(derivative, height=0)[0]
        negative_inflection_points = find_peaks(-derivative, height=0.2)[0]
        
        for peak in negative_inflection_points:
            for AP_peak in AP_peaks:
                if abs(peak - AP_peak) < acquisition_frequency / 1000:
                    negative_inflection_points = negative_inflection_points[negative_inflection_points != AP_peak]
        
        for inflection in negative_inflection_points:
            baseline_start = int(inflection * acquisition_frequency/1000) - int(0.5 * acquisition_frequency/1000)
            baseline_voltage = np.mean(processed_data[baseline_start:int(inflection * acquisition_frequency/1000)])
            delete_start_index = max(0, inflection - int(delete_start_stim * acquisition_frequency / 1000))
            interp_start_index = min(len(processed_data), inflection - int(delete_start_stim * acquisition_frequency / 1000))
            delete_end_index = min(len(processed_data), inflection + int(delete_end_stim * acquisition_frequency / 1000))
            if any(processed_data[delete_end_index:] > baseline_voltage):
                interp_end_index = np.where(processed_data[delete_end_index:] > baseline_voltage)[0][0] + delete_end_index
            else:
                interp_end_index = delete_end_index
            current_window = np.arange(interp_start_index, interp_end_index)
            processed_data[current_window] = np.nan
            processed_data = np.interp(np.arange(0, len(processed_data)), np.arange(0, len(processed_data))[~np.isnan(processed_data)], processed_data[~np.isnan(processed_data)])
        
        return processed_data
    
    print("\n" + "="*70)
    print("GENERATING FIGURE 6 EXAMPLE TRACES")
    print("="*70)
    
    # Configuration is passed as 'selected_cells' argument
    
    acq_freq = 20000
    start_ms = 400 # Adjusted to include 100ms baseline (Stim starts at 500)
    end_ms = 1500
    start_idx = int(start_ms * acq_freq / 1000)
    end_idx = int(end_ms * acq_freq / 1000)
    
    example_traces = {'WT': {}, 'GNB1': {}}
    
    for geno, config in selected_cells.items():
        file_path = os.path.join(data_dir, config['file'])
        if not os.path.exists(file_path):
            print(f"  ⚠ Warning: {geno} file not found: {config['file']}")
            continue
            
        try:
            df = pd.read_pickle(file_path)
            apply_noise = config.get('apply_noise_removal', False)
            
            for pathway, row_idx in config['rows'].items():
                raw_sweep = df.iloc[row_idx]['sweep']
                normalized = normalize_plateau_trace(raw_sweep)
                
                if apply_noise:
                    # Remove Input R pulse (50-150ms) first, then remove stim artifacts
                    denoised = remove_noise(normalized, 50, acq_freq, 100)
                    cleaned = remove_artifacts_automated(denoised, acq_freq, 0.3, 1.3)
                else:
                    # Just remove stim artifacts
                    cleaned = remove_artifacts_automated(normalized, acq_freq, 0.3, 1.3)
                
                # Slice to display window (500-1500ms)
                example_traces[geno][pathway] = cleaned[start_idx:end_idx]
                
            print(f"  ✓ {geno}: Processed {len(config['rows'])} pathways from {config['file']}")
            
        except Exception as e:
            print(f"  ✗ Error processing {geno}: {e}")
    
    # Save to pickle
    import pickle
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(example_traces, f)
    print(f"  ✓ Saved example traces to: {output_path}")
    
    return example_traces

#----------------------------------------------------------------------------------
# GABAb Analysis for Stratum Oriens (Basal Pathway) - Figure 5
#----------------------------------------------------------------------------------

def analyze_gabab_stratum_oriens(E_I_basal_traces, master_df, condition_to_plot='gabazine', 
                                  ax=None, save_path=None):
    """
    Analyze the GABAb component from Stratum Oriens (Basal) pathway unitary traces.
    
    This is similar to analyze_gabab_component but specifically for basal pathway data
    extracted via get_E_I_traces_basal().
    
    Parameters:
        E_I_basal_traces: Dict from get_E_I_traces_basal() with structure:
                          {cell_id: {ISI_time: {'channel_1': {condition: trace_dict}}}}
        master_df: Master dataframe for genotype lookup
        condition_to_plot: Condition to analyze (default 'gabazine' for GABAb component)
        ax: Optional matplotlib axes for plotting
        save_path: Optional path to save figure
    
    Returns:
        gabab_measurements: Dict with measurements per cell/genotype
        export_data: Dict with Mean and SEM traces by genotype for CSV export
    """
    import matplotlib.pyplot as plt
    
    # Build genotype lookup
    genotype_dict = {}
    for _, row in master_df.iterrows():
        cell_id = str(row['Cell_ID'])
        genotype_dict[cell_id] = row.get('Genotype', 'Unknown')
    
    gabab_measurements = []
    traces_by_geno = {'WT': [], 'GNB1': []}
    dt = 1/20000  # 20kHz sampling
    
    # Extract traces from basal pathway data
    for cell_id, isi_data in E_I_basal_traces.items():
        if 300 not in isi_data:
            continue
            
        genotype = genotype_dict.get(cell_id, 'Unknown')
        if genotype not in ['WT', 'GNB1']:
            continue
            
        for channel, cond_data in isi_data[300].items():
            for condition, trace_dict in cond_data.items():
                # Match condition (case-insensitive)
                if condition.lower() != condition_to_plot.lower():
                    continue
                    
                trace = trace_dict.get('unitary_average_traces')
                
                # Safety checks
                if trace is None:
                    continue
                if isinstance(trace, list) and len(trace) == 0:
                    continue
                if not isinstance(trace, np.ndarray):
                    trace = np.array(trace)
                if len(trace) == 0:
                    continue
                
                # Store trace for averaging
                traces_by_geno[genotype].append(trace)
                
                # Calculate GABAb measurements
                time = np.arange(len(trace)) * dt * 1000  # Convert to ms
                
                # Find trough (most negative point)
                trough_amplitude_neg = np.min(trace)
                trough_amplitude_abs = np.abs(trough_amplitude_neg)
                trough_time = time[np.argmin(trace)]
                
                # Calculate integral below zero (GABAb area)
                negative_trace = np.where(trace < 0, trace, 0)
                integral_below_zero = -np.trapz(negative_trace, dx=dt * 1000)  # mV·ms
                
                gabab_measurements.append({
                    'Cell_ID': cell_id,
                    'Genotype': genotype,
                    'Pathway': 'Stratum Oriens',
                    'Condition': condition,
                    'Trough_Amplitude_mV': trough_amplitude_abs,
                    'Trough_Time_ms': trough_time,
                    'Integral_mV_ms': integral_below_zero
                })
    
    # Convert to DataFrame
    df_measurements = pd.DataFrame(gabab_measurements)
    
    print(f"\n=== GABAb Analysis: Stratum Oriens ({condition_to_plot}) ===")
    print(f"  WT cells: {len(traces_by_geno['WT'])}")
    print(f"  GNB1 cells: {len(traces_by_geno['GNB1'])}")
    
    # Calculate mean/SEM traces for export
    export_data = {}
    
    for geno in ['WT', 'GNB1']:
        traces = traces_by_geno[geno]
        if not traces:
            continue
            
        # Pad traces to same length
        max_len = max(len(t) for t in traces)
        padded = [np.pad(t, (0, max_len - len(t)), mode='constant', constant_values=0) 
                  for t in traces]
        arr = np.array(padded)
        
        mean_trace = np.mean(arr, axis=0)
        sem_trace = np.std(arr, axis=0) / np.sqrt(len(arr))
        time = np.arange(max_len) * dt * 1000
        
        export_data[f"{geno}_Time"] = time
        export_data[f"{geno}_Mean"] = mean_trace
        export_data[f"{geno}_SEM"] = sem_trace
        export_data[f"{geno}_N"] = len(traces)
    
    # Plot if axes provided
    if ax is not None or save_path:
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
            show_at_end = True
        else:
            show_at_end = False
        
        colors = {'WT': 'black', 'GNB1': 'red'}
        
        for geno in ['WT', 'GNB1']:
            if f"{geno}_Mean" not in export_data:
                continue
            time = export_data[f"{geno}_Time"]
            mean = export_data[f"{geno}_Mean"]
            sem = export_data[f"{geno}_SEM"]
            n = export_data[f"{geno}_N"]
            
            ax.plot(time, mean, color=colors[geno], label=f"{geno} (n={n})", linewidth=1.5)
            ax.fill_between(time, mean - sem, mean + sem, color=colors[geno], alpha=0.2)
        
        ax.axhline(0, color='gray', linestyle='--', linewidth=0.5)
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Voltage (mV)')
        ax.set_title(f'GABAb Component - Stratum Oriens ({condition_to_plot})')
        ax.legend(frameon=False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        if show_at_end:
            plt.tight_layout()
            if save_path:
                plt.savefig(save_path)
                plt.close()
                print(f"  ✓ Figure saved to: {save_path}")
            else:
                plt.show()
    
    return df_measurements, export_data

def export_gabab_stratum_oriens_data(df_measurements, export_data, output_dir):
    """
    Export GABAb Stratum Oriens analysis results to CSV files.
    
    Parameters:
        df_measurements: DataFrame from analyze_gabab_stratum_oriens()
        export_data: Dict with mean/SEM traces from analyze_gabab_stratum_oriens()
        output_dir: Directory to save CSV files
    
    Returns:
        Paths to saved files
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Save measurements
    measurements_path = os.path.join(output_dir, 'GABAb_Stratum_Oriens_Measurements.csv')
    df_measurements.to_csv(measurements_path, index=False)
    print(f"  ✓ Saved: {measurements_path}")
    
    # Save traces (for plotting)
    if export_data:
        traces_path = os.path.join(output_dir, 'GABAb_Stratum_Oriens_Traces.csv')
        
        # Find max length to pad all arrays
        max_len = 0
        for key, arr in export_data.items():
            if isinstance(arr, np.ndarray) and len(arr) > max_len:
                max_len = len(arr)
        
        # Pad and create DataFrame
        padded_data = {}
        for key, arr in export_data.items():
            if isinstance(arr, np.ndarray):
                if len(arr) < max_len:
                    padded_data[key] = np.pad(arr, (0, max_len - len(arr)), 
                                               mode='constant', constant_values=np.nan)
                else:
                    padded_data[key] = arr
            else:
                padded_data[key] = arr
        
        df_traces = pd.DataFrame(padded_data)
        df_traces.to_csv(traces_path, index=False)
        print(f"  ✓ Saved: {traces_path}")
        
        return measurements_path, traces_path
    
    return measurements_path, None

def get_coarse_fi_traces_by_condition(dir_path, conditions_of_interest, master_df=None):
    """
    Extracts FI data (Current vs Firing Rate) for specific conditions from .pkl files.
    
    Args:
        dir_path: Directory containing .pkl files.
        conditions_of_interest: List of condition strings to search for (e.g. ['gabazine', 'gabazine + baclofen']).
        master_df: Optional dataframe filter.
        
    Returns:
        Dict: { condition: { cell_id: { current: firing_rate } } }
    """
    import dill
    import os
    import numpy as np
    import pandas as pd
    
    FI_data = {cond: {} for cond in conditions_of_interest}
    
    valid_ids = None
    if master_df is not None and 'Cell_ID' in master_df.columns:
        valid_ids = set(master_df['Cell_ID'].astype(str))

    files = [f for f in os.listdir(dir_path) if f.endswith('.pkl')]
    
    for name in files:
        cell_id = convert_pkl_filename_to_cell_id(name)
        if cell_id is None: continue
        
        if valid_ids is not None and cell_id not in valid_ids:
            continue
            
        try:
            with open(os.path.join(dir_path, name), 'rb') as f:
                data = pd.read_pickle(f)
                
            if isinstance(data, pd.DataFrame):
                iterable = range(len(data))
                get_stim = lambda i: data['stimulus_metadata_dict'].iloc[i]
                get_analysis = lambda i: data['analysis_dict'].iloc[i]
            else:
                 if 'analysis_dict' not in data: continue
                 iterable = range(len(data['analysis_dict']))
                 get_stim = lambda i: data['stimulus_metadata_dict'][i]
                 get_analysis = lambda i: data['analysis_dict'][i]

            for i in iterable:
                stim_meta = get_stim(i)
                if not stim_meta: continue
                
                condition = stim_meta.get('condition', '').lower().strip()
                if not condition: continue
                
                # Sort conditions by length to match the most specific first (e.g. "gabazine + baclofen" before "gabazine")
                sorted_conditions = sorted(conditions_of_interest, key=len, reverse=True)
                matched_cond = None
                
                for target_cond in sorted_conditions:
                    if target_cond.lower() in condition:
                        matched_cond = target_cond
                        break
                
                if matched_cond:
                    analysis = get_analysis(i)
                    if not analysis: continue
                    
                    coarse_f_I = analysis.get('Coarse_FI') or analysis.get('IV_stim')
                    if not coarse_f_I: continue
                    
                    # Extract data
                    current_amplitudes = np.round(coarse_f_I['current_amplitudes'], 1)
                    firing_rates = coarse_f_I['firing_rates']
                    
                    cell_fi_data = {}
                    for amp, rate in zip(current_amplitudes, firing_rates):
                        if amp in cell_fi_data:
                            cell_fi_data[amp] = (cell_fi_data[amp] + rate) / 2 # Average duplicates
                        else:
                            cell_fi_data[amp] = rate
                    
                    # Store
                    if cell_id not in FI_data[matched_cond]:
                        FI_data[matched_cond][cell_id] = cell_fi_data
                    else:
                        existing = FI_data[matched_cond][cell_id]
                        for amp, rate in cell_fi_data.items():
                            if amp in existing:
                                existing[amp] = (existing[amp] + rate) / 2
                            else:
                                existing[amp] = rate

        except Exception as e:
            continue
            
    return FI_data

def generate_EI_summary_files(amplitudes_csv_path, output_dir):
    """Generate summary CSV files for Figure 4 and Supplemental Figure 1 from E_I_amplitudes.csv
    Cross-references with master_df to only count cells with 'Full E/I' for apical pathways"""
    import pandas as pd
    import os
    
    df = pd.read_csv(amplitudes_csv_path)
    
    # Load master_df to filter cells with "Full E/I"
    master_df = pd.read_csv('master_df.csv')
    full_ei_cells = set(master_df[master_df['Experiment Notes'].str.contains('Full E/I', na=False, case=False)]['Cell_ID'].values)
    
    pathways = df['Pathway'].unique()
    genotypes = ['WT', 'GNB1']
    
    # Figure 4 Summary
    fig4_rows = []
    for pathway in pathways:
        for genotype in genotypes:
            subset = df[(df['Pathway'] == pathway) & (df['Genotype'] == genotype)]
            if len(subset) == 0:
                continue
            
            # For apical pathways, filter to only cells with "Full E/I"
            # But track ALL cells with any Gabazine for partial count
            all_cells_this_pathway = subset.copy()
            if pathway in ['Perforant', 'Schaffer']:
                subset = subset[subset['Cell_ID'].isin(full_ei_cells)]
            
            n_cells = subset['Cell_ID'].nunique()
            cells_with_control = subset[subset['Control_Amplitude'].notna()]['Cell_ID'].nunique()
            cells_with_gabazine = subset[subset['Gabazine_Amplitude'].notna()]['Cell_ID'].nunique()
            
            control_cells = set(subset[subset['Control_Amplitude'].notna()]['Cell_ID'].unique())
            gabazine_cells = set(subset[subset['Gabazine_Amplitude'].notna()]['Cell_ID'].unique())
            cells_with_both = len(control_cells & gabazine_cells)
            
            # Count partial Gabazine cells (cells with Gabazine but NOT in "Full E/I" set)
            if pathway in ['Perforant', 'Schaffer']:
                all_gabazine_cells = set(all_cells_this_pathway[all_cells_this_pathway['Gabazine_Amplitude'].notna()]['Cell_ID'].unique())
                partial_gabazine_cells = all_gabazine_cells - gabazine_cells
                n_partial_gabazine = len(partial_gabazine_cells)
            else:
                n_partial_gabazine = 0
            
            # Panel C/D: Calculate N range (includes both full and partial Gabazine)
            # Full Gabazine cells
            full_gab_n_per_isi = []
            for isi in [10, 25, 50, 100, 300]:
                n = subset[subset['ISI'] == isi]['Gabazine_Amplitude'].notna().sum()
                if n > 0:
                    full_gab_n_per_isi.append(n)
            
            # Full + Partial Gabazine cells (for apical only)
            if pathway in ['Perforant', 'Schaffer'] and n_partial_gabazine > 0:
                combined_gab_n_per_isi = []
                for isi in [10, 25, 50, 100, 300]:
                    n = all_cells_this_pathway[all_cells_this_pathway['ISI'] == isi]['Gabazine_Amplitude'].notna().sum()
                    if n > 0:
                        combined_gab_n_per_isi.append(n)
                
                panel_cd_max = max(combined_gab_n_per_isi) if combined_gab_n_per_isi else 0
                panel_cd_min = min(full_gab_n_per_isi) if full_gab_n_per_isi else 0
            else:
                panel_cd_max = max(full_gab_n_per_isi) if full_gab_n_per_isi else 0
                panel_cd_min = min(full_gab_n_per_isi) if full_gab_n_per_isi else 0
            
            panel_cd_range = f"{panel_cd_min}-{panel_cd_max}"
            
            fig4_rows.append({
                'Pathway': pathway, 'Genotype': genotype, 'Full_EI_Count': n_cells,
                'Cells_With_Control': cells_with_control, 'Cells_With_Gabazine': cells_with_gabazine,
                'Cells_With_Partial_Gabazine': n_partial_gabazine,
                'Cells_With_Both': cells_with_both, 'Panel_C_D_Total_Gabazine': panel_cd_max,
                'Panel_C_D_Range': panel_cd_range, 'Panel_E_Total_Both': cells_with_both,
                'Panel_E_Range': f"{cells_with_both}-{cells_with_both}"
            })
    
    fig4_df = pd.DataFrame(fig4_rows)
    fig4_df.to_csv(os.path.join(output_dir, 'Figure_4_Full_EI_Summary.csv'), index=False)
    print(f"✓ Generated Figure 4 EI Summary")
    
    # Supplemental Figure 1 Summary
    supp_rows = []
    
    # Define cell sets once
    full_ei_cell_ids = set(full_ei_cells)
    
    # Define valid control cells (Full E/I or E/I Up to...)
    control_mask = master_df['Experiment Notes'].str.contains('E/I Up to', na=False, case=False) | \
                   master_df['Experiment Notes'].str.contains('Full E/I', na=False, case=False)
    valid_control_cells = set(master_df[control_mask]['Cell_ID'])

    for pathway in ['Schaffer', 'Perforant', 'Basal_Stratum_Oriens']:
        pathway_df = df[df['Pathway'] == pathway]
        
        for genotype in genotypes:
            subset = pathway_df[pathway_df['Genotype'] == genotype]
            
            if len(subset) == 0:
                continue
            
            # --- Define Cell Populations ---
            # 1. Full E/I Population (for Gabazine & Estimated Inhibition)
            if pathway in ['Schaffer', 'Perforant']:
                full_ei_subset = subset[subset['Cell_ID'].isin(full_ei_cell_ids)]
            else:
                # For Basal, we assume all are valid/full unless specified otherwise,
                # or we trust the input DF which has already filtered valid cells.
                # If Basal also needs strict filtering, add logic here.
                full_ei_subset = subset 

            # 2. Control Population (Includes Full E/I + "E/I Up to" / Control-Only)
            # Filter subset to only valid annotations
            control_subset = subset[subset['Cell_ID'].isin(valid_control_cells)]
            
            ranges_dict = {}
            
            # --- Calculate N (Single Value) for each Metric ---
            # NOTE: We report the Total Number of Unique Cells contributing to ANY of the summation ISIs [10, 25, 50, 100].
            # ISI 300 (Unitary) is strictly excluded.
            
            summation_isis = [10, 25, 50, 100]
            
            # A. Control (With Inhibition) - Uses LARGER population (Full + Partial)
            n_control = control_subset[
                (control_subset['ISI'].isin(summation_isis)) & 
                (control_subset['Control_Amplitude'].notna())
            ]['Cell_ID'].nunique()
            ranges_dict['Control_N_Range'] = str(n_control)
            
            # B. Gabazine (No Inhibition) - Uses STRICT Full E/I population
            n_gabazine = full_ei_subset[
                (full_ei_subset['ISI'].isin(summation_isis)) & 
                (full_ei_subset['Gabazine_Amplitude'].notna())
            ]['Cell_ID'].nunique()
            ranges_dict['Gabazine_N_Range'] = str(n_gabazine)
            
            # C. Expected (Linear Sum) - Calculated from Unitary (ISI 300)
            n_expected = full_ei_subset[
                (full_ei_subset['ISI'].isin(summation_isis)) & 
                (full_ei_subset['Expected_EPSP_Amplitude'].notna())
            ]['Cell_ID'].nunique()
            ranges_dict['Expected_N_Range'] = str(n_expected)
            
            # D. Estimated Inhibition (Gabazine - Control) - Requires BOTH, so strictly Full E/I population
            n_est_inh = full_ei_subset[
                (full_ei_subset['ISI'].isin(summation_isis)) & 
                (full_ei_subset['Control_Amplitude'].notna()) & 
                (full_ei_subset['Gabazine_Amplitude'].notna())
            ]['Cell_ID'].nunique()
            ranges_dict['Est_Inhibition_N_Range'] = str(n_est_inh)
            
            row = {'Pathway': pathway, 'Genotype': genotype}
            row.update(ranges_dict)
            supp_rows.append(row)
    
    supp_df = pd.DataFrame(supp_rows)
    supp_df.to_csv(os.path.join(output_dir, 'Supplemental_Figure_1_EI_Summary.csv'), index=False)
    print(f"✓ Generated Supplemental Figure 1 EI Summary")
    
    return fig4_df, supp_df
