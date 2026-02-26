"intracellular_analysis functions "
import numpy as np
import pandas
import matplotlib as mpl
import os 
import matplotlib.pyplot as plt
import yaml

mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['font.size'] = 12.
mpl.rcParams['font.sans-serif'] = 'Arial'
mpl.rcParams['text.usetex'] = False

#for pulling data for NWB files
'''Deprecated - need to reinstall older python (3.9 or 3.7) or nwb loader doesnt work'''
# import pynwb
# from pynwb.icephys import VoltageClampStimulusSeries, VoltageClampSeries
# from pynwb import NWBHDF5IO
# from nwbwidgets import nwb2widget

#for analysis of NWB files
from scipy import signal 
from sklearn.linear_model import LinearRegression
from scipy.stats import linregress 
from scipy.signal import chirp, find_peaks, peak_widths
import scipy
import pandas as pd 
import igor2 as igor
from igor.packed import load as loadpxp
from igor.record.wave import WaveRecord

def read_from_yaml(file_path, Loader=None):
    """
    Import a python dict from .yaml
    :param file_path: str (should end in '.yaml')
    :param Loader: :class:'yaml.Loader'
    :return: dict
    """
    if Loader is None:
        Loader = yaml.FullLoader
    if os.path.isfile(file_path):
        with open(file_path, 'r') as stream:
            data = yaml.load(stream, Loader=Loader)
        return data
    else:
        raise Exception('File: {} does not exist.'.format(file_path))


def recursive_convert_dict_bytes_to_str(input_dict):
    """
    Recursively convert all bytes in a dictionary to strings
    :param input_dict: dict
    :return: dict
    """
    if isinstance(input_dict, bytes):
        return input_dict.decode('utf-8')
    elif isinstance(input_dict, dict):
        return {recursive_convert_dict_bytes_to_str(key): recursive_convert_dict_bytes_to_str(value) for key, value in input_dict.items()}
    elif isinstance(input_dict, (list, tuple)):
        return [recursive_convert_dict_bytes_to_str(element) for element in input_dict]
    else:
        return input_dict

def load_NM_data_create_sweeps(data_file, yaml_file_path):
    data = loadpxp(data_file)
    config_dict = read_from_yaml(yaml_file_path) 
    
    #Define the meta data and data dictionaries 
    _, waves_dict = data
    root_waves_dict = waves_dict['root']
    root_waves_dict = recursive_convert_dict_bytes_to_str(root_waves_dict)

    clamp_notes_dict = root_waves_dict['ClampNotes']
    experiment_type = clamp_notes_dict['F_Stim']
    if 'F_experiment_description' in clamp_notes_dict:
        experiment_description = clamp_notes_dict['F_experiment_description']
    else:
        experiment_description = experiment_type
    # experiment_type = experiment_type_byte.decode('utf-8')
    R_inp = clamp_notes_dict['F_Relectrode']
    
    Neuromatic_metadata = root_waves_dict[experiment_type] #must be in byte format, i.e - b'IV_stim' 
    
    #Create metadata
    acquisition_metadata_dict = {}
    analysis_dict = {}
    stimulus_metadata_dict = {}

    acquisition_metadata_dict['date'] = root_waves_dict['FileDate']
    acquisition_metadata_dict['inter trial interval (ms)'] = Neuromatic_metadata['InterStimTime'] #is ms
    acquisition_metadata_dict['total recording window (ms)'] = Neuromatic_metadata['WaveLength'] # is ms
    acquisition_metadata_dict['samples per wave'] = Neuromatic_metadata['SamplesPerWave']
    acquisition_metadata_dict['experiment start time'] = root_waves_dict['FileTime']
    acquisition_metadata_dict['experiment end time'] = root_waves_dict['FileFinish']
    acquisition_metadata_dict['global wall clock'] = root_waves_dict['FileDateTime']
    acquisition_metadata_dict['total experiment time (s)'] = Neuromatic_metadata[ 'TotalTime']
    acquisition_metadata_dict['stim type'] = experiment_type
    acquisition_metadata_dict['experiment description'] = experiment_description

    dt = root_waves_dict['SampleInterval'] 
    acquisition_frequency = 1/(dt/1000)
    
    acquisition_metadata_dict['dt'] = dt
    acquisition_metadata_dict['acquisition_frequency'] = acquisition_frequency
    
    #Pull out sweep and data
    sweeps = []
    for key, val in root_waves_dict.items():
        if isinstance(val, WaveRecord) and 'RecordA' in key:
            sweeps.append(val.wave['wave']['wData'])
    #pull out the recorded stimulus channel
    stim_record = []
    for key, val in root_waves_dict.items():
        if isinstance(val, WaveRecord) and 'RecordB' in key:
            stim_record.append(val.wave['wave']['wData'])

    #pull out the stimulus command but in some instances the stim command is just the test pulse so pull out the first command
    stim_command = []
    for key, val in Neuromatic_metadata.items():
        if isinstance(val, WaveRecord) and 'uDAC_0_' in key:
            stim_command.append(val.wave['wave']['wData'])
    
    analysis_dict['R_inp'] = R_inp

    # get the NM_clamp_notes_keys expected for this experiment type
    if experiment_type in config_dict['stimulus_meta_data_keys']:
        stimulus_metadata_keys = config_dict['stimulus_meta_data_keys'][experiment_type]
        # iterate over NM_clamp_notes_keys, and populate the stimulus_metadata_dict
        for NM_key, stimulus_metadata_dict_key in stimulus_metadata_keys.items():
            if NM_key in clamp_notes_dict:
                val = clamp_notes_dict[NM_key]
                stimulus_metadata_dict[stimulus_metadata_dict_key] = val

    return sweeps, stim_record, stim_command, acquisition_metadata_dict, stimulus_metadata_dict, analysis_dict

def create_traces_and_exclude(sweeps, exclude_trace_list):
    
    response_data = sweeps

    #Create traces and sweep number array
    raw_traces = []
    sweep_numbers = []
    for i, sweep in enumerate(response_data): 
        trace = response_data[i]
        sweep_number = i + 1
        sweep_numbers.append(sweep_number)
        raw_traces.append(trace)
        
    #Exclude and add back to traces and sweep numbers
    #add to exclude list traces that are unwanted
    excluded_sweep_numbers =  exclude_trace_list #must be a list
    excluded_traces = []
    for sweep_number in excluded_sweep_numbers:
        #this function is similar to np.where, it returns the index of the sweep number in the sweep_numbers list
        index = sweep_numbers.index(sweep_number)
        excluded_trace = raw_traces[index] 
        excluded_traces.append(excluded_trace) 
        #Remove operator 
        #Remove excluded traces from the each list of values
        del raw_traces[index]
        del sweep_numbers[index]
    
    return raw_traces, sweep_numbers, excluded_sweep_numbers, excluded_traces

def create_data_dict(path):
    file_dict = {}
    files = []
    file_names = []
    for file in os.listdir(path):
        if file.endswith(f".pkl"):
            # print(file)
            unpickled_df = pd.read_pickle(f'{path}/{file}')
            file_names.append(file)
            files.append(unpickled_df)

    file_dict = dict(zip(file_names, files))
    return file_dict

def sweep_number_to_index_array(sweep_numbers, sweep_number_subset_start, sweep_number_subset_end):
    index_array = []
    for sweep_number in range(sweep_number_subset_start, sweep_number_subset_end, 1):
        try:
            index = sweep_numbers.index(sweep_number) 
            index_array.append(index)
        except:
            pass
    return np.array(index_array)

def get_IV_stim_traces(data_dict):
    IV_stim_traces = {}
    for index, stim in enumerate(data_dict['stim_type']):
        if stim == b'IV_stim':
            # print(data_dict['experiment_index'][index])
            IV_stim_traces[data_dict['experiment_index'][index]] = data_dict['traces'][index]
    count = len(IV_stim_traces)
    return IV_stim_traces, count

def plot_temp_IV_traces(temp_count, temp_traces, fig, ax):
    keys = list(temp_traces.keys())

    if temp_count == 1:
        ax.set_title(f'Experiment_Index {keys[0]}')
    else:
        for i in range(temp_count):
            ax[i].set_title(f'Experiment_Index {keys[i]}') 
    
    if temp_count == 1:
        for i in range(len(temp_traces[keys[0]])):
            ax.plot(temp_traces[keys[0]][i])
            ax.set_xlabel('Time (ms)')
            ax.set_ylabel('Voltage (mV)')
            ax.set_ylim(-130, 50)
    else:
        for count, traces in enumerate(temp_traces.values()):
                for i in range(len(traces)):
                        ax[count].plot(traces[i])
                        ax[count].set_xlabel('Time (ms)')
                        ax[count].set_ylabel('Voltage (mV)')
                        ax[count].set_ylim(-130, 50)
    fig.tight_layout()

def calculate_AP_numbers(data, spike_threshold, acquisition_frequency, stim_length):
        traces = []
        avg_firing_rates = []
        for i in range(len(data)):
                current_trace = data[i]
                traces.append(current_trace)

        AP_numbers = []
        for trace in traces:
                trace = trace.copy()
                AP_peaks_idx = find_peaks(trace, height=spike_threshold)[0]
                AP_peaks_time = AP_peaks_idx * 1000 /acquisition_frequency
                AP_numbers.append(np.sum(np.shape(AP_peaks_time))) 
        
        for number in AP_numbers:
                current_firing_rate = number/stim_length * 1000
                avg_firing_rates.append(np.mean(current_firing_rate))
                
        return AP_numbers, avg_firing_rates

def analyze_AP_AHP_properties_custom(traces, stim_length, stims, acquisition_frequency, positive_sweep_number_range, Vm_rest_start, Vm_rest_end, AP_peak_window_start, AP_peak_window_end):
    #for debugging
    import matplotlib.pyplot as plt
    # fig, ax = plt.subplots(1, 2 , figsize=(8, 4))   
    #initialize the variables to store the results of the
    #Action Potential Properties
    V_rest = []
    AP_traces = []
    AP_peaks = []
    AP_peaks_indices = []
    AP_thresholds = []
    AP_ISI_list = []
    AP_rheobase = []
    AP_peaks_dict = {}
    AP_firing_rates_dict = {}
    AP_sizes = []
    AP_halfwidth_list = [] 

    #After Hyperpolarization Properties
    AHP_sizes = []
    AHP_halfwidth_list = [] 

    for sweep in positive_sweep_number_range:
        current_trace = traces[sweep]
        #Make a copy of the current trace to avoid modifying the original trace
        current_trace = current_trace.copy()
        #Find the AP Peak for analysis

        try:
            AP_peaks_idx = find_peaks(current_trace, height=0)[0]
            print(f'AP_peaks_idx: {AP_peaks_idx}')
            AP_peaks_indices.append(AP_peaks_idx)

            #Get the number of AP peaks 
            AP_peaks_dict[sweep] = len(AP_peaks_idx)

            #Action Potential Fire Rates (Hz)
            AP_firing_rates_dict[sweep] = len(AP_peaks_idx)/(stim_length/1000)

            #Find the ISIs in ms
            AP_ISI_time = np.diff(AP_peaks_idx)/acquisition_frequency * 1000
            AP_ISI_list.append(AP_ISI_time)
        except:
             pass
            #  print(f'No spikes found in sweep {sweep}')

        prev_peak_idx = 0
        for i, peak_idx in enumerate(AP_peaks_idx):

            #Get rest potential
            try:
                if i == 0:
                    Vm_rest_start = int(Vm_rest_start * acquisition_frequency/1000)
                    Vm_rest_end = int(Vm_rest_end * acquisition_frequency/1000)

                    V_rest.append(np.mean(current_trace[Vm_rest_start:Vm_rest_end]))
            except:
                pass
                # print(f'No rest potential found in sweep {sweep}')
            
            #AP peak voltage
            AP_peak_idx = peak_idx
            AP_peak_voltage = current_trace[AP_peak_idx]

            # Find the AP window
            AP_peak_window_start = int(AP_peak_window_start * acquisition_frequency/1000)
            AP_peak_window_end = int(AP_peak_window_end * acquisition_frequency/1000)

            window_start = peak_idx - AP_peak_window_start
            if prev_peak_idx > window_start: #avoid measuring the previous AP in the current window
                window_start = prev_peak_idx + int(np.round(0.5*acquisition_frequency)) #if firing rate is very high, set cutoff to 0.5ms after the previous AP
            window_end = peak_idx + AP_peak_window_end

            AP_trace = current_trace[window_start:window_end]
            AP_traces.append(AP_trace)

            prev_peak_idx = peak_idx

            #Find the AP threshold - the greatest rate of change in the AP trace (second derivative or third derivative)
            derivative = np.diff(AP_trace, n=2)
            current_derivative = derivative.copy()
            # filter the derivative to smooth the trace  
            derivative_filtered = signal.savgol_filter(current_derivative, 15, 3, mode='mirror')
            
            try:
                    AP_threshold_idx = find_peaks(derivative_filtered, height=0.5)[0] # Find peaks in the filtered derivative
                    if len(AP_threshold_idx) > 0:
                        AP_threshold_idx = AP_threshold_idx[0] # Get the index of the first peak
                        AP_threshold = AP_trace[AP_threshold_idx] # Get the threshold value from the trace
                        AP_thresholds.append(AP_threshold) # Append the threshold to the list
                    else:
                        AP_threshold = np.NaN
            except:
                pass

            try:
                AP_peak_idx = AP_peak_window_start # how many ms after the threshold or the AP_peak_window_start
                AP_peak_voltage = AP_trace[AP_peak_idx] # peak in mV
                AP_peaks.append(AP_peak_voltage)
            except:
                pass
                # print(f'No peak found in sweep {sweep}')
            
            # Action Potential Sizes
            AP_size = AP_peak_voltage - AP_threshold # AP in mV
            try:
                AP_sizes.append(float(AP_size)) 
            except:
                AP_sizes.append(np.NaN)

            #Action Potential Halfwidths - halfwidth center point is from the center of the AP to the threshold, width is the distance on both sides of the center point
            AP_halfwidth_voltage = AP_threshold + (AP_size)/2 # halfwidth in mV; AP_peak_voltage - AP_threshold = voltage at the center line, thus add threshold (smaller value) to get the distance from center to the threshold
            try:
                AP_halfwidth_start_idx = np.where(AP_trace >= AP_halfwidth_voltage)[0][0]
                AP_halfwidth_end_idx = AP_halfwidth_start_idx + np.where(AP_trace[AP_halfwidth_start_idx:] <= AP_halfwidth_voltage)[0][0] # starting from the halfwidth start, cut off the beginning artifact of the start of the trace, find the first time the voltage is slightly smaller than haldwidth start, as the value on both sides can not be exactly the same
            except:
                # print(f'No halfwidth found in sweep {sweep}')
                AP_halfwidth_start_idx = np.NaN
                AP_halfwidth_end_idx = np.NaN
        
            AP_halfwidth = (AP_halfwidth_end_idx - AP_halfwidth_start_idx)/acquisition_frequency # halfwidth in ms
            AP_halfwidth_list.append(AP_halfwidth)

            try: 
                # Get the mirror of the AP trace using (-) sign - for After Hyperpolarization
                if len(find_peaks(-AP_trace[int(AP_threshold_idx):])[0]) == 0: # if there is no trough after the threshold
                    AHP_halfwidth = np.NaN
                    AHP_size = np.NaN
                else:
                    AHP_trough_idx = find_peaks(-AP_trace[int(AP_threshold_idx):])[0][0] + AP_threshold_idx # starting from the threshold, find the first trough (negative peak)
                    AHP_voltage = AP_trace[AHP_trough_idx] # AHP in mV
                    AHP_size = AP_threshold - AHP_voltage # AHP size in mV 
                
                #After Hyperpolarization Halfwidths
                AHP_halfwidth_voltage = AHP_voltage + (AP_threshold - AHP_voltage)/2 # AHP halfwidth in mV; AP_threshold - AHP_voltage = the voltage at the center line, thus add the AHP voltage (smaller value) to get the distance from center to the AHP voltage
                AHP_halfwidth_start_idx = AP_peak_idx + np.where(AP_trace[AP_peak_idx:] <= AHP_halfwidth_voltage)[0][0] #find the values smaller than or equal to center line voltage, cut of the values in the beginning so start past the peak 
                
                if len(np.where(AP_trace[AHP_halfwidth_start_idx:]>=AHP_halfwidth_voltage)[0]) == 0: #if the AHP halfwidth voltage is not found in the trace, set the halfwidth to NaN
                    AHP_halfwidth = np.NaN
                else:
                    AHP_halfwidth_end_idx = AHP_halfwidth_start_idx + np.where(AP_trace[AHP_halfwidth_start_idx:] >= AHP_halfwidth_voltage)[0][0] # starting from the halfwidth start, cut off the beginning artifact of the start of the trace, find the first time the voltage is slightly larger than haldwidth start, as the value on both sides can not be exactly the same
                    AHP_halfwidth = (AHP_halfwidth_end_idx - AHP_halfwidth_start_idx)/acquisition_frequency # AHP halfwidth in ms
            except: 
                # print(f'No AHP found in sweep {sweep}')
                # print(f'No AHP halfwidth found in sweep {sweep}')
                AHP_halfwidth = np.NaN
                AHP_size = np.NaN
            
            AHP_sizes.append(AHP_size)
            AHP_halfwidth_list.append(AHP_halfwidth)

                #Find current rheobase
            try:
                if abs(AP_threshold) > 0:
                    rheobase = stims[sweep] * 1000
                    AP_rheobase.append(rheobase[window_start:window_end])
            except:
                pass 
                # print(f'No rheobase found in sweep {sweep}')    
   
    try:
        AP_ISI_list = AP_ISI_list[9]  # ISI in ms
        AP_rheobase = abs(np.nanmean(AP_rheobase[0]))  # rheobase in pA
    except:
        AP_ISI_list = np.NaN
        AP_rheobase = np.NaN

    avg_Vm_rest = float(np.nanmean(V_rest))  # in mV

    try:
        avg_AP_size = float(np.nanmean(AP_sizes))  # in mV
    except:
        avg_AP_size = np.NaN

    try:
        avg_AHP_size = float(np.nanmean(AHP_sizes))  # in mV
    except:
        avg_AHP_size = np.NaN  # Assign NaN if the array is empty

    AP_Total_Number = len(AP_sizes)  # total number of APs

    try:
        avg_AHP_halfwidth = float(np.nanmean(AHP_halfwidth_list))  # in ms
    except:
        avg_AHP_halfwidth = np.NaN  # Assign NaN if the array is empty

    try:
        avg_AP_threshold = float(np.nanmean(AP_thresholds))  # in mV
    except:
        avg_AP_threshold = np.NaN  # Assign NaN if the array is empty

    AP_properties_spikes_dict = {
        'AP_threshold (mV)': AP_thresholds,
        'Vm_rest_per_sweep (mV)': V_rest,
        'AP_halfwidth (ms)': AP_halfwidth_list,
        'AP_size (mV)': AP_sizes,
        'AHP_halfwidth (ms)': AHP_halfwidth_list,
        'AHP_size (mV)': AHP_sizes}

    AP_properties_avg_dict = {
        'AP_peaks_indices': AP_peaks_indices,
        'Sweep_Range': [positive_sweep_number_range],
        'AP_peaks_per_sweep': AP_peaks_dict,
        'AP_firing_rates': AP_firing_rates_dict,
        'AP_Total_Number': AP_Total_Number,
        'AP_rheobase (pA)': AP_rheobase,
        'AP_ISI_time (ms)': AP_ISI_list,
        'Avg_Vm_rest (mV)': avg_Vm_rest,
        'Avg_AP_threshold': avg_AP_threshold,
        'Avg_AP_halfwidth (ms)': avg_AHP_halfwidth,
        'Avg_AP_size (mv)': avg_AP_size,
        'Avg_AHP_halfwidth (ms)': avg_AHP_halfwidth,
        'Avg_AHP_size (mV)': avg_AHP_size} 
    
    return AP_properties_spikes_dict, AP_properties_avg_dict

# def calculate_input_resistance(voltage_trace, current_pulse_amp, acquisition_frequency, dt, start_time, end_time):
#     vm_baseline = np.mean(voltage_trace[int(start_time * acquisition_frequency/1000) - int(10 * acquisition_frequency/1000):int(start_time * acquisition_frequency/1000)])
#     end_vm = np.mean(voltage_trace[int(end_time * acquisition_frequency/1000) - int(10 * acquisition_frequency/1000):int(end_time * acquisition_frequency/1000)])
#     delta_voltage = (end_vm - vm_baseline)/1000 #convert to volts
#     delta_current = current_pulse_amp * 10**-12 #convert to amps
#     input_resistance = (delta_voltage / -delta_current) * 10**-6 #convert to megaohms
#     return input_resistance

def calculate_input_resistances(data, start_time, end_time, acquisition_frequency, current_pulse_amp):
        traces = []
        for i in range(len(data)):
                current_trace = data[i]
                traces.append(current_trace)
        input_resistances = []
        for trace in traces: 
                vm_baseline = np.mean(trace[int(start_time * acquisition_frequency/1000) - int(10 * acquisition_frequency/1000):int(start_time * acquisition_frequency/1000)])
                end_vm = np.mean(trace[int(end_time * acquisition_frequency/1000) - int(10 * acquisition_frequency/1000):int(end_time * acquisition_frequency/1000)])
                delta_voltage = (end_vm - vm_baseline)/1000 #convert to volts
                delta_current = current_pulse_amp * 10**-12 #convert to amps
                input_resistance = (delta_voltage / -delta_current) * 10**-6 #convert to megaohms
                input_resistances.append(-input_resistance)
        return input_resistances

def sweep_number_to_index_array(sweep_numbers, sweep_number_subset_start, sweep_number_subset_end):
    index_array = []
    for sweep_number in range(sweep_number_subset_start, sweep_number_subset_end, 1):
        try:
            index = sweep_numbers.index(sweep_number) 
            index_array.append(index)
        except:
            pass
    return np.array(index_array)

def offset(raw_traces, acquisition_frequency, zeroing_sample_start, zeroing_sample_end):
    '''This function offsets the data and applies a low pass filter to the data
    
    :param sweep_number_range: the sweep numbers to be analyzed, array of int
    :param acquisition_frequency: the sampling frequency of the data, Hz;int
    :param recording_total_points: the total number of points in the data, int
    :param zeroing_sample_start: the time point of baseline to start for offsetting the data, ms;int
    :param zeroing_sample_end: the time point of baseline to end for offsetting the data, ms;int
 
    '''
    offset_data = []
    for trace in raw_traces: 
        offset_window = trace[int(zeroing_sample_start/1000*acquisition_frequency):int(zeroing_sample_end/1000*acquisition_frequency)] 
        current_trace = trace - np.mean(offset_window)
        offset_data.append(current_trace)
    return np.array(offset_data)

def remove_stim_artifacts(trace, acquisition_frequency, dt, height, delete_start_duration, delete_end_duration):
    '''This function to remove the stimulation artifacts from a single data trace''' 
    processed_trace = np.copy(trace)
    time = np.arange(0, len(trace) * dt, dt)
    trace = trace.copy()
    artifact_indexes = find_peaks(trace, height=height)[0]
    for peak_idx in artifact_indexes:
        delete_idx_pos = np.arange(peak_idx - int(delete_start_duration * acquisition_frequency/1000), peak_idx + int(delete_end_duration * acquisition_frequency/1000), 1) 
        #set the values to nan for an index that is the length of the stim artifact window
        processed_trace[delete_idx_pos] = np.nan
        is_nan_bool_array = np.isnan(processed_trace)
        #Need to interpolate across nans that are boolean true
        #create a line a line between the two points before and after the nan that is at a value of 0
        processed_trace[is_nan_bool_array] = np.interp(time[is_nan_bool_array], time[~is_nan_bool_array], processed_trace[~is_nan_bool_array])
       

    return processed_trace

def LowPass_filter(raw_traces, acquisition_frequency, filter_order, filter_type, filter_output, critical_frequency):
    '''This function offsets the data and applies a low pass filter to the data
    
    :param sweep_number_range: the sweep numbers to be analyzed, array of int
    :param acquisition_frequency: the sampling frequency of the data, Hz;int
    :param recording_total_points: the total number of points in the data, int
    :param zeroing_sample_start: the time point of baseline to start for offsetting the data, ms;int
    :param zeroing_sample_end: the time point of baseline to end for offsetting the data, ms;int
    :param filter_order: the order of the filter, int
    :param filter_type: the type of filter, string
    :param filter_output: the output of the filter - numerator/denominator ('ba'), pole-zero ('zpk'), or second-order sections ('sos')., string
    :param critical_frequency: the critical frequency to filter the data, in the same units as acquisition frequency, Hz; int
    :return: filterd_offset_data
    '''
    filtered_offset_data = [] 
    for trace in raw_traces: 
        #butter(N = The order of the filter, Wn = The critical frequency or frequencies., btype='low', analog=False, output='ba', fs=The sampling frequency of the digital system.)
        #Wn is thus in half cycles / sample and defined as 2*critical frequencies / fs 
        filter = signal.butter(filter_order, critical_frequency, filter_type, fs = acquisition_frequency, output = filter_output)
        filtered_offset_trace = signal.sosfilt(filter, trace) 
        filtered_offset_data.append(filtered_offset_trace)
    return filtered_offset_data

def batch_remove_stim_artifacts(offset_data, acquisition_frequency, dt, height, delete_start_duration, delete_end_duration):
    '''This function to remove the stimulation artifacts from all data traces
    '''
    processed_data = []
    for trace in offset_data:
        processed_trace = remove_stim_artifacts(trace, acquisition_frequency, dt, height, delete_start_duration, delete_end_duration)
        processed_data.append(processed_trace)

    return processed_data

def create_analysis_windows(analysis_start_time, buffer_start_duration, analysis_end_time, buffer_end_duration, inter_channel_interval, single_pulse_duration, interpulse_duration) : 
    '''  
    Function to pull out single pulses from a trace with defind windows to analyze 

    '''

    #To find windows for single pulses with given channel start times that may change from experiment to experiment
    analysis_start_time = analysis_start_time + buffer_start_duration #ms
    analysis_end_time = analysis_end_time + buffer_end_duration #ms
    channel_start_times = np.arange(analysis_start_time, analysis_end_time, inter_channel_interval) #ms
    single_pulse_duration = single_pulse_duration #ms
    interpulse_duration = interpulse_duration #ms

    first_pulse_window_list = []
    second_pulse_window_list = []
    for channel_start_time in channel_start_times:
            #Find the first pulses in the whole sweep
            first_pulse_windows = np.arange(int(channel_start_time*acquisition_frequency/1000), int((channel_start_time + single_pulse_duration) * acquisition_frequency/1000) , 1)  #ms
            current_first_pulse_end = int((channel_start_time + interpulse_duration) * acquisition_frequency/1000)
            first_pulse_window_list.append(first_pulse_windows)

            #find the second pulses in the whole sweep
            second_pulse_windows = np.arange(current_first_pulse_end, current_first_pulse_end + int(single_pulse_duration * acquisition_frequency/1000) , 1)  #ms
            second_pulse_window_list.append(second_pulse_windows)
            # paired_pulse_duration = (second_pulse_windows[-1]/ acquisition_frequency) * 1000  - channel_start_time #ms
            # print(f'Paired Pulse Duration channel {i+1}: {paired_pulse_duration} ms')
    
    return first_pulse_window_list, second_pulse_window_list

def get_EPSP_amplitudes(data, analysis_windows, vm_baseline_start, vm_baseline_end):
    peaks = {}
    max_peaks = {}
    max_peaks_index = {}
    EPSP_amplitudes = {}
    for i, trace in enumerate(data):
        peaks[i] = {}
        max_peaks_index[i] = {}
        max_peaks[i] = {}
        EPSP_amplitudes[i] = {}
        for channel in analysis_windows:
            peaks[i][channel] = {}
            max_peaks[i][channel] = {}
            max_peaks_index[i][channel] = {}
            EPSP_amplitudes[i][channel] = {}
            for pulse in analysis_windows[channel]:
                if pulse == 'first_pulse':
                    peaks[i][channel][pulse] = find_peaks(trace[analysis_windows[channel][pulse]], -65)
                    current_peak = peaks[i][channel][pulse]
                    #get baseline vm
                    baseline_vm = np.mean(trace[int(vm_baseline_start * acquisition_frequency/1000) : int(vm_baseline_end * acquisition_frequency/1000)])
                    try:
                        max_peaks[i][channel][pulse] = np.max(trace[analysis_windows[channel][pulse]][current_peak[0]])
                        max_peaks_index[i][channel][pulse] = np.where(trace[analysis_windows[channel][pulse]] == max_peaks[i][channel][pulse])
                        max_peaks_index[i][channel][pulse] = max_peaks_index[i][channel][pulse][0][0]

                        #calculate EPSP amplitude
                        EPSP_amplitudes[i][channel][pulse] =  (-baseline_vm) - (-max_peaks[i][channel][pulse]) 
                    except:
                        max_peaks[i][channel][pulse] = np.nan
                        max_peaks_index[i][channel][pulse] = np.nan
                        EPSP_amplitudes[i][channel][pulse] = np.nan
                elif pulse == 'second_pulse':
                    peaks[i][channel][pulse] = find_peaks(trace[analysis_windows[channel][pulse]], -65)
                    current_peak = peaks[i][channel][pulse]
                    #get baseline vm
                    baseline_vm = np.mean(trace[int(vm_baseline_start * acquisition_frequency/1000) : int(vm_baseline_end * acquisition_frequency/1000)])
                    try:
                        max_peaks[i][channel][pulse] = np.max(trace[analysis_windows[channel][pulse]][current_peak[0]])
                        max_peaks_index[i][channel][pulse] = np.where(trace[analysis_windows[channel][pulse]] == max_peaks[i][channel][pulse])
                        max_peaks_index[i][channel][pulse] = max_peaks_index[i][channel][pulse][0][0]

                        #calculate EPSP amplitude
                        EPSP_amplitudes[i][channel][pulse] =  (-baseline_vm) - (-max_peaks[i][channel][pulse]) 
                    except:
                        max_peaks[i][channel][pulse] = np.nan
                        max_peaks_index[i][channel][pulse] = np.nan
                        EPSP_amplitudes[i][channel][pulse] = np.nan

    return EPSP_amplitudes

    

