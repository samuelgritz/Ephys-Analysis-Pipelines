__author__ = 'Samuel Gritz' 

import numpy as np
import pandas as pd
import matplotlib as mpl
import yaml
import os

import tkinter as tk
from tkinter import ttk 
from tkinter import simpledialog, filedialog, ttk, messagebox

import matplotlib.figure as figure
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from PIL import Image, ImageTk

# Import custom functions
import scipy.signal as signal
from scipy.signal import find_peaks, savgol_filter


#Updates
#TODO: Add feature to be able to open new data automatically - resetting the intermediate to 'raw' data


# Set matplotlib configurations
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['font.size'] = 12.
mpl.rcParams['font.sans-serif'] = 'Arial'
mpl.rcParams['text.usetex'] = False

#Add user entry boxes for all values that need to be set or maintained
def read_from_yaml(file_path, Loader=None):
    """Import a python dict from .yaml

    Parameters
    ----------
    file_path : str
        Path to the .yaml file.
    Loader : class, optional
        YAML loader class, by default None.

    Returns
    -------
    dict
        Dictionary containing the data from the .yaml file.

    Raises
    ------
    Exception
        If the file does not exist.
    """
    if Loader is None:
        Loader = yaml.FullLoader
    if os.path.isfile(file_path):
        with open(file_path, 'r') as stream:
            data = yaml.load(stream, Loader=Loader)
        return data
    else:
        raise Exception('File: {} does not exist.'.format(file_path))

def analyze_AP_AHP_properties_trace(trace, time, AP_peak_idx, stim_end, stim_length, acquisition_frequency, Vm_rest_start, Vm_rest_end, AP_peak_window_start, AP_peak_window_end):

    """Analyzes action potential (AP) and afterhyperpolarization (AHP) properties from a single trace.

    Parameters
    ----------
    trace : np.array
        The voltage trace data.
    time : np.array
        The time array corresponding to the voltage trace.
    AP_peak_idx : list or int
        A list of indices or a single index corresponding to the detected AP peaks.
    stim_end : int or float
        The time in milliseconds when the stimulation ends.
    stim_length : int or float
        The duration of the stimulation in milliseconds.
    acquisition_frequency : int or float
        The acquisition frequency of the data in Hz.
    Vm_rest_start : int or float
        Start time in milliseconds of the window to measure resting membrane potential.
    Vm_rest_end : int or float
        End time in milliseconds of the window to measure resting membrane potential.
    AP_peak_window_start : int or float
        Duration in milliseconds before the AP peak to define the analysis window.
    AP_peak_window_end : int or float
        Duration in milliseconds after the AP peak to define the analysis window.

    Returns
    -------
    dict
        A dictionary containing the analyzed AP and AHP properties.
    """

    # plt.figure()
    #Make a copy of the current trace to avoid modifying the original trace
    current_trace = trace.copy()
    time = time.copy()
    # plt.plot(time, current_trace, color='black')

    #Resting membrane potential
    Vm_rest = np.mean(current_trace[Vm_rest_start:Vm_rest_end])

    #AP properties
    AP_number = []
    AP_firing_rate = []
    AP_traces = []
    AP_thresholds = []
    AP_threshold_indices = []
    AP_ISI_list = []
    AP_sizes = []
    AP_halfwidth_list = []

    #AHP properties
    AHP_indices = []
    AHP_sizes = []
    AHP_halfwidth_list = []
    
    # print('AP_peak_idx:', AP_peak_idx)

    #if AP_peak_idx is a single value, then convert it to a list
    if type(AP_peak_idx) == int:
        AP_peak_idx = [AP_peak_idx]

    if len(AP_peak_idx) == 0:
        AP_number = len(AP_peak_idx)
        AP_firing_rate = AP_number/(stim_length/1000)

        # print('AP_number:', AP_number)
        # print('AP_firing_rate:', AP_firing_rate)
    else:
        AP_number = len(AP_peak_idx)
        AP_firing_rate = AP_number/(stim_length/1000)

        # print('AP_number:', AP_number)
        # print('AP_firing_rate:', AP_firing_rate)
    
    # Define duration of time before and after the AP peak
    AP_peak_window_start = int(AP_peak_window_start * acquisition_frequency/1000) #duration of time before the AP
    AP_peak_window_end = int(AP_peak_window_end * acquisition_frequency/1000) #duration of time after the AP

    # #Calculate the time between the peaks
    spike_times = time[(AP_peak_idx)]

    prev_peak_idx = 0

    for i, peak_idx in enumerate(AP_peak_idx):
        AP_ISI = spike_times[i] - spike_times[i-1] if i > 0 else np.NaN
        if i > 0:
            AP_ISI_list.append(AP_ISI)
        
        AP_peak_voltage = current_trace[peak_idx]

        window_start = peak_idx - AP_peak_window_start # default value of window start

        temporal_peak_offset = int(np.round(0.5*acquisition_frequency/1000)) # pick an index that is 0.5ms after the previous peak_idx
        #pick the larger of the two values between the previous peak + offset or the windoe start
        # Assuming prev_peak_idx and temporal_peak_offset are numpy arrays
        indices = np.where(prev_peak_idx + temporal_peak_offset > window_start)[0]
        if indices.size > 0:
            window_start = np.max(prev_peak_idx[indices] + temporal_peak_offset[indices])
        # window_start = max(prev_peak_idx+temporal_peak_offset, window_start)#avoid measuring the previous AP in the current window if the previous peak is within the window then pick a value that start just after the previous peak
        window_end = peak_idx + AP_peak_window_end

        # Find the AP threshold - the greatest rate of change in the AP trace (second derivative or third derivative)
        AP_trace = current_trace[window_start:window_end]
        AP_traces.append(AP_trace)

        #try the second derivative
        try:
            derivative = np.diff(AP_trace, n=2)
            current_derivative = derivative.copy()
            # filter the derivative to smooth the trace
            derivative_filtered = signal.savgol_filter(current_derivative, 15, 3, mode='mirror')
            try:
                threshold_idx_in_AP_trace = find_peaks(derivative_filtered, height=0.5)[0][0] # Find peaks in the filtered derivative
            except:
                continue
                threshold_idx_in_AP_trace = 0

            AP_threshold_idx = window_start + threshold_idx_in_AP_trace # Get the threshold index in the original trace
            AP_threshold_indices.append(AP_threshold_idx)
            AP_threshold = current_trace[AP_threshold_idx] # Get the threshold value from the trace
            if AP_threshold > 0: # Fix the syntax error here
                AP_threshold = np.NaN

            AP_thresholds.append(AP_threshold)

        #if second derivative does not work, try the first derivative
        except:
            derivative = np.diff(AP_trace)
            current_derivative = derivative.copy()
            # filter the derivative to smooth the trace
            derivative_filtered = signal.savgol_filter(current_derivative, 15, 3, mode='mirror')
            try:
                threshold_idx_in_AP_trace = find_peaks(derivative_filtered, height=0.5)[0][0] # Find peaks in the filtered derivative
            except:
                continue
                threshold_idx_in_AP_trace = 0

            threshold_idx_in_AP_trace = find_peaks(derivative_filtered, height=0.5)[0][0]
            AP_threshold_idx = window_start + threshold_idx_in_AP_trace
            AP_threshold_indices.append(AP_threshold_idx)
            AP_threshold = current_trace[AP_threshold_idx]
            if AP_threshold > 0: # Fix the syntax error here

                AP_threshold = np.NaN

        try:
            AP_peak_voltage = current_trace[peak_idx] # peak in mV
            AP_size = AP_peak_voltage - AP_threshold # AP in mV
            AP_sizes.append(AP_size)
        except:
            AP_size = np.NaN
            AP_sizes.append(np.NaN)

        try:
            #Action Potential Halfwidths - halfwidth center point is from the center of the AP to the threshold, width is the distance on both sides of the center point
            AP_halfwidth_voltage = AP_threshold + (AP_size)/2 # halfwidth in mV; AP_peak_voltage - AP_threshold = voltage at the center line, thus add threshold (smaller value) to get the distance from center to the threshold
            AP_halfwidth_start_idx = np.where(AP_trace >= AP_halfwidth_voltage)[0][0]
            AP_halfwidth_points = np.where(AP_trace[AP_halfwidth_start_idx:] <= AP_halfwidth_voltage)[0][0] # starting from the halfwidth start, cut off the beginning artifact of the start of the trace, find the first time the voltage is slightly smaller than haldwidth start, as the value on both sides can not be exactly the same
            AP_halfwidth = AP_halfwidth_points/acquisition_frequency * 1000 # halfwidth in ms
            AP_halfwidth_list.append(AP_halfwidth)
            
            # Get the mirror of the AP trace using (-) sign - for After Hyperpolarization
            #if the index is past time 850ms, then cut off the trace at 850ms as the AHP is not relevant
            if window_end > int(stim_end * acquisition_frequency/1000):
                window_end = int(stim_end * acquisition_frequency/1000)

            AHP_trough_idx = np.argmin(current_trace[peak_idx:window_end]) + peak_idx # starting from the peak_idx, find the first trough (negative peak)
            # print('AHP_trough_idx:', AHP_trough_idx)
            AHP_indices.append(AHP_trough_idx)
         
            AHP_voltage = current_trace[AHP_trough_idx] # AHP in mV
            # print('AHP_voltage:', AHP_voltage)
            AHP_size = AP_threshold - AHP_voltage # AHP size in mV 
            AHP_sizes.append(AHP_size)
        except:
            AHP_size = np.NaN
            AHP_sizes.append(np.NaN)
            AHP_indices.append(np.NaN)

        try:
            #After Hyperpolarization Halfwidths
            AHP_halfwidth_voltage = (AP_threshold + AHP_voltage)/2 # AHP halfwidth in mV; AP_threshold - AHP_voltage = the voltage at the center line, thus add the AHP voltage (smaller value) to get the distance from center to the AHP voltage
            #find where values are smaller than the haldwidth voltage and get the index of the first value
            # print(AHP_halfwidth_voltage)
            #find this relative to the beginning of the AHP trace
            AHP_halfwidth_start_idx = peak_idx + np.where(current_trace[peak_idx:window_end] <= AHP_halfwidth_voltage)[0][0] 
            # print('AHP_halfwidth_start_idx:', AHP_halfwidth_start_idx)

            # if len(np.where(current_trace[AHP_halfwidth_start_idx:]>=AHP_halfwidth_voltage)[0]) == 0: #if the AHP halfwidth voltage is not found in the trace, set the halfwidth to NaN
            #     AHP_halfwidth = np.NaN
            # else:
            #where does the trace go above the AHP halfwidth voltage is the end so take that index and add to start index to get the width        
            # print(np.where(current_trace[AHP_halfwidth_start_idx:] >= AHP_halfwidth_voltage)[0])    
            AHP_halfwidth_end_idx = peak_idx + np.where(current_trace[AHP_halfwidth_start_idx:] >= AHP_halfwidth_voltage)[0][0] # starting from the halfwidth start, cut off the beginning artifact of the start of the trace, find the first time the voltage is slightly larger than haldwidth start, as the value on both sides can not be exactly the same
            # print('AHP_halfwidth_end_idx:', AHP_halfwidth_end_idx)
            AHP_halfwidth = (AHP_halfwidth_end_idx - AHP_halfwidth_start_idx)/acquisition_frequency * 1000 # AHP halfwidth in ms
            # print('AHP_halfwidth:', AHP_halfwidth)
            AHP_halfwidth_list.append(AHP_halfwidth)
        except:
            AHP_halfwidth = np.NaN
            AHP_halfwidth_list.append(np.NaN)

    AP_properties_spikes_dict = {
        'AP_threshold': AP_thresholds, #mV
        'AP_threshold_indices': AP_threshold_indices, #index
        'AP_peak_indices': AP_peak_idx, #index
        'AP_firing_rate': AP_firing_rate, #Hz
        'AP_number': AP_number, #number
        'AP_ISI_time': AP_ISI_list, #ms
        'AP_halfwidth': AP_halfwidth_list, #ms
        'AP_size': AP_sizes, #mV
        'AHP_size': AHP_sizes, #mV
        'AHP_halfwidth': AHP_halfwidth_list, #ms
        'AHP_indices': AHP_indices #index 
    }
    
    return AP_properties_spikes_dict

#Old version
def interp_spikes_old(AP_peak_window_start, AP_peak_window_end, spike_width, acquisition_frequency, voltage_trace, time):
    """Interpolates over spikes in a voltage trace using an older method.

    Parameters
    ----------
    AP_peak_window_start : float
        Duration in milliseconds before the AP peak to define the analysis window.
    AP_peak_window_end : float
        Duration in milliseconds after the AP peak to define the analysis window.
    spike_width : float
        The expected width of the spike in milliseconds.
    acquisition_frequency : int or float
        The acquisition frequency of the data in Hz.
    voltage_trace : np.array
        The voltage trace data.
    time : np.array
        The time array corresponding to the voltage trace.

    Returns
    -------
    np.array
        The voltage trace with spikes interpolated.
    """
    spike_onsets = []
    interp_ends = []

    working_trace = voltage_trace.copy()
    AP_peaks = find_peaks(working_trace, height=-40, width=(10,1000), distance=50, prominence=7)[0] 
    prev_peak_idx = 0
    time = np.arange(0, len(voltage_trace) * 0.05, 0.05)

    for peak_idx in AP_peaks:
        #plt.figure()
        window_start = peak_idx - int(AP_peak_window_start * acquisition_frequency / 1000)
        temporal_peak_offset = int(np.round(0.5 * acquisition_frequency / 1000))  # 0.5 ms after previous peak_idx
        indices = np.where(prev_peak_idx + temporal_peak_offset > window_start)[0]
        if indices.size > 0:
            window_start = np.max(prev_peak_idx[indices] + temporal_peak_offset[indices])
        window_end = peak_idx + int(AP_peak_window_end * acquisition_frequency / 1000)
        AP_trace = voltage_trace[window_start:window_end]
        #plt.plot(AP_trace)

        try:
            derivative = np.diff(AP_trace, n=1)
            derivative_filtered = savgol_filter(derivative, 15, 3, mode='mirror')
            peaks, _ = find_peaks(derivative_filtered, height=0.2, prominence=0.1, distance=80, width=(5, 1000))
            if peaks.size > 0:
                threshold_idx_in_AP_trace = peaks[0]
                AP_threshold_idx = window_start + threshold_idx_in_AP_trace
                AP_threshold_idx -= int(spike_width * acquisition_frequency / 1000)
            else:
                #print("No peaks found in first derivative, trying second derivative")
                derivative = np.diff(AP_trace, n=2)
                derivative_filtered = savgol_filter(derivative, 15, 3, mode='mirror')
                peaks, _ = find_peaks(derivative_filtered, height=0.2, prominence=0.1, distance=80, width=(5, 1000))
                if peaks.size > 0:
                    threshold_idx_in_AP_trace = peaks[0]
                    AP_threshold_idx = window_start + threshold_idx_in_AP_trace
                    AP_threshold_idx = peak_idx - (int(spike_width * acquisition_frequency / 1000) + int(4 * acquisition_frequency / 1000))
                else:
                    #print("No peaks found in second derivative either. Skipping this iteration.")
                    continue
        except Exception as e:
            print(f"Error occurred: {e}")
            continue

        # Append AP_threshold_idx only if it has been successfully assigned
        if 'AP_threshold_idx' in locals():
            spike_onsets.append(AP_threshold_idx)

    #print(spike_onsets) 
    for i, idx in enumerate(spike_onsets):
        start = AP_peaks[i]
        baseline = np.mean(voltage_trace[(idx - int(0.1 * acquisition_frequency / 1000)):idx])
        max_end = idx + int(AP_peak_window_end * acquisition_frequency / 1000)
        min_idx = np.argmin(voltage_trace[start:max_end]) + start

        for end in range(start, max_end):
            if voltage_trace[end] <= baseline:
                break
            if end == min_idx:
                break
            try:
                if end == spike_onsets[i + 1]:
                    break
            except IndexError:
                pass
        interp_ends.append(end)

    for i, idx in enumerate(spike_onsets):
        end_index = interp_ends[i]
        voltage_trace[idx + 1:end_index] = np.nan

    is_nan_bool_array = np.isnan(voltage_trace)
    voltage_trace[is_nan_bool_array] = np.interp(time[is_nan_bool_array], time[~is_nan_bool_array], voltage_trace[~is_nan_bool_array])

    return voltage_trace #, AP_peaks, spike_onsets

#new version
def interp_spikes(AP_peak_window_start, AP_peak_window_end, spike_width, acquisition_frequency, voltage_trace, time):
    """Interpolates over spikes in a voltage trace using a new method.

    Parameters
    ----------
    AP_peak_window_start : float
        Duration in milliseconds before the AP peak to define the analysis window.
    AP_peak_window_end : float
        Duration in milliseconds after the AP peak to define the analysis window.
    spike_width : float
        The expected width of the spike in milliseconds.
    acquisition_frequency : int or float
        The acquisition frequency of the data in Hz.
    voltage_trace : np.array
        The voltage trace data.
    time : np.array
        The time array corresponding to the voltage trace.

    Returns
    -------
    np.array
        The voltage trace with spikes interpolated.
    """
    spike_onsets = []
    interp_ends = []

    working_trace = voltage_trace.copy()
    AP_peaks = find_peaks(working_trace, height=-50, width=(10,1000), distance=50, prominence=20)[0]  
    
    print(f"Detected {len(AP_peaks)} peaks")

    time = np.arange(0, len(voltage_trace) * 0.05, 0.05)

    for i, peak_idx in enumerate(AP_peaks):
        window_start = peak_idx - int(AP_peak_window_start * acquisition_frequency / 1000)
        window_end = peak_idx + int(AP_peak_window_end * acquisition_frequency / 1000)
        if window_start < 0 or window_end >= len(voltage_trace):
            continue
        
        # Get the local trace around the spike
        AP_trace = voltage_trace[window_start:window_end]

        # First derivative to detect onset
        derivative = np.diff(AP_trace, n=1)
        derivative_filtered = savgol_filter(derivative, 15, 3, mode='mirror')
        peaks, _ = find_peaks(derivative_filtered, height=0.2, prominence=0.1, distance=80, width=(5, 1000))

        if peaks.size > 0:
            threshold_idx_in_AP_trace = peaks[0]
            AP_threshold_idx = window_start + threshold_idx_in_AP_trace
            spike_onsets.append(AP_threshold_idx)
        else:
            AP_threshold_idx = peak_idx - int(1 * acquisition_frequency / 1000)  # Fallback threshold
            spike_onsets.append(AP_threshold_idx)

    # Find the endpoint for each spike (where it returns to baseline or minimum)
    for i, idx in enumerate(spike_onsets):
        start = idx
        baseline = np.mean(voltage_trace[max(0, idx - int(50 * acquisition_frequency / 1000)):idx])
        max_end = min(idx + int(AP_peak_window_end * acquisition_frequency / 1000), len(voltage_trace) - 1)
        
        # First search for return to baseline
        end = max_end
        for j in range(start, max_end):
            if voltage_trace[j] <= baseline:
                end = j
                break
        
        # If it doesn’t return to baseline, take the minimum point after peak
        if end == max_end:
            min_idx = np.argmin(voltage_trace[start:max_end]) + start
            end = min_idx
        
        interp_ends.append(end)

    for i, idx in enumerate(spike_onsets):
        end_index = interp_ends[i]
        if end_index > idx:
            # Flatten the peak by setting the entire region from the onset to the peak to baseline value
            baseline = np.mean(voltage_trace[max(0, idx - int(50 * acquisition_frequency / 1000)):idx])
            voltage_trace[idx:end_index] = baseline
        else:
            print(f"Invalid clipping range: idx={idx}, end_index={end_index}")

    # Step back further and interpolate across a longer range
    for i, idx in enumerate(spike_onsets):
        end_index = interp_ends[i]
        if end_index > idx:
            # Increase the range for interpolation by stepping back further
            extended_start = max(0, idx - int(4 * acquisition_frequency / 1000))  # Step back before the onset
            extended_end = min(len(voltage_trace) - 1, end_index + int(4 * acquisition_frequency / 1000))  # Step forward after the end
            
            # Interpolate across the extended range
            voltage_trace[extended_start:extended_end] = np.interp(
                time[extended_start:extended_end], 
                [time[extended_start-1], time[extended_end+1]], 
                [voltage_trace[extended_start-1], voltage_trace[extended_end+1]]
            )

    return voltage_trace

def calculate_input_resistance(voltage_trace, current_pulse_amp, acquisition_frequency, start_time, end_time):
        """Calculates the input resistance of a cell.

        Parameters
        ----------
        voltage_trace : np.array
            The voltage trace data.
        current_pulse_amp : int
            The amplitude of the current pulse in pA.
        acquisition_frequency : int
            The acquisition frequency of the data in Hz.
        start_time : int or float
            The start time of the current pulse in milliseconds.
        end_time : int or float
            The end time of the current pulse in milliseconds.

        Returns
        -------
        float
            The calculated input resistance in MΩ.
        """
        #iterate through the traces
        try:
            #convert the start, end time and acquisition frequency to integers
            start_time = int(start_time)
            end_time = int(end_time)
            acquisition_frequency = int(acquisition_frequency)
            current_pulse_amp = int(current_pulse_amp)
            # calculate the input resistance
            #convert start and end times to indices
            start_time_index = int(start_time * acquisition_frequency/1000)
            end_time_index = int(end_time * acquisition_frequency/1000)
            # #calculate the baseline voltage
            baseline_start = start_time_index - int(10 * acquisition_frequency/1000)
            vm_baseline = np.mean(voltage_trace[baseline_start:start_time_index])
            # #calculate the end voltage
            end_vm = np.mean(voltage_trace[end_time_index - baseline_start:end_time_index])
            delta_voltage = (end_vm - vm_baseline)/1000 #convert to volts
            delta_current = current_pulse_amp * 10**-12 #convert to amps
            input_resistance = abs(delta_voltage / -delta_current) * 10**-6 #convert to megaohms
            return input_resistance
        except Exception as e:
            print("Error calculating input resistance:", e)
            pass

def calculate_voltage_sag(experiment_type, voltage_trace, acquisition_frequency, start_time, end_time):
    """Calculates the voltage sag ratio.

    Parameters
    ----------
    experiment_type : str
        The type of the experiment. Must be 'voltage_sag'.
    voltage_trace : np.array
        The voltage trace data.
    acquisition_frequency : int or float
        The acquisition frequency of the data in Hz.
    start_time : int or float
        The start time of the hyperpolarizing step in milliseconds.
    end_time : int or float
        The end time of the hyperpolarizing step in milliseconds.

    Returns
    -------
    float
        The calculated voltage sag ratio in percentage.
    """
    try:
        if experiment_type != 'voltage_sag':
            #have a pop up that says the experiment type is not voltage sag
            messagebox.showinfo("Error", "The experiment type is not voltage sag, please select the correct experiment type")
        else:
            #convert the start, end time and acquisition frequency to integers
            start_time = int(start_time * 20000/1000)
            end_time = int(end_time * 20000/1000) 
            print('start_time:', start_time)
            print('end_time:', end_time)
            baseline_start = start_time - int(10 * 20000/1000)
            #calculate the baseline voltage
            baseline_voltage = np.mean(voltage_trace[baseline_start:start_time])
            print('baseline_voltage:', baseline_voltage)
            # print('baseline_voltage:', baseline_voltage)
            # Find the minimum voltage (sag minimum) during the hyperpolarizing step
            min_voltage_during_sag = min(voltage_trace[start_time:end_time])
            print('min_voltage_during_sag:', min_voltage_during_sag)
            # print('min_voltage_during_sag:', min_voltage_during_sag)
            # Find the steady-state voltage at the end of the hyperpolarizing step 50ms before the end of the step
            steady_state_voltage = np.mean(voltage_trace[end_time - int(10 * 20000/1000):end_time])
            print('steady_state_voltage:', steady_state_voltage)
            # print('steady_state_voltage:', steady_state_voltage)
            # Calculate the sag ratio
            sag_ratio = (steady_state_voltage - min_voltage_during_sag) / (baseline_voltage - min_voltage_during_sag)
            print('sag_ratio:', sag_ratio)
            sag_ratio = abs(sag_ratio) * 100

            return sag_ratio
        
    except Exception as e:
        print("Error calculating voltage sag:", e)
        pass

def remove_artifacts_custom(data, artifact_times, acquisition_frequency, delete_start_stim, delete_end_stim):
    """Removes stimulation artifacts from data with known stimulation times.

    Parameters
    ----------
    data : np.array
        The data containing stimulation artifacts.
    artifact_times : dict
        A dictionary with channels as keys and lists of artifact times in milliseconds.
    acquisition_frequency : int or float
        The acquisition frequency of the data in Hz.
    delete_start_stim : int or float
        The time in milliseconds to start deleting before the stimulation artifact.
    delete_end_stim : int or float
        The time in milliseconds to stop deleting after the stimulation artifact.

    Returns
    -------
    np.array
        The data with artifacts removed and interpolated.
    """
    processed_data = np.copy(data)  # Make a copy of the data to avoid modifying the original
    stim_times = artifact_times
    for channel in stim_times.keys():
        stim_times_list = stim_times[channel]
        for stim_time in stim_times_list:
            #convert stim time from a float to an integer
            baseline_start = int(stim_time * acquisition_frequency/1000) - int(0.5 * acquisition_frequency/1000)
            baseline_voltage = np.mean(processed_data[baseline_start:int(stim_time * acquisition_frequency/1000)])
            current_stim_index = int(stim_time * acquisition_frequency / 1000)
            delete_start_index = max(0, current_stim_index - int(delete_start_stim * acquisition_frequency / 1000))
            interp_start_index = min(len(processed_data), current_stim_index - int(delete_start_stim * acquisition_frequency / 1000))
            delete_end_index = min(len(processed_data), current_stim_index + int(delete_end_stim * acquisition_frequency / 1000))
            #interpolate to the first time that the Vm is above the (local) baseline, starting at the end of the delete_end_index
            #if the Vm is above the baseline, then interpolate to the first time that the Vm is above the baseline
            if any(processed_data[delete_end_index:] > baseline_voltage ):
                interp_end_index = np.where(processed_data[delete_end_index:] > baseline_voltage)[0][0] + delete_end_index
            else:
                interp_end_index = delete_end_index
            current_window = np.arange(interp_start_index, interp_end_index)
            processed_data[current_window] = np.nan
            processed_data = np.interp(np.arange(0, len(processed_data)), np.arange(0, len(processed_data))[~np.isnan(processed_data)], processed_data[~np.isnan(processed_data)])

    return processed_data

def remove_artifacts_automated(data, acquisition_frequency, delete_start_stim, delete_end_stim):
    """Removes stimulation artifacts from data automatically without knowing the stimulation times.

    Parameters
    ----------
    data : np.array
        The data containing stimulation artifacts.
    acquisition_frequency : int or float
        The acquisition frequency of the data in Hz.
    delete_start_stim : int or float
        The time in milliseconds to start deleting before the stimulation artifact.
    delete_end_stim : int or float
        The time in milliseconds to stop deleting after the stimulation artifact.

    Returns
    -------
    np.array
        The data with artifacts removed and interpolated.
    """
    ''' Deprecated function to remove stimulation artifacts from the data automatically without knowing the stimulation times'''
    processed_data = np.copy(data)  # Make a copy of the data to avoid modifying the original

    # stim_removal_threshold = stim_removal_threshold
    #first get the derivative of the data
    derivative = np.diff(processed_data, n=1)
    #find all possible APs first
    AP_peaks = find_peaks(derivative, height=0)[0]
    #Then find the negative inflection points
    negative_inflection_points = find_peaks(-derivative, height=0.2)[0] #0.2 is a good threshold for negative inflection points
    #If a peak occurs 1ms from an AP_peak, remove it from all peaks
    for peak in negative_inflection_points:
        for AP_peak in AP_peaks:
            if abs(peak - AP_peak) < acquisition_frequency / 1000:
                negative_inflection_points = negative_inflection_points[negative_inflection_points != AP_peak]
    # plt.figure()
    # plt.plot(derivative)
    # plt.scatter(negative_inflection_points, derivative[negative_inflection_points], color='r')
    # # #get now use those negative inflection points as the start of the window
    for inflection in negative_inflection_points:
        baseline_start = int(inflection * acquisition_frequency/1000) - int(0.5 * acquisition_frequency/1000)
        baseline_voltage = np.mean(processed_data[baseline_start:int(inflection * acquisition_frequency/1000)])
        delete_start_index = max(0, inflection - int(delete_start_stim * acquisition_frequency / 1000))
        interp_start_index = min(len(processed_data), inflection - int(delete_start_stim * acquisition_frequency / 1000))
        delete_end_index = min(len(processed_data), inflection + int(delete_end_stim * acquisition_frequency / 1000))
        if any(processed_data[delete_end_index:] > baseline_voltage ):
                interp_end_index = np.where(processed_data[delete_end_index:] > baseline_voltage)[0][0] + delete_end_index
        else:
            interp_end_index = delete_end_index
        current_window = np.arange(interp_start_index, interp_end_index)
        # plt.plot(current_window, derivative[current_window], color='g')
        processed_data[current_window] = np.nan
        processed_data = np.interp(np.arange(0, len(processed_data)), np.arange(0, len(processed_data))[~np.isnan(processed_data)], processed_data[~np.isnan(processed_data)])
    
    return processed_data

def remove_noise(data, noise_times, acquisition_frequency, delete_noise_duration_list):
    """Removes noise artifacts from data.

    Parameters
    ----------
    data : np.array
        The data containing noise artifacts.
    noise_times : list or int or float
        A list of noise times or a single noise time in milliseconds.
    acquisition_frequency : int or float
        The acquisition frequency of the data in Hz.
    delete_noise_duration_list : list or int or float
        A list of durations or a single duration to delete around each noise time in milliseconds.

    Returns
    -------
    np.array
        The data with noise artifacts removed and interpolated.
    """
    processed_data = np.copy(data)  # Make a copy of the data to avoid modifying the original

    # Ensure noise_times and delete_noise_duration_list are lists
    noise_times = [noise_times] if isinstance(noise_times, (int, float)) else noise_times
    delete_noise_duration_list = [delete_noise_duration_list] if isinstance(delete_noise_duration_list, (
    int, float)) else delete_noise_duration_list

    # Ensure noise_times and delete_noise_duration_list have the same length
    if len(noise_times) != len(delete_noise_duration_list):
        raise ValueError("noise_times and delete_noise_duration_list must have the same length.")

    # Loop through noise times and durations to process noise
    for noise_time, delete_noise_duration in zip(noise_times, delete_noise_duration_list):
        # Convert noise time and duration to indices
        current_noise_index = int(noise_time * acquisition_frequency / 1000)
        delete_start_index = max(0, current_noise_index - int(delete_noise_duration * acquisition_frequency / 1000))
        delete_end_index = min(len(processed_data),
                               current_noise_index + int(delete_noise_duration * acquisition_frequency / 1000))

        # Set the current window (region affected by noise) to NaN
        processed_data[delete_start_index:delete_end_index] = np.nan

        # Perform interpolation to fill NaN values
        not_nan_indices = np.arange(0, len(processed_data))[~np.isnan(processed_data)]
        processed_data = np.interp(np.arange(0, len(processed_data)), not_nan_indices,
                                   processed_data[~np.isnan(processed_data)])

    return processed_data

def remove_artifacts_plateau(data, time, acquisition_frequency, delete_start_stim, delete_end_stim, height):
        """Removes stimulation artifacts from plateau experiments.

        Parameters
        ----------
        data : np.array
            The data containing stimulation artifacts.
        time : np.array
            The time array corresponding to the data.
        acquisition_frequency : int or float
            The acquisition frequency of the data in Hz.
        delete_start_stim : int or float
            The time in milliseconds to start deleting before the stimulation artifact.
        delete_end_stim : int or float
            The time in milliseconds to stop deleting after the stimulation artifact.
        height : float
            The minimum voltage height to consider a peak.

        Returns
        -------
        np.array or None
            The data with artifacts removed and interpolated, or None if an error occurs.
        """
        try:
            current_trace = np.copy(data)
            #get the forth derivative using np.diff
            forth_derivative = np.diff(current_trace, n=4) 
            #find AP peaks
            AP_peaks = find_peaks(current_trace, height=height)[0]
            #get the time of the action potential peak
            AP_peaks_times = time[AP_peaks]
            #find all peaks in the forth derivative
            all_peaks = find_peaks(forth_derivative, height=1.5)[0]
            #If a peak occurs 1ms from an AP_peak, remove it from all peaks
            for peak in all_peaks:
                if any(abs(peak - AP_peaks) < int(1 * acquisition_frequency/1000) + 1):
                    all_peaks = np.delete(all_peaks, np.where(all_peaks == peak))
            #redefine the stim times
            stim_times = time[all_peaks]
            #create a processed trace
            processed_data = np.copy(current_trace)
            #remove the stimulation artifacts
            artifact_times = stim_times

            for artifact_time in artifact_times:
                current_stim_index = int(artifact_time * acquisition_frequency / 1000)
                delete_start_index = max(0, current_stim_index - int(delete_start_stim * acquisition_frequency / 1000))
                delete_end_index = min(len(processed_data), current_stim_index + int(delete_end_stim * acquisition_frequency / 1000))
                current_window = np.arange(delete_start_index, delete_end_index)
                processed_data[current_window] = np.nan
                processed_data = np.interp(np.arange(0, len(processed_data)), np.arange(0, len(processed_data))[~np.isnan(processed_data)], processed_data[~np.isnan(processed_data)])
            
            return processed_data
        except Exception as e:
            print("Error in remove_artifacts_plateau:", e)
            return None
    
def find_peaks_in_window(voltage_trace, delete_start_time, window_size, acquisition_frequency):
    """Finds the maximum peak and its index in a specified window of a single trace.

    Parameters
    ----------
    voltage_trace : np.array
        The voltage trace data.
    delete_start_time : int or float
        The start time of the window in milliseconds.
    window_size : int or float
        The duration of the window in milliseconds.
    acquisition_frequency : int or float
        The acquisition frequency of the data in Hz.

    Returns
    -------
    tuple
        A tuple containing a list with the maximum peak value and a list with the maximum peak index.
    """
    # Find the peaks in a window of a single trace
    delete_start_index = int(delete_start_time * acquisition_frequency / 1000)
    current_trace = voltage_trace

    window_size = window_size
    current_window = current_trace[delete_start_index: int ( delete_start_index + (window_size * acquisition_frequency / 1000) )]

    max_peak_value = []
    max_peak_idx = []
    peaks, _ = find_peaks(current_window, height=1)
    if peaks.size > 0:
        # Get the max peak index
        max_peak = max(current_window[peaks])
        max_peak_index = np.where(current_window == max_peak)[0][0]
        max_peak_value.append(max_peak)

        # Get the index of the max peak
        max_peak_idx.append(max_peak_index + int(delete_start_index * acquisition_frequency / 1000))

    return max_peak_value, max_peak_idx

# Main class
class GUI(object):
    """Main class for the GUI. Instances of this GUI receive default metadata and contain methods for processing data and plotting.
    
    Attributes
    ----------
    sweep_properties : list
        List of sweep properties.
    cell_meta_data_fields : list
        List of cell metadata fields.
    cell_properties : list
        List of cell properties.
    analysis_types : dict
        Dictionary of analysis types.
    defaults : dict
        Dictionary of default values.
    data_df : pd.DataFrame
        DataFrame to store and manipulate data.
    analysis_steps_list : list
        List of analysis steps.
    processed_data_label : dict
        Dictionary to store processed data labels.
    plot_widgets : dict
        Dictionary to store plot widgets.
    stim_start_dict : dict
        Dictionary to store stimulation start times.
    EPSP_list : list
        List of EPSP experiment types.
    plateau_list : list
        List of plateau experiment types.
    ISI_time : list
        List of ISI times.
    offset_traces : dict
        Dictionary to store offset traces.
    temp_data : None
        Temporary data variable.
    current_index : int
        Current index of the data frame.
    params_root : tk.Toplevel
        Tkinter root for parameters.
    plot_data_root : tk.Toplevel
        Tkinter root for data plots.
    plot_metadata_root : tk.Toplevel
        Tkinter root for plot metadata.
    plot_r_inp_root : tk.Toplevel
        Tkinter root for input resistance plots.
    load_navigate_frame : ttk.Frame
        Frame for load and navigation widgets.
    export_frame : ttk.Frame
        Frame for export widgets.
    analysis_steps_frame : ttk.Frame
        Frame for analysis steps widgets.
    experiment_metadata_frame : ttk.Frame
        Frame for experiment metadata widgets.
    plot_data_frame : ttk.Frame
        Frame for data plot.
    plot_metadata_frame : ttk.Frame
        Frame for plot metadata.
    plot_r_inp_frame : ttk.Frame
        Frame for input resistance plot.
    clear_plot : tk.BooleanVar
        Boolean variable for clearing the plot.
    detect_spikes_check : tk.BooleanVar
        Boolean variable for detecting spikes.
    AP_AHP_analysis_check : tk.BooleanVar
        Boolean variable for AP/AHP analysis.
    plotting_checks : dict
        Dictionary of boolean variables for plotting.
    label_checks : dict
        Dictionary of boolean variables for labeling.
    plot_check_buttons : dict
        Dictionary of plot check buttons.
    label_check_buttons : dict
        Dictionary of label check buttons.
    stim_removal_check : tk.BooleanVar
        Boolean variable for stimulation removal.
    stim_removal_automated_check : tk.BooleanVar
        Boolean variable for automated stimulation removal.
    noise_removal_check : tk.BooleanVar
        Boolean variable for noise removal.
    interpolated_spikes_check : tk.BooleanVar
        Boolean variable for interpolated spikes.
    interpolated_spikes_check_old_method : tk.BooleanVar
        Boolean variable for old interpolated spikes method.
    offset_traces_check : tk.BooleanVar
        Boolean variable for offset traces.
    partition_traces_check : tk.BooleanVar
        Boolean variable for partitioning traces.
    analyze_EPSP_peaks_check : tk.BooleanVar
        Boolean variable for EPSP peak analysis.
    plateau_area_check : tk.BooleanVar
        Boolean variable for plateau area analysis.
    stim_removal_pressed : bool
        Flag for stimulation removal.
    noise_removal_pressed : bool
        Flag for noise removal.
    interpolated_spikes_pressed : bool
        Flag for interpolated spikes.
    interpolated_spikes_old_method_pressed : bool
        Flag for old interpolated spikes method.
    offset_traces_pressed : bool
        Flag for offset traces.
    partition_traces_pressed : bool
        Flag for partitioning traces.
    plot_labels_legend_list : list
        List of plot label entries.
    raw_label_toggle : tk.BooleanVar
        Boolean variable for raw data label.
    processed_label_toggle : tk.BooleanVar
        Boolean variable for processed data label.
    raw_data_plot_toggle : tk.BooleanVar
        Boolean variable for plotting raw data.
    average_plot_toggle : tk.BooleanVar
        Boolean variable for plotting average.
    offset_plot_toggle : tk.BooleanVar
        Boolean variable for plotting offset.
    offset_average_plot_toggle : tk.BooleanVar
        Boolean variable for plotting offset average.
    average_traces_pressed : tk.BooleanVar
        Boolean variable for average traces.
    load_navigate_label : ttk.Label
        Label for load and navigate frame.
    select_index_label : ttk.Label
        Label for index selection.
    index_entry : ttk.Entry
        Entry for index.
    uncheck_processing_checks_label : ttk.Label
        Label for unchecking processing checks.
    start_index : int
        Starting index.
    current_experiment_type : str
        Current experiment type.
    experiment_list : list
        List of experiment types.
    experiment_type_var : tk.StringVar
        String variable for experiment type.
    experiment_type_dropdown : ttk.Combobox
        Dropdown for experiment type.
    stim_label : tk.Label
        Label for stimulus information.
    plot_button : ttk.Button
        Button for plotting.
    clear_processing_steps_button : ttk.Button
        Button for clearing processing steps.
    uncheck_processing_checks_buttons : ttk.Button
        Button for unchecking processing checks.
    save_data_button : ttk.Button
        Button for saving data.
    open_new_data_button : ttk.Button
        Button for opening new data.
    open_saved_data_button : ttk.Button
        Button for opening saved data.
    experiment_preprocessing_label : ttk.Label
        Label for experiment preprocessing.
    analysis_steps_label : ttk.Label
        Label for analysis steps.
    analysis_steps_entry : ttk.Entry
        Entry for analysis steps.
    current_trace_key_label : ttk.Label
        Label for current trace key.
    current_trace_key_dropdown : ttk.Combobox
        Dropdown for current trace key.
    analyze_AP_AHP_properties_check_button : ttk.Checkbutton
        Check button for AP/AHP analysis.
    calculate_intrinsic_properties_button : ttk.Button
        Button for calculating intrinsic properties.
    preprocessing_label : ttk.Label
        Label for preprocessing steps.
    stim_removal_check_button : ttk.Checkbutton
        Check button for stimulation removal.
    stim_removal_automated_check_button : ttk.Checkbutton
        Check button for automated stimulation removal.
    noise_removal_check_button : ttk.Checkbutton
        Check button for noise removal.
    interpolated_spikes_check_button : ttk.Checkbutton
        Check button for interpolated spikes.
    interpolated_spikes_check_old_method_button : ttk.Checkbutton
        Check button for old interpolated spikes method.
    post_processing_label : ttk.Label
        Label for post-processing steps.
    partition_traces_check_button : ttk.Checkbutton
        Check button for partitioning traces.
    offset_data_check_button : ttk.Checkbutton
        Check button for offset data.
    analyze_EPSP_peaks_button : ttk.Checkbutton
        Check button for EPSP peak analysis.
    analyze_plateau_button : ttk.Checkbutton
        Check button for plateau area analysis.
    AP_properties_label : ttk.Label
        Label for AP properties.
    Vm_rest_start_label : ttk.Label
        Label for Vm rest start.
    Vm_rest_start_entry : ttk.Entry
        Entry for Vm rest start.
    Vm_rest_end_label : ttk.Label
        Label for Vm rest end.
    Vm_rest_end_entry : ttk.Entry
        Entry for Vm rest end.
    AP_peak_window_start_label : ttk.Label
        Label for AP peak window start.
    AP_peak_window_start_entry : ttk.Entry
        Entry for AP peak window start.
    AP_peak_window_end_label : ttk.Label
        Label for AP peak window end.
    AP_peak_window_end_entry : ttk.Entry
        Entry for AP peak window end.
    stim_removal_label : ttk.Label
        Label for stimulation removal.
    delete_start_stim_label : ttk.Label
        Label for delete start stim.
    delete_start_stim_entry : ttk.Entry
        Entry for delete start stim.
    delete_end_stim_label : ttk.Label
        Label for delete end stim.
    delete_end_stim_entry : ttk.Entry
        Entry for delete end stim.
    noise_removal_label : ttk.Label
        Label for noise removal.
    delete_noise_duration_label : ttk.Label
        Label for delete noise duration.
    delete_noise_duration_entry : ttk.Entry
        Entry for delete noise duration.
    noise_times_label : ttk.Label
        Label for noise times.
    noise_times_entry : ttk.Entry
        Entry for noise times.
    interpolated_spikes_label : ttk.Label
        Label for interpolated spikes.
    spike_width_label : ttk.Label
        Label for spike width.
    spike_width_entry : ttk.Entry
        Entry for spike width.
    AP_peak_window_start_label_interp : ttk.Label
        Label for interpolated AP peak window start.
    AP_peak_window_start_entry_interp : ttk.Entry
        Entry for interpolated AP peak window start.
    AP_peak_window_end_label_interp : ttk.Label
        Label for interpolated AP peak window end.
    AP_peak_window_end_entry_interp : ttk.Entry
        Entry for interpolated AP peak window end.
    interpolate_spike_height_label : ttk.Label
        Label for interpolated spike height.
    interpolate_spike_height_entry : ttk.Entry
        Entry for interpolated spike height.
    experiment_metadata_label : ttk.Label
        Label for experiment metadata.
    test_pulse_label : ttk.Label
        Label for test pulse.
    test_pulse_start_label : ttk.Label
        Label for test pulse start.
    test_pulse_start_entry : ttk.Entry
        Entry for test pulse start.
    test_pulse_end_label : ttk.Label
        Label for test pulse end.
    test_pulse_end_entry : ttk.Entry
        Entry for test pulse end.
    test_pulse_amp_label : ttk.Label
        Label for test pulse amplitude.
    test_pulse_amp_entry : ttk.Entry
        Entry for test pulse amplitude.
    experiment_metadata_fill_label : ttk.Label
        Label for filling experiment metadata.
    fill_metadata_button : ttk.Button
        Button for filling metadata.
    update_metadata_button : ttk.Button
        Button for updating metadata.
    pathway_label : ttk.Label
        Label for pathway.
    pathway_entry_1 : ttk.Entry
        Entry for pathway 1.
    pathway_entry_2 : ttk.Entry
        Entry for pathway 2.
    pathway_entry_3 : ttk.Entry
        Entry for pathway 3.
    pathway_entry_4 : ttk.Entry
        Entry for pathway 4.
    amplitude_label : ttk.Label
        Label for amplitude.
    amplitude_entry_1 : ttk.Entry
        Entry for amplitude 1.
    amplitude_entry_2 : ttk.Entry
        Entry for amplitude 2.
    amplitude_entry_3 : ttk.Entry
        Entry for amplitude 3.
    amplitude_entry_4 : ttk.Entry
        Entry for amplitude 4.
    channel_label : ttk.Label
        Label for channel.
    channel_entry_1 : ttk.Entry
        Entry for channel 1.
    channel_entry_2 : ttk.Entry
        Entry for channel 2.
    channel_entry_3 : ttk.Entry
        Entry for channel 3.
    channel_entry_4 : ttk.Entry
        Entry for channel 4.
    ISI_label : ttk.Label
        Label for ISI.
    ISI_entry : ttk.Entry
        Entry for ISI.
    experiment_type_label : ttk.Label
        Label for experiment type.
    experiment_type_entry : ttk.Entry
        Entry for experiment type.
    experiment_description_label : ttk.Label
        Label for experiment description.
    experiment_description_entry : ttk.Entry
        Entry for experiment description.
    condition_label : ttk.Label
        Label for condition.
    condition_entry : ttk.Entry
        Entry for condition.
    canvas : FigureCanvasTkAgg
        Canvas for plotting data.
    ax : matplotlib.axes.Axes
        Axes for plotting data.
    toolbar : NavigationToolbar2Tk
        Toolbar for data plot.
    nav_button_frame : ttk.Frame
        Frame for navigation buttons.
    previous_button : ttk.Button
        Button for previous sweep.
    next_button : ttk.Button
        Button for next sweep.
    clear_plot_check_button : ttk.Checkbutton
        Check button for clearing the plot.
    clear_all_plots_check_button : ttk.Button
        Button for clearing all plots.
    plot_title_entry_label : ttk.Label
        Label for plot title entry.
    plot_title_entry : ttk.Entry
        Entry for plot title.
    raw_data_frame : ttk.Frame
        Frame for raw data widgets.
    raw_data_plot_check : ttk.Checkbutton
        Check button for raw data plot.
    raw_data_label_entry : ttk.Entry
        Entry for raw data label.
    raw_label_check : ttk.Checkbutton
        Check button for raw data label.
    dynamic_elements_frame : ttk.Frame
        Frame for dynamically added elements.
    canvas_r_inp : FigureCanvasTkAgg
        Canvas for plotting input resistance.
    ax_r_inp : matplotlib.axes.Axes
        Axes for plotting input resistance.
    toolbar_r_inp : NavigationToolbar2Tk
        Toolbar for input resistance plot.
    label_input_resistance : ttk.Label
        Label for input resistance calculation.
    calculate_input_resistance_button : ttk.Button
        Button for plotting input resistance.
    time : np.array
        Time array for plotting.
    channel_names : dict
        Dictionary of channel names.
    partition_data_dict : dict
        Dictionary for partitioned data.
    partition_time : np.array
        Time array for partitioned data.
    offset_data_dict : dict
        Dictionary for offset data.
    average_data : np.array
        Average of multiple traces.
    average_offset_traces : dict
        Average of multiple offset traces.
    AP_peaks_idx : np.array
        Indices of detected AP peaks.
    AP_properties : dict
        Dictionary of AP properties.
    current_trace : np.array
        The current trace being analyzed.
    
    Methods
    -------
    __init__(self, sweep_properties, cell_meta_data_fields, cell_properties, analysis_types_legend, defaults)
        Initializes the GUI.
    widgets_load_navigate_frame(self)
        Initializes the load and navigate frame widgets.
    widgets_export_frame(self)
        Initializes the export frame widgets.
    widgets_analysis_steps_frame(self)
        Initializes the analysis steps frame widgets.
    widgets_experiment_metadata_frame(self)
        Initializes the experiment metadata frame widgets.
    widgets_plot_data_frame(self)
        Initializes the data plot frame widgets.
    widgets_plot_metadata_frame(self)
        Initializes the plot metadata frame widgets.
    widgets_plot_r_inp_frame(self)
        Initializes the input resistance plot frame widgets.
    update_index_and_plot(self)
        Updates the index and plots the data.
    update_metadata(self)
        Updates the metadata for the current sweep.
    fill_metadata(self)
        Fills the metadata entries with default values.
    update_description_and_type(self)
        Updates the experiment description and type.
    change_text(self)
        Changes the text of labels in the GUI.
    update_entry(self, entry, index, key, suffix='')
        Helper method to update a single entry widget.
    update_pathway_entries(self, index)
        Helper method to update pathway entries.
    update_amplitude_entries(self, index)
        Helper method to update amplitude entries.
    open_file(self)
        Opens a pickle file.
    load_image(self, image_path)
        Loads an image from a file path.
    get_plot_information(self, sweep)
        Gets plotting information for a given sweep.
    plot_partitioned_and_offset_traces(self, traces, time, acquisition_frequency, label_toggle, additional_label)
        Plots partitioned and offset traces.
    update_plot(self)
        Updates the plot based on selected options.
    average_multiple_plots(self)
        Averages multiple plots.
    apply_partition_data(self)
        Partitions the data based on experiment type.
    toggle_partition_data(self)
        Toggles partitioning data.
    apply_offset_data(self)
        Offsets the data by subtracting the mean of a baseline window.
    toggle_offset_data(self)
        Toggles offsetting data.
    clear_all_plots(self)
        Clears all plots on the canvas.
    next(self)
        Moves to the next sweep and updates the plot.
    previous(self)
        Moves to the previous sweep and updates the plot.
    apply_analysis_steps(self)
        Applies selected analysis steps.
    get_input_resistances(self)
        Gets input resistance values for all sweeps.
    plot_input_resistance(self)
        Plots input resistance across experiments.
    apply_AP_AHP_analysis(self)
        Applies AP and AHP analysis.
    toggle_AP_AHP_analysis(self)
        Toggles AP and AHP analysis.
    calculate_intrinsic_properties(self)
        Calculates intrinsic cell properties.
    apply_analyze_EPSP_peaks(self)
        Applies EPSP peak analysis.
    toggle_analyze_EPSP_peaks(self)
        Toggles EPSP peak analysis.
    apply_plateau_area_under_curve(self)
        Applies plateau area under curve analysis.
    toggle_plateau_area_under_curve(self)
        Toggles plateau area under curve analysis.
    get_stim_times(self)
        Gets stimulation times.
    apply_stim_removal(self)
        Applies stimulation removal.
    toggle_stim_removal(self)
        Toggles stimulation removal.
    apply_noise_removal(self)
        Applies noise removal.
    toggle_noise_removal(self)
        Toggles noise removal.
    apply_interpolated_spikes(self)
        Applies spike interpolation.
    toggle_interpolated_spikes(self)
        Toggles spike interpolation.
    uncheck_processing_steps(self)
        Unchecks all processing step checkboxes.
    clear_processing_steps(self)
        Clears all intermediate traces and analysis steps.
    save_data(self)
        Saves the data to a pickle file.
    open_new_data(self)
        Opens a new data file.
    open_saved_data(self)
        Opens a saved data file.
    """
    def __init__(self, sweep_properties, cell_meta_data_fields, cell_properties, 
                 analysis_types_legend, defaults):
      
        self.sweep_properties = sweep_properties
        self.cell_meta_data_fields = cell_meta_data_fields
        self.cell_properties = cell_properties
        self.analysis_types = {}
        self.defaults = defaults
        self.data_df = self.open_file()
        self.analysis_steps_list = []
        self.processed_data_label = {}
        self.plot_widgets = {}
        self.stim_start_dict = {}

        #if the data_df does not have the 'analysis_dict' column, then add it
        if 'analysis_dict' not in self.data_df.columns:
            self.data_df['analysis_dict'] = [{} for _ in range(len(self.data_df))]
            self.data_df['intermediate_traces'] = [{} for _ in range(len(self.data_df))]
            self.data_df['current_trace_key'] = ['raw' for _ in range(len(self.data_df))]
        
        #create experiment_type lists for conditionals
        self.EPSP_list = []
        self.plateau_list = []
        #for EPSP_list have it be instances where num_pulses is equal to 2 or 3 but less than 5
        #for plateau_list have it be instances where num_pulses 5 or greater
        #first check if the key in self.defaults['analysis'] has 'channel' in it
        #if it does, then check if the value of 'num_pulses' is 2 or 3
        #if it is, then add the key to EPSP_list
        #if the value of 'num_pulses' is 5 or greater, then add the key to plateau_list
        for key in self.defaults['analysis'].keys():
            #if the string channel is contained in any of the nested keys under key
            if any('channel' in nested_key for nested_key in self.defaults['analysis'][key].keys()):
                for channel in self.defaults['analysis'][key].keys():
                    if self.defaults['analysis'][key][channel]['num_pulses'] == 2 or self.defaults['analysis'][key][channel]['num_pulses'] == 3:
                        self.EPSP_list.append(key)
                    elif self.defaults['analysis'][key][channel]['num_pulses'] >= 5:
                        self.plateau_list.append(key)
        
        #make sure that the EPSP_list and plateau_list are unique
        self.EPSP_list = list(set(self.EPSP_list))
        self.plateau_list = list(set(self.plateau_list))

        #store temporary data values 
        self.ISI_time = []
        self.offset_traces = {}
        # self.average_data = {}
        # self.average_offset_traces = {}
        self.temp_data = None
        self.current_index = 0

        #Root for dealing with singles traces analysis, plotting, and data manipulation
        self.params_root = tk.Toplevel()
        self.params_root.title("Electrophysiology Visualization Tool Parameters")

        #Plotting roots
        self.plot_data_root = tk.Toplevel()
        self.plot_data_root.title("Plot Data")

        self.plot_metadata_root = tk.Toplevel()
        self.plot_metadata_root.title("Plot Metadata")

        self.plot_r_inp_root = tk.Toplevel()
        self.plot_r_inp_root.title("Plot Input Resistance")

        #set up the structure of the GUI for single trace analysis
        self.load_navigate_frame = ttk.Frame(self.params_root, borderwidth=1, relief=tk.GROOVE)
        self.export_frame = ttk.Frame(self.params_root, borderwidth=0, relief=tk.GROOVE)
        self.analysis_steps_frame = ttk.Frame(self.params_root, borderwidth=1, relief=tk.GROOVE)
        self.experiment_metadata_frame = ttk.Frame(self.params_root, borderwidth=1, relief=tk.GROOVE)

        self.load_navigate_frame.grid(row=0, column=0, sticky='nsew')
        self.export_frame.grid(row=0, column=1, sticky='nsew')
        self.analysis_steps_frame.grid(row=1, column=0, sticky='nsew')
        self.experiment_metadata_frame.grid(row=1, column=1, sticky='nsew')

        #plotting on seperate frames
        self.plot_data_frame = ttk.Frame(self.plot_data_root, borderwidth=0, relief=tk.GROOVE)
        self.plot_metadata_frame = ttk.Frame(self.plot_metadata_root, borderwidth=0, relief=tk.GROOVE)
        self.plot_r_inp_frame = ttk.Frame(self.plot_r_inp_root, borderwidth=0, relief=tk.GROOVE)

        self.plot_data_frame.grid(row=0, column=0, sticky='nsew')
        self.plot_metadata_frame.grid(row=0, column=0, sticky='nsew')
        self.plot_r_inp_frame.grid(row=0, column=0, sticky='nsew')

        #check box boolean states
        self.clear_plot = tk.BooleanVar(value=True)
        #FI and cell properties
        self.detect_spikes_check = tk.BooleanVar(value=False)
        self.AP_AHP_analysis_check = tk.BooleanVar(value=False)

        #dictionary to store boolean values based on trace key - this will be dynamically appended to 
        self.plotting_checks = {}
        self.label_checks = {}

        #dictionary to store the check buttons for plotting and labeling
        self.plot_check_buttons = {}
        self.label_check_buttons = {}

        self.stim_removal_check = tk.BooleanVar(value=False)
        self.stim_removal_automated_check = tk.BooleanVar(value=False)
        self.noise_removal_check = tk.BooleanVar(value=False)
        self.interpolated_spikes_check = tk.BooleanVar(value=False)
        self.interpolated_spikes_check_old_method = tk.BooleanVar(value=False)
        self.offset_traces_check = tk.BooleanVar(value=False)
        self.partition_traces_check = tk.BooleanVar(value=False)
        self.analyze_EPSP_peaks_check = tk.BooleanVar(value=False)
        self.plateau_area_check = tk.BooleanVar(value=False) 

        #preprocessing for EPSP and plateau
        self.stim_removal_pressed = False
        self.noise_removal_pressed = False
        self.interpolated_spikes_pressed = False
        self.interpolated_spikes_old_method_pressed = False
        #postprocessing for EPSP and plateau
        self.offset_traces_pressed = False
        self.partition_traces_pressed = False

        #check buttons for plotting labels and plots
        self.plot_labels_legend_list = []
        self.raw_label_toggle = tk.BooleanVar(value=False)
        self.processed_label_toggle = tk.BooleanVar(value=False)

        #toggle to plot the raw data
        self.raw_data_plot_toggle = tk.BooleanVar(value=True)
        #check for plotting the average and the offset
        self.average_plot_toggle = tk.BooleanVar(value=False)
        self.offset_plot_toggle = tk.BooleanVar(value=False)
        self.offset_average_plot_toggle = tk.BooleanVar(value=False)

        #toggle for the offset and average traces
        # self.plot_offset_traces_pressed = False
        self.average_traces_pressed = tk.BooleanVar(value=False)

        #load the frames for the GUI for single trace analysis
        self.widgets_load_navigate_frame()
        self.widgets_export_frame()
        self.widgets_analysis_steps_frame()
        self.widgets_experiment_metadata_frame()
        self.widgets_plot_data_frame()
        self.widgets_plot_metadata_frame()
        self.widgets_plot_r_inp_frame()

        # #load the frames for the GUI for multiple trace analysis
        # self.widgets_multiple_load_navigate_frame()

        #run the mainloop to keep the GUI running
        self.params_root.mainloop()
        # self.multiple_params_root.mainloop()
        self.plot_data_root.mainloop()
        self.plot_metadata_root.mainloop()
        self.plot_r_inp_root.mainloop()
    
    def widgets_load_navigate_frame(self):
        #label for load and navigate frame
        self.load_navigate_label = ttk.Label(self.load_navigate_frame, text="Load and Navigate Data", font=('Arial', 16, 'bold'))
        self.load_navigate_label.grid(row=0, column=0, sticky='nsew')
        #label for select index
        self.select_index_label = ttk.Label(self.load_navigate_frame, text="Select Index")
        self.select_index_label.grid(row=1, column=0, sticky='nsew')

        #add user input box for index 
        self.index_entry = ttk.Entry(self.load_navigate_frame, width=10)
        #set the default value to 0 if zero exists otherwise set to 1
        self.index_entry.insert(0, '0') 
        self.index_entry.grid(row=2, column=0, sticky='nsew')

        self.uncheck_processing_checks_label = ttk.Label(self.load_navigate_frame, text="When moving to another experiment, uncheck the processing checks")
        self.uncheck_processing_checks_label.grid(row=2, column=2, sticky='nsew')

        #create the first index for plotting
        # if self.data_df.experiment_index.iloc[0] == 0:
        #     self.start_index = 0
        # elif self.data_df.experiment_index.iloc[1] == 1:
        #     self.start_index = 1
        self.start_index = 0

        # Dropdown menu to select experiment type
        self.current_experiment_type = self.data_df['stim_type'].iloc[int(self.index_entry.get())]
        self.experiment_list = list(self.defaults['analysis'].keys())
        self.experiment_type_var = tk.StringVar(self.load_navigate_frame)
        self.experiment_type_dropdown = ttk.Combobox(self.load_navigate_frame, textvariable=self.experiment_type_var, values=self.experiment_list)
        #populate the first value of the dropdown menu with the experiment_list[1]
        self.experiment_type_var.set(self.experiment_list[0])
        self.experiment_type_dropdown.grid(row=4, column=0, columnspan=2, sticky='nsew')

        self.stim_label = tk.Label(self.plot_data_frame, text="")
        self.stim_label.pack()

        self.plot_button = ttk.Button(self.load_navigate_frame, text='Plot Selected Index', command=self.update_index_and_plot)
        self.plot_button.grid(row=3, column=0, sticky='nsew')

        self.clear_processing_steps_button = ttk.Button(self.load_navigate_frame, text='Clear Processing Steps (when moving to another experiment)', command=self.clear_processing_steps)
        self.clear_processing_steps_button.grid(row=3, column=1, sticky='nsew')

        self.uncheck_processing_checks_buttons = ttk.Button(self.load_navigate_frame, text='Uncheck Processing Checks', command=self.uncheck_processing_steps) 
        self.uncheck_processing_checks_buttons.grid(row=3, column=2, sticky='nsew')

    def widgets_export_frame(self):
        self.save_data_button = ttk.Button(self.export_frame, text='Save Data', command=self.save_data)
        self.save_data_button.pack(side=tk.TOP, pady=1)

        self.open_new_data_button = ttk.Button(self.export_frame, text='Open New Data', command=self.open_new_data)
        self.open_new_data_button.pack(side=tk.TOP, pady=1)

        self.open_saved_data_button = ttk.Button(self.export_frame, text='Open Saved Data', command=self.open_saved_data)
        self.open_saved_data_button.pack(side=tk.TOP, pady=1)

    def widgets_analysis_steps_frame(self):
        #add label for experiment preprocessing
        self.experiment_preprocessing_label = ttk.Label(self.analysis_steps_frame, text="Experiment Preprocessing", font=('Arial', 16, 'bold'))
        self.experiment_preprocessing_label.grid(row=0, column=0, sticky='nsew')

        #analysis_steps entry to be filled in with current_trace keys
        self.analysis_steps_label = ttk.Label(self.analysis_steps_frame, text="Analysis Steps", font=('Arial', 14, 'bold'))
        self.analysis_steps_label.grid(row=1, column=0, sticky='nsew')
        self.analysis_steps_entry = ttk.Entry(self.analysis_steps_frame, width=40)

        #add raw to the list
        self.analysis_steps_list.append('raw')
        self.analysis_steps_entry.insert(0, 'raw,')
        self.analysis_steps_entry.grid(row=1, column=1, sticky='nsew')

        #current_trace_key label dropdown 
        self.current_trace_key_label = ttk.Label(self.analysis_steps_frame, text="Current Trace Key", font=('Arial', 14, 'bold'))
        self.current_trace_key_label.grid(row=2, column=0, sticky='nsew')

        #current_trace_key dropdown menu
        self.current_trace_key_var = tk.StringVar(self.analysis_steps_frame)
        self.current_trace_key_dropdown = ttk.Combobox(self.analysis_steps_frame, textvariable=self.current_trace_key_var, values=['raw'], width=30)
        # insert the first value of the dropdown menu with the current_trace_key_var
        self.current_trace_key_dropdown.insert(0, 'raw')
        self.current_trace_key_dropdown.grid(row=2, column=1, sticky='nsew')

        '''Processing Steps for FI analysis and Cell Properties'''

        #detect spikes check button to plot spikes
        # self.detect_spikes_check_button = ttk.Checkbutton(self.analysis_steps_frame, text="Detect Spikes", variable=self.detect_spikes_check, command=self.toggle_detect_spikes)
        # self.detect_spikes_check_button.grid(row=3, column=0, sticky='nsew')

        #check button to analyze AP and AHP properties
        self.analyze_AP_AHP_properties_check_button = ttk.Checkbutton(self.analysis_steps_frame, text="Analyze AP Properties (make sure to choose the right experiment type)", variable=self.AP_AHP_analysis_check, command=self.toggle_AP_AHP_analysis)
        self.analyze_AP_AHP_properties_check_button.grid(row=4, column=0, sticky='nsew')

        #button to calculate input resistance and voltage sag
        self.calculate_intrinsic_properties_button = ttk.Button(self.analysis_steps_frame, text="Calculate Input Resistance and Voltage Sag", command=self.calculate_intrinsic_properties)
        self.calculate_intrinsic_properties_button.grid(row=5, column=0, sticky='nsew')

        '''Pre-procesing Steps for EPSP and Plateau Analysis'''

        #preprocessing label
        self.preprocessing_label = ttk.Label(self.analysis_steps_frame, text="Preprocessing Steps", font=('Arial', 14, 'bold'))
        self.preprocessing_label.grid(row=6, column=0, sticky='nsew')

        #add stim removal button check
        self.stim_removal_check_button = ttk.Checkbutton(self.analysis_steps_frame, text="Stim Removal - Please Select Experiment Type First and Fill Metadata",
                                                            variable=self.stim_removal_check, command=self.toggle_stim_removal)
        
        self.stim_removal_check_button.grid(row=7, column=0, sticky='nsew')

        #if it is pressed then add the stim_removal_pressed to True
        if self.stim_removal_check.get() == True:
            self.stim_removal_pressed = True
        
        self.stim_removal_automated_check_button = ttk.Checkbutton(self.analysis_steps_frame, text="Automated Artifact Removal", variable=self.stim_removal_automated_check, command=self.toggle_stim_removal)
        self.stim_removal_automated_check_button.grid(row=8, column=0, sticky='nsew')

        if self.stim_removal_automated_check.get() == True:
            self.stim_removal_pressed = True

        #add noise removal check button
        self.noise_removal_check_button = ttk.Checkbutton(self.analysis_steps_frame, text="Noise Removal", variable=self.noise_removal_check, command=self.toggle_noise_removal)
        self.noise_removal_check_button.grid(row=9, column=0, sticky='nsew')

        if self.noise_removal_check.get() == True:
            self.noise_removal_pressed = True

        #add interpolated spikes check button
        self.interpolated_spikes_check_button = ttk.Checkbutton(self.analysis_steps_frame, text="Interpolated Spikes", variable=self.interpolated_spikes_check, command=self.toggle_interpolated_spikes)
        self.interpolated_spikes_check_button.grid(row=10, column=0, sticky='nsew')

        #add interpolated spikes old method button
        self.interpolated_spikes_check_old_method_button = ttk.Checkbutton(self.analysis_steps_frame, text="Interpolated Spikes Old Method", variable=self.interpolated_spikes_check_old_method, command=self.toggle_interpolated_spikes)
        self.interpolated_spikes_check_old_method_button.grid(row=10, column=1, sticky='nsew')


        if self.interpolated_spikes_check.get() == True:
            #self.noise_removal_pressed = True
            self.interpolated_spikes_pressed = True
        
        if self.interpolated_spikes_check_old_method.get() == True:
            self.interpolated_spikes_old_method_pressed = True

        '''Post-processing Steps for EPSP and Plateau Analysis'''

        #post processing label
        self.post_processing_label = ttk.Label(self.analysis_steps_frame, text="Post Processing Steps", font=('Arial', 14, 'bold'))
        self.post_processing_label.grid(row=11, column=0, sticky='nsew')

        #partition_trace check button
        self.partition_traces_check_button = ttk.Checkbutton(self.analysis_steps_frame, text="Partition Trace", variable=self.partition_traces_check, command=self.toggle_partition_data)
        self.partition_traces_check_button.grid(row=12, column=0, sticky='nsew')

        if self.partition_traces_check.get() == True:
            self.partition_traces_pressed = True

        #add offset data check button
        self.offset_data_check_button = ttk.Checkbutton(self.analysis_steps_frame, text="Offset Data", variable=self.offset_traces_check, command=self.toggle_offset_data)
        self.offset_data_check_button.grid(row=13, column=0, sticky='nsew')

        if self.offset_traces_check.get() == True:
            self.offset_traces_pressed = True

        '''Labels and entries for pre and post processing steps'''

        #button to analyze EPSP peaks
        self.analyze_EPSP_peaks_button = ttk.Checkbutton(self.analysis_steps_frame, text="Analyze EPSP Peaks", variable=self.analyze_EPSP_peaks_check, command=self.toggle_analyze_EPSP_peaks)
        self.analyze_EPSP_peaks_button.grid(row=14, column=0, sticky='nsew')
        self.analyze_plateau_button = ttk.Checkbutton(self.analysis_steps_frame, text="Analyze Plateau", variable=self.plateau_area_check, command=self.toggle_plateau_area_under_curve) 
        self.analyze_plateau_button.grid(row=15, column=0, sticky='nsew')

        #analyze AP properties title label
        self.AP_properties_label = ttk.Label(self.analysis_steps_frame, text="Analyze AP Properties", font=('Arial', 14, 'bold'))
        self.AP_properties_label.grid(row=16, column=0, sticky='nsew')

        self.Vm_rest_start_label = ttk.Label(self.analysis_steps_frame, text="Vm Rest Start (ms)")
        self.Vm_rest_start_entry = ttk.Entry(self.analysis_steps_frame, width=10)
        self.Vm_rest_start_entry.insert(200, '200')
        #entry for Vm_rest_end
        self.Vm_rest_end_label = ttk.Label(self.analysis_steps_frame, text="Vm Rest End (ms)")
        self.Vm_rest_end_entry = ttk.Entry(self.analysis_steps_frame, width=10)
        self.Vm_rest_end_entry.insert(300, '300')
        #entry for AP_peak_window_start
        self.AP_peak_window_start_label = ttk.Label(self.analysis_steps_frame, text="AP Peak Window Start (ms)")
        self.AP_peak_window_start_entry = ttk.Entry(self.analysis_steps_frame, width=10)    
        self.AP_peak_window_start_entry.insert(5, '5')

        #entry for AP_peak_window_end
        self.AP_peak_window_end_label = ttk.Label(self.analysis_steps_frame, text="AP Peak Window End (ms)")
        self.AP_peak_window_end_entry = ttk.Entry(self.analysis_steps_frame, width=10)
        self.AP_peak_window_end_entry.insert(20, '20')

        #all the boxes to the GUI
        self.Vm_rest_start_label.grid(row=17, column=1, sticky='nsew')
        self.Vm_rest_start_entry.grid(row=17, column=0, sticky='nsew')

        self.Vm_rest_end_label.grid(row=18, column=1, sticky='nsew')
        self.Vm_rest_end_entry.grid(row=18, column=0, sticky='nsew')

        self.AP_peak_window_start_label.grid(row=19, column=1, sticky='nsew')
        self.AP_peak_window_start_entry.grid(row=19, column=0, sticky='nsew')

        #TODO add one to each row
        self.AP_peak_window_end_label.grid(row=20, column=1, sticky='nsew')
        self.AP_peak_window_end_entry.grid(row=20, column=0, sticky='nsew')

        #stim removal label
        self.stim_removal_label = ttk.Label(self.analysis_steps_frame, text="Stim Removal", font=('Arial', 14, 'bold'))
        self.stim_removal_label.grid(row=21, column=0, sticky='nsew')

        #entry for delete_start_stim
        self.delete_start_stim_label = ttk.Label(self.analysis_steps_frame, text="Delete Stim Artifacts Start (ms)")
        self.delete_start_stim_entry = ttk.Entry(self.analysis_steps_frame, width=10)
        default_start_stim = self.defaults['filtering']['stim_removal']['start']
        self.delete_start_stim_entry.insert(0, f'{default_start_stim}')

        #entry for delete_end_stim
        self.delete_end_stim_label = ttk.Label(self.analysis_steps_frame, text="Delete Stim Artifacts End Duration (ms)")
        self.delete_end_stim_entry = ttk.Entry(self.analysis_steps_frame, width=10)
        default_end_stim = self.defaults['filtering']['stim_removal']['end']
        self.delete_end_stim_entry.insert(0, f'{default_end_stim}')

        #all the boxes to the GUI
        self.delete_start_stim_label.grid(row=22, column=1, sticky='nsew')
        self.delete_start_stim_entry.grid(row=22, column=0, sticky='nsew')
        self.delete_end_stim_label.grid(row=23, column=1, sticky='nsew')
        self.delete_end_stim_entry.grid(row=23, column=0, sticky='nsew')

        #noise removal label
        self.noise_removal_label = ttk.Label(self.analysis_steps_frame, text="Noise Removal", font=('Arial', 14, 'bold'))
        self.noise_removal_label.grid(row=24, column=0, sticky='nsew')

        #deletion time duration
        self.delete_noise_duration_label = ttk.Label(self.analysis_steps_frame, text="Delete Noise Duration (ms)")
        self.delete_noise_duration_entry = ttk.Entry(self.analysis_steps_frame, width=10)
        default_delete_noise_duration = self.defaults['filtering']['noise_removal']['delete_duration']
        self.delete_noise_duration_entry.insert(0, f'{default_delete_noise_duration}')

        self.delete_noise_duration_label.grid(row=25, column=1, sticky='nsew')
        self.delete_noise_duration_entry.grid(row=25, column=0, sticky='nsew')

        self.noise_times_label = ttk.Label(self.analysis_steps_frame, text="Noise Times (ms)")
        self.noise_times_entry = ttk.Entry(self.analysis_steps_frame, width=10)
        default_noise_times = self.defaults['filtering']['noise_removal']['noise_time']
        self.noise_times_entry.insert(0, f'{default_noise_times}')

        self.noise_times_label.grid(row=26, column=1, sticky='nsew')
        self.noise_times_entry.grid(row=26, column=0, sticky='nsew')

        #interpolated spikes label
        self.interpolated_spikes_label = ttk.Label(self.analysis_steps_frame, text="Interpolated Spikes", font=('Arial', 14, 'bold'))
        self.interpolated_spikes_label.grid(row=27, column=0, sticky='nsew')

        #entry for spike_width
        self.spike_width_label = ttk.Label(self.analysis_steps_frame, text="Spike Width (ms)")
        self.spike_width_entry = ttk.Entry(self.analysis_steps_frame, width=10)
        default_spike_width = self.defaults['filtering']['interpolated_spikes']['spike_width']
        self.spike_width_entry.insert(0, f'{default_spike_width}')

        #entry for AP_peak_window_start
        self.AP_peak_window_start_label_interp = ttk.Label(self.analysis_steps_frame, text="AP Peak Window Start (ms)")
        self.AP_peak_window_start_entry_interp = ttk.Entry(self.analysis_steps_frame)
        #use the default value for the AP_peak_window_start
        self.AP_peak_window_start_entry_interp.insert(5, '5')

        #entry for AP_peak_window_end
        self.AP_peak_window_end_label_interp = ttk.Label(self.analysis_steps_frame, text="AP Peak Window End (ms)")
        self.AP_peak_window_end_entry_interp = ttk.Entry(self.analysis_steps_frame, width=10)
        #use the default value for the AP_peak_window_end
        self.AP_peak_window_end_entry_interp.insert(10, '10')

        #all the boxes to the GUI
        self.spike_width_label.grid(row=28, column=1, sticky='nsew')
        self.spike_width_entry.grid(row=28, column=0, sticky='nsew')

        self.AP_peak_window_start_label_interp.grid(row=29, column=1, sticky='nsew')
        self.AP_peak_window_start_entry_interp.grid(row=29, column=0, sticky='nsew')

        self.AP_peak_window_end_label_interp.grid(row=30, column=1, sticky='nsew')
        self.AP_peak_window_end_entry_interp.grid(row=30, column=0, sticky='nsew')

        self.interpolate_spike_height_label = ttk.Label(self.analysis_steps_frame, text="Interpolated Spike Height (mV)")
        self.interpolate_spike_height_entry = ttk.Entry(self.analysis_steps_frame, width=10)

    def widgets_experiment_metadata_frame(self):
        #add a label for the experiment metadata
        self.experiment_metadata_label = ttk.Label(self.experiment_metadata_frame, text="Experiment Metadata", font=('Arial', 16, 'bold'))
        self.experiment_metadata_label.grid(row=0, column=0, sticky='nsew')

        #test pulse
        self.test_pulse_label = ttk.Label(self.experiment_metadata_frame, text="Test Pulse")
        self.test_pulse_label.grid(row=1, column=0, sticky='nsew')

        self.test_pulse_start_label = ttk.Label(self.experiment_metadata_frame, text="Start (ms)")
        self.test_pulse_start_label.grid(row=2, column=1, sticky='nsew')
        self.test_pulse_start_entry = ttk.Entry(self.experiment_metadata_frame, width=10)
        self.test_pulse_start_entry.grid(row=2, column=0, sticky='nsew')

        self.test_pulse_end_label = ttk.Label(self.experiment_metadata_frame, text="Duration (ms)")
        self.test_pulse_end_label.grid(row=3, column=1, sticky='nsew')
        self.test_pulse_end_entry = ttk.Entry(self.experiment_metadata_frame, width=10)
        self.test_pulse_end_entry.grid(row=3, column=0, sticky='nsew')

        self.test_pulse_amp_label = ttk.Label(self.experiment_metadata_frame, text="Amplitude (pA)")
        self.test_pulse_amp_label.grid(row=4, column=1, sticky='nsew')
        self.test_pulse_amp_entry = ttk.Entry(self.experiment_metadata_frame, width=10)
        self.test_pulse_amp_entry.grid(row=4, column=0, sticky='nsew')

        #insert values for the test pulse convert values to integers
        self.test_pulse_start_entry.insert(0, f'{int(self.defaults["analysis"]["test_pulse"]["step_start"])}')
        self.test_pulse_end_entry.insert(0, f'{int(self.defaults["analysis"]["test_pulse"]["step_duration"])}')
        self.test_pulse_amp_entry.insert(0, f'{int(self.defaults["analysis"]["test_pulse"]["current_amplitude"])}')

        #label for filling in the experiment metadata
        self.experiment_metadata_fill_label = ttk.Label(self.experiment_metadata_frame, text="Fill in the Experiment Metadata")
        self.experiment_metadata_fill_label.grid(row=5, column=0, sticky='nsew')

        #buttons to fill in the metadata
        self.fill_metadata_button = ttk.Button(self.experiment_metadata_frame, text="Fill Metadata", command=self.fill_metadata)
        self.fill_metadata_button.grid(row=5, column=1, sticky='nsew')

        self.update_metadata_button = ttk.Button(self.experiment_metadata_frame, text="Update Metadata if Data is Missing", command=self.update_metadata)
        self.update_metadata_button.grid(row=5, column=2, sticky='nsew')

        #add a labels for the experiment metadata
        #pathway label in column 1
        self.pathway_label = ttk.Label(self.experiment_metadata_frame, text="Pathway")
        self.pathway_label.grid(row=6, column=0, sticky='nsew')

        #pathway entry in column 0
        self.pathway_entry_1 = ttk.Entry(self.experiment_metadata_frame, width=10)
        self.pathway_entry_1.grid(row=7, column=0, sticky='nsew')
        self.pathway_entry_2 = ttk.Entry(self.experiment_metadata_frame, width= 10)
        self.pathway_entry_2.grid(row=8, column=0, sticky='nsew')
        self.pathway_entry_3 = ttk.Entry(self.experiment_metadata_frame, width=10)
        self.pathway_entry_3.grid(row=9, column=0, sticky='nsew')
        self.pathway_entry_4 = ttk.Entry(self.experiment_metadata_frame, width=10)
        self.pathway_entry_4.grid(row=10, column=0, sticky='nsew')

        #Amplitude entries in column 1
        self.amplitude_label = ttk.Label(self.experiment_metadata_frame, text="Amplitude")
        self.amplitude_label.grid(row=6, column=1, sticky='nsew')

        self.amplitude_entry_1 = ttk.Entry(self.experiment_metadata_frame, width=10)
        self.amplitude_entry_1.grid(row=7, column=1, sticky='nsew')
        self.amplitude_entry_2 = ttk.Entry(self.experiment_metadata_frame, width=10)
        self.amplitude_entry_2.grid(row=8, column=1, sticky='nsew')
        self.amplitude_entry_3 = ttk.Entry(self.experiment_metadata_frame, width=10)
        self.amplitude_entry_3.grid(row=9, column=1, sticky='nsew')
        self.amplitude_entry_4 = ttk.Entry(self.experiment_metadata_frame, width=10)
        self.amplitude_entry_4.grid(row=10, column=1, sticky='nsew')

        #channel entries
        self.channel_label = ttk.Label(self.experiment_metadata_frame, text="Channel")
        self.channel_label.grid(row=6, column=2, sticky='nsew')

        self.channel_entry_1 = ttk.Entry(self.experiment_metadata_frame, width=10)
        self.channel_entry_1.grid(row=7, column=2, sticky='nsew')
        self.channel_entry_2 = ttk.Entry(self.experiment_metadata_frame, width=10)
        self.channel_entry_2.grid(row=8, column=2, sticky='nsew')
        self.channel_entry_3 = ttk.Entry(self.experiment_metadata_frame, width=10)
        self.channel_entry_3.grid(row=9, column=2, sticky='nsew')
        self.channel_entry_4 = ttk.Entry(self.experiment_metadata_frame, width=10)
        self.channel_entry_4.grid(row=10, column=2, sticky='nsew')

        #ISI entry
        self.ISI_label = ttk.Label(self.experiment_metadata_frame, text="ISI (For E/I Experiments)")
        self.ISI_label.grid(row=11, column=0, sticky='nsew')
        self.ISI_entry = ttk.Entry(self.experiment_metadata_frame, width=10)
        self.ISI_entry.grid(row=11, column=1, sticky='nsew')

        #experiment type and description
        self.experiment_type_label = ttk.Label(self.experiment_metadata_frame, text="Experiment Type")
        self.experiment_type_label.grid(row=12, column=0, sticky='nsew')

        self.experiment_type_entry = ttk.Entry(self.experiment_metadata_frame, width=10)
        self.experiment_type_entry.grid(row=12, column=1, sticky='nsew')

        self.experiment_description_label = ttk.Label(self.experiment_metadata_frame, text="Experiment Description")
        self.experiment_description_label.grid(row=13, column=0, sticky='nsew')

        self.experiment_description_entry = ttk.Entry(self.experiment_metadata_frame, width=10)
        self.experiment_description_entry.grid(row=13, column=1, sticky='nsew')

        #Condition labels
        #make text Bolded
        self.condition_label = ttk.Label(self.experiment_metadata_frame, text="Condition") 
        self.condition_label.grid(row=14, column=0, sticky='nsew')
        #condition entry
        self.condition_entry = ttk.Entry(self.experiment_metadata_frame, width=10)
        self.condition_entry.grid(row=14, column=1, sticky='nsew')

        #metadata should be filled in with the default values
        self.fill_metadata()

    def widgets_plot_data_frame(self):
        '''Plot the data'''
        self.canvas = FigureCanvasTkAgg(figure.Figure(figsize=(10, 5)), master=self.plot_data_frame)
        self.ax = self.canvas.figure.add_subplot(111)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.plot_data_frame)
        self.toolbar.update()
        self.toolbar.pack(side=tk.TOP, fill=tk.X)

        self.ax.set_xlabel('Time (s)')
        self.ax.set_ylabel('Voltage (mV)')
    
    def widgets_plot_metadata_frame(self):
        # Navigation Buttons
        nav_button_frame = ttk.Frame(self.plot_metadata_frame) 
        nav_button_frame.grid(row=0, column=0, sticky='nsew')

        self.previous_button = ttk.Button(nav_button_frame, text="Previous", command=self.previous)
        self.previous_button.grid(row=0, column=0, sticky='nsew')

        self.next_button = ttk.Button(nav_button_frame, text="Next", command=self.next)
        self.next_button.grid(row=0, column=1, sticky='nsew')

        # Checkbox to clear the current plot
        self.clear_plot_check_button = ttk.Checkbutton(nav_button_frame, text="Clear Current Plot", variable=self.clear_plot)
        self.clear_plot_check_button.grid(row=0, column=2, sticky='nsew')

        # Checkbox to clear all plots
        self.clear_all_plots_check_button = ttk.Button(nav_button_frame, text="Clear All Plots", command=self.clear_all_plots)
        self.clear_all_plots_check_button.grid(row=0, column=3, sticky='nsew')

        # Create a frame to hold the title entry and label
        title_frame = ttk.Frame(self.plot_metadata_frame)
        title_frame.grid(row=1, column=0, sticky='nsew')

        self.plot_title_entry_label = ttk.Label(title_frame, text="Plot Title")
        self.plot_title_entry_label.grid(row=1, column=0, sticky='nsew')

        # Add an entry box for the title
        # default_title = 'Plotting Current Sweep'
        self.plot_title_entry = ttk.Entry(title_frame, width=20)
        # self.plot_title_entry.insert(0, default_title)
        self.plot_title_entry.grid(row=1, column=1, sticky='nsew')

        # Create a frame and entry for the raw data label on the plot
        raw_data_frame = ttk.Frame(self.plot_metadata_frame)
        raw_data_frame.grid(row=2, column=0, sticky='nsew')

        # Add the raw data plot check button
        self.raw_data_plot_check = ttk.Checkbutton(raw_data_frame, text='Plot Raw Data', variable=self.raw_data_plot_toggle, command=self.update_plot)
        self.raw_data_plot_check.grid(row=2, column=0, sticky='nsew')

        # Add the raw data label entry
        self.raw_data_label_entry = ttk.Entry(raw_data_frame, width=20)
        self.raw_data_label_entry.grid(row=2, column=1, sticky='nsew')

        # Insert the default value for the raw data label
        self.raw_data_label_entry.insert(0, 'Raw Data')

        # Add the raw label check button
        self.raw_label_check = ttk.Checkbutton(raw_data_frame, text='Label toggle', variable=self.raw_label_toggle, command=self.update_plot)
        self.raw_label_check.grid(row=2, column=2, sticky='nsew')

        # Add to the label legend list
        self.plot_labels_legend_list.append(self.raw_data_label_entry)

        # Create a frame for dynamically added elements
        self.dynamic_elements_frame = ttk.Frame(self.plot_metadata_frame)
        self.dynamic_elements_frame.grid(row=3, column=0, sticky='nsew')

        self.update_plot()

    def widgets_plot_r_inp_frame(self):
        self.canvas_r_inp = FigureCanvasTkAgg(figure.Figure(figsize=(10, 5)), master=self.plot_r_inp_frame)
        self.ax_r_inp = self.canvas_r_inp.figure.add_subplot(111)
        self.canvas_r_inp.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.toolbar_r_inp = NavigationToolbar2Tk(self.canvas_r_inp, self.plot_r_inp_frame)
        self.toolbar_r_inp.update()
        
        self.label_input_resistance = ttk.Label(self.plot_r_inp_frame, text="Calculate Input Resistance Across Experiments")
        self.label_input_resistance.pack(side=tk.TOP, pady=1)

        self.calculate_input_resistance_button = ttk.Button(self.plot_r_inp_frame, text='Plot Input Resistance', command=self.plot_input_resistance)
        self.calculate_input_resistance_button.pack(side=tk.TOP, pady=1)
    
    """Functions for handling single traces and general functions"""

    def update_index_and_plot(self):
        '''Update the index and plot the data'''
        self.current_index = int(self.index_entry.get())
        self.update_plot()
    
    def update_metadata(self):
        # Fetch the current values from the tkinter widgets
        experiment_description = self.experiment_description_entry.get()
        experiment_type_var = self.experiment_type_entry.get()

        # Ensure we're working with the original DataFrame
        if experiment_description:
            self.data_df.loc[self.current_index, 'experiment_description'] = experiment_description

        # Update the experiment type based on current and new values
        current_experiment_type = self.data_df.loc[self.current_index, 'stim_type']
        
        if experiment_type_var:
            # If the experiment type has been manually changed, update it
            if experiment_type_var != current_experiment_type:
                self.data_df.loc[self.current_index, 'stim_type'] = experiment_type_var
        else:
            # Keep the existing value if no new value is provided
            self.data_df.loc[self.current_index, 'stim_type'] = current_experiment_type

        # Update the condition and ISI if they are not empty
        if self.condition_entry.get():
            self.data_df.at[self.current_index, 'stimulus_metadata_dict']['condition'] = self.condition_entry.get()
        if self.ISI_entry.get():
            self.data_df.at[self.current_index, 'stimulus_metadata_dict']['ISI'] = self.ISI_entry.get()

        # Update the pathway entries if they are not empty
        if self.pathway_entry_1.get():
            self.data_df.at[self.current_index, 'stimulus_metadata_dict']['channel_1_label'] = self.pathway_entry_1.get()
        if self.pathway_entry_2.get():
            self.data_df.at[self.current_index, 'stimulus_metadata_dict']['channel_2_label'] = self.pathway_entry_2.get()
        if self.pathway_entry_3.get():
            self.data_df.at[self.current_index, 'stimulus_metadata_dict']['channel_3_label'] = self.pathway_entry_3.get()
        if self.pathway_entry_4.get():
            self.data_df.at[self.current_index, 'stimulus_metadata_dict']['channel_4_label'] = self.pathway_entry_4.get()

        # Update the amplitude entries if they are not empty
        if self.amplitude_entry_1.get():
            self.data_df.at[self.current_index, 'stimulus_metadata_dict']['channel_1_amp'] = self.amplitude_entry_1.get()
        if self.amplitude_entry_2.get():
            self.data_df.at[self.current_index, 'stimulus_metadata_dict']['channel_2_amp'] = self.amplitude_entry_2.get()
        if self.amplitude_entry_3.get():
            self.data_df.at[self.current_index, 'stimulus_metadata_dict']['channel_3_amp'] = self.amplitude_entry_3.get()
        if self.amplitude_entry_4.get():
            self.data_df.at[self.current_index, 'stimulus_metadata_dict']['channel_4_amp'] = self.amplitude_entry_4.get()

        # Update the channel entries if they are not empty
        if self.channel_entry_1.get():
            self.data_df.at[self.current_index, 'stimulus_metadata_dict']['channel_1_label'] = self.channel_entry_1.get()
        if self.channel_entry_2.get():
            self.data_df.at[self.current_index, 'stimulus_metadata_dict']['channel_2_label'] = self.channel_entry_2.get()
        if self.channel_entry_3.get():
            self.data_df.at[self.current_index, 'stimulus_metadata_dict']['channel_3_label'] = self.channel_entry_3.get()
        if self.channel_entry_4.get():
            self.data_df.at[self.current_index, 'stimulus_metadata_dict']['channel_4_label'] = self.channel_entry_4.get()

        # Update all of the entries if the user changes the values
        if self.ISI_entry.get():
            self.update_entry(self.ISI_entry.get(), self.current_index, 'ISI')
        if self.condition_entry.get():
            self.update_entry(self.condition_entry.get(), self.current_index, 'condition')
        if self.experiment_type_entry.get():
            self.update_entry(self.experiment_type_entry.get(), self.current_index, 'stim_type')
        if self.experiment_description_entry.get():
            self.update_entry(self.experiment_description_entry.get(), self.current_index, 'experiment_description')

    def fill_metadata(self):
        # Fill in the metadata for the experiment
        self.update_description_and_type()
        self.change_text()
    
    def update_description_and_type(self):
        '''Update the experiment description and type when the index is changed using after()'''
        try:
            if self.index_entry.get() != self.current_index:
                index = self.current_index
                # Only update the description and type if the entry fields are empty
                if not self.experiment_description_entry.get():
                    self.experiment_description_entry.delete(0, tk.END)
                    self.experiment_description_entry.insert(0, self.data_df['experiment_description'].iloc[index])
                if not self.experiment_type_entry.get():
                    self.experiment_type_entry.delete(0, tk.END)
                    self.experiment_type_entry.insert(0, self.data_df['stim_type'].iloc[index])
        except Exception as e:
            print(f"Error: {e}")


    def change_text(self):
        '''Change the text of all the labels in the GUI when the index is changed using after()'''
        try:
            if self.index_entry.get() != self.current_index:
                self.current_index = int(self.index_entry.get())
                #update multiple entries
                self.update_pathway_entries(self.current_index)
                self.update_amplitude_entries(self.current_index)
                #update single entries
                self.update_entry(self.ISI_entry, self.current_index, 'ISI')
                self.update_entry(self.condition_entry, self.current_index, 'condition')
            # self.experiment_metadata_frame.after(100, self.change_text)
        except Exception as e:
            print(f"Error: {e}")
    
    def update_entry(self, entry, index, key, suffix=''):
        '''Helper method to update a single entry widget.'''
        try:
            # Check if the entry is an instance of a tkinter Entry widget
            if isinstance(entry, ttk.Entry):
                # Clear the entry widget
                entry.delete(0, tk.END)
            elif isinstance(entry, float):
                # Round the float entry to 2 decimal places
                entry = round(entry, 2)
            elif isinstance(entry, str):
                # Keep the string entry as is
                pass
            else:
                raise ValueError("Unsupported entry type")

            metadata = self.data_df['stimulus_metadata_dict'].iloc[index]

            default_metadata = {'channel_1_amp': '', 'channel_2_amp': '', 'channel_3_amp': '', 'channel_4_amp': '',
                                'channel_1_label': '', 'channel_2_label': '', 'channel_3_label': '', 'channel_4_label': '',
                                'condition': '', 'ISI': ''}

            # Check if the metadata is a dictionary and is empty
            if not isinstance(metadata, dict):
                metadata = default_metadata

            # Ensure the key is in the metadata dictionary
            if key not in metadata:
                metadata[key] = default_metadata[key]

            # Insert the value into the entry
            value = metadata[key]

            # If the entry is a tkinter Entry widget, insert the value
            if isinstance(entry, ttk.Entry):
                entry.insert(0, f'{(value)}{suffix}')
            else:
                # Otherwise, just assign the value
                entry = f'{(value)}{suffix}'

            # Also add the value to the data_df for that index and key
            self.data_df['stimulus_metadata_dict'][index][key] = value

        except Exception as e:
            print(f"Error updating entry {key}: {e}")

    def update_pathway_entries(self, index):
        '''Helper method to update the pathway entries.'''
        pathway_keys = ['channel_1_label', 'channel_2_label', 'channel_3_label', 'channel_4_label']
        pathway_entries = [self.pathway_entry_1, self.pathway_entry_2, self.pathway_entry_3, self.pathway_entry_4]

        for key, entry in zip(pathway_keys, pathway_entries):
            self.update_entry(entry, index, key)

    def update_amplitude_entries(self, index):
        '''Helper method to update the amplitude entries.'''
        amplitude_keys = ['channel_1_amp', 'channel_2_amp', 'channel_3_amp', 'channel_4_amp']
        amplitude_entries = [self.amplitude_entry_1, self.amplitude_entry_2, self.amplitude_entry_3, self.amplitude_entry_4]

        for key, entry in zip(amplitude_keys, amplitude_entries):
            self.update_entry(entry, index, key)
        
    def open_file(self):
        try:
            self.file_path = filedialog.askopenfilename(filetypes=[("Pickle files", "*.pkl")])
            if self.file_path:
                self.data_df = pd.read_pickle(self.file_path)
        
            return self.data_df
        
        except Exception as e:
            print(e)
            print('No file was selected.')

    def load_image(self, image_path):
        '''Load an image from a file path'''
        try:
            image = Image.open(image_path)
            return image
        except Exception as e:
            print(f"Error loading image: {e}")
            return None

    def get_plot_information(self, sweep):
        current_experiment = self.experiment_type_var.get()
        IV_stim_list = ['Coarse_FI', 'Fine_FI', 'voltage_sag']
        EPSP_stim_list = ['EPSP_test_pulses', 'E_I_pulse', 'theta_burst', 'tetanus']
        
        try:
            if current_experiment in IV_stim_list:
                stim_start = self.defaults['analysis'][current_experiment]['step_start']
                stim_end = self.defaults['analysis'][current_experiment]['step_duration'] + stim_start
                current_stim_array = self.data_df['stim_command'][sweep]
                stim_start_index = int(stim_start * self.data_df['acquisition_frequency'][sweep] / 1000)
                stim_end_index = int(stim_end * self.data_df['acquisition_frequency'][sweep] / 1000)
                current_stim_value = np.mean(current_stim_array[stim_start_index:stim_end_index])
                #round up to 4 decimal places
                current_stim_value = round(current_stim_value, 2)
                
            elif current_experiment in EPSP_stim_list:
                stim_start = np.nan
                stim_end = np.nan
                current_stim_value = np.nan

            else:
                stim_start = np.nan
                stim_end = np.nan
                current_stim_value = np.nan

            return stim_start, stim_end, current_stim_value
    
        except Exception as e:
            print(f"Error in get_plot_information: {e}")
            stim_start = np.nan
            stim_end = np.nan
            current_stim_value = np.nan
            return stim_start, stim_end, current_stim_value
        
    def plot_partitioned_and_offset_traces(self, traces, time, acquisition_frequency, label_toggle, additional_label):
        """
        Plot the offset traces for the different types of experiments.

        Parameters:
            traces (dict): A dictionary of the offset traces for the different channels.
            time (np.array): The time array for the current trace.
            acquisition_frequency (int): The acquisition frequency of the data.
            label_toggle (bool): A boolean to determine if the legend should be added to the plot.
            additional_label (str): An additional label to append to each trace label.
        """
        # for plotting ISI 300 experiments
        def plot_unitary_traces(add_legend):
            used_channels = set()
            for color, channel in zip(['black', 'red'], traces):
                baseline_window_duration = self.defaults['analysis']['E_I_pulse'][channel]['baseline_window_duration']
                ISI_correction_time = self.defaults['analysis']['E_I_pulse'][channel]['ISI_correction']
                for stim in traces[channel]:
                    current_trace = traces[channel][stim][0]
                    current_time = np.linspace(stim - baseline_window_duration, stim + (self.ISI_time - ISI_correction_time), len(current_trace))
                    channel_label = f"{self.channel_names[channel]} {additional_label}" if channel not in used_channels and add_legend else None
                    self.ax.plot(current_time, current_trace, color=color, label=channel_label)
                    used_channels.add(channel)
            if add_legend:
                self.ax.legend()

        if self.experiment_type_var.get() in self.EPSP_list:
            if self.ISI_time == 300:
                plot_unitary_traces(add_legend=label_toggle)
            else:
                used_channels = set()
                for color, channel in zip(['black', 'red'], traces):
                    baseline_window_duration = self.defaults['analysis']['E_I_pulse'][channel]['baseline_window_duration']
                    current_trace = traces[channel]
                    print(self.stim_start_dict)
                    num_pulses = self.defaults['analysis'][self.experiment_type_var.get()][channel]['num_pulses']
                    # stim_time = self.stim_start_dict[channel]
                    # #get the first value of the stim time list
                    stim_time = self.stim_start_dict[channel]
                    print(stim_time)    
                    if num_pulses > 2:
                        partition_time_duration = self.defaults['analysis']['E_I_pulse'][channel]['partition_time_duration']
                        current_time = self.partition_time
                        stim_time = stim_time[0]
                        current_time = np.linspace(stim_time - baseline_window_duration, stim_time + (partition_time_duration), len(current_trace))
                        pathway_name = f"{self.channel_names[channel]} {additional_label}" if channel not in used_channels and label_toggle else None
                        used_channels.add(channel)
                        self.ax.plot(current_time, current_trace, color=color, label=pathway_name)
                    else:
                        num_pulses = self.defaults['analysis'][self.experiment_type_var.get()][channel]['num_pulses']
                        ISI_time = self.defaults['analysis'][self.experiment_type_var.get()][channel]['ISI'] * num_pulses
                        ISI_correction_time = self.defaults['analysis'][self.experiment_type_var.get()][channel]['ISI_correction']
                        current_time = np.linspace(stim_time - baseline_window_duration, stim_time + (ISI_time + ISI_correction_time), len(current_trace))
                        pathway_name = f"{self.channel_names[channel]} {additional_label}" if channel not in used_channels and label_toggle else None
                        used_channels.add(channel)
                        self.ax.plot(current_time, current_trace, color=color, label=pathway_name)
                if label_toggle:
                    self.ax.legend()
        else:
            if label_toggle:
                self.ax.plot(time, traces, label=f'Offset Trace for {self.experiment_description_entry.get()} {additional_label}', color='k', linewidth=1)
                self.ax.legend()
            else:
                self.ax.plot(time, traces, color='k', linewidth=1)

    def update_plot(self):
        '''Update the plot based on the selected index.'''

        self.get_stim_times()

        if self.index_entry.get() != '':
            index = int(self.index_entry.get())
        else:
            index = self.current_index

        if self.clear_plot.get():
            self.ax.clear()

        if self.raw_data_label_entry.get() != '':
            raw_label = self.raw_data_label_entry.get()
        else:
            raw_label = 'Raw Data'

        # Get the channel pathway names for the current index
        self.channel_names = {key: self.defaults['analysis']['E_I_pulse'][key]['label'] for key in self.defaults['analysis']['E_I_pulse']}

        default_title = f'Plotting Current Sweep {index}'
        if self.plot_title_entry.get() != '':
            title = self.plot_title_entry.get()
        else:
            title = default_title

        '''Plotting Raw Data'''

        data = self.data_df['sweep'][index]
        dt = self.data_df['dt'][index]
        self.time = np.arange(0, len(data) * dt, dt)

        # Make sure remove label is not empty and if check button is checked, plot with the label
        if self.raw_data_plot_toggle.get():
            if self.raw_label_toggle.get():
                self.ax.plot(self.time, data, color='grey', linewidth=1, label=raw_label, alpha=0.8)
                # Add the legend with this label
                self.ax.legend()
            else:
                self.ax.plot(self.time, data, color='grey', linewidth=1, alpha=0.8)

        '''Plotting AP analysis'''
        self.get_plot_information(index)

        stim_start, stim_end, stim_value = self.get_plot_information(index)

        if not np.isnan(stim_start) and not np.isnan(stim_end) and not np.isnan(stim_value):
            self.stim_label.config(text=f"IV Stim Start: {stim_start} ms, IV Stim End: {stim_end} ms, IV Stim Value: {stim_value} pA")
        else:
            self.stim_label.config(text="Stim information not available")

        if 'AP' in self.data_df['analysis_dict'].at[index]:
            AP_peak_idx = self.data_df['analysis_dict'].at[index]['AP']['AP_peak_indices']
            AP_threshold_idx = self.data_df['analysis_dict'].at[index]['AP']['AP_threshold_indices']
            AHP_idx = self.data_df['analysis_dict'].at[index]['AP']['AHP_indices']

            used_labels = set()

            # Plot AP Peaks
            self.ax.scatter(self.time[AP_peak_idx], data[AP_peak_idx], marker='o', color='red', label='AP Peaks')
            used_labels.add('AP Peaks')

            # Plot AP Thresholds, only non-NaN values
            for idx in AP_threshold_idx:
                if not np.isnan(idx):
                    label = 'AP Threshold' if 'AP Threshold' not in used_labels else '_nolegend_'
                    self.ax.scatter(self.time[idx], data[idx], marker='o', color='blue', label=label)
                    used_labels.add('AP Threshold')

            # Plot AHPs, only non-NaN values
            for idx in AHP_idx:
                if not np.isnan(idx):
                    label = 'AHP' if 'AHP' not in used_labels else '_nolegend_'
                    self.ax.scatter(self.time[idx], data[idx], marker='o', color='green', label=label)
                    used_labels.add('AHP')

            # Add the legend
            if used_labels:
                self.ax.legend()

        # if 'E_I_pulse' in self.data_df['analysis_dict'].at[index]:
        #     #scatter the EPSP peaks
        #     # EPSP_peak_idx = self.data_df['analysis_dict'].at[index]['E_I_pulse']['EPSP_peak_indices']
        #     EPSP_value = self.data_df['analysis_dict'].at[index]['E_I_pulse']['EPSP_amplitude']

        #     self.ax.text( 0.05, 0.95, f'Average Input Resistance: {EPSP_value}',  transform=self.ax.transAxes, fontsize=10, verticalalignment='top')) 

        
        # if self.experiment_type_var.get() in self.data_df['analysis_dict'].at[index]:
        #     #fill between the plateau area
        #     for channel in self.defaults['analysis'][self.experiment_type_var.get()]:
        #         stim_start = self.defaults['analysis'][self.experiment_type_var.get()][channel]['stim_start']
        #         plateau_area = self.data_df['analysis_dict'].at[index][self.experiment_type_var.get()]['plateau_area']
        #         plateau_area = plateau_area - plateau_area[0]
        #         plateau_area = plateau_area + stim_start
        #         self.ax.fill_between(self.time, data, where=(plateau_area[0] <= np.arange(len(data)) <= plateau_area[-1]), color='grey', alpha=0.5, label='Plateau Area')

        # Iterate over all the traces that it finds in current_trace keys for intermediate traces
        #'intermediate_traces is a dictionary of dictionaries'
        if 'intermediate_traces' in self.data_df:
            trace_keys_list = list(self.data_df['intermediate_traces'][index].keys())
            for trace_key in trace_keys_list:
                # Make sure it only does it once
                if trace_key not in self.plot_check_buttons.keys():
                    # Generate all the GUI elements needed for plot checkboxes and label textboxes for this trace_key
                    # Create the new frame for the elements
                    new_elements_frame = ttk.Frame(self.dynamic_elements_frame)
                    new_elements_frame.pack(side=tk.TOP, fill=tk.X, pady=2)

                    # Add the check button boolean for that trace key to the analysis_check_buttons dictionary
                    self.plotting_checks[trace_key] = tk.BooleanVar(value=False)
                    #capitilize the first letter of the trace key for the plot text
                    plot_text = trace_key.capitalize()
                    check_button = ttk.Checkbutton(new_elements_frame, text=plot_text, variable=self.plotting_checks[trace_key], command=self.update_plot)
                    check_button.pack(side=tk.LEFT, padx=5, pady=2)
                    self.plot_check_buttons[trace_key] = check_button

                    # Create an entry for the processed data label
                    stim_data_label = ttk.Entry(new_elements_frame, width=15)
                    stim_data_label.pack(side=tk.LEFT, padx=5, pady=2)
                    stim_data_label.insert(0, f'{trace_key}')
                    self.processed_data_label[trace_key] = stim_data_label

                    # Pack a check button for the processed data label
                    self.label_checks[trace_key] = tk.BooleanVar(value=False)
                    label_check = ttk.Checkbutton(new_elements_frame, text='Label toggle', variable=self.label_checks[trace_key], command=self.update_plot)
                    label_check.pack(side=tk.LEFT, padx=5, pady=2)
                    self.label_check_buttons[trace_key] = label_check

                    # Add the label to the legend list
                    self.plot_labels_legend_list.append(stim_data_label)

                    # Add the new widgets to the plot_widgets dictionary
                    if trace_key not in self.plot_widgets:
                        self.plot_widgets[trace_key] = []
                    self.plot_widgets[trace_key].extend([new_elements_frame, stim_data_label, label_check, check_button])
                
                #if the trace key does not exist then destroy it from the new_elements_frame
                for trace_key in self.plot_widgets.keys():
                    if trace_key not in list(self.data_df['intermediate_traces'][self.current_index].keys()):
                        #set check box boolean to false
                        self.plotting_checks[trace_key].set(False)
                        self.label_checks[trace_key].set(False)
                    # #when opening a file with existing traces, set the check box to true
                    # else:
                    #     self.plotting_checks[trace_key].set(True)
                    #     self.label_checks[trace_key].set(True)
                    
            # Plotting selected intermediate traces
            for trace_key in trace_keys_list:
                if self.plotting_checks[trace_key].get():
                    label_toggle = self.label_checks[trace_key].get()
                    if trace_key == 'partitioned_trace' or trace_key == 'offset_trace':
                        traces = self.data_df['intermediate_traces'][index][trace_key] 
                        dt = self.data_df['dt'][index]
                        time = np.arange(0, len(self.data_df['sweep'][index]) * self.data_df['dt'][index], self.data_df['dt'][index])
                        if label_toggle:
                            processed_label = self.processed_data_label[trace_key].get()
                            self.plot_partitioned_and_offset_traces(traces, time, self.data_df['acquisition_frequency'][index], label_toggle=label_toggle, additional_label=processed_label)
                        else:
                            self.plot_partitioned_and_offset_traces(traces, time, self.data_df['acquisition_frequency'][index], label_toggle=label_toggle, additional_label='')
                    else:
                        processed_data = self.data_df['intermediate_traces'][index][trace_key]
                        dt = self.data_df['dt'][index]
                        time = np.arange(0, len(processed_data) * dt, dt)
                        if label_toggle:
                            processed_label = self.processed_data_label[trace_key].get()
                            self.ax.plot(time, processed_data, color='blue', linewidth=1, label=processed_label)
                            self.ax.legend()
                        else:
                            self.ax.plot(time, processed_data, color='blue', linewidth=1)

        self.ax.set_title(title)
        self.ax.set_xlabel('Time (s)')
        self.ax.set_ylabel('Voltage (mV)')
        self.canvas.draw()

    def average_multiple_plots(self):
        #make an instance for if the traces have been offset already            
        #average the multiple plots if they are not offset
        self.average_traces_pressed = True
        if ',' in self.index_entry_multiple.get():
            temp_plot_indices = self.index_entry_multiple.get().split(',')
            temp_plot_indices = [int(index) for index in temp_plot_indices]

            temp_traces = []
            for i in temp_plot_indices:
                temp_traces.append(self.data_df['sweep'][i])
            
            temp_plot_traces = []
            for trace in temp_traces:
                temp_plot_traces.append(trace[:len(temp_traces[0])])
            
            if self.offset_traces == {}:
                self.average_data = np.mean(temp_plot_traces, axis=0)
            
            elif self.offset_traces != {}:
                offset_traces = self.offset_traces
                if self.ISI_time == 300:
                    #create a temp dictionary of each average trace per channel at each stim time and for each index
                    self.average_offset_traces = {}
                    for index in offset_traces:
                        self.average_offset_traces[index] = {}
                        for channel in offset_traces[index]:
                            self.average_offset_traces[index][channel] = {}
                            for stim in offset_traces[index][channel]:
                                self.average_offset_traces[index][channel][stim] = np.mean(offset_traces[index][channel][stim], axis=0)
                else:
                    self.average_offset_traces = {}
                    for index in offset_traces:
                        self.average_offset_traces[index] = {}
                        for channel in offset_traces[index]:
                            self.average_offset_traces[index][channel] = np.mean(offset_traces[index][channel], axis=0)
            
            else:
                self.average_data = np.mean(temp_plot_traces, axis=0)
    
    def apply_partition_data(self):
        '''partition the data based on the experiment type'''

        index = self.current_index
        acquisition_frequency = self.data_df['acquisition_frequency'][self.start_index]

        if self.data_df['current_trace_key'][index] == 'raw':
                self.current_trace = self.data_df['sweep'][index]

        else:
            self.current_trace = self.data_df['intermediate_traces'][index][self.data_df['current_trace_key'][index]]
        
        acquisition_frequency = self.data_df['acquisition_frequency'][self.start_index]
        self.stim_start_dict = {key: self.defaults['analysis'][self.experiment_type_var.get()][key]['stim_start'] for key in self.defaults['analysis'][self.experiment_type_var.get()]}

        if self.experiment_type_var.get() in self.EPSP_list:

            #check how many pulses are in the experiment for the first channel (key)
            self.num_pulses = self.defaults['analysis'][self.experiment_type_var.get()][list(self.defaults['analysis'][self.experiment_type_var.get()].keys())[0]]['num_pulses']

            #First handle E/I experiments -using the ISI time given either by the stimulus metadata or by the user - should be self.ISI_time
            self.partition_data_dict = {key: [] for key in self.defaults['analysis']['E_I_pulse']}

            #for E/I Experiments - if the ISI is 300 (unitary, then organize by channel, stim time)
            #if the ISI is not 300, then organize by channel only
            if self.num_pulses > 2:
                
                if self.ISI_time is None:
                    self.ISI_time = int(float(self.ISI_entry.get()))
                    self.ISI_time = self.ISI_time
                else:
                    self.ISI_time = self.ISI_time
                
                self.stim_times = self.get_stim_times()

                if self.ISI_time == 300:
                    #organize by channel then stim time for the specific experiment
                    for channel in self.partition_data_dict:
                        stim_times_list = self.stim_times[channel]
                        self.partition_data_dict[channel] = {key: [] for key in stim_times_list}
                    #fill the dictionary with the partitioned data
                    for channel in self.partition_data_dict:
                        baseline_window_duration = self.defaults['analysis']['E_I_pulse'][channel]['baseline_window_duration']
                        baseline_window_duration = int(baseline_window_duration * acquisition_frequency / 1000)
                        for stim_time_key in self.partition_data_dict[channel]:
                            partition_time_duration = self.defaults['analysis']['E_I_pulse'][channel]['partition_time_duration']
                            partition_time_duration = int(partition_time_duration * acquisition_frequency / 1000)
                            #convery stim_time to integer
                            current_stim_time = int(stim_time_key)
                            stim_index = int(current_stim_time * acquisition_frequency / 1000)
                            partition_start_index = stim_index - baseline_window_duration
                            partition_end_index = stim_index + partition_time_duration
                            self.partition_data_dict[channel][stim_time_key].append(self.current_trace[partition_start_index:partition_end_index])
                            self.partition_time = np.linspace(partition_start_index, partition_end_index, len(self.partition_data_dict[channel][stim_time_key][0]))
                else:
                    for channel in self.partition_data_dict:
                        # print(channel)
                        # print(self.defaults['analysis']['E_I_pulse'][channel]) 
                        baseline_window_duration = self.defaults['analysis']['E_I_pulse'][channel]['baseline_window_duration']
                        # print(baseline_window_duration)
                        # print( int(baseline_window_duration * acquisition_frequency / 1000))
                        baseline_window_duration = int(baseline_window_duration * acquisition_frequency / 1000)
                        stim_time = self.stim_start_dict[channel]
                        #get the first value of the stim time list
                        # print(stim_time)
                        stim_time = int(stim_time[0])
                        stim_index = int(stim_time * acquisition_frequency / 1000)

                        partition_start_index = stim_index - baseline_window_duration
                        partition_time_duration = self.defaults['analysis']['E_I_pulse'][channel]['partition_time_duration']
                        partition_end_index = stim_index + int(partition_time_duration * acquisition_frequency / 1000)
                        self.partition_data_dict[channel] = self.current_trace[partition_start_index: partition_end_index]
                        self.partition_time = np.linspace(partition_start_index, partition_end_index, len(self.partition_data_dict[channel]))
          
            #Plateau type experiments - no ISI time
            else:
                for channel in self.partition_data_dict:
                    baseline_window_duration = self.defaults['analysis'][self.experiment_type_var.get()][channel]['baseline_window_duration']
                    baseline_window_duration = int(baseline_window_duration * acquisition_frequency / 1000)
                    stim_time = self.stim_start_dict[channel]
                    stim_time = int(stim_time)
                    stim_index = int(stim_time * acquisition_frequency / 1000)
                    partition_start_index = stim_index - baseline_window_duration
                    partition_time_duration = self.defaults['analysis'][self.experiment_type_var.get()][channel]['partition_time_duration']
                    partition_end_index = stim_index + int(partition_time_duration * acquisition_frequency / 1000)
                    self.partition_data_dict[channel] = self.current_trace[partition_start_index: partition_end_index]
                    self.partition_time = np.linspace(partition_start_index, partition_end_index, len(self.partition_data_dict[channel]))


                # #handle unitary EPSPs
                # if self.ISI_time == 300:
                #     for channel in self.partition_data_dict:
                #         current_stim_times_channel = []
                #         stim_start_time_channel = self.stim_start_dict[channel]

                #         ISI_correction_time = self.defaults['analysis']['E_I_pulse'][channel]['ISI_correction']

                #         for pulse in range(self.defaults['analysis']['E_I_pulse'][channel]['num_pulses']):
                #             current_stim_times_channel.append(stim_start_time_channel + (pulse * (self.ISI_time - ISI_correction_time)))
                        
                #         self.partition_data_dict[channel] = {key: [] for key in current_stim_times_channel}

                #     for channel in self.partition_data_dict:
                #             baseline_window_duration = self.defaults['analysis']['E_I_pulse'][channel]['baseline_window_duration']
                #             baseline_window_duration = int(baseline_window_duration * acquisition_frequency / 1000)
                #             for stim_time in self.partition_data_dict[channel]:
                #                 start_stim_index = int(stim_time * acquisition_frequency / 1000)
                #                 #start 20ms before the stim time 
                #                 partition_start_index = start_stim_index - baseline_window_duration
                #                 partition_end_index = start_stim_index + int((self.ISI_time - ISI_correction_time) * acquisition_frequency / 1000)
                #                 self.partition_data_dict[channel][stim_time].append(self.current_trace[partition_start_index:partition_end_index])
            
            #     for other experiments E/I not unitary EPSPs there is just a channel key
            #     else:
            #         for channel in self.partition_data_dict:
            #             baseline_window_duration = self.defaults['analysis']['E_I_pulse'][channel]['baseline_window_duration']
            #             baseline_window_duration = int(baseline_window_duration * acquisition_frequency / 1000)
            #             stim_time = self.stim_start_dict[channel]
            #             stim_index = int(stim_time * acquisition_frequency / 1000)
            #             # #take 10ms before the stim time
            #             partition_start_index = stim_index - baseline_window_duration
            #             partition_time_duration = self.defaults['analysis']['E_I_pulse'][channel]['partition_time_duration']
            #             partition_end_index = stim_index + int((partition_time_duration) * acquisition_frequency / 1000)
            #             self.partition_time = np.linspace(partition_start_index, partition_end_index, len(self.partition_data_dict[channel]))

            #             # #get a trace that is 300ms from the stim start time
            #             self.partition_data_dict[channel] = self.current_trace[partition_start_index: partition_end_index]
            #             # self.current_trace = self.partition_data_dict
            # else: 
            #     for channel in self.partition_data_dict:
            #         baseline_window_duration = self.defaults['analysis'][self.experiment_type_var.get()][channel]['baseline_window_duration']
            #         baseline_window_duration = int(baseline_window_duration * acquisition_frequency / 1000)
            #         # ISI_time = self.defaults['analysis'][self.experiment_type_var.get()][channel]['ISI']
            #         stim_time = self.stim_start_dict[channel]
            #         stim_index = int(stim_time * acquisition_frequency / 1000)
            #         partition_start_index = stim_index - baseline_window_duration
            #         partition_time_duration = self.defaults['analysis'][self.experiment_type_var.get()][channel]['partition_time_duration']
            #         partition_end_index = stim_index + int(partition_time_duration * acquisition_frequency / 1000)
            #         self.partition_time = np.linspace(partition_start_index, partition_end_index, len(self.partition_data_dict[channel]))

            #         self.partition_data_dict[channel] = self.current_trace[partition_start_index: partition_end_index]
            #         # self.partition_time = np.linspace(partition_start_index, partition_end_index, len(self.partition_data_dict[channel]))
            #         # self.current_trace = self.partition_data_dict
    
        #update the intermediate trace for the current index
        self.data_df['intermediate_traces'][self.current_index]['partitioned_trace'] = self.partition_data_dict
        self.data_df['current_trace_key'].at[index] = 'partitioned_trace'

    def toggle_partition_data(self):
        if self.partition_traces_check.get():
            self.analysis_steps_list.append('partitioned_trace')
            self.analysis_steps_entry.insert(tk.END, 'partitioned_trace,')
            self.current_trace_key_dropdown['values'] = self.analysis_steps_list
            #set the current trace key in the dropdown to partitioned_trace
            self.current_trace_key_dropdown.set('partitioned_trace')
        else:
            if 'partitioned_trace' in self.analysis_steps_list:
                self.analysis_steps_list.remove('partitioned_trace')
                self.analysis_steps_entry.delete(0, tk.END)
                for element in self.analysis_steps_list:
                    self.analysis_steps_entry.insert(tk.END, f'{element},')

        self.apply_analysis_steps()
        self.update_plot()
    
    def apply_offset_data(self):
        '''Offset the data by subtracting the mean of a window before the stimulus'''
        index = self.current_index
        if self.data_df['current_trace_key'][index] == 'raw':
            self.current_trace = self.data_df['sweep'][index]

        else:
            self.current_trace = self.data_df['intermediate_traces'][index][self.data_df['current_trace_key'][index]]
            acquisition_frequency = self.data_df['acquisition_frequency'][self.start_index]

        self.stim_start_dict = {key: self.defaults['analysis'][self.experiment_type_var.get()][key]['stim_start'] for key in self.defaults['analysis'][self.experiment_type_var.get()]}

        if self.experiment_type_var.get() in self.EPSP_list:
            self.offset_data_dict = {key: [] for key in self.defaults['analysis']['E_I_pulse']}
            self.num_pulses = self.defaults['analysis'][self.experiment_type_var.get()][list(self.defaults['analysis'][self.experiment_type_var.get()].keys())[0]]['num_pulses']

            #Unitary EPSPs are organized by channel then stim time, other experiments are just by channel
            if self.num_pulses > 2:
                if self.ISI_time is None:
                    self.ISI_time = int(float(self.ISI_entry.get()))
                    self.ISI_time = self.ISI_time
                else:
                    self.ISI_time = self.ISI_time

                self.stim_times = self.get_stim_times()

                if self.ISI_time == 300:
                    #first create the dictionary for the offset traces
                    for channel in self.offset_data_dict:
                        stim_times_list = self.stim_times[channel]
                        self.offset_data_dict[channel] = {key: [] for key in stim_times_list}
                    #fill the dictionary with the partitioned data
                    for channel in self.offset_data_dict:
                        baseline_window_duration = self.defaults['analysis']['E_I_pulse'][channel]['baseline_window_duration']
                        for stim_time in self.partition_data_dict[channel]:
                            current_trace = self.partition_data_dict[channel][stim_time][0]
                            offset_window_samples = int(baseline_window_duration * acquisition_frequency / 1000)
                            start_index = 0
                            end_index = min(offset_window_samples, len(current_trace))
                            offset_window_trace = current_trace[start_index:end_index]
                            mean_offset_value = np.mean(offset_window_trace)
                            offset_corrected_trace = current_trace - mean_offset_value
                            self.offset_data_dict[channel][stim_time].append(offset_corrected_trace)

                    self.offset_traces = self.offset_data_dict
                    self.current_trace = self.offset_traces
                
                else:
                    for channel in self.offset_data_dict:
                        baseline_window_duration = self.defaults['analysis']['E_I_pulse'][channel]['baseline_window_duration']
                        current_trace = self.partition_data_dict[channel]
                        offset_window_samples = int(baseline_window_duration * acquisition_frequency / 1000)
                        start_index = 0
                        end_index = min(offset_window_samples, len(current_trace))
                        offset_window_trace = current_trace[start_index:end_index]
                        mean_offset_value = np.mean(offset_window_trace)
                        offset_corrected_trace = current_trace - mean_offset_value
                        self.offset_data_dict[channel] = offset_corrected_trace
                    
                    self.offset_traces = self.offset_data_dict
                    self.current_trace = self.offset_traces

            #for pair pulse experiments
            else:
                for channel in self.partition_data_dict:
                    baseline_window_duration = self.defaults['analysis'][self.experiment_type_var.get()][channel]['baseline_window_duration']
                    current_trace = self.partition_data_dict[channel]
                    offset_window_samples = int(baseline_window_duration * acquisition_frequency / 1000)  # 15 milliseconds offset
                    start_index = 0
                    end_index = min(offset_window_samples, len(current_trace))
                    offset_window_trace = current_trace[start_index:end_index]
                    mean_offset_value = np.mean(offset_window_trace)
                    offset_corrected_trace = current_trace - mean_offset_value
                    self.offset_data_dict[channel] = offset_corrected_trace

                self.offset_traces = self.offset_data_dict
                self.current_trace = self.offset_traces
        
        #for other experiments such as plateau experiments
        if self.experiment_type_var.get() in self.plateau_list:
            current_trace = self.current_trace
            stim_start = self.stim_start_dict[list(self.stim_start_dict.keys())[0]]
            stim_start_index = int(stim_start * acquisition_frequency / 1000)
            offset_window = current_trace[stim_start_index - int(10 * acquisition_frequency / 1000): stim_start_index]
            offset_value = np.mean(offset_window)
            offset_trace = current_trace - offset_value
            self.offset_traces = offset_trace
            self.current_trace = self.offset_traces
        

            # if self.ISI_time == 300:
            #     for channel in self.offset_data_dict:
            #         current_stim_times_channel = []
            #         stim_start_time_channel = self.stim_start_dict[channel]
            #         ISI_correction_time = self.defaults['analysis']['E_I_pulse'][channel]['ISI_correction']

            #         for pulse in range(self.defaults['analysis']['E_I_pulse'][channel]['num_pulses']):
            #             current_stim_times_channel.append(stim_start_time_channel + (pulse * (self.ISI_time - ISI_correction_time)))
                    
            #         self.offset_data_dict[channel] = {key: [] for key in current_stim_times_channel}

                # for channel in self.partition_data_dict:
                #     baseline_window_duration = self.defaults['analysis']['E_I_pulse'][channel]['baseline_window_duration']
                #     for stim_time in self.partition_data_dict[channel]:
                #         current_trace = self.partition_data_dict[channel][stim_time][0]
                #         offset_window_samples = int(baseline_window_duration * acquisition_frequency / 1000)  
                #         # Ensure the start index is non-negative and the window is within trace length
                #         start_index = 0
                #         end_index = min(offset_window_samples, len(current_trace))
                #         # Get the offset window trace
                #         offset_window_trace = current_trace[start_index:end_index]
                #         # Calculate the mean of the offset window
                #         mean_offset_value = np.mean(offset_window_trace)
                #         # Subtract the mean from the current trace to offset it
                #         offset_corrected_trace = current_trace - mean_offset_value
                #         self.offset_data_dict[channel][stim_time].append(offset_corrected_trace)

                # self.offset_traces = self.offset_data_dict
                # self.current_trace = self.offset_traces  
        
        #update the intermediate trace for the current index
        self.data_df['intermediate_traces'][self.current_index]['offset_trace'] = self.current_trace
        self.data_df['current_trace_key'].at[index] = 'offset_trace'
    
    def toggle_offset_data(self):
        if self.offset_traces_check.get():
            self.analysis_steps_list.append('offset_trace')
            self.analysis_steps_entry.insert(tk.END, 'offset_trace,')
            self.current_trace_key_dropdown['values'] = self.analysis_steps_list
            #set the current trace key in the dropdown to offset_trace
            self.current_trace_key_dropdown.set('offset_trace')
        else:
            if 'offset_trace' in self.analysis_steps_list:
                self.analysis_steps_list.remove('offset_trace')
                self.analysis_steps_entry.delete(0, tk.END)
                for element in self.analysis_steps_list:
                    self.analysis_steps_entry.insert(tk.END, f'{element},')

        self.apply_analysis_steps()
        self.update_plot()

    def clear_all_plots(self):
        '''Clear all plots on the canvas'''
        self.ax.clear()
        self.canvas.draw()
    
    def next(self):
        '''Move to the next index and update the plot'''
        self.current_index = int(self.index_entry.get())
        #should also refill the metadata entries
        if self.current_index < len(self.data_df) - 1:
            self.previous_index = self.current_index
            self.fill_metadata()
            self.index_entry.delete(0, tk.END)
            self.index_entry.insert(0, str(self.current_index + 1))
            self.current_index += 1
            self.analysis_types = self.data_df['analysis_dict'][self.current_index]
            self.apply_analysis_steps()
            self.update_plot()
        
    def previous(self):
        '''Move to the previous index and update the plot'''
        self.current_index = int(self.index_entry.get())
        if self.current_index > 0:
            self.previous_index = self.current_index
            self.fill_metadata()
            self.index_entry.delete(0, tk.END)
            self.index_entry.insert(0, str(self.current_index - 1))
            self.current_index -= 1
            self.analysis_types = self.data_df['analysis_dict'][self.current_index]
            self.apply_analysis_steps()
            self.update_plot()
    
    def apply_analysis_steps(self):
        '''Check the toggle states and run the appropriate functions'''
            
        for trace_key in self.analysis_steps_list:
            if trace_key == 'raw':
                self.data_df['current_trace_key'].at[self.current_index] = trace_key
            elif trace_key == 'stim_removed_trace':
                if 'stim_removed_trace' not in self.data_df['intermediate_traces'][self.current_index]:
                    self.apply_stim_removal()
                else:
                    #TODO: # wrapper function that does this and changes the current trace key pulldown
                    self.data_df['current_trace_key'].at[self.current_index] = trace_key        
            elif trace_key == 'noise_removed_trace':
                if 'noise_removed_trace' not in self.data_df['intermediate_traces'][self.current_index]:
                    self.apply_noise_removal()
                else:
                    self.data_df['current_trace_key'].at[self.current_index] = trace_key
            elif trace_key == 'interpolated_spikes_trace':
                if 'interpolated_spikes_trace' not in self.data_df['intermediate_traces'][self.current_index]:
                    self.apply_interpolated_spikes()
                else:
                    self.data_df['current_trace_key'].at[self.current_index] = trace_key
            elif trace_key == 'partitioned_trace':
                if 'partitioned_trace' not in self.data_df['intermediate_traces'][self.current_index]:
                    self.apply_partition_data()
                else:
                    self.data_df['current_trace_key'].at[self.current_index] = trace_key
            elif trace_key == 'offset_trace':
                if 'offset_trace' not in self.data_df['intermediate_traces'][self.current_index]:
                    self.apply_offset_data()
                else:
                    self.data_df['current_trace_key'].at[self.current_index] = trace_key
            #make sure that the current trace is used for analysis if it is processed (i.e - noise removed)

            elif trace_key == 'AP_AHP_analysis':
                if 'AP' not in self.analysis_types: 
                    self.apply_AP_AHP_analysis()

            # elif trace_key == 'detect_spikes':
            #     if 'detect_spikes' not in self.data_df['intermediate_traces'][self.current_index]:
            #         self.apply_d

            elif trace_key == 'analyze_EPSP_peaks':
                if 'EPSP_peaks' not in self.analysis_types:
                    self.apply_analyze_EPSP_peaks()
            
            elif trace_key == 'plateau_area':
                if 'plateau_area' not in self.analysis_types:
                    self.apply_plateau_area_under_curve()
    
    # #Get the input resistance across every sweep
    def get_input_resistances(self):
        #Get the input resistance across every sweep
        acquisition_frequency = self.data_df['acquisition_frequency'][self.start_index]
        current_pulse_amp = self.test_pulse_amp_entry.get()
        start_time = self.test_pulse_start_entry.get()
        end_time = self.test_pulse_end_entry.get()
        #have running list of input resistance values
        input_resistances = []

        all_sweeps = self.data_df['sweep'][:]
        #Get the array from all the sweeps and store them in a list of traces
        traces = []
        for sweep in all_sweeps:
            traces.append(sweep)

        for trace in traces:
            input_resistance = calculate_input_resistance(trace, current_pulse_amp, acquisition_frequency, start_time, end_time)
            input_resistances.append(input_resistance)
        
        return input_resistances
    
    # #Plot the input resistance across all the sweeps using experiment time
    def plot_input_resistance(self):
        input_resistances = self.get_input_resistances()

        #create the time array
        time_points = []
        #enumerate through all the sweeps
        for i in range(len(self.data_df)):
            first_time = self.data_df['global_wall_clock'][0]
            time_point = (self.data_df['global_wall_clock'][i] - first_time)/60
            time_points.append(time_point)

        #now plot the input resistance for each experiment against the cumulative time points 
        self.ax_r_inp.scatter(time_points, input_resistances, color='black', linewidth=1, label='Input Resistance')
        self.ax_r_inp.set_xlabel('Time (min)')
        self.ax_r_inp.set_ylabel('Input Resistance (MΩ)')
        self.ax_r_inp.set_title('Input Resistance Across Experiments')
        self.ax_r_inp.legend()
        self.canvas_r_inp.draw()

    # #Toggle analyze AP AHP properties to plot the AP threshold and AHP
    def apply_AP_AHP_analysis(self):
        index = self.current_index
        #run the AP AHP analysis function

        time = np.arange(0, len(self.data_df['sweep'][index]) * self.data_df['dt'][index], self.data_df['dt'][index])
        Vm_rest_start = int(self.Vm_rest_start_entry.get())
        Vm_rest_end = int(self.Vm_rest_end_entry.get())
        AP_peak_window_start = int(self.AP_peak_window_start_entry.get())
        AP_peak_window_end = int(self.AP_peak_window_end_entry.get())
        stim_start, stim_end, stim_value = self.get_plot_information(index)

        if self.data_df['current_trace_key'][index] == 'raw':
            self.current_trace = self.data_df['sweep'][index]

        else:
            self.current_trace = self.data_df['intermediate_traces'][index][self.data_df['current_trace_key'][index]]

        current_trace = self.current_trace.copy()
        self.AP_peaks_idx = find_peaks(current_trace, height=-10)[0]
        temp_AP_peak_idx = self.AP_peaks_idx

        # # trace, time, AP_peak_idx, stim_length, acquisition_frequency, Vm_rest_start, Vm_rest_end, AP_peak_window_start, AP_peak_window_end)
        self.AP_properties = analyze_AP_AHP_properties_trace(self.current_trace, time, temp_AP_peak_idx, stim_end, stim_end-stim_start, self.data_df['acquisition_frequency'][index], 
                                                             Vm_rest_start, Vm_rest_end, AP_peak_window_start, AP_peak_window_end)

        # 'AP_threshold': AP_thresholds, #mV
        # 'AP_threshold_indices': AP_threshold_indices, #index
        # 'AP_peak_indices': AP_peak_idx, #index
        # 'AP_firing_rate': AP_firing_rate, #Hz
        # 'AP_number': AP_number, #number
        # 'AP_ISI_time': AP_ISI_list, #ms
        # 'AP_halfwidth': AP_halfwidth_list, #ms
        # 'AP_size': AP_sizes, #mV
        # 'AHP_size:': AHP_sizes, #mV
        # 'AHP_halfwidth': AHP_halfwidth_list, #ms
        # 'AHP_indices': AHP_indices #index 

        # print(self.AP_properties.keys())
        
        self.analysis_types['AP'] = {}   # key:None for key in self.analysis_types['AP']}
        # for key in self.analysis_types['AP']:
        #     self.analysis_types['AP'][key] = self.AP_properties[key][:]

        self.analysis_types['AP']['AP_threshold'] = self.AP_properties['AP_threshold'][:]
        self.analysis_types['AP']['AP_threshold_indices'] = self.AP_properties['AP_threshold_indices'][:]
        self.analysis_types['AP']['AP_firing_rate'] = self.AP_properties['AP_firing_rate']
        self.analysis_types['AP']['AP_number'] = self.AP_properties['AP_number']
        self.analysis_types['AP']['AP_ISI_time'] = self.AP_properties['AP_ISI_time'][:]
        self.analysis_types['AP']['AHP_indices'] = self.AP_properties['AHP_indices'][:]
        self.analysis_types['AP']['AP_halfwidth'] = self.AP_properties['AP_halfwidth'][:]
        self.analysis_types['AP']['AP_size'] = self.AP_properties['AP_size'][:]
        self.analysis_types['AP']['AHP_halfwidth'] = self.AP_properties['AHP_halfwidth'][:]
        self.analysis_types['AP']['AHP_size'] = self.AP_properties['AHP_size'][:]
        self.analysis_types['AP']['AP_peak_indices'] = self.AP_properties['AP_peak_indices'][:]

        #get the current amplitude and firing rate for FI plot
        current_stim_value = stim_value

        #get the experiment type from the data_df ['stim_type]
        experiment_type = self.data_df['stim_type'][index]

        self.analysis_types[experiment_type] = {}
        self.analysis_types[experiment_type]['current_amplitudes'] = current_stim_value
        self.analysis_types[experiment_type]['firing_rates'] = self.AP_properties['AP_firing_rate']

        #save this dictionary to the data_df of analysis_dict
        self.data_df['analysis_dict'].at[index].update(self.analysis_types)
        
    def toggle_AP_AHP_analysis(self):
        if self.AP_AHP_analysis_check.get():
            self.analysis_steps_list.append('AP_AHP_analysis')
            self.analysis_steps_entry.insert(tk.END, 'AP_AHP_analysis,')
            #set the current trace key in the dropdown to AP_AHP_analysis
            self.current_trace_key_dropdown['values'] = self.analysis_steps_list
            #set the current trace key in the dropdown to offset_trace
            self.current_trace_key_dropdown.set('AP_AHP_analysis')
        else:
            if 'AP_AHP_analysis' in self.analysis_steps_list:
                self.analysis_steps_list.remove('AP_AHP_analysis')
                self.analysis_steps_entry.delete(0, tk.END)
                for element in self.analysis_steps_list:
                    self.analysis_steps_entry.insert(tk.END, f'{element},')

        self.apply_analysis_steps()
        self.update_plot()
    
    def calculate_intrinsic_properties(self):
        #calculate the intrinsic properties of the cell including the input resistance, voltage sag...
        index= self.current_index
        experiment_type = self.experiment_type_var.get()
        
        if self.data_df['current_trace_key'][index] == 'raw':
            self.current_trace = self.data_df['sweep'][index]
        else:
            self.current_trace = self.data_df['intermediate_traces'][index][self.data_df['current_trace_key'][index]]

        #get the input resistance
        input_resistances = self.get_input_resistances()
        average_input_resistance = np.mean(input_resistances, axis=0)

        self.analysis_types['Intrinsic_cell'] = {}
        self.analysis_types['Intrinsic_cell']['steady_state_input_resistance'] = average_input_resistance

        #calculate the voltage sag
        start_time = self.defaults['analysis']['voltage_sag']['step_start']
        end_time = self.defaults['analysis']['voltage_sag']['step_duration'] + start_time

        print(f'Start Time: {start_time}, End Time: {end_time}')
        # print(f'Start Time: {start_time}, End Time: {end_time}')
        voltage_sag = calculate_voltage_sag(experiment_type, self.current_trace, self.data_df['acquisition_frequency'][index], start_time, end_time)
        self.analysis_types['Intrinsic_cell']['Voltage_sag'] = voltage_sag

        self.data_df['analysis_dict'].at[index].update(self.analysis_types)
        
        print(f'Average Input Resistance: {average_input_resistance} MΩ')
        print(f'Voltage Sag: {voltage_sag} %')

        #plot the input resistance and voltage sag as text on the canvas
        self.ax.text( 0.05, 0.95, f'Average Input Resistance: {np.round(average_input_resistance,2)} MΩ and Voltage Sag: {np.round(voltage_sag,2)} %',  transform=self.ax.transAxes, fontsize=10, verticalalignment='top')
        self.canvas.draw()
    
    def apply_analyze_EPSP_peaks(self):
        #get the EPSP peaks - for this will take in dictionary of traces that have been partitioned and offset
        #for a single trace it will find peaks for each partitioned trace and for multiple traces it will find peaks for each partitioned trace of the average trace

        #analyze the EPSP peaks
        acquisition_frequency = self.data_df['acquisition_frequency'][self.current_index]
        delete_start = self.defaults['filtering']['stim_removal']['start']
        delete_start_index = int(delete_start * acquisition_frequency/1000)
        traces = self.offset_data_dict

        #get the EPSP peaks
       # get the EPSP peaks
        if self.ISI_time == 300:
            for channel in traces:
                EPSP_peak_list = []
                EPSP_peak_idx_list = []
                for stim in traces[channel]:
                    window_size = 100
                    current_trace = traces[channel][stim][0]
                    EPSP_peak, EPSP_peak_idx = find_peaks_in_window(current_trace, delete_start_index, window_size, acquisition_frequency)
                    EPSP_peak_list.append(EPSP_peak)
                    EPSP_peak_idx_list.append(EPSP_peak_idx)

                # Update the analysis dictionary for the current channel
                if 'E_I_pulse' not in self.analysis_types:
                    self.analysis_types['E_I_pulse'] = {}

                self.analysis_types['E_I_pulse'][channel] = {
                    'EPSP_amplitude': EPSP_peak_list,
                    'EPSP_peak_indices': EPSP_peak_idx_list
                }

                # Update the DataFrame with the new analysis types
                self.data_df['analysis_dict'].at[self.current_index].update(self.analysis_types)

                # Print the EPSP peak list size for the current channel
                print(f'EPSP size for channel {channel} in mV: {EPSP_peak_list}')

                # Add text to the plot for the current channel's EPSP peak sizes
                self.ax.text(0.5, 0.95, f'EPSP Peak Size in mV for channel {channel}: {EPSP_peak_list}', transform=self.ax.transAxes, fontsize=10, verticalalignment='top')
            
            # Redraw the canvas to update the plot
            self.canvas.draw()

        else:
                for channel in traces:
                    window_size = 300
                    current_trace = traces[channel]
                    if self.num_pulses > 2:
                        EPSP_peak, EPSP_peak_idx = find_peaks_in_window(current_trace, delete_start_index, window_size, acquisition_frequency)

                        if 'E_I_pulse' not in self.analysis_types:
                            self.analysis_types['E_I_pulse'] = {}
                        
                        self.analysis_types['E_I_pulse'][channel] = {
                            'EPSP_amplitude': EPSP_peak,
                            'EPSP_peak_indices': EPSP_peak_idx
                        }

                        self.data_df['analysis_dict'].at[self.current_index].update(self.analysis_types)

                        self.ax.text( 0.5, 0.95, f'EPSP Peak Size in mV: {EPSP_peak}', transform=self.ax.transAxes, fontsize=10, verticalalignment='top')
                        print(f'EPSP size for channel {channel} in mV: {EPSP_peak}')
                        self.canvas.draw()
                       
                    else: 
                        EPSP_peak, EPSP_peak_idx = find_peaks_in_window(current_trace, delete_start_index, window_size, acquisition_frequency)

                        if 'E_I_pulse' not in self.analysis_types:
                            self.analysis_types['Paired_Pulse'] = {}

                        self.analysis_types['Paired_Pulse'][channel] = {   
                            'EPSP_amplitude': EPSP_peak,
                            'EPSP_peak_indices': EPSP_peak_idx
                        }

                        self.ax.text( 0.5, 0.95, f'EPSP Peak Size in mV: {EPSP_peak}', transform=self.ax.transAxes, fontsize=10, verticalalignment='top')
                        print(f'EPSP size for channel {channel} in mV: {EPSP_peak}')
                        self.canvas.draw()
            
    def toggle_analyze_EPSP_peaks(self):
        if self.analyze_EPSP_peaks_check.get():
            self.analysis_steps_list.append('analyze_EPSP_peaks')
            self.analysis_steps_entry.insert(tk.END, 'analyze_EPSP_peaks,')
            #set the current trace key in the dropdown to analyze_EPSP_peaks
            self.current_trace_key_dropdown['values'] = self.analysis_steps_list
            self.current_trace_key_dropdown.set('analyze_EPSP_peaks')
        else:
            if 'analyze_EPSP_peaks' in self.analysis_steps_list:
                self.analysis_steps_list.remove('analyze_EPSP_peaks')
                self.analysis_steps_entry.delete(0, tk.END)
                for element in self.analysis_steps_list:
                    self.analysis_steps_entry.insert(tk.END, f'{element},')

        self.apply_analysis_steps()
        self.update_plot()
    
    #fix this function - not giving the correct values
    def apply_plateau_area_under_curve(self):
        # Calculate the area under the curve for plateau potential and EPSP with spikes
        experiment_type = self.experiment_type_var.get()
        try:
            if experiment_type in self.plateau_list:

                index = self.current_index

                if self.data_df['current_trace_key'][index] == 'raw':
                    self.current_trace = self.data_df['sweep'][index]

                else:
                    self.current_trace = self.data_df['intermediate_traces'][index][self.data_df['current_trace_key'][index]]
                
                acquisition_frequency = self.data_df['acquisition_frequency'][index]
                
                for channel in self.defaults['analysis'][experiment_type]:
                    stim_start = self.defaults['analysis'][experiment_type][channel]['stim_start']
                    stim_start_idx = int(stim_start * acquisition_frequency / 1000)
                    num_pulses = self.defaults['analysis'][experiment_type][channel]['num_pulses']
                    ISI_time = self.defaults['analysis'][experiment_type][channel]['ISI']
                    stim_end = stim_start + num_pulses * ISI_time
                    stim_end_idx = int(stim_end * acquisition_frequency / 1000)
                    current_data = self.current_trace[stim_start_idx:stim_end_idx]

                    #if area under curve is below zero set values to 0
                    if np.sum(current_data) < 0:
                        current_data[current_data < 0] = 0

                    #get area under curve 
                    plateau_area = np.trapz(current_data, dx=1/acquisition_frequency)


                self.analysis_types[experiment_type] = {}
                self.analysis_types[experiment_type]['plateau_area'] = plateau_area

                self.data_df['analysis_dict'].at[index].update(self.analysis_types)

                #plot filled between for the actual plateau area not the plateau duration
                # self.ax.fill_between(time[stim_start_idx:stim_end_idx], current_data, color='blue', alpha=0.5, label='Plateau Area')
                # self.ax.legend()
                # self.canvas.draw()

        except Exception as e:
            print('Error calculating area under the curve:', e)
    
    def toggle_plateau_area_under_curve(self):
        if self.plateau_area_check.get():
            self.analysis_steps_list.append('plateau_area')
            self.analysis_steps_entry.insert(tk.END, 'plateau_area,')
            #set the current trace key in the dropdown to plateau_area
            self.current_trace_key_dropdown['values'] = self.analysis_steps_list
            self.current_trace_key_dropdown.set('plateau_area')
        else:
            if 'plateau_area' in self.analysis_steps_list:
                self.analysis_steps_list.remove('plateau_area')
                self.analysis_steps_entry.delete(0, tk.END)
                for element in self.analysis_steps_list:
                    self.analysis_steps_entry.insert(tk.END, f'{element},')

        self.apply_analysis_steps()
        self.update_plot()

    def get_stim_times(self):
        experiment_type = self.experiment_type_var.get()
        non_E_I_pulse_stims = ['EPSP_test_pulses', 'theta_burst', 'tetanus']

        # self.stim_start_dict = {key: self.defaults['analysis'][experiment_type][key]['stim_start'] for key in self.defaults['analysis'][experiment_type]}
        #populate the stim times dictionary and also stim_start_dict with the first stim time
        self.stim_times = {}
        try:
            if experiment_type in non_E_I_pulse_stims:
                for channel in self.defaults['analysis'][experiment_type]:
                    stim_times = self.defaults['analysis'][experiment_type][channel]['stim_times']
                    #stim times per channel
                    self.stim_times[channel] = stim_times
                    self.stim_start_dict[channel] = stim_times[0]

                    # stim_start = self.defaults['analysis'][experiment_type][channel]['stim_start']
                    # for i in range(self.defaults['analysis'][experiment_type][channel]['num_pulses']):
                    #     self.stim_times.append(stim_start + i*self.defaults['analysis'][experiment_type][channel]['ISI'])

            if experiment_type == 'E_I_pulse':

                #get the ISI time entered by the user
                if self.ISI_entry.get() != '':
                    self.ISI_time = self.ISI_entry.get()
                    #convert to float then int
                    self.ISI_time = int(float(self.ISI_time))
                
                else:
                    self.ISI_time = self.data_df['stimulus_metadata_dict'][self.current_index]['ISI']
                    #convert to float then int
                    self.ISI_time = int(float(self.ISI_time))
                
                for channel in self.defaults['analysis'][experiment_type]:
                    stim_times = self.defaults['analysis'][experiment_type][channel]['ISI_stim_times'][self.ISI_time]
                    self.stim_times[channel] = stim_times[0]   
                    self.stim_start_dict[channel] = stim_times[0]

                    # stim_start = self.defaults['analysis'][experiment_type][channel]['stim_start']
                    # for i in range(self.defaults['analysis'][experiment_type][channel]['num_pulses']):
                    #     if ISI_time in self.defaults['analysis'][experiment_type][channel]['ISI_list']:
                    #         self.stim_times.append(stim_start + i*ISI_time)  

            return self.stim_times
        except Exception as e:
            print('Error getting stim times:', e)

    def apply_stim_removal(self):   
        index = self.current_index

        # test_pulse_start = self.defaults['analysis']['test_pulse']['step_start']
        # test_pulse_start_index = int(test_pulse_start * self.data_df['acquisition_frequency'][index] / 1000)
        # test_pulse_end_index = int(self.defaults['analysis']['test_pulse']['step_start'] * self.data_df['acquisition_frequency'][index] / 1000) + test_pulse_start_index

        if self.data_df['current_trace_key'][index] == 'raw':
            self.current_trace = self.data_df['sweep'][index]

        else:
            self.current_trace = self.data_df['intermediate_traces'][index][self.data_df['current_trace_key'][index]]

        if self.experiment_type_var.get() in self.EPSP_list:
                stim_times = self.get_stim_times()
                if self.stim_removal_check.get():
                    temp_data = remove_artifacts_custom(self.current_trace, stim_times, self.data_df['acquisition_frequency'][index], float(self.delete_start_stim_entry.get()), float(self.delete_end_stim_entry.get()))
                    self.current_trace = temp_data
                elif self.stim_removal_automated_check.get():
                    temp_data = remove_artifacts_automated(self.current_trace, self.data_df['acquisition_frequency'][index], float(self.delete_start_stim_entry.get()), float(self.delete_end_stim_entry.get()))
                    self.current_trace = temp_data

        if self.experiment_type_var.get() in self.plateau_list:
            stim_times = self.get_stim_times()
            if self.stim_removal_check.get():
                temp_data = remove_artifacts_custom(self.current_trace, stim_times, self.data_df['acquisition_frequency'][index], float(self.delete_start_stim_entry.get()), float(self.delete_end_stim_entry.get()))
                self.current_trace = temp_data
            elif self.stim_removal_automated_check.get():
                temp_data = remove_artifacts_automated(self.current_trace, self.data_df['acquisition_frequency'][index], float(self.delete_start_stim_entry.get()), float(self.delete_end_stim_entry.get()))
                self.current_trace = temp_data
            # time = np.arange(0, len(self.current_trace) * self.data_df['dt'][index], self.data_df['dt'][index])
            # temp_data = remove_artifacts_plateau(self.current_trace, time, acquisition_frequency=self.data_df['acquisition_frequency'][index], 
            #                                     delete_start_stim=float(self.delete_start_stim_entry.get()), delete_end_stim=float(self.delete_end_stim_entry.get()), height=0)
            self.current_trace = temp_data

        #insert into the intermediate traces dictionary of the data frame for the current index
        self.data_df['intermediate_traces'][index]['stim_removed_trace'] = self.current_trace
        self.data_df['current_trace_key'].at[index] = 'stim_removed_trace'

    def toggle_stim_removal(self):
        if self.stim_removal_check.get():
            self.analysis_steps_list.append('stim_removed_trace')
            self.analysis_steps_entry.insert(tk.END, 'stim_removed_trace,')
            self.current_trace_key_dropdown['values'] = self.analysis_steps_list
            #set the current trace key in the dropdown to offset_trace
            self.current_trace_key_dropdown.set('stim_removed_trace')
        elif self.stim_removal_automated_check.get():
            self.analysis_steps_list.append('stim_removed_trace')
            self.analysis_steps_entry.insert(tk.END, 'stim_removed_trace,')
            self.current_trace_key_dropdown['values'] = self.analysis_steps_list
            #set the current trace key in the dropdown to offset_trace
            self.current_trace_key_dropdown.set('stim_removed_trace')
        else:
            if 'stim_removed_trace' in self.analysis_steps_list:
                self.analysis_steps_list.remove('stim_removed_trace')
                self.analysis_steps_entry.delete(0, tk.END)
                for element in self.analysis_steps_list:
                    self.analysis_steps_entry.insert(tk.END, f'{element},')

        self.apply_analysis_steps()
        self.update_plot() 
    
    def apply_noise_removal(self):
        index = self.current_index
        if self.data_df['current_trace_key'][index] == 'raw':
            self.current_trace = self.data_df['sweep'][index]

        else:
            self.current_trace = self.data_df['intermediate_traces'][index][self.data_df['current_trace_key'][index]]
            
        noise_times = self.noise_times_entry.get()
        delete_duration_list = self.delete_noise_duration_entry.get() 

        #if there are multiple noise times and delete durations
        if ',' in noise_times:
            noise_times = noise_times.split(',')
            delete_duration_list = delete_duration_list.split(',')
            noise_times = [int(noise_time) for noise_time in noise_times]
            delete_duration_list = [int(delete_duration) for delete_duration in delete_duration_list]
        
        else:
            noise_times = int(noise_times)
            delete_duration_list = int(delete_duration_list)

        print(f'Noise Times: {noise_times}, Delete Duration: {delete_duration_list}')

        # remove_noise(data, noise_times, acquisition_frequency, delete_noise_duration)
        temp_data = remove_noise(self.current_trace, noise_times, self.data_df['acquisition_frequency'][index], delete_duration_list)
        self.data_df['intermediate_traces'][index]['noise_removed_trace'] = temp_data
        self.current_trace = temp_data
        
        self.data_df['intermediate_traces'][index]['noise_removed_trace'] = self.current_trace
        self.data_df['current_trace_key'].at[index] = 'noise_removed_trace'

    def toggle_noise_removal(self):
        if self.noise_removal_check.get():
            self.analysis_steps_list.append('noise_removed_trace')
            self.analysis_steps_entry.insert(tk.END, 'noise_removed_trace,')
            self.current_trace_key_dropdown['values'] = self.analysis_steps_list
            #set the current trace key in the dropdown to offset_trace
            self.current_trace_key_dropdown.set('noise_removed_trace')
        
        else:
            if 'noise_removed_trace' in self.analysis_steps_list:
                self.analysis_steps_list.remove('noise_removed_trace')
                self.analysis_steps_entry.delete(0, tk.END)
                #remove the element from the analysis steps entry
                for element in self.analysis_steps_list:
                    self.analysis_steps_entry.insert(tk.END, f'{element},')
                #remove the element from the current trace key dropdown
                self.current_trace_key_dropdown['values'] = self.analysis_steps_list

        self.apply_analysis_steps()
        self.update_plot() 
    
    def apply_interpolated_spikes(self):
        index = self.current_index

        if self.data_df['current_trace_key'][index] == 'raw':
            self.current_trace = self.data_df['sweep'][index]

        else:
            self.current_trace = self.data_df['intermediate_traces'][index][self.data_df['current_trace_key'][index]]

        dt = self.data_df['dt'][index]
        acquisition_frequency = self.data_df['acquisition_frequency'][index]
        time = np.arange(0, len(self.current_trace) * dt, dt)
        spike_width = float(self.spike_width_entry.get())
        AP_peak_window_start_interp = float(self.AP_peak_window_start_entry_interp.get()) # 1
        AP_peak_window_end_interp = float(self.AP_peak_window_end_entry_interp.get()) # 2
        
        if self.interpolated_spikes_check_old_method.get() == True:
            temp_data = interp_spikes_old(AP_peak_window_start_interp, AP_peak_window_end_interp, spike_width, acquisition_frequency, self.current_trace, time) 
        
        else:
            temp_data = interp_spikes(AP_peak_window_start_interp, AP_peak_window_end_interp, spike_width, acquisition_frequency, self.current_trace, time) 
        
        self.current_trace = temp_data
        self.data_df['intermediate_traces'][index]['interpolated_spikes_trace'] = self.current_trace
        self.data_df['current_trace_key'].at[index] = 'interpolated_spikes_trace'

    def toggle_interpolated_spikes(self):
        if self.interpolated_spikes_check.get():
            self.analysis_steps_list.append('interpolated_spikes_trace')
            self.analysis_steps_entry.insert(tk.END, 'interpolated_spikes_trace,')
            self.current_trace_key_dropdown['values'] = self.analysis_steps_list
            #set the current trace key in the dropdown to offset_trace
            self.current_trace_key_dropdown.set('interpolated_spikes_trace')
        
        elif self.interpolated_spikes_check_old_method.get():
            self.analysis_steps_list.append('interpolated_spikes_trace')
            self.analysis_steps_entry.insert(tk.END, 'interpolated_spikes_trace,')
            self.current_trace_key_dropdown['values'] = self.analysis_steps_list
            #set the current trace key in the dropdown to offset_trace
            self.current_trace_key_dropdown.set('interpolated_spikes_trace')
        
        else:
            if 'interpolated_spikes_trace' in self.analysis_steps_list:
                self.analysis_steps_list.remove('interpolated_spikes_trace')
                self.analysis_steps_entry.delete(0, tk.END)
                for element in self.analysis_steps_list:
                    self.analysis_steps_entry.insert(tk.END, f'{element},')

        self.apply_analysis_steps()
        self.update_plot() 
    
    def uncheck_processing_steps(self):
        self.stim_removal_check.set(False)
        self.stim_removal_automated_check.set(False)
        self.noise_removal_check.set(False)
        self.interpolated_spikes_check.set(False)
        self.offset_traces_check.set(False)
        self.partition_traces_check.set(False)
        self.analyze_EPSP_peaks_check.set(False)
        self.plateau_area_check.set(False)
        self.AP_AHP_analysis_check.set(False)

    def clear_processing_steps(self):
        #clear all of the intermediate traces and the current trace key
        index = self.current_index

        # # trace_keys_list = list(self.data_df['intermediate_traces'][index].keys())
        # self.data_df['analysis_dict'] = [{} for _ in range(len(self.data_df))]
        # self.data_df['intermediate_traces'] = [{} for _ in range(len(self.data_df))]
        # self.data_df['current_trace_key'] = ['raw' for _ in range(len(self.data_df))]

        self.analysis_steps_list = []
        self.analysis_steps_entry.delete(0, tk.END)
        self.analysis_steps_entry.insert(0, 'raw,')
        self.current_trace_key_dropdown['values'] = self.analysis_steps_list
        self.current_trace_key_dropdown.set('raw')

        #clear the analysis dictionary for the current index
        self.data_df['analysis_dict'].at[index] = {}
        self.data_df['intermediate_traces'].at[index] = {}
        self.data_df['current_trace_key'].at[index] = 'raw'
        self.analysis_types = {}

        self.stim_removal_check.set(False)
        self.stim_removal_automated_check.set(False)
        self.noise_removal_check.set(False)
        self.interpolated_spikes_check.set(False)
        self.offset_traces_check.set(False)
        self.partition_traces_check.set(False)
        self.analyze_EPSP_peaks_check.set(False)
        self.plateau_area_check.set(False)
        self.AP_AHP_analysis_check.set(False)

        self.update_plot()

    #Save the data to a pickle file
    def save_data(self):
        try:
            data_pickle_name = simpledialog.askstring("Data Pickle File", "Enter a name (date_cell):") 
            save_path = simpledialog.askstring("Save Path", "Enter a save path (e.g. /content/drive/My Drive/):")
            if data_pickle_name is not None and save_path is not None:
                data_df = self.data_df  
                data_df.to_pickle(f'{save_path}/{data_pickle_name}.pkl')
        except Exception as e:
            print("Error saving data:", e)
    
    def open_new_data(self):
        try:
            self.data_df = self.open_file()
            self.fill_metadata()
        
            #clear the analysis steps and the current trace key
            self.analysis_steps_list = []
            self.analysis_steps_entry.delete(0, tk.END)
            self.analysis_steps_entry.insert(0, 'raw,')

            self.current_trace_key_dropdown['values'] = self.analysis_steps_list
            self.current_trace_key_dropdown.set('raw')

            self.data_df['analysis_dict'] = [{} for _ in range(len(self.data_df))]
            self.data_df['intermediate_traces'] = [{} for _ in range(len(self.data_df))]
            self.data_df['current_trace_key'] = ['raw' for _ in range(len(self.data_df))]
            
        except Exception as e:
            print(f"Error opening new data: {e}")
    
    def open_saved_data(self):
        try:
            data_pickle_name = simpledialog.askstring("Data Pickle File path", "Enter the name of the data pickle file:") 
            if data_pickle_name is not None:
                self.data_df = pd.read_pickle(data_pickle_name)
                self.fill_metadata()

        except Exception as e:
            print("Error opening saved data:", e)