import numpy as np
import pandas as pd 
import os
import re
from fnmatch import fnmatch
from pathlib import Path
import matplotlib.pyplot as plt

import igor2 as igor
from igor.packed import load as loadpxp
from igor.record.wave import WaveRecord

from Intracellular_analysis_functions_05032024 import load_NM_data_create_sweeps 

from tkinter import simpledialog
import tkinter as tk
from tkinter import simpledialog
ROOT = tk.Tk()
# conda activate /opt/anaconda3/envs/Ephys_Analysis --> Deprecated - need to reinstall older python (3.9 or 3.7) or pxp loader doesnt work
#use base with older python instead!

config_file_path = '/Users/samgritz/Desktop/Rutgers/Milstein_Lab/Code/Rutgers-Neuroscience-PhD/Ephys_Analysis/Updated_Code/Intracellular_Analysis/dataframe_defaults.yaml'

myDir = simpledialog.askstring("File Path", "Enter a directory Path (Please make sure no log0.pxp and backup.pxp files exist and that files end in index not date):")
save_dir = simpledialog.askstring("File Path", "Enter a directory Path to save the dataframes:")


def find_folders_with_pattern_and_date(myDir, pattern):
    matching_folders = []
    date_pattern = r'\d{2}\d{2}\d{4}'  # Date pattern (MMDDYYYY)
    for path, subdirs, files in os.walk(myDir):
        for subdir in subdirs:
            if pattern in subdir:
                date_match = re.search(date_pattern, subdir)
                if date_match:
                    date = date_match.group()
                else:
                    date = "Date not found in folder name"
                matching_folders.append((os.path.join(path, subdir), date))
    return matching_folders

# Define the directory to search and the pattern to match
pattern_top = "CA1_Patch"

# Find matching folders and their dates
matching_folders_with_dates = find_folders_with_pattern_and_date(myDir, pattern_top)
pattern = "*.pxp"

pxp_files = {}
save_files = {}

for folder, date in matching_folders_with_dates:
    folder_name = os.path.basename(folder)
    pxp_files[folder_name] = []
    save_files[folder_name] = []
    for path, subdirs, files in os.walk(folder):
        pxp_files[folder_name].append({})
        save_files[folder_name].append({})
        for subdir in subdirs:
            pxp_files[folder_name][0][subdir] = []
            save_files[folder_name][0][subdir] = []
            subdir_path = os.path.join(path, subdir)
            for files in os.listdir(subdir_path):
                # Skip backup files and files matching the pattern
                if fnmatch(files, pattern) and '_backup.pxp' not in files:
                    pxp_files[folder_name][-1][subdir].append(os.path.join(subdir_path, files))
            
            # Build list of valid numeric indices
            index_list = []
            valid_files = []
            for pxp_file in pxp_files[folder_name][-1][subdir]:
                index = str(pxp_file[-7:-4])
                # Only include if index can be converted to integer
                try:
                    int(index)
                    index_list.append(index)
                    valid_files.append(pxp_file)
                except ValueError:
                    print(f"Skipping file with non-numeric index: {pxp_file}")
                    continue
            
            # Sort by numeric index
            sorted_index_list = sorted(index_list, key=int, reverse=False)
            sorted_files = []
            for index in sorted_index_list:
                current_sorted_file = f'/{valid_files[0][1:-7]}{index}.pxp'
                sorted_files.append(current_sorted_file)
            
            pxp_files[folder_name][-1][subdir] = sorted_files
            
            master_df = pd.DataFrame()
            for file in pxp_files[folder_name][-1][subdir]:
                try:
                    sweeps, stim_records, stim_commands, acquisition_metadata_dict, stimulus_metadata_dict, analysis_dict =  load_NM_data_create_sweeps(file, yaml_file_path=config_file_path)
                    print(acquisition_metadata_dict)
                    for index, sweep in enumerate(sweeps):
                        if index == 0:
                            initial_global_wall_clock = acquisition_metadata_dict['global wall clock']
                        sweep_duration = acquisition_metadata_dict['samples per wave'] * acquisition_metadata_dict['dt'] / 1000  # seconds
                        inter_trial_delay = acquisition_metadata_dict['inter trial interval (ms)'] / 1000  # seconds
                        current_global_wall_clock = initial_global_wall_clock + index * (sweep_duration + inter_trial_delay)
                        dict_metadata = {}

                        try:
                            dict_metadata['experiment_index'] = int(file[-7:-4])
                            dict_metadata['sweep_index'] = index
                            dict_metadata['cell_number'] = file[-10:-8]
                            dict_metadata['file_path']= file
                            dict_metadata['stim_type'] = acquisition_metadata_dict['stim type']
                            dict_metadata['experiment_description'] = acquisition_metadata_dict['experiment description']
                            dict_metadata['date'] = acquisition_metadata_dict['date']
                            dict_metadata['dt'] = acquisition_metadata_dict['dt']
                            dict_metadata['acquisition_frequency'] = acquisition_metadata_dict['acquisition_frequency']
                            dict_metadata['sweep_duration'] = acquisition_metadata_dict['total recording window (ms)']
                            dict_metadata['total_experiment_time'] = acquisition_metadata_dict['total experiment time (s)']
                            dict_metadata['inter_trial_interval'] = acquisition_metadata_dict['inter trial interval (ms)']
                            dict_metadata['samples_per_wave'] = acquisition_metadata_dict['samples per wave']
                            dict_metadata['global_wall_clock'] = current_global_wall_clock
                            dict_metadata['experiment_start_time'] = acquisition_metadata_dict['experiment start time']
                            dict_metadata['experiment_end_time'] = acquisition_metadata_dict['experiment end time']
                            dict_metadata['sweep'] = sweep

                            if (acquisition_metadata_dict['stim type'] == 'Plateau_Stim') or (acquisition_metadata_dict['stim type'] == 'EPSP_stim'):
                                if isinstance(stim_commands, list):
                                    if stim_commands is not None and stim_records is not None:
                                        dict_metadata['stim_record'] = stim_records
                                        dict_metadata['stim_command'] = stim_commands
                                    else:
                                        dict_metadata['stim_record'] = None
                                        dict_metadata['stim_command'] = None
                                else:
                                    if stim_commands is not None and stim_records is not None:
                                        dict_metadata['stim_record'] = stim_records
                                        dict_metadata['stim_command'] = stim_commands
                                    else:
                                        dict_metadata['stim_record'] = None
                                        dict_metadata['stim_command'] = None
                            else:
                                if isinstance(stim_commands[index], list):
                                    if any(stim_commands[index]) and any(stim_records[index]):
                                        dict_metadata['stim_record'] = stim_records[index]
                                        dict_metadata['stim_command'] = stim_commands[index]
                                    else:
                                        dict_metadata['stim_record'] = None
                                        dict_metadata['stim_command'] = None
                                else:
                                    if stim_commands[index] is not None and stim_records[index] is not None:
                                        dict_metadata['stim_record'] = stim_records[index]
                                        dict_metadata['stim_command'] = stim_commands[index]
                                    else:
                                        dict_metadata['stim_record'] = None
                                        dict_metadata['stim_command'] = None

                            dict_metadata['analysis_dict'] = analysis_dict
                            dict_metadata['stimulus_metadata_dict'] = stimulus_metadata_dict
                        except:
                            print(f'Error processing {file}, sweep {index}')
                            continue
                        
                        this_df_row = pd.DataFrame([dict_metadata])
                        master_df = pd.concat([master_df, this_df_row], ignore_index=True)

                except Exception as e:
                    print(f'Error with {file}: {e}')
                    continue  # Continue to the next file
            
            # Save the DataFrame to a pickle file
            save_files[folder_name][-1][subdir] = f'{save_dir}/{folder_name}_{subdir}.pkl'
            master_df.to_pickle(save_files[folder_name][-1][subdir])
