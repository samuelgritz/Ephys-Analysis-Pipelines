__authors__ = 'Samuel Gritz'
import tkinter as tk
from tkinter import ttk 
from tkinter import simpledialog, filedialog, ttk, messagebox

import matplotlib.pyplot as plt
import numpy as np
import yaml
import os    
import click

from processing_gui_working_dev import GUI
from processing_gui_working_dev import *


#python Run_GUI.py --config_file_path /Users/samgritz/Desktop/Rutgers/Milstein_Lab/Code/Rutgers-Neuroscience-PhD/Ephys_Analysis/Updated_Code/Intracellular_Analysis/Default_Metadata_new_stims_dev_080724.yaml --interactive --debug

@click.command()
@click.option('--config_file_path', type=click.Path(exists=True), required=True,
              default='/Users/samgritz/Desktop/Rutgers/Milstein_Lab/Code/Rutgers-Neuroscience-PhD/Ephys_Analysis/Updated_Code/Intracellular_Analysis/Default_Metadata_new_stims_dev_080724.yaml')
@click.option('--debug', is_flag=True)
@click.option('--interactive', is_flag=True)
def main(config_file_path, debug, interactive):

    """
    Main function for running GUI for intracellular analysis.

    :param config_file_path: path to YAML configuration file
    :param debug: flag to print debug statements
    """
    config_dict = read_from_yaml(config_file_path) 
    gui = GUI(**config_dict)

    # if debug:
    #     print(network.data)

    if interactive:
        globals().update(locals())


if __name__ == "__main__":
    main(standalone_mode=False)