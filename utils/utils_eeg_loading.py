# -*- coding: utf-8 -*-
"""
Created on Mon Mar  3 02:14:56 2025

@author: 18307
"""

import os

import numpy as np
import pandas as pd

import mne

from . import utils_basic_reading

# %% Read Original EEG/.mat
def read_eeg_original_dataset(dataset, identifier=None, object_type='pandas_dataframe'):
    """
    Read original EEG data from specified dataset.
    
    Parameters:
    dataset (str): Dataset name ('SEED' or 'DREAMER').
    identifier (str, optional): Subject/session identifier, required for SEED dataset, ignored for DREAMER.
    object_type (str): Desired output format ('pandas_dataframe', 'numpy_array', 'mne', or 'fif').
    
    Returns:
    dict or mat object: The loaded EEG data in the specified format.
    
    Raises:
    ValueError: If dataset or object_type is invalid, or if identifier is missing when required.
    FileNotFoundError: If the expected file does not exist.
    """
    import os
    
    # Valid options
    valid_datasets = ['SEED', 'DREAMER']
    valid_object_types = ['pandas_dataframe', 'numpy_array', 'mne', 'fif']
    
    # Normalize inputs
    dataset = dataset.upper()
    object_type = object_type.lower()
    
    # Validate dataset and object type
    if dataset not in valid_datasets:
        raise ValueError(f"Invalid dataset: {dataset}. Choose from {', '.join(valid_datasets)}.")
    
    if object_type not in valid_object_types:
        raise ValueError(f"Invalid object type: {object_type}. Choose from {', '.join(valid_object_types)}.")
    
    # Dataset-specific validation
    if dataset == 'SEED' and not identifier:
        raise ValueError("Identifier parameter is required for SEED dataset.")
    
    # Normalize identifier if needed
    if identifier and dataset == 'SEED':
        identifier = identifier.lower()
    
    # Construct base path
    path_parent_parent = os.path.dirname(os.path.dirname(os.getcwd()))
    base_path = os.path.join(path_parent_parent, 'Research_Data', dataset, 'original eeg')
    
    try:
        if dataset == 'SEED':
            path_file = os.path.join(base_path, 'Preprocessed_EEG', f'{identifier}.mat')
            eeg = utils_basic_reading.read_mat(path_file)
        elif dataset == 'DREAMER':
            path_file = os.path.join(base_path, 'DREAMER.mat')
            eeg = utils_basic_reading.read_mat(path_file)['DREAMER']
        
        return eeg
    
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {path_file}. Check the path and file existence.")
    except Exception as e:
        raise Exception(f"Error reading {dataset} data: {str(e)}")

def read_and_parse_seed(identifier):
    """
    Read and parse SEED dataset EEG data from a .mat file.

    Parameters:
    identifier (str): Identifier for the subject/session.

    Returns:
    numpy.ndarray:
        - A concatenated NumPy array of EEG data across all available segments.

    Raises:
    FileNotFoundError: If the specified file does not exist.
    """
    # Construct file path
    path_base = os.path.abspath(os.path.join(os.getcwd(), "../../Research_Data/SEED/original eeg/Preprocessed_EEG"))
    path_data = os.path.join(path_base, f"{identifier}.mat")

    # Load data
    if not os.path.exists(path_data):
        raise FileNotFoundError(f"File not found: {path_data}. Check the identifier and directory structure.")

    data_temp = utils_basic_reading.read_mat(path_data)

    # Concatenate EEG segments
    data = np.hstack([data_temp[key] for key in data_temp])

    return data

def read_and_parse_dreamer(identifier):
    """
    Read and parse DREAMER dataset EEG data from a .mat file.

    Returns:
    tuple:
        - dict: Original DREAMER dataset structure.
        - list of numpy.ndarray: List of EEG data matrices (transposed) for each trial.
        - list: List of electrode names.

    Raises:
    FileNotFoundError: If the expected .mat file does not exist.
    """
    # Construct file path
    path_base = os.path.abspath(os.path.join(os.getcwd(), "../../Research_Data/DREAMER/original eeg"))
    path_data = os.path.join(path_base, "DREAMER.mat")

    # Load data
    if not os.path.exists(path_data):
        raise FileNotFoundError(f"File not found: {path_data}. Ensure the DREAMER dataset is correctly placed.")

    dreamer = utils_basic_reading.read_mat(path_data)

    # Extract EEG data
    eeg_list = [np.vstack(trial["EEG"]["stimuli"]) for trial in dreamer["DREAMER"]["Data"]]
    eeg_list_transposed = [matrix.T for matrix in eeg_list]
    eeg_dict = {i: eeg_list_transposed[i] for i in range(len(eeg_list_transposed))}
    
    # Extract specific EEG
    key = utils_basic_reading.get_last_number(identifier) - 1
    eeg = eeg_dict[key]
    
    return eeg

# %% Read Filtered EEG/.fif
def read_eeg_filtered(dataset, identifier, freq_band='joint', object_type='pandas_dataframe'):
    """
    Read filtered EEG data for the specified experiment and frequency band.

    Parameters:
    dataset (str): Dataset name (e.g., 'SEED', 'DREAMER').
    identifier (str): Identifier for the subject/session.
    freq_band (str): Frequency band to load ("alpha", "beta", "gamma", "delta", "theta", or "joint").
                     Default is "joint", which loads all bands.
    object_type (str): Desired output format: 'pandas_dataframe', 'numpy_array', or 'mne'.

    Returns:
    mne.io.Raw | dict | pandas.DataFrame | numpy.ndarray:
        - If 'mne', returns the MNE Raw object (or a dictionary of them for 'joint').
        - If 'pandas_dataframe', returns a DataFrame with EEG data.
        - If 'numpy_array', returns a NumPy array with EEG data.

    Raises:
    ValueError: If the specified frequency band is not valid.
    FileNotFoundError: If the expected file does not exist.
    """
    # Valid options
    valid_datasets = ['SEED', 'DREAMER']
    valid_bands = ['alpha', 'beta', 'gamma', 'delta', 'theta', 'joint']
    valid_object_types = ['pandas_dataframe', 'numpy_array', 'mne', 'fif']
    
    # Normalize inputs
    dataset = dataset.upper()
    identifier = identifier.lower()
    freq_band = freq_band.lower()
    object_type = object_type.lower()
    
    # Validate inputs
    if dataset not in valid_datasets:
        raise ValueError(f"Invalid dataset: {dataset}. Choose from {', '.join(valid_datasets)}.")
    
    if freq_band not in valid_bands:
        raise ValueError(f"Invalid frequency band: {freq_band}. Choose from {', '.join(valid_bands)}.")
    
    if object_type not in valid_object_types:
        raise ValueError(f"Invalid object type: {object_type}. Choose from {', '.join(valid_object_types)}.")
    
    # Construct base path
    path_parent_parent = os.path.dirname(os.path.dirname(os.getcwd()))
    base_path = os.path.join(path_parent_parent, 'Research_Data', dataset, 'original eeg', 'Filtered_EEG')
    
    # Function to process a single frequency band
    def process_band(band):
        file_path = os.path.join(base_path, f'{identifier}_{band.capitalize()}_eeg.fif')
        try:
            raw_data = mne.io.read_raw_fif(file_path, preload=True)
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {file_path}. Check the path and file existence.")
            
        if object_type == 'pandas_dataframe':
            return pd.DataFrame(raw_data.get_data(), index=raw_data.ch_names)
        elif object_type == 'numpy_array':
            return raw_data.get_data()
        else:  # Default to MNE Raw object / .fif object
            return raw_data
    
    # Handle joint vs. single band request
    if freq_band == 'joint':
        result = {}
        for band in ['alpha', 'beta', 'gamma', 'delta', 'theta']:
            result[band] = process_band(band)
        return result
    else:
        return process_band(freq_band)

# %% Example Usage
if __name__ == '__main__':
    # EEG from original dataset
    eeg_dreamer = read_eeg_original_dataset(dataset='dreamer', identifier=None)
    eeg_dreamer_ = read_and_parse_dreamer('sub1')
    eeg_seed_sample = read_eeg_original_dataset(dataset='seed', identifier='sub1ex1')
    eeg_seed_sample_ = read_and_parse_seed('sub1ex1')
    
    # Filtered EEG
    filtered_eeg_dreamer_sample1 = read_eeg_filtered(dataset='dreamer', identifier='sub1', freq_band='alpha')
    filtered_eeg_dreamer_sample2 = read_eeg_filtered(dataset='dreamer', identifier='sub1', freq_band='beta')
    filtered_eeg_seed_sample1 = read_eeg_filtered(dataset='seed', identifier='sub1ex1', freq_band='alpha')
    filtered_eeg_seed_sample2 = read_eeg_filtered(dataset='seed', identifier='sub1ex2', freq_band='beta')