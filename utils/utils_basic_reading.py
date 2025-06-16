# -*- coding: utf-8 -*-
"""
Created on Mon Mar  3 16:38:27 2025

@author: usouu
"""
import os
import h5py
import scipy
import numpy as np
import pandas as pd

# %% Basic File Reading Functions
def read_txt(path_file):
    """
    Reads a text file and returns its content as a Pandas DataFrame.
    
    Parameters:
    - path_file (str): Path to the text file.
    
    Returns:
    - pd.DataFrame: DataFrame containing the parsed text data.
    """
    return pd.read_csv(path_file, sep=r'\s+', engine='python')

def read_xlsx(path_file):
    xls = pd.ExcelFile(path_file)
    dfs = {sheet: xls.parse(sheet) for sheet in xls.sheet_names}
    return dfs

def read_hdf5(path_file):
    """
    Reads an HDF5 file and returns its contents as a dictionary.
    
    Parameters:
    - path_file (str): Path to the HDF5 file.
    
    Returns:
    - dict: Parsed data from the HDF5 file.
    
    Raises:
    - FileNotFoundError: If the file does not exist.
    - TypeError: If the file is not a valid HDF5 format.
    """
    if not os.path.exists(path_file):
        raise FileNotFoundError(f"File not found: {path_file}")

    try:
        with h5py.File(path_file, 'r') as f:
            return {key: simplify_mat_structure(f[key]) for key in f.keys()}
    except OSError:
        raise TypeError(f"File '{path_file}' is not in HDF5 format.")

def read_mat(path_file, simplify=True):
    """
    Reads a MATLAB .mat file, supporting both HDF5 and older formats.
    
    Parameters:
    - path_file (str): Path to the .mat file.
    - simplify (bool): Whether to simplify the data structure (default: True).
    
    Returns:
    - dict: Parsed MATLAB file data.
    
    Raises:
    - FileNotFoundError: If the file does not exist.
    - TypeError: If the file format is invalid.
    """
    if not os.path.exists(path_file):
        raise FileNotFoundError(f"File not found: {path_file}")
    
    try:
        # Attempt to read as HDF5 format
        with h5py.File(path_file, 'r') as f:
            return {key: simplify_mat_structure(f[key]) for key in f.keys()} if simplify else f
    except OSError:
        try:
            # Read as non-HDF5 .mat file
            mat_data = scipy.io.loadmat(path_file, squeeze_me=simplify, struct_as_record=not simplify)
            return {key: simplify_mat_structure(value) for key, value in mat_data.items() if not key.startswith('_')} if simplify else mat_data
        except Exception as e:
            raise TypeError(f"Failed to read '{path_file}': {e}")

def simplify_mat_structure(data):
    """
    Recursively processes and simplifies MATLAB data structures.
    
    Converts:
    - HDF5 datasets to NumPy arrays or scalars.
    - HDF5 groups to Python dictionaries.
    - MATLAB structs to Python dictionaries.
    - Cell arrays to Python lists.
    - NumPy arrays are squeezed to remove unnecessary dimensions.
    
    Parameters:
    - data: Input data (HDF5, MATLAB struct, NumPy array, etc.).
    
    Returns:
    - Simplified Python data structure.
    """
    if isinstance(data, h5py.Dataset):
        return data[()]
    elif isinstance(data, h5py.Group):
        return {key: simplify_mat_structure(data[key]) for key in data.keys()}
    elif isinstance(data, scipy.io.matlab.mat_struct):
        return {field: simplify_mat_structure(getattr(data, field)) for field in data._fieldnames}
    elif isinstance(data, np.ndarray):
        if data.dtype == 'object':
            return [simplify_mat_structure(item) for item in data]
        return np.squeeze(data)
    return data

# %% Tools
import re

def get_last_number(text):
    matches = re.findall(r'\d+', text)
    return int(matches[-1]) if matches else None