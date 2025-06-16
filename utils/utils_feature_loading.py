# -*- coding: utf-8 -*-
"""
Created on Sat Mar  1 00:17:25 2025

@author: 18307
"""

import os

from . import utils_basic_reading

# %% Read Feature Functions
def read_cfs(dataset, identifier, feature, band='joint'):
    """
    Reads channel feature data (CFS) from an HDF5 file.
    
    Parameters:
    - dataset (str): Dataset name (e.g., 'SEED').
    - identifier (str): Subject/Experiment identifier.
    - feature (str): Feature type.
    - band (str): Frequency band (default: 'joint').
    
    Returns:
    - dict: Parsed CFS data.
    """
    dataset, identifier, feature, band = dataset.upper(), identifier.lower(), feature.lower(), band.lower()
    path_parent_parent = os.path.dirname(os.path.dirname(os.getcwd()))
    path_file = os.path.join(path_parent_parent, 'Research_Data', dataset, 'channel features', f'{feature}_h5', f"{identifier}.h5")
    cfs_temp = utils_basic_reading.read_hdf5(path_file)
    return cfs_temp if band == 'joint' else cfs_temp.get(band, {})

def read_fcs(dataset, identifier, feature, band='joint'):
    """
    Reads functional connectivity data (FCS) from an HDF5 file.
    
    Parameters:
    - dataset (str): Dataset name (e.g., 'SEED').
    - identifier (str): Subject/Experiment identifier.
    - feature (str): Feature type.
    - band (str): Frequency band (default: 'joint').
    
    Returns:
    - dict: Parsed FCS data.
    """
    dataset, identifier, feature, band = dataset.upper(), identifier.lower(), feature.lower(), band.lower()
    path_parent_parent = os.path.dirname(os.path.dirname(os.getcwd()))
    path_file = os.path.join(path_parent_parent, 'Research_Data', dataset, 'functional connectivity', f'{feature}_h5', f"{identifier}.h5")
    fcs_temp = utils_basic_reading.read_hdf5(path_file)
    return fcs_temp if band == 'joint' else fcs_temp.get(band, {})

def read_fcs_global_average(dataset, feature, band='joint', source='mat'):
    dataset, feature, band, source = dataset.upper(), feature.lower(), band.lower(), source.lower()
    path_parent_parent = os.path.dirname(os.path.dirname(os.getcwd()))
    path_file = os.path.join(path_parent_parent, 'Research_Data', dataset, 'functional connectivity', 
                             'global_averaged_h5', f'fc_global_averaged_{feature}_{source}.h5')
    fcs_temp = utils_basic_reading.read_hdf5(path_file)
    return fcs_temp if band == 'joint' else fcs_temp.get(band, {})

def read_fcs_mat(dataset, identifier, feature, band='joint'):
    dataset, identifier, feature, band = dataset.upper(), identifier.lower(), feature.lower(), band.lower()
    path_grandparent = os.path.abspath(os.path.join(os.getcwd(), "../.."))
    path_file = os.path.join(path_grandparent, 'Research_Data', dataset, 'functional connectivity', f'{feature}_mat', f'{identifier}.mat')
    fcs_mat = utils_basic_reading.read_mat(path_file)
    
    return fcs_mat if band == 'joint' else fcs_mat.get(band, {})

# %% Read Labels Functions
def read_labels(dataset):
    """
    Reads emotion labels for a specified dataset.
    
    Parameters:
    - dataset (str): The dataset name (e.g., 'SEED', 'DREAMER').
    
    Returns:
    - pd.DataFrame: DataFrame containing label data.
    
    Raises:
    - ValueError: If the dataset is not supported.
    """
    path_parent_parent = os.path.dirname(os.path.dirname(os.getcwd()))
    if dataset.lower() == 'seed':
        path_labels = os.path.join(path_parent_parent, 'Research_Data', 'SEED', 'labels', 'labels_seed.txt')
    elif dataset.lower() == 'dreamer':
        path_labels = os.path.join(path_parent_parent, 'Research_Data', 'DREAMER', 'labels', 'labels_dreamer.txt')
    else:
        raise ValueError('Currently only support SEED and DREAMER')
    return utils_basic_reading.read_txt(path_labels)

# %% Read Distributions
def read_distribution(dataset, mapping_method='auto'):
    """
    Read the electrode distribution file for a given EEG dataset and mapping method.

    Parameters:
    dataset (str): The EEG dataset name ('SEED' or 'DREAMER').
    mapping_method (str): The mapping method ('auto' for automatic mapping, 'manual' for manual mapping).
                          Default is 'auto'.

    Returns:
    list or pandas.DataFrame:
        - The parsed electrode distribution data, depending on how `utils_basic_reading.read_txt` processes it.

    Raises:
    ValueError: If the dataset or mapping method is invalid.
    FileNotFoundError: If the distribution file does not exist.
    """
    # Define valid parameters
    valid_datasets = ['SEED', 'DREAMER']
    valid_mapping_methods = ['auto', 'manual']

    # Normalize inputs
    dataset = dataset.upper()
    mapping_method = mapping_method.lower()

    # Validate inputs
    if dataset not in valid_datasets:
        raise ValueError(f"Invalid dataset: {dataset}. Choose from {', '.join(valid_datasets)}.")
    
    if mapping_method not in valid_mapping_methods:
        raise ValueError(f"Invalid mapping method: {mapping_method}. Choose from {', '.join(valid_mapping_methods)}.")

    # Define the base path
    base_path = os.path.abspath(os.path.join(os.getcwd(), "../../Research_Data", dataset, "electrode distribution"))

    # Determine the correct file based on dataset and mapping method
    file_map = {
        ('SEED', 'auto'): "biosemi62_64_channels_original_distribution.txt",
        ('SEED', 'manual'): "biosemi62_64_channels_manual_distribution.txt",
        ('DREAMER', 'auto'): "biosemi62_14_channels_original_distribution.txt",
        ('DREAMER', 'manual'): "biosemi62_14_channels_manual_distribution.txt",
    }

    path_distr = os.path.join(base_path, file_map[(dataset, mapping_method)])

    # Check if file exists before reading
    if not os.path.exists(path_distr):
        raise FileNotFoundError(f"Distribution file not found: {path_distr}. Check dataset and mapping method.")

    # Read and return the distribution file
    distribution = utils_basic_reading.read_txt(path_distr)
    
    return distribution

# %% Read Channel Rankings
def read_ranking(ranking='all'):
    """
    Read electrode ranking information from a predefined Excel file.
    
    Parameters:
    ranking (str): The type of ranking to return. Options:
                  - 'label_driven_mi'
                  - 'data_driven_mi'
                  - 'data_driven_pcc' 
                  - 'data_driven_plv'
                  - 'all': returns all rankings (default)
    
    Returns:
    pandas.DataFrame or pandas.Series: The requested ranking data.
    
    Raises:
    ValueError: If an invalid ranking type is specified.
    FileNotFoundError: If the ranking file cannot be found.
    """
    import os
    
    # Valid ranking options
    valid_rankings = ['label_driven_mi', 'data_driven_mi', 'data_driven_pcc', 'data_driven_plv', 'all']
    
    # Validate input
    if ranking not in valid_rankings:
        raise ValueError(f"Invalid ranking type: '{ranking}'. Choose from {', '.join(valid_rankings)}.")
    
    # Define path
    path_current = os.getcwd()
    path_ranking = os.path.join(path_current, 'Distribution', 'electrodes_ranking.xlsx')
    
    # Check if file exists
    if not os.path.exists(path_ranking):
        raise FileNotFoundError(f"Ranking file not found at: {path_ranking}")
    
    try:
        # Read xlsx; electrodes ranking
        if ranking == 'all':
            result = utils_basic_reading.read_xlsx(path_ranking)
        else:
            result = utils_basic_reading.read_xlsx(path_ranking)[ranking]
            
        return result
    
    except KeyError:
        raise KeyError(f"Ranking type '{ranking}' not found in the Excel file.")
    except Exception as e:
        raise Exception(f"Error reading ranking data: {str(e)}")

# %% Example Usage
if __name__ == "__main__":
    # %% cfs
    dataset, experiment_sample, feature_sample, freq_sample = 'seed', 'sub1ex1', 'de_LDS', 'joint'
    seed_cfs_sample = read_cfs(dataset, experiment_sample, feature_sample, freq_sample)
    
    # %% fcs
    dataset, experiment_sample, feature_sample, freq_sample = 'seed', 'sub1ex1', 'pcc', 'joint'
    seed_fcs_sample_seed = read_fcs(dataset, experiment_sample, feature_sample, freq_sample)
    
    dataset, experiment_sample = 'dreamer', 'sub1'
    seed_fcs_sample_dreamer = read_fcs(dataset, experiment_sample, feature_sample, freq_sample)
    
    # %% read labels
    labels_seed_ = read_labels('seed')
    labels_dreamer_ = read_labels('dreamer')

    # %% Read Distribution
    distribution = read_distribution(dataset='seed')
    
    # %% Read Ranking
    ranking = read_ranking()