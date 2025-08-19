# -*- coding: utf-8 -*-
"""
Created on Sat Mar  1 00:17:25 2025

@author: 18307
"""

import os

from . import utils_basic_reading

# %% Read Feature Functions
def read_cfs(dataset, identifier, feature, band='joint', file_type='.h5'):
    """
    Reads channel feature data (CFS) from a file (.h5 or .mat).

    Parameters:
    - dataset (str): Dataset name (e.g., 'SEED').
    - identifier (str): Subject or experiment identifier.
    - feature (str): Feature type (e.g., 'power', 'entropy').
    - band (str): Frequency band (default: 'joint').
    - file_type (str): File extension indicating format, either '.h5' or '.mat'.

    Returns:
    - dict: Parsed CFS data for the specified band. If the band is not found, returns an empty dict.
    
    Raises:
    - ValueError: If the specified file_type is unsupported.
    - FileNotFoundError: If the file does not exist.
    """
    dataset = dataset.upper()
    identifier = identifier.lower()
    feature = feature.lower()
    band = band.lower()

    base_path = os.path.abspath(os.path.join(os.getcwd(), "../.."))
    base_dir = os.path.join(base_path, 'Research_Data', dataset, 'channel features')

    if file_type == '.h5':
        path_file = os.path.join(base_dir, f'{feature}_h5', f"{identifier}.h5")
        cfs_data = utils_basic_reading.read_hdf5(path_file)
    elif file_type == '.mat':
        path_file = os.path.join(base_dir, f'{feature}_mat', f"{identifier}.mat")
        cfs_data = utils_basic_reading.read_mat(path_file)
    else:
        raise ValueError(f"Unsupported file_type: {file_type}. Supported types are '.h5' and '.mat'.")

    return cfs_data if band == 'joint' else cfs_data.get(band, {})

def read_fcs(dataset, identifier, feature, band='joint', file_type='.h5'):
    """
    Reads functional connectivity (FCS) data from a file (HDF5 or MAT format).

    Parameters:
    - dataset (str): Dataset name (e.g., 'SEED').
    - identifier (str): Subject or experiment identifier.
    - feature (str): Feature type (e.g., 'pcc', 'pli').
    - band (str): Frequency band to extract (default: 'joint').
    - file_type (str): File extension indicating format, either '.h5' or '.mat'.

    Returns:
    - dict: FCS data for the specified band. If the band is not found, returns an empty dict.
    
    Raises:
    - ValueError: If the specified file_type is unsupported.
    - FileNotFoundError: If the corresponding file does not exist.
    """
    dataset = dataset.upper()
    identifier = identifier.lower()
    feature = feature.lower()
    band = band.lower()

    base_path = os.path.abspath(os.path.join(os.getcwd(), "../.."))
    base_dir = os.path.join(base_path, 'Research_Data', dataset, 'functional connectivity')

    if file_type == '.h5':
        path_file = os.path.join(base_dir, f'{feature}_h5', f"{identifier}.h5")
        fcs_data = utils_basic_reading.read_hdf5(path_file)
    elif file_type == '.mat':
        path_file = os.path.join(base_dir, f'{feature}_mat', f"{identifier}.mat")
        fcs_data = utils_basic_reading.read_mat(path_file)
    else:
        raise ValueError(f"Unsupported file_type: {file_type}. Supported types are '.h5' and '.mat'.")

    return fcs_data if band == 'joint' else fcs_data.get(band, {})

def read_fcs_global_average(dataset, feature, band='joint', sub_range=range(1, 16)):
    dataset, feature, band = dataset.upper(), feature.lower(), band.lower()
    
    if dataset == 'SEED':
        total_subjects = 15
    elif dataset == 'DREAMER':
        total_subjects = 23
    
    path_parent_parent = os.path.dirname(os.path.dirname(os.getcwd()))
    path_file = os.path.join(path_parent_parent, 'Research_Data', dataset, 'functional connectivity', 
                             f'{feature}_h5', f'global_averaged_{sub_range.stop-1}_{total_subjects}.h5')
    fcs_temp = utils_basic_reading.read_hdf5(path_file)
    return fcs_temp if band == 'joint' else fcs_temp.get(band, {})

# %% Read Labels Functions
def read_labels(dataset, header=False):
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
    return utils_basic_reading.read_txt(path_labels, header)

# %% Read Distributions
def read_distribution(dataset, mapping_method='auto', header=True):
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
        ('SEED', 'auto'): "biosemi64_62_channels_original_distribution.txt",
        ('SEED', 'manual'): "biosemi64_62_channels_manual_distribution.txt",
        ('DREAMER', 'auto'): "biosemi64_14_channels_original_distribution.txt",
        ('DREAMER', 'manual'): "biosemi64_14_channels_manual_distribution.txt",
    }

    path_distr = os.path.join(base_path, file_map[(dataset, mapping_method)])

    # Check if file exists before reading
    if not os.path.exists(path_distr):
        raise FileNotFoundError(f"Distribution file not found: {path_distr}. Check dataset and mapping method.")

    # Read and return the distribution file
    distribution = utils_basic_reading.read_txt(path_distr, header)
    
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