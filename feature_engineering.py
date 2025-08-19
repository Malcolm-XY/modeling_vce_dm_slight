# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 23:15:11 2025

@author: 18307
"""

import os
import time
import h5py
import numpy as np
import matplotlib.pyplot as plt

import mne
from scipy.signal import hilbert

from utils import utils_feature_loading, utils_visualization, utils_eeg_loading

# %% Filter EEG
def filter_eeg(eeg, freq=128, verbose=False):
    """
    Filter raw EEG data into standard frequency bands using MNE.

    Parameters:
    eeg (numpy.ndarray): Raw EEG data array with shape (n_channels, n_samples).
    freq (int): Sampling frequency of the EEG data. Default is 128 Hz.
    verbose (bool): If True, prints progress messages. Default is False.

    Returns:
    dict:
        A dictionary where keys are frequency band names ("Delta", "Theta", "Alpha", "Beta", "Gamma")
        and values are the corresponding MNE Raw objects filtered to that band.
    """
    # Create MNE info structure and Raw object from the EEG array
    info = mne.create_info(ch_names=[f"Ch{i}" for i in range(eeg.shape[0])], sfreq=freq, ch_types='eeg')
    mne_eeg = mne.io.RawArray(eeg, info)
    
    # Define frequency bands
    freq_bands = {
        "Delta": (0.5, 4),
        "Theta": (4, 8),
        "Alpha": (8, 13),
        "Beta": (13, 30),
        "Gamma": (30, 50),
    }
    
    band_filtered_eeg = {}
    
    # Filter EEG data for each frequency band
    for band, (low_freq, high_freq) in freq_bands.items():
        filtered_eeg = mne_eeg.copy().filter(l_freq=low_freq, h_freq=high_freq, method="fir", phase="zero-double")
        band_filtered_eeg[band] = filtered_eeg
        if verbose:
            print(f"{band} band filtered: {low_freq}–{high_freq} Hz")
    
    return band_filtered_eeg

def filter_eeg_seed(identifier, freq=200, verbose=True, save=False):
    """
    Load, filter, and optionally save SEED dataset EEG data into frequency bands.

    Parameters:
    identifier (str): Identifier for the subject/session.
    freq (int): SEED: 200 Hz. DREAMER: 128 Hz.
    verbose (bool): If True, prints progress messages. Default is True.
    save (bool): If True, saves the filtered EEG data to disk. Default is False.

    Returns:
    dict:
        A dictionary where keys are frequency band names and values are the filtered MNE Raw objects.

    Raises:
    FileNotFoundError: If the SEED data file cannot be found.
    """
    # Load raw EEG data using the provided utility function
    eeg = utils_eeg_loading.read_and_parse_seed(identifier)
    
    # Construct the output folder path for filtered data
    base_path = os.path.abspath(os.path.join(os.getcwd(), "../../Research_Data/SEED/original eeg/Filtered_EEG"))
    os.makedirs(base_path, exist_ok=True)
    
    # Filter the EEG data into different frequency bands
    filtered_eeg_dict = filter_eeg(eeg, freq=freq, verbose=verbose)
    
    # Save filtered EEG data if requested
    if save:
        for band, filtered_eeg in filtered_eeg_dict.items():
            path_file = os.path.join(base_path, f"{identifier}_{band}_eeg.fif")
            filtered_eeg.save(path_file, overwrite=True)
            if verbose:
                print(f"Saved {band} band filtered EEG to {path_file}")
    
    return filtered_eeg_dict

def filter_eeg_dreamer(identifier, freq=128, verbose=True, save=False):
    """
    Load, filter, and optionally save DREAMER dataset EEG data into frequency bands.

    Parameters:
    identifier (str): Identifier for the trial/session.
    freq (int): SEED: 200 Hz. DREAMER: 128 Hz.
    verbose (bool): If True, prints progress messages. Default is True.
    save (bool): If True, saves the filtered EEG data to disk. Default is False.

    Returns:
    dict:
        A dictionary where keys are frequency band names and values are the filtered MNE Raw objects.

    Raises:
    FileNotFoundError: If the DREAMER data file cannot be found.
    """
    # Load raw EEG data using the provided utility function for DREAMER
    eeg = utils_eeg_loading.read_and_parse_dreamer(identifier)
    
    # Construct the output folder path for filtered data
    base_path = os.path.abspath(os.path.join(os.getcwd(), "../../Research_Data/DREAMER/original eeg/Filtered_EEG"))
    os.makedirs(base_path, exist_ok=True)
    
    # Filter the EEG data into different frequency bands
    filtered_eeg_dict = filter_eeg(eeg, freq=freq, verbose=verbose)
    
    # Save filtered EEG data if requested
    if save:
        for band, filtered_eeg in filtered_eeg_dict.items():
            path_file = os.path.join(base_path, f"{identifier}_{band}_eeg.fif")
            filtered_eeg.save(path_file, overwrite=True)
            if verbose:
                print(f"Saved {band} band filtered EEG to {path_file}")
    
    return filtered_eeg_dict

def filter_eeg_and_save_circle(dataset, subject_range, experiment_range=None, verbose=True, save=False):
    # Normalize parameters
    dataset = dataset.upper()

    valid_dataset = ['SEED', 'DREAMER']
    if dataset not in valid_dataset:
        raise ValueError(f"{dataset} is not a valid dataset. Valid datasets are: {valid_dataset}")

    if dataset == 'SEED' and subject_range is not None and experiment_range is not None:
        for subject in subject_range:
            for experiment in experiment_range:
                identifier = f'sub{subject}ex{experiment}'
                print(f"Processing: {identifier}.")
                filter_eeg_seed(identifier, verbose=verbose, save=save)
    elif dataset == 'DREAMER' and subject_range is not None and experiment_range is None:
        for subject in subject_range:
            print(f"Processing Subject: {subject}.")
            filter_eeg_dreamer(subject, verbose=verbose, save=save)
    else:
        raise ValueError("Error of unexpected subject or experiment range designation.")

# %% Feature Engineering
def compute_distance_matrix(dataset, projection_params=None, visualize=False, c='blue'):
    if projection_params is None:
        projection_params = {}

    proj_type = projection_params.get('type', '3d_euclidean')
    source = projection_params.get('source', 'auto')
    resolution = projection_params.get('resolution', None)

    dist = utils_feature_loading.read_distribution(dataset, source)
    ch_names = dist['channel']
    x, y, z = map(np.array, (dist['x'], dist['y'], dist['z']))
    coords3d = np.stack([x, y, z], axis=-1)

    coords2d, dist_mat = None, None

    if proj_type == '3d_euclidean':
        diff = coords3d[:, None, :] - coords3d[None, :, :]
        dist_mat = np.linalg.norm(diff, axis=-1)
        coords2d = np.stack([x, y], axis=-1)

    elif proj_type == '3d_spherical':
        unit = coords3d / np.linalg.norm(coords3d, axis=1, keepdims=True)
        dot = np.clip(unit @ unit.T, -1.0, 1.0)
        dist_mat = np.arccos(dot)
        coords2d = np.stack([x, y], axis=-1)

    elif proj_type == '2d_flat':
        coords2d = np.stack([x, y], axis=-1)
        diff = coords2d[:, None, :] - coords2d[None, :, :]
        dist_mat = np.linalg.norm(diff, axis=-1)

    elif proj_type == '2d_stereographic':
        f = projection_params.get('focal_length', 1.0)
        m = projection_params.get('max_scaling', 5.0)
        z_norm = (z - z.min()) / (z.max() - z.min() + 1e-6)
        scale = f / (f - z_norm + 1e-6)
        scale = np.clip(scale, 0, m)
        coords2d = np.stack([x * scale, y * scale], axis=-1)
        diff = coords2d[:, None, :] - coords2d[None, :, :]
        dist_mat = np.linalg.norm(diff, axis=-1)

    elif proj_type == '2d_azimuthal':
        unit = coords3d / np.linalg.norm(coords3d, axis=1, keepdims=True)
        xu, yu, zu = unit[:, 0], unit[:, 1], unit[:, 2]
        theta = np.arccos(zu)
        phi = np.arctan2(yu, xu)
        x_proj = theta * np.cos(phi)
        y_proj = theta * np.sin(phi)

        x_norm = (x_proj - np.min(x_proj)) / (np.ptp(x_proj) + 1e-6)
        y_norm = (y_proj - np.min(y_proj)) / (np.ptp(y_proj) + 1e-6)

        factor = projection_params.get('y_compression_factor', 1.0)
        direction = projection_params.get('y_compression_direction', 'positive')
        y_offset = y_norm - 0.5
        if factor != 1.0:
            if direction == 'positive':
                y_offset[y_offset > 0] *= factor
            elif direction == 'negative':
                y_offset[y_offset < 0] *= factor
            y_norm = 0.5 + y_offset

        coords2d = np.stack([x_norm, y_norm], axis=-1)
        diff = coords2d[:, None, :] - coords2d[None, :, :]
        dist_mat = np.linalg.norm(diff, axis=-1)

    else:
        raise ValueError(f"Unsupported projection type: {proj_type}")

    # == 可选栅格图输出 ==
    proj_grid = None
    if resolution is not None:
        H, W = (resolution, resolution) if isinstance(resolution, int) else resolution
        proj_grid = np.zeros((H, W), dtype=np.uint8)

        uv = coords2d.astype(np.float64).copy()
        uv -= np.mean(uv, axis=0)
        scale = np.max(np.abs(uv)) + 1e-6
        uv = 0.5 + uv / (2 * scale)

        ix = np.clip((uv[:, 0] * (W - 1)).round().astype(int), 0, W - 1)
        iy = np.clip((uv[:, 1] * (H - 1)).round().astype(int), 0, H - 1)
        proj_grid[iy, ix] = 1

    # == 可视化 ==
    if visualize:
        # ---- 1. Scatter plot ----
        fig, ax = plt.subplots(figsize=(6, 6), facecolor='white')
        ax.set_facecolor('white')
        ax.scatter(coords2d[:, 0], coords2d[:, 1], c=c, s=30)
        for i, name in enumerate(ch_names):
            ax.text(coords2d[i, 0], coords2d[i, 1], name, fontsize=8, ha='right', va='bottom')
        ax.set_title(f"Projection: {proj_type}", fontsize=12)
        ax.axis('equal')
        ax.grid(True, linestyle='--', color='lightgray', linewidth=0.5)
        plt.tight_layout()
        plt.show()

        # ---- 2. Grid image with cell borders ----
        if proj_grid is not None:
            H, W = proj_grid.shape
            iy, ix = np.nonzero(proj_grid)
        
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.set_facecolor("white")
            ax.scatter(ix, iy, c=c, s=60)  # 点大小可调（建议 s=20~50）
        
            # 网格线
            ax.set_xticks(np.arange(-0.5, W, 1), minor=True)
            ax.set_yticks(np.arange(-0.5, H, 1), minor=True)
            ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)
            ax.tick_params(which='minor', size=0)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlim(-0.5, W - 0.5)
            ax.set_ylim(-0.5, H - 0.5)
            ax.invert_yaxis()
            ax.set_title(f"Projection Grid {H}×{W}")
            plt.gca().invert_yaxis()  # 坐标方向和 imshow 一致
            plt.tight_layout()
            plt.show()
        
    # return ch_names, dist_mat, proj_grid

    return ch_names, dist_mat

def fc_matrices_circle(dataset, subject_range=range(1, 2), experiment_range=range(1, 2),
                       feature='pcc', band='joint', save=False, verbose=True):
    """
    Computes functional connectivity matrices for EEG datasets.

    Features:
    - Computes connectivity matrices based on the selected feature and frequency band.
    - Records total and average computation time.
    - Optionally saves results in HDF5 format.

    Parameters:
    - dataset (str): Dataset name ('SEED' or 'DREAMER').
    - subject_range (range): Range of subject IDs (default: range(1, 2)).
    - experiment_range (range): Range of experiment IDs (default: range(1, 2)).
    - feature (str): Connectivity feature ('pcc', 'plv', 'mi').
    - band (str): Frequency band ('delta', 'theta', 'alpha', 'beta', 'gamma', or 'joint').
    - save (bool): Whether to save results (default: False).
    - verbose (bool): Whether to print timing information (default: True).

    Returns:
    - dict: Dictionary containing computed functional connectivity matrices.
    """

    dataset = dataset.upper()
    feature = feature.lower()
    band = band.lower()

    valid_datasets = {'SEED', 'DREAMER'}
    valid_features = {'pcc', 'plv', 'mi', 'pli', 'wpli'}
    valid_bands = {'joint', 'theta', 'delta', 'alpha', 'beta', 'gamma'}

    if dataset not in valid_datasets:
        raise ValueError(f"Invalid dataset '{dataset}'. Supported datasets: {valid_datasets}")
    if feature not in valid_features:
        raise ValueError(f"Invalid feature '{feature}'. Supported features: {valid_features}")
    if band not in valid_bands:
        raise ValueError(f"Invalid band '{band}'. Supported bands: {valid_bands}")

    def eeg_loader(subject, experiment=None):
        """Loads EEG data for a given subject and experiment."""
        identifier = f"sub{subject}ex{experiment}" if dataset == 'SEED' else f"sub{subject}"
        eeg_data = utils_eeg_loading.read_eeg_filtered(dataset, identifier)
        return identifier, eeg_data

    fc_matrices = {}
    start_time = time.time()
    total_experiment_time = 0
    experiment_count = 0

    if dataset == 'SEED': 
        sampling_rate = 200
        experiments = experiment_range
    elif dataset == 'DREAMER':
        sampling_rate = 128
        experiments = [None]

    for subject in subject_range:
        for experiment in experiments:
            experiment_start = time.time()
            experiment_count += 1

            identifier, eeg_data = eeg_loader(subject, experiment)
            bands_to_process = ['delta', 'theta', 'alpha', 'beta', 'gamma'] if band == 'joint' else [band]

            fc_matrices[identifier] = {} if band == 'joint' else None

            for current_band in bands_to_process:
                data = np.array(eeg_data[current_band])

                if feature == 'pcc':
                    result = compute_corr_matrices(data, sampling_rate)
                elif feature == 'plv':
                    result = compute_plv_matrices(data, sampling_rate)
                elif feature == 'mi':
                    result = compute_mi_matrices(data, sampling_rate)
                elif feature == 'pli':
                    result = compute_pli_matrices(data, sampling_rate)
                elif feature == 'wpli':
                    result = compute_wpli_matrices(data, sampling_rate)
                    
                if band == 'joint':
                    fc_matrices[identifier][current_band] = result
                else:
                    fc_matrices[identifier] = result

            experiment_duration = time.time() - experiment_start
            total_experiment_time += experiment_duration

            if verbose:
                print(f"Experiment {identifier} completed in {experiment_duration:.2f} seconds")

            if save:
                save_results(dataset, feature, identifier, fc_matrices[identifier])

    total_time = time.time() - start_time
    avg_experiment_time = total_experiment_time / experiment_count if experiment_count else 0

    if verbose:
        print(f"\nTotal time taken: {total_time:.2f} seconds")
        print(f"Average time per experiment: {avg_experiment_time:.2f} seconds")

    return fc_matrices

def save_results(dataset, feature, identifier, data):
    """Saves functional connectivity matrices to an HDF5 file."""
    path_parent = os.path.dirname(os.getcwd())
    path_parent_parent = os.path.dirname(path_parent)
    base_path = os.path.join(path_parent_parent, 'Research_Data', dataset, 'functional connectivity', f'{feature}_h5')
    os.makedirs(base_path, exist_ok=True)
    
    file_path = os.path.join(base_path, f"{identifier}.h5")
    with h5py.File(file_path, 'w') as f:
        if isinstance(data, dict):  # Joint band case
            for band, matrix in data.items():
                f.create_dataset(band, data=matrix, compression="gzip")
        else:  # Single band case
            f.create_dataset("connectivity", data=data, compression="gzip")

    print(f"Data saved to {file_path}")

from tqdm import tqdm  # 确保在文件顶部导入
def compute_corr_matrices(eeg_data, sampling_rate, window=1, overlap=0, verbose=True, visualization=True):
    """
    Compute correlation matrices for EEG data using a sliding window approach.
    
    Parameters:
        eeg_data (numpy.ndarray): EEG data with shape (channels, time_samples).
        sampling_rate (int): Sampling rate of the EEG data in Hz.
        window (float): Window size in seconds for segmenting EEG data.
        overlap (float): Overlap fraction between consecutive windows (0 to 1).
        verbose (bool): If True, shows progress bar.
        visualization (bool): If True, displays correlation matrices.
    
    Returns:
        list of numpy.ndarray: List of correlation matrices for each window.
    """
    # Compute step size and segment length
    step = int(sampling_rate * window * (1 - overlap))
    segment_length = int(sampling_rate * window)

    # Generate overlapping segments
    split_segments = [
        eeg_data[:, i:i + segment_length]
        for i in range(0, eeg_data.shape[1] - segment_length + 1, step)
    ]

    # Compute correlation matrices with tqdm progress bar
    corr_matrices = []
    iterator = tqdm(enumerate(split_segments), total=len(split_segments), disable=not verbose, desc="Computing Corr Matrices")

    for idx, segment in iterator:
        if segment.shape[1] < segment_length:
            continue
        corr_matrix = np.corrcoef(segment)
        corr_matrices.append(corr_matrix)

    # Visualization
    if visualization and corr_matrices:
        avg_corr_matrix = np.mean(corr_matrices, axis=0)
        utils_visualization.draw_projection(avg_corr_matrix)

    return corr_matrices

def compute_plv_matrices(eeg_data, sampling_rate, window=1, overlap=0, verbose=True, visualization=True):
    """
    Compute Phase Locking Value (PLV) matrices for EEG data using a sliding window approach.

    Parameters:
        eeg_data (numpy.ndarray): EEG data with shape (channels, time_samples).
        sampling_rate (int): Sampling rate of the EEG data in Hz.
        window (float): Window size in seconds for segmenting EEG data.
        overlap (float): Overlap fraction between consecutive windows (0 to 1).
        verbose (bool): If True, shows progress bar.
        visualization (bool): If True, displays average PLV matrix.

    Returns:
        list of numpy.ndarray: List of PLV matrices for each window.
    """
    step = int(sampling_rate * window * (1 - overlap))
    segment_length = int(sampling_rate * window)

    # Split EEG data into overlapping windows
    split_segments = [
        eeg_data[:, i:i + segment_length]
        for i in range(0, eeg_data.shape[1] - segment_length + 1, step)
    ]

    plv_matrices = []

    iterator = tqdm(enumerate(split_segments), total=len(split_segments), disable=not verbose, desc="Computing PLV Matrices")

    for idx, segment in iterator:
        if segment.shape[1] < segment_length:
            continue  # Skip incomplete segments

        # Hilbert transform to extract phase
        analytic_signal = hilbert(segment, axis=1)
        phase_data = np.angle(analytic_signal)

        num_channels = phase_data.shape[0]
        plv_matrix = np.zeros((num_channels, num_channels))

        for ch1 in range(num_channels):
            for ch2 in range(num_channels):
                phase_diff = phase_data[ch1, :] - phase_data[ch2, :]
                plv_matrix[ch1, ch2] = np.abs(np.mean(np.exp(1j * phase_diff)))

        plv_matrices.append(plv_matrix)

    # Visualization
    if visualization and plv_matrices:
        avg_plv_matrix = np.mean(plv_matrices, axis=0)
        utils_visualization.draw_projection(avg_plv_matrix)

    return plv_matrices

def compute_pli_matrices(eeg_data, sampling_rate, window=1, overlap=0, verbose=True, visualization=True):
    """
    Compute Phase Lag Index (PLI) matrices for EEG data using a sliding window approach.

    Parameters:
        eeg_data (numpy.ndarray): EEG data with shape (channels, time_samples).
        sampling_rate (int): Sampling rate of the EEG data in Hz.
        window (float): Window size in seconds for segmenting EEG data.
        overlap (float): Overlap fraction between consecutive windows (0 to 1).
        verbose (bool): If True, shows progress bar.
        visualization (bool): If True, displays average PLI matrix.

    Returns:
        list of numpy.ndarray: List of PLI matrices for each window.
    """
    step = int(sampling_rate * window * (1 - overlap))
    segment_length = int(sampling_rate * window)

    # Generate overlapping segments
    split_segments = [
        eeg_data[:, i:i + segment_length]
        for i in range(0, eeg_data.shape[1] - segment_length + 1, step)
    ]

    pli_matrices = []

    iterator = tqdm(enumerate(split_segments), total=len(split_segments), disable=not verbose, desc="Computing PLI Matrices")

    for idx, segment in iterator:
        if segment.shape[1] < segment_length:
            continue

        analytic_signal = hilbert(segment, axis=1)
        phase_data = np.angle(analytic_signal)

        num_channels = phase_data.shape[0]
        pli_matrix = np.zeros((num_channels, num_channels))

        for ch1 in range(num_channels):
            for ch2 in range(num_channels):
                if ch1 == ch2:
                    continue
                phase_diff = phase_data[ch1] - phase_data[ch2]
                pli = np.abs(np.mean(np.sign(np.sin(phase_diff))))
                pli_matrix[ch1, ch2] = pli

        pli_matrices.append(pli_matrix)

    if visualization and pli_matrices:
        avg_pli_matrix = np.mean(pli_matrices, axis=0)
        utils_visualization.draw_projection(avg_pli_matrix)

    return pli_matrices

def compute_wpli_matrices(eeg_data, sampling_rate, window=1, overlap=0, verbose=True, visualization=True):
    """
    Compute weighted Phase Lag Index (wPLI) matrices for EEG data using a sliding window approach.

    Parameters:
        eeg_data (numpy.ndarray): EEG data with shape (channels, time_samples).
        sampling_rate (int): Sampling rate of the EEG data in Hz.
        window (float): Window size in seconds for segmenting EEG data.
        overlap (float): Overlap fraction between consecutive windows (0 to 1).
        verbose (bool): If True, shows progress bar.
        visualization (bool): If True, displays average wPLI matrix.

    Returns:
        list of numpy.ndarray: List of wPLI matrices for each window.
    """
    step = int(sampling_rate * window * (1 - overlap))
    segment_length = int(sampling_rate * window)

    # Create sliding window segments
    split_segments = [
        eeg_data[:, i:i + segment_length]
        for i in range(0, eeg_data.shape[1] - segment_length + 1, step)
    ]

    wpli_matrices = []
    iterator = tqdm(enumerate(split_segments), total=len(split_segments), disable=not verbose, desc="Computing wPLI Matrices")

    for idx, segment in iterator:
        if segment.shape[1] < segment_length:
            continue

        analytic_signal = hilbert(segment, axis=1)
        num_channels = analytic_signal.shape[0]
        wpli_matrix = np.zeros((num_channels, num_channels))

        for ch1 in range(num_channels):
            for ch2 in range(num_channels):
                if ch1 == ch2:
                    continue

                csd = analytic_signal[ch1] * np.conj(analytic_signal[ch2])
                im_part = np.imag(csd)

                numerator = np.abs(np.mean(im_part))
                denominator = np.mean(np.abs(im_part)) + 1e-10  # avoid divide-by-zero
                wpli = numerator / denominator
                wpli_matrix[ch1, ch2] = wpli

        wpli_matrices.append(wpli_matrix)

    if visualization and wpli_matrices:
        avg_wpli_matrix = np.mean(wpli_matrices, axis=0)
        utils_visualization.draw_projection(avg_wpli_matrix)

    return wpli_matrices

from sklearn.metrics import mutual_info_score
def compute_mi_matrices(eeg_data, sampling_rate, window=1, overlap=0, verbose=True, visualization=True, bins=16):
    """
    Compute Mutual Information (MI) matrices for EEG data using a sliding window approach.

    Parameters:
        eeg_data (numpy.ndarray): EEG data with shape (channels, time_samples).
        sampling_rate (int): Sampling rate of the EEG data in Hz.
        window (float): Window size in seconds for segmenting EEG data.
        overlap (float): Overlap fraction between consecutive windows (0 to 1).
        verbose (bool): If True, shows progress bar.
        visualization (bool): If True, displays average MI matrix.
        bins (int): Number of bins for discretizing EEG signals before MI computation.

    Returns:
        list of numpy.ndarray: List of MI matrices for each window.
    """
    step = int(sampling_rate * window * (1 - overlap))
    segment_length = int(sampling_rate * window)

    # Create overlapping segments
    split_segments = [
        eeg_data[:, i:i + segment_length]
        for i in range(0, eeg_data.shape[1] - segment_length + 1, step)
    ]

    mi_matrices = []
    iterator = tqdm(enumerate(split_segments), total=len(split_segments), disable=not verbose, desc="Computing MI Matrices")

    for idx, segment in iterator:
        if segment.shape[1] < segment_length:
            continue

        num_channels = segment.shape[0]
        mi_matrix = np.zeros((num_channels, num_channels))

        # Discretize each channel
        discretized = np.array([
            np.digitize(segment[ch], bins=np.histogram_bin_edges(segment[ch], bins=bins))
            for ch in range(num_channels)
        ])

        for ch1 in range(num_channels):
            for ch2 in range(num_channels):
                if ch1 == ch2:
                    continue
                mi = mutual_info_score(discretized[ch1], discretized[ch2])
                mi_matrix[ch1, ch2] = mi

        mi_matrices.append(mi_matrix)

    if visualization and mi_matrices:
        avg_mi_matrix = np.mean(mi_matrices, axis=0)
        utils_visualization.draw_projection(avg_mi_matrix)

    return mi_matrices

def compute_average_fcs(dataset, subjects=range(1, 16), experiments=range(1, 4), 
                        feature='pcc', band='joint', in_file_type='.h5', 
                        save=False, verbose=False, visualization=False):
    """
    Computes and optionally saves or visualizes the averaged functional connectivity matrices.

    Parameters
    ----------
    dataset : str
        Dataset name (e.g., 'seed').
    subjects : iterable
        List or range of subject indices.
    experiments : iterable
        List or range of experiment indices.
    feature : str
        Feature type, e.g., 'pcc', 'plv', 'pli'.
    band : str
        Frequency band or 'joint' for all bands.
    in_file_type : str
        Input file type, '.h5' or '.mat'.
    out_file_type : str
        Output file type, '.h5' or '.mat'.
    save : bool
        Whether to save the result.
    verbose : bool
        Whether to print verbose output.
    visualization : bool
        Whether to visualize the global averaged matrix.

    Returns
    -------
    np.ndarray
        The global averaged functional connectivity matrix.
    """
    
    assert dataset.lower() in {'seed', 'dreamer'}, "Unsupported dataset."
    assert feature.lower() in {'pcc', 'plv', 'mi', 'pli', 'wpli'}, "Invalid feature."
    assert band.lower() in {'joint', 'alpha', 'beta', 'gamma', 'delta', 'theta'}, "Invalid band."
    assert in_file_type in {'.h5', '.mat'}, "Unsupported input file type."

    fcs_averaged_dict, fcs_averaged_dict_ = [], {'alpha': [], 'beta': [], 'gamma': [], 'delta': [], 'theta': []}
    
    for subject in subjects:
        for experiment in experiments:
            
            if dataset == 'seed':
                identifier, total_subjects = f"sub{subject}ex{experiment}", 15
            elif dataset == 'dreamer':
                identifier, total_subjects = f"sub{subject}", 23
                
            if verbose:
                print(f"Processing: {identifier}")
            
            features = utils_feature_loading.read_fcs(dataset, identifier, feature, band, in_file_type)
            
            if band == 'joint':
                try:
                    avg_bands = [{"average": np.mean(features[b], axis=0), 
                                  "band": b, "subject": subject, "experiment": experiment} 
                                 for b in ['alpha', 'beta', 'gamma', 'delta', 'theta']]
                    
                    fcs_averaged_dict.append(avg_bands)
                    for entry in avg_bands:                        
                        fcs_averaged_dict_[entry["band"]].append(entry["average"])
                    
                    # Correct theta/delta swap if necessary
                except KeyError:
                    avg_bands = [{"average": np.mean(features[b], axis=0), 
                                  "band": b, "subject": subject, "experiment": experiment} 
                                 for b in ['alpha', 'beta', 'gamma']]
                    
                    fcs_averaged_dict.append(avg_bands)
                    for entry in avg_bands:                        
                        fcs_averaged_dict_[entry["band"]].append(entry["average"])

    # Compute global average
    global fcs_global_averaged
    
    try:
        fcs_global_averaged = {b: np.mean(fcs_averaged_dict_[b], axis=0)
                               for b in ['alpha', 'beta', 'gamma', 'delta', 'theta']}
    except KeyError:        
        fcs_global_averaged = {b: np.mean(fcs_averaged_dict_[b], axis=0)
                               for b in ['alpha', 'beta', 'gamma']}
        
    if visualization:
        for fc in fcs_global_averaged.values():
            utils_visualization.draw_projection(fc)

    if save:
        save_results(dataset, feature, f'global_averaged_{subject}_{total_subjects}', fcs_global_averaged)
        
        if verbose:
            print("Results saved to .h5 and .mat")

    return fcs_global_averaged, fcs_averaged_dict_
        
# %% Label Engineering
def generate_labels(sampling_rate=128):
    dreamer = utils_eeg_loading.read_eeg_original_dataset('dreamer')

    # labels
    score_arousal = 0
    score_dominance = 0
    score_valence = 0
    index = 0
    eeg_all = []
    for data in dreamer['Data']:
        index += 1
        score_arousal += data['ScoreArousal']
        score_dominance += data['ScoreDominance']
        score_valence += data['ScoreValence']
        eeg_all.append(data['EEG']['stimuli'])

    labels = [1, 3, 5]
    score_arousal_labels = normalize_to_labels(score_arousal, labels)
    score_dominance_labels = normalize_to_labels(score_dominance, labels)
    score_valence_labels = normalize_to_labels(score_valence, labels)

    # data
    eeg_sample = eeg_all[0]
    labels_arousal = []
    labels_dominance = []
    labels_valence = []
    for eeg_trial in range(0, len(eeg_sample)):
        label_container = np.ones(len(eeg_sample[eeg_trial]))

        label_arousal = label_container * score_arousal_labels[eeg_trial]
        label_dominance = label_container * score_dominance_labels[eeg_trial]
        label_valence = label_container * score_valence_labels[eeg_trial]

        labels_arousal = np.concatenate((labels_arousal, label_arousal))
        labels_dominance = np.concatenate((labels_dominance, label_dominance))
        labels_valence = np.concatenate((labels_valence, label_valence))

    labels_arousal = labels_arousal[::sampling_rate]
    labels_dominance = labels_dominance[::sampling_rate]
    labels_valence = labels_valence[::sampling_rate]

    return labels_arousal, labels_dominance, labels_valence

def normalize_to_labels(array, labels):
    """
    Normalize an array to discrete labels.

    Parameters:
        array (np.ndarray): The input array.
        labels (list): The target labels to map to (e.g., [1, 3, 5]).

    Returns:
        np.ndarray: The normalized array mapped to discrete labels.
    """
    # Step 1: Normalize array to [0, 1]
    array_min = np.min(array)
    array_max = np.max(array)
    normalized = (array - array_min) / (array_max - array_min)

    # Step 2: Map to discrete labels
    bins = np.linspace(0, 1, len(labels))
    discrete_labels = np.digitize(normalized, bins, right=True)

    # Map indices to corresponding labels
    return np.array([labels[i - 1] for i in discrete_labels])

# %% interpolation
import scipy.interpolate
def interpolate_matrices(
    data: dict[str, np.ndarray], 
    scale_factor: tuple[float, float] = (1.0, 1.0), 
    method: str = 'nearest'
) -> dict[str, np.ndarray]:
    """
    Perform interpolation on dictionary-formatted data, scaling each channel's (samples, w, h) data.

    Parameters:
    - data: dict, format {ch: numpy.ndarray}, where each value has shape (samples, w, h).
    - scale_factor: tuple (float, float), interpolation scaling factor (new_w/w, new_h/h).
    - method: str, interpolation method, options:
        - 'nearest' (nearest neighbor)
        - 'linear' (bilinear interpolation)
        - 'cubic' (bicubic interpolation)

    Returns:
    - new_data: dict, format {ch: numpy.ndarray}, interpolated data with shape (samples, new_w, new_h).
    """

    if not isinstance(scale_factor, tuple):
        raise TypeError("scale_factor must be a tuple of two floats (scale_w, scale_h).")
    
    if method not in {'nearest', 'linear', 'cubic'}:
        raise ValueError("Invalid interpolation method. Choose from 'nearest', 'linear', or 'cubic'.")

    new_data = {}  # Store interpolated data
    
    for ch, array in data.items():
        if array.ndim != 3:
            raise ValueError(f"Each array in data must have 3 dimensions (samples, w, h), but got {array.shape} for channel {ch}.")

        samples, w, h = array.shape
        new_w, new_h = int(w * scale_factor[0]), int(h * scale_factor[1])

        # Ensure valid shape
        if new_w <= 0 or new_h <= 0:
            raise ValueError("Interpolated dimensions must be positive integers.")

        # Generate original and target grid points
        x_old = np.linspace(0, 1, w)
        y_old = np.linspace(0, 1, h)
        x_new = np.linspace(0, 1, new_w)
        y_new = np.linspace(0, 1, new_h)

        xx_old, yy_old = np.meshgrid(x_old, y_old, indexing='ij')
        xx_new, yy_new = np.meshgrid(x_new, y_new, indexing='ij')

        old_points = np.column_stack([xx_old.ravel(), yy_old.ravel()])
        new_points = np.column_stack([xx_new.ravel(), yy_new.ravel()])

        # Initialize new array
        new_array = np.empty((samples, new_w, new_h), dtype=array.dtype)

        # Perform interpolation for each sample
        for i in range(samples):
            values = array[i].ravel()
            interpolated = scipy.interpolate.griddata(old_points, values, new_points, method=method)
            new_array[i] = interpolated.reshape(new_w, new_h)

        new_data[ch] = new_array

    return new_data

def interpolate_matrices_(data, scale_factor=(1.0, 1.0), method='nearest'):
    """
    对形如 samples x channels x w x h 的数据进行插值，使每个 w x h 矩阵放缩

    参数:
    - data: numpy.ndarray, 形状为 (samples, channels, w, h)
    - scale_factor: float 或 (float, float)，插值的缩放因子
    - method: str，插值方法，可选：
        - 'nearest' (最近邻)
        - 'linear' (双线性)
        - 'cubic' (三次插值)

    返回:
    - new_data: numpy.ndarray, 插值后的数据，形状 (samples, channels, new_w, new_h)
    """
    samples, channels, w, h = data.shape
    new_w, new_h = int(w * scale_factor[0]), int(h * scale_factor[1])

    # 目标尺寸
    output_shape = (samples, channels, new_w, new_h)
    new_data = np.zeros(output_shape, dtype=data.dtype)

    # 原始网格点 (w, h)
    x_old = np.linspace(0, 1, w)
    y_old = np.linspace(0, 1, h)
    xx_old, yy_old = np.meshgrid(x_old, y_old, indexing='ij')

    # 目标网格点 (new_w, new_h)
    x_new = np.linspace(0, 1, new_w)
    y_new = np.linspace(0, 1, new_h)
    xx_new, yy_new = np.meshgrid(x_new, y_new, indexing='ij')

    # 插值
    for i in range(samples):
        for j in range(channels):
            old_points = np.column_stack([xx_old.ravel(), yy_old.ravel()])  # 原始点坐标
            new_points = np.column_stack([xx_new.ravel(), yy_new.ravel()])  # 目标点坐标
            values = data[i, j].ravel()  # 原始像素值

            # griddata 进行插值
            interpolated = scipy.interpolate.griddata(old_points, values, new_points, method=method)
            new_data[i, j] = interpolated.reshape(new_w, new_h)

    return new_data

# %% padding
def global_padding(matrix, width=81, verbose=True):
    """
    Pads a 2D, 3D or 4D matrix to the specified width while keeping the original data centered.
    For shape of: width x height, samples x width x height, samples x channels x width x height.

    Parameters:
        matrix (np.ndarray): The input matrix to be padded.
        width (int): The target width/height for padding.
        verbose (bool): If True, prints the original and padded shapes.

    Returns:
        np.ndarray: The padded matrix with the specified width.
    """
    if len(matrix.shape) == 2:
        width_input, _ = matrix.shape
        total_padding = width - width_input
        pad_before = total_padding // 2
        pad_after = total_padding - pad_before

        padded_matrix = np.pad(
            matrix,
            pad_width=((pad_before, pad_after), (pad_before, pad_after)),
            mode='constant',
            constant_values=0
        )

    elif len(matrix.shape) == 3:
        _, width_input, _ = matrix.shape
        total_padding = width - width_input
        pad_before = total_padding // 2
        pad_after = total_padding - pad_before

        padded_matrix = np.pad(
            matrix,
            pad_width=((0, 0), (pad_before, pad_after), (pad_before, pad_after)),
            mode='constant',
            constant_values=0
        )

    elif len(matrix.shape) == 4:
        _, _, width_input, _ = matrix.shape
        total_padding = width - width_input
        pad_before = total_padding // 2
        pad_after = total_padding - pad_before

        padded_matrix = np.pad(
            matrix,
            pad_width=((0, 0), (0, 0), (pad_before, pad_after), (pad_before, pad_after)),
            mode='constant',
            constant_values=0
        )

    else:
        raise ValueError("Input matrix must be either 2D, 3D or 4D.")

    if verbose:
        print("Original shape:", matrix.shape)
        print("Padded shape:", padded_matrix.shape)

    return padded_matrix

# %% Normalize
from scipy.stats import boxcox, yeojohnson
def normalize_matrix(matrix, method='minmax', epsilon=1e-8, param=None):
    """
    对矩阵或批量矩阵进行归一化或变换处理。
    
    支持方法包括：minmax, max, mean, z-score, boxcox, yeojohnson, sqrt, log, none。
    可输入单个矩阵 (H, W) 或批量矩阵 (N, H, W)。

    参数:
        matrix (np.ndarray): 输入矩阵或批量矩阵。
        method (str): 归一化方法。
        epsilon (float): 防止除零的极小值。
        param (dict): 额外参数，如 target_range 或 lmbda。
    """
    if param is None:
        param = {}
    a, b = param.get('target_range', (0, 1))
    lmbda = param.get('lmbda', None)

    # 判断是否批处理
    is_batch = matrix.ndim == 3
    matrices = matrix if is_batch else matrix[None, ...]

    normalized = []
    for mat in matrices:
        mat = mat.copy()

        if method == 'minmax':
            min_val, max_val = np.min(mat), np.max(mat)
            scale = max(max_val - min_val, epsilon)
            mat = ((mat - min_val) / scale) * (b - a) + a

        elif method == 'max':
            max_val = max(np.max(np.abs(mat)), epsilon)
            mat = mat / max_val

        elif method == 'mean':
            mean_val = max(np.mean(mat), epsilon)
            mat = mat / mean_val

        elif method == 'z-score':
            mean_val, std_val = np.mean(mat), np.std(mat)
            mat = (mat - mean_val) / max(std_val, epsilon)

        elif method == 'boxcox':
            mat += epsilon
            if np.any(mat <= 0):
                raise ValueError("Box-Cox 要求所有值 > 0")
            mat = boxcox(mat.flatten(), lmbda=lmbda)[0].reshape(mat.shape)

        elif method == 'yeojohnson':
            mat = yeojohnson(mat.flatten(), lmbda=lmbda)[0].reshape(mat.shape)

        elif method == 'sqrt':
            if np.any(mat < 0):
                raise ValueError("平方根要求非负值")
            mat = np.sqrt(mat + epsilon)

        elif method == 'log':
            if np.any(mat <= 0):
                raise ValueError("对数要求值 > 0")
            mat = np.log(mat + epsilon)

        elif method == 'none':
            pass

        else:
            raise ValueError(f"不支持的归一化方法: {method}")

        normalized.append(mat)

    result = np.stack(normalized) if is_batch else normalized[0]
    return result

def rebuild_features(data, coordinates, param, visualize=False):
    """
    用 IDW 或 Gaussian 对坏通道进行重建，支持连接矩阵 (n,n) 或特征向量 (n,)

    Args:
        data (np.ndarray): 输入数据，shape 为 (n, n) 或 (n,)
        coordinates (dict): {'x': [...], 'y': [...], 'z': [...]}
        param (dict): 同上，支持 method / threshold / kernel / sigma / manual_bad_idx

    Returns:
        np.ndarray: 重建后的数据
    """
    if visualize:
        try:
            utils_visualization.draw_projection(data, 'Before Spatial Gaussian Rebuilding')
        except ModuleNotFoundError: 
            print("utils_visualization not found")
    
    data = data.copy()
    coords = np.vstack([coordinates['x'], coordinates['y'], coordinates['z']]).T
    # n = coords.shape[0]

    # === 自动检测坏通道（按平均值做异常点检测）
    if data.ndim == 2:
        mean_val = np.mean(np.abs(data), axis=1)
    elif data.ndim == 1:
        mean_val = np.abs(data)
    else:
        raise ValueError("Only supports 1D or 2D array")

    if param['method'] == 'zscore':
        z = (mean_val - np.mean(mean_val)) / (np.std(mean_val) + 1e-8)
        bad_idx = np.where(np.abs(z) > param['threshold'])[0]
    elif param['method'] == 'iqr':
        q1, q3 = np.percentile(mean_val, [25, 75])
        iqr = q3 - q1
        lower, upper = q1 - param['threshold'] * iqr, q3 + param['threshold'] * iqr
        bad_idx = np.where((mean_val < lower) | (mean_val > upper))[0]
    else:
        raise ValueError("param['method'] must be 'zscore' or 'iqr'")

    # === 合并手动坏通道
    manual = np.array(param.get('manual_bad_idx', []), dtype=int)
    bad_idx = np.unique(np.concatenate([bad_idx, manual]))

    print(f"[INFO] Detected bad channels: {bad_idx.tolist()}")
    if len(bad_idx) == 0:
        return data

    # === 开始重建
    for i in bad_idx:
        dists = np.linalg.norm(coords[i] - coords, axis=1)
        dists[i] = np.inf

        if param['kernel'] == 'idw':
            weights = 1 / (dists + 1e-8)
        elif param['kernel'] == 'gaussian':
            sigma = param.get('sigma', 1.0)
            weights = np.exp(-dists**2 / (2 * sigma**2))
        else:
            raise ValueError("Unsupported kernel type")

        weights[i] = 0
        weights /= weights.sum()

        if data.ndim == 1:
            data[i] = weights @ data
        elif data.ndim == 2:
            data[i, :] = weights @ data
            data[:, i] = data[i, :]  # 对称处理

    if visualize:
        try:
            utils_visualization.draw_projection(data, 'After Spatial Gaussian Rebuilding')
        except ModuleNotFoundError: 
            print("utils_visualization not found")

    return data

from scipy.spatial.distance import cdist
def spatial_gaussian_smoothing_on_vector(A, coordinates, sigma):
    coords = np.vstack([coordinates['x'], coordinates['y'], coordinates['z']]).T
    dists = cdist(coords, coords)
    weights = np.exp(- (dists ** 2) / (2 * sigma ** 2))
    weights /= weights.sum(axis=1, keepdims=True)
    A_smooth = weights @ A
    return A_smooth

def spatial_gaussian_smoothing_on_fc_matrix(A, coordinates, sigma, visualize=False):
    """
    Applies spatial Gaussian smoothing to a symmetric functional connectivity (FC) matrix.

    Parameters
    ----------
    A : np.ndarray of shape (N, N)
        Symmetric functional connectivity matrix.
    coordinates : dict with keys 'x', 'y', 'z'
        Each value is a list or array of length N, giving 3D coordinates for each channel.
    sigma : float
        Standard deviation of the spatial Gaussian kernel.

    Returns
    -------
    A_smooth : np.ndarray of shape (N, N)
        Symmetrically smoothed functional connectivity matrix.
    """
    if visualize:
        try:
            utils_visualization.draw_projection(A, 'Before Spatial Gaussian Smoothing')
        except ModuleNotFoundError: 
            print("utils_visualization not found")
    
    # Step 1: Stack coordinate vectors to (N, 3)
    coords = np.vstack([coordinates['x'], coordinates['y'], coordinates['z']]).T  # shape (N, 3)

    # Step 2: Compute Euclidean distance matrix between channels
    dists = cdist(coords, coords)  # shape (N, N)

    # Step 3: Compute spatial Gaussian weights
    weights = np.exp(- (dists ** 2) / (2 * sigma ** 2))  # shape (N, N)
    weights /= weights.sum(axis=1, keepdims=True)       # normalize per row

    # Step 4: Apply spatial smoothing to both rows and columns
    A_smooth = weights @ A @ weights.T

    # Step 5 (optional): Enforce symmetry
    # A_smooth = 0.5 * (A_smooth + A_smooth.T)
    
    if visualize:
        try:
            utils_visualization.draw_projection(A_smooth, 'After Spatial Gaussian Smoothing')
        except ModuleNotFoundError: 
            print("utils_visualization not found")
    
    return A_smooth

# %% Tools
def remove_idx_manual(A, manual_idxs=[]):
    if len(A.shape) == 1:
        A = np.delete(A, manual_idxs, axis=0)
    elif len(A.shape) == 2:
        A = np.delete(A, manual_idxs, axis=0)
        A = np.delete(A, manual_idxs, axis=1)
    elif len(A.shape) == 3:
        A = np.delete(A, manual_idxs, axis=1)
        A = np.delete(A, manual_idxs, axis=2)
    return A

def insert_idx_manual(A, manual_idxs=[], value=0):
    if len(A.shape) == 1:
        for idx in manual_idxs:
            if idx >= len(A):
                A = np.append(A, value)
            else:
                A = np.insert(A, idx, value)
                
    return A

# %% Example usage
if __name__ == "__main__":
    # filter_eeg_and_save_circle('seed', subject_range=range(1,2), experiment_range=range(1,2), verbose=True, save=False)
    
    # %% Filter EEG
    # eeg = utils_eeg_loading.read_eeg_originaldataset('seed', 'sub1ex1')
    # filtered_eeg_seed_sample = filter_eeg_seed('sub1ex1')
    
    # filter_eeg_and_save_circle('seed', range(1,2), range(1,4), save=False)
    
    # eeg = utils_eeg_loading.read_eeg_originaldataset('dreamer', 'sub1')
    # filtered_eeg_seed_sample = filter_eeg_dreamer('sub1')    
    
    # %% Feature Engineering; Distance Matrix
    # channel_names, distance_matrix = compute_distance_matrix('seed')
    # utils_visualization.draw_projection(distance_matrix)
    
    # channel_names, distance_matrix = compute_distance_matrix('dreamer')
    # utils_visualization.draw_projection(distance_matrix)
    
    # %% Feature Engineering; Compute functional connectivities
    # eeg_sample_seed = utils_eeg_loading.read_and_parse_seed('sub1ex1')
    # pcc_sample_seed = compute_corr_matrices(eeg_sample_seed, sampling_rate=200)
    # plv_sample_seed = compute_plv_matrices(eeg_sample_seed, samplingrate=200)
    # # mi_sample_seed = compute_mi_matrices(eeg_sample_seed, samplingrate=200)
    
    # eeg_sample_dreamer = utils_eeg_loading.read_and_parse_dreamer('sub1')
    # pcc_sample_dreamer = compute_corr_matrices(eeg_sample_dreamer, samplingrate=128)
    # plv_sample_dreamer = compute_plv_matrices(eeg_sample_dreamer, samplingrate=128)
    # mi_sample_dreamer = compute_mi_matrices(eeg_sample_dreamer, samplingrate=128)
    
    # %% Label Engineering
    # labels_seed = utils_feature_loading.read_labels('seed')
    # labels_dreamer = utils_feature_loading.read_labels('dreamer')
    # labels_dreamer_ = generate_labels()
    
    # %% Interpolation
    
    # %% Feature Engineering; Computation circles
    # fc_pcc_matrices_seed = fc_matrices_circle('SEED', feature='pcc', save=False, subject_range=range(1, 2), experiment_range=range(1, 2))
    # fc_plv_matrices_seed = fc_matrices_circle('SEED', feature='plv', save=False, subject_range=range(1, 2), experiment_range=range(1, 2))
    # fc_pli_matrices_seed = fc_matrices_circle('SEED', feature='pli', save=False, subject_range=range(1, 2), experiment_range=range(1, 2))
    # fc_wpli_matrices_seed = fc_matrices_circle('SEED', feature='wpli', save=False, subject_range=range(1, 2), experiment_range=range(1, 2))
    # fc_mi_matrices_seed = fc_matrices_circle('SEED', feature='mi', save=True, subject_range=range(1, 2), experiment_range=range(1, 4))

    # fc_pcc_matrices_dreamer = fc_matrices_circle('dreamer', feature='pcc', save=True, subject_range=range(1, 2))
    # fc_plv_matrices_dreamer = fc_matrices_circle('dreamer', feature='plv', save=True, subject_range=range(1, 2))
    # fc_mi_matrices_dreamer = fc_matrices_circle('dreamer', feature='pli', save=True, subject_range=range(1, 2))
    # fc_mi_matrices_dreamer = fc_matrices_circle('dreamer', feature='wpli', save=True, subject_range=range(1, 2))
    # fc_mi_matrices_dreamer = fc_matrices_circle('dreamer', feature='mi', save=True, subject_range=range(1, 2))
    
    # %% Feature Engineering; Compute Average CM   
    # SEED
    # fcs_global_averaged = compute_average_fcs('seed', subjects=range(1, 6), experiments=range(1, 4), 
    #                         feature='wpli', band='joint', in_file_type='.h5',
    #                         save=True, verbose=False, visualization=True)
    
    # fcs_global_averaged = compute_average_fcs('seed', subjects=range(1, 11), experiments=range(1, 4), 
    #                         feature='wpli', band='joint', in_file_type='.h5',
    #                         save=True, verbose=False, visualization=True)
    
    # fcs_global_averaged = compute_average_fcs('seed', subjects=range(1, 16), experiments=range(1, 4), 
    #                         feature='wpli', band='joint', in_file_type='.h5',
    #                         save=True, verbose=False, visualization=True)
    
    # fcs_global_averaged_ = utils_feature_loading.read_fcs_global_average('seed', 'wpli', sub_range=range(1, 6))
    
    # DREAMER
    # fcs_global_averaged = compute_average_fcs('dreamer', subjects=range(1, 24), experiments=range(1, 2), 
    #                         feature='pcc', band='joint', in_file_type='.h5',
    #                         save=True, verbose=False, visualization=True)
    
    # fcs_global_averaged_ = utils_feature_loading.read_fcs_global_average('dreamer', 'pcc', sub_range=range(1, 24))
    
    # fcs_global_averaged = compute_average_fcs('dreamer', subjects=range(1, 24), experiments=range(1, 2), 
    #                         feature='plv', band='joint', in_file_type='.h5',
    #                         save=True, verbose=False, visualization=False)
    
    # fcs_global_averaged_ = utils_feature_loading.read_fcs_global_average('dreamer', 'plv', sub_range=range(1, 24))
    
    # %% End program actions
    utils_visualization.end_program_actions(play_sound=True, shutdown=False, countdown_seconds=120)