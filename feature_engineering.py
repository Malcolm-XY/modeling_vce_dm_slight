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

import joblib

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

def filter_eeg_seed(identifier, verbose=True, save=False):
    """
    Load, filter, and optionally save SEED dataset EEG data into frequency bands.

    Parameters:
    identifier (str): Identifier for the subject/session.
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
    filtered_eeg_dict = filter_eeg(eeg, verbose=verbose)
    
    # Save filtered EEG data if requested
    if save:
        for band, filtered_eeg in filtered_eeg_dict.items():
            path_file = os.path.join(base_path, f"{identifier}_{band}_eeg.fif")
            filtered_eeg.save(path_file, overwrite=True)
            if verbose:
                print(f"Saved {band} band filtered EEG to {path_file}")
    
    return filtered_eeg_dict

def filter_eeg_dreamer(identifier, verbose=True, save=False):
    """
    Load, filter, and optionally save DREAMER dataset EEG data into frequency bands.

    Parameters:
    identifier (str): Identifier for the trial/session.
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
    filtered_eeg_dict = filter_eeg(eeg, verbose=verbose)
    
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
def compute_distance_matrix(dataset, projection_params=None, visualize=True):
    projection_type = projection_params.get('type')

    # 读取电极数据
    distribution = utils_feature_loading.read_distribution(dataset)
    channel_names = distribution['channel']
    x, y, z = np.array(distribution['x']), np.array(distribution['y']), np.array(distribution['z'])

    coords = np.vstack((x, y, z)).T

    # temporal---start---
    if projection_type == '2d':
        distribution = utils_feature_loading.read_distribution(dataset, 'manual')
        
        coords = np.vstack((x, y, z)).T
        diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
        distance_matrix = np.sqrt(np.sum(diff ** 2, axis=-1))
        return channel_names, distance_matrix
    # temporal---end---

    if projection_type == '3d':
        # 计算球面距离（单位球半径 R = 1）
        norms = np.linalg.norm(coords, axis=1, keepdims=True)
        unit_coords = coords / norms  # 标准化到单位球面
        dot_products = np.clip(unit_coords @ unit_coords.T, -1.0, 1.0)
        distance_matrix = np.arccos(dot_products)

    elif projection_type == 'euclidean':
        # 欧氏距离
        diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
        distance_matrix = np.sqrt(np.sum(diff ** 2, axis=-1))

    else:
        raise ValueError(f"Unsupported projection type: {projection_type}")

    return channel_names, distance_matrix

def compute_distance_matrix_(dataset, projection_params=None, visualize=True):
    """
    使用指定投影方法将EEG电极坐标从3D投影到2D平面，或直接在3D空间中计算距离矩阵。

    Args:
        dataset (str): 数据集名称，用于读取电极坐标。
        projection_params (dict): 投影参数，包含以下字段：
            - 'type': 'stereo'、'azimuthal' 或 '3d'
            - 'focal_length': stereo 参数
            - 'max_scaling': stereo 参数
            - 'y_compression_factor': float, y轴压缩/拉伸因子（默认1.0，仅限azimuthal）
            - 'y_compression_direction': 'positive' 或 'negative'，仅作用一侧
        visualize (bool): 是否显示投影图（仅2D模式有效）。

    Returns:
        tuple: (channel_names, distance_matrix)
    """
    projection_type = projection_params.get('type')

    # 读取电极数据
    distribution = utils_feature_loading.read_distribution(dataset)
    channel_names = distribution['channel']
    x, y, z = np.array(distribution['x']), np.array(distribution['y']), np.array(distribution['z'])

    if projection_type == '3d':
        coords = np.vstack((x, y, z)).T
        diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
        distance_matrix = np.sqrt(np.sum(diff ** 2, axis=-1))
        return channel_names, distance_matrix

    if projection_type == 'stereo':
        focal_length = projection_params.get('focal_length', 1.0)
        max_scaling = projection_params.get('max_scaling', 5.0)
        epsilon = 1e-6

        z_norm = (z - np.min(z)) / (np.max(z) - np.min(z))
        scaling = focal_length / (focal_length - z_norm + epsilon)
        scaling = np.clip(scaling, 0, max_scaling)

        x_proj = x * scaling
        y_proj = y * scaling

    elif projection_type == 'azimuthal':
        coords = np.vstack((x, y, z)).T.astype(np.float64)
        coords /= np.linalg.norm(coords, axis=1, keepdims=True)
        x_unit, y_unit, z_unit = coords[:, 0], coords[:, 1], coords[:, 2]

        theta = np.arccos(z_unit)
        phi = np.arctan2(y_unit, x_unit)

        x_proj = theta * np.cos(phi)
        y_proj = theta * np.sin(phi)
    else:
        raise ValueError(f"未知的投影类型: {projection_type}")

    # === 归一化投影坐标 ===
    x_norm = (x_proj - np.min(x_proj)) / (np.max(x_proj) - np.min(x_proj))
    y_norm = (y_proj - np.min(y_proj)) / (np.max(y_proj) - np.min(y_proj))

    # === Azimuthal 特有：仅单侧压缩 y 轴 ===
    if projection_type == 'azimuthal':
        y_compression_factor = projection_params.get('y_compression_factor', 1.0)
        y_compression_direction = projection_params.get('y_compression_direction', 'positive')

        if y_compression_factor != 1.0:
            y_offset = y_norm - 0.5
            if y_compression_direction == 'positive':
                y_offset[y_offset > 0] *= y_compression_factor
            elif y_compression_direction == 'negative':
                y_offset[y_offset < 0] *= y_compression_factor
            y_norm = 0.5 + y_offset

    # 合并投影坐标
    proj_coords = np.vstack((x_norm, y_norm)).T

    if visualize:
        plt.figure(figsize=(6, 6))
        plt.scatter(x_norm, y_norm, c='blue')
        for i, name in enumerate(channel_names):
            plt.text(x_norm[i], y_norm[i], name, fontsize=8, ha='right', va='bottom')
        plt.title(f"Projection: {projection_type}")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.axis("equal")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    # === 计算距离矩阵 ===
    diff = proj_coords[:, np.newaxis, :] - proj_coords[np.newaxis, :, :]
    distance_matrix = np.sqrt(np.sum(diff ** 2, axis=-1))

    return channel_names, distance_matrix

def compute_distance_matrix__(dataset, normalize=False,
                            normalization_method='minmax', projection_params=None,
                            visualize=True):
    """
    使用指定投影方法将EEG电极坐标从3D投影到2D平面，并计算距离矩阵。

    Args:
        dataset (str): 数据集名称，用于读取电极坐标。
        normalize (bool): 是否对输出距离矩阵归一化。
        normalization_method (str): 使用的归一化方法。
        projection_params (dict): 投影参数，包含以下字段：
            - 'type': 'stereo' 或 'azimuthal'
            - 'focal_length': stereo 参数
            - 'max_scaling': stereo 参数
            - 'y_compression_factor': float, y轴压缩/拉伸因子（默认1.0）
            - 'y_compression_direction': 'positive' 或 'negative'，仅作用一侧
        visualize (bool): 是否显示投影图。

    Returns:
        tuple: (channel_names, distance_matrix)
    """
    projection_type = projection_params.get('type', 'stereo') if projection_params else 'stereo'

    # 读取电极数据
    distribution = utils_feature_loading.read_distribution(dataset)
    channel_names = distribution['channel']
    x, y, z = np.array(distribution['x']), np.array(distribution['y']), np.array(distribution['z'])

    if projection_type == 'stereo':
        focal_length = projection_params.get('focal_length', 1.0)
        max_scaling = projection_params.get('max_scaling', 5.0)
        epsilon = 1e-6

        z_norm = (z - np.min(z)) / (np.max(z) - np.min(z))
        scaling = focal_length / (focal_length - z_norm + epsilon)
        scaling = np.clip(scaling, 0, max_scaling)

        x_proj = x * scaling
        y_proj = y * scaling

    elif projection_type == 'azimuthal':
        coords = np.vstack((x, y, z)).T.astype(np.float64)
        coords /= np.linalg.norm(coords, axis=1, keepdims=True)
        x_unit, y_unit, z_unit = coords[:, 0], coords[:, 1], coords[:, 2]

        theta = np.arccos(z_unit)
        phi = np.arctan2(y_unit, x_unit)

        x_proj = theta * np.cos(phi)
        y_proj = theta * np.sin(phi)
    else:
        raise ValueError(f"未知的投影类型: {projection_type}")

    # === 归一化投影坐标 ===
    x_norm = (x_proj - np.min(x_proj)) / (np.max(x_proj) - np.min(x_proj))
    y_norm = (y_proj - np.min(y_proj)) / (np.max(y_proj) - np.min(y_proj))

    # === Azimuthal 特有：仅单侧压缩 y 轴 ===
    if projection_type == 'azimuthal':
        y_compression_factor = projection_params.get('y_compression_factor', 1.0)
        y_compression_direction = projection_params.get('y_compression_direction', 'positive')

        if y_compression_factor != 1.0:
            y_offset = y_norm - 0.5
            if y_compression_direction == 'positive':
                y_offset[y_offset > 0] *= y_compression_factor  # 仅上半区压缩/伸展
            elif y_compression_direction == 'negative':
                y_offset[y_offset < 0] *= y_compression_factor  # 仅下半区压缩/伸展
            y_norm = 0.5 + y_offset

    # 合并投影坐标
    proj_coords = np.vstack((x_norm, y_norm)).T

    if visualize:
        plt.figure(figsize=(6, 6))
        plt.scatter(x_norm, y_norm, c='blue')
        for i, name in enumerate(channel_names):
            plt.text(x_norm[i], y_norm[i], name, fontsize=8, ha='right', va='bottom')
        plt.title(f"Projection: {projection_type}")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.axis("equal")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    # === 计算距离矩阵 ===
    diff = proj_coords[:, np.newaxis, :] - proj_coords[np.newaxis, :, :]
    distance_matrix = np.sqrt(np.sum(diff ** 2, axis=-1))

    if normalize:
        distance_matrix = normalize_matrix(distance_matrix, method=normalization_method)

    return channel_names, distance_matrix

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
    valid_features = {'pcc', 'plv', 'mi'}
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

def fc_matrices_circle_(dataset, subject_range=range(1, 2), experiment_range=range(1, 2), feature='pcc', band='joint', save=False, verbose=True):
    """
    计算 SEED 数据集的相关矩阵，并可选保存。
    
    **新增功能**:
    - 记录总时间
    - 记录每个 experiment 的平均时间

    参数：
    dataset (str): 数据集名称（目前仅支持 'SEED'）。
    subject_range (range): 被试 ID 范围，默认 1~2。
    experiment_range (range): 实验 ID 范围，默认 1~2。
    freq_band (str): 频带类型，可选 'alpha', 'beta', 'gamma' 或 'joint'（默认）。
    save (bool): 是否保存结果，默认 False。
    verbose (bool): 是否打印计时信息，默认 True。

    返回：
    dict: 计算得到的相关矩阵字典。
    """
    # Normalize parameters
    dataset = dataset.upper()
    feature = feature.lower()
    band = band.lower()

    valid_dataset = ['SEED', 'DREAMER']
    if not dataset in valid_dataset:
        raise ValueError("Currently only support SEED and DREAMER datasets")
    valid_feature = ['pcc', 'plv', 'mi']
    if feature not in valid_feature:
        raise ValueError(f"{feature} is not a valid feature. Valid features are: {valid_feature}")
    valid_bands = ['joint', 'theta', 'delta', 'alpha', 'beta', 'gamma']
    if band not in valid_bands:
        raise ValueError(f"{band} is not a valid band. Valid bands are: {valid_bands}")

    def eeg_loader(dataset, subject, experiment):
        if dataset == 'SEED':
            identifier = f'sub{subject}ex{experiment}'
        elif dataset == 'DREAMER':
            identifier = f'sub{subject}'
            
        eeg = utils_eeg_loading.read_eeg_filtered(dataset, identifier)
        
        return identifier, eeg

    fc_matrices_dict = {}

    # **开始计时**
    start_time = time.time()
    experiment_count = 0  # 计数 experiment 计算次数
    total_experiment_time = 0  # 累计 experiment 计算时间

    # For processing DREAMER dataset
    if dataset == 'DREAMER':
        for subject in subject_range:
            experiment_start_time = time.time()  # 记录单次 experiment 开始时间
            experiment_count += 1

            identifier, eeg_data = eeg_loader(dataset, subject, experiment=None)

            if band.lower() in ['delta', 'theta', 'alpha', 'beta', 'gamma']:
                data = np.array(eeg_data[band.lower()])
                if feature.lower() == 'pcc':
                    fc_matrices_dict[identifier] = compute_corr_matrices(data, sampling_rate=200)
                elif feature.lower() == 'plv':
                    fc_matrices_dict[identifier] = compute_plv_matrices(data, sampling_rate=200)
                elif feature.lower() == 'mi':
                    fc_matrices_dict[identifier] = compute_mi_matrices(data, sampling_rate=200)

            elif band.lower() == 'joint':
                fc_matrices_dict[identifier] = {}  # 确保是字典
                for band in ['delta', 'theta', 'alpha', 'beta', 'gamma']:
                    data = np.array(eeg_data[band])
                    if feature.lower() == 'pcc':
                        fc_matrices_dict[identifier][band] = compute_corr_matrices(data, sampling_rate=200)
                    elif feature.lower() == 'plv':
                        fc_matrices_dict[identifier][band] = compute_plv_matrices(data, sampling_rate=200)
                    elif feature.lower() == 'mi':
                        fc_matrices_dict[identifier][band] = compute_mi_matrices(data, sampling_rate=200)

            # **记录单个 experiment 计算时间**
            experiment_time = time.time() - experiment_start_time
            total_experiment_time += experiment_time
            if verbose:
                print(f"Experiment {identifier} completed in {experiment_time:.2f} seconds")

            # **保存计算结果**
            if save:
                path_current = os.getcwd()
                path_parent = os.path.dirname(path_current)
                path_parent_parent = os.path.dirname(path_parent)

                path_folder = os.path.join(path_parent_parent, 'Research_Data', dataset, 'functional connectivity',
                                           f'{feature}_h5')

                """
                将不同频段的功能连接矩阵存储为 HDF5 文件。
                
                参数：
                - fc_matrices_dict (dict): 功能连接矩阵数据。
                - path_folder (str): 存储文件的目标文件夹路径。
                - identifier (str): 数据标识符（如实验名称）。
                
                返回：
                - None
                """
                os.makedirs(path_folder, exist_ok=True)
                file_path_h5 = os.path.join(path_folder, f"{identifier}.h5")

                with h5py.File(file_path_h5, 'w') as f:
                    for band in ["delta", "theta", "alpha", "beta", "gamma"]:
                        f.create_dataset(band, data=fc_matrices_dict[identifier][band], compression="gzip")

                print(f"Data saved to {file_path_h5}")

    elif dataset == 'SEED':
        for subject in subject_range:
            for experiment in experiment_range:
                experiment_start_time = time.time()  # 记录单次 experiment 开始时间
                experiment_count += 1

                identifier, eeg_data = eeg_loader(dataset, subject, experiment)

                if band.lower() in ['delta', 'theta', 'alpha', 'beta', 'gamma']:
                    data = np.array(eeg_data[band.lower()])
                    if feature.lower() == 'pcc':
                        fc_matrices_dict[identifier] = compute_corr_matrices(data, sampling_rate=200)
                    elif feature.lower() == 'plv':
                        fc_matrices_dict[identifier] = compute_plv_matrices(data, sampling_rate=200)
                    elif feature.lower() == 'mi':
                        fc_matrices_dict[identifier] = compute_mi_matrices(data, sampling_rate=200)

                elif band.lower() == 'joint':
                    fc_matrices_dict[identifier] = {}  # 确保是字典
                    for band in ['delta', 'theta', 'alpha', 'beta', 'gamma']:
                        data = np.array(eeg_data[band])
                        if feature.lower() == 'pcc':
                            fc_matrices_dict[identifier][band] = compute_corr_matrices(data, sampling_rate=200)
                        elif feature.lower() == 'plv':
                            fc_matrices_dict[identifier][band] = compute_plv_matrices(data, sampling_rate=200)
                        elif feature.lower() == 'mi':
                            fc_matrices_dict[identifier][band] = compute_mi_matrices(data, sampling_rate=200)

                # **记录单个 experiment 计算时间**
                experiment_time = time.time() - experiment_start_time
                total_experiment_time += experiment_time
                if verbose:
                    print(f"Experiment {identifier} completed in {experiment_time:.2f} seconds")

                # **保存计算结果**
                if save:
                    path_current = os.getcwd()
                    path_parent = os.path.dirname(path_current)
                    path_parent_parent = os.path.dirname(path_parent)

                    path_folder = os.path.join(path_parent_parent, 'Research_Data', 'SEED', 'functional connectivity', f'{feature}_h5')

                    """
                    将不同频段的功能连接矩阵存储为 HDF5 文件。
                
                    参数：
                    - fc_matrices_dict (dict): 功能连接矩阵数据。
                    - path_folder (str): 存储文件的目标文件夹路径。
                    - identifier (str): 数据标识符（如实验名称）。
                
                    返回：
                    - None
                    """
                    os.makedirs(path_folder, exist_ok=True)
                    file_path_h5 = os.path.join(path_folder, f"{identifier}.h5")

                    with h5py.File(file_path_h5, 'w') as f:
                        for band in ["delta", "theta", "alpha", "beta", "gamma"]:
                            f.create_dataset(band, data=fc_matrices_dict[identifier][band], compression="gzip")

                    print(f"Data saved to {file_path_h5}")

    # **计算总时间 & 平均 experiment 时间**
    total_time = time.time() - start_time
    avg_experiment_time = total_experiment_time / experiment_count if experiment_count > 0 else 0

    if verbose:
        print(f"\nTotal time taken: {total_time:.2f} seconds")
        print(f"Average time per experiment: {avg_experiment_time:.2f} seconds")
    
    return fc_matrices_dict

def compute_corr_matrices(eeg_data, sampling_rate, window=1, overlap=0, verbose=True, visualization=True):
    """
    Compute correlation matrices for EEG data using a sliding window approach.
    
    Parameters:
        eeg_data (numpy.ndarray): EEG data with shape (channels, time_samples).
        sampling_rate (int): Sampling rate of the EEG data in Hz.
        window (float): Window size in seconds for segmenting EEG data.
        overlap (float): Overlap fraction between consecutive windows (0 to 1).
        verbose (bool): If True, prints progress.
        visualization (bool): If True, displays correlation matrices.
    
    Returns:
        list of numpy.ndarray: List of correlation matrices for each window.
    """
    # Compute step size based on overlap
    step = int(sampling_rate * window * (1 - overlap))  # Step size for moving window
    segment_length = int(sampling_rate * window)

    # Split EEG data into overlapping windows
    split_segments = [
        eeg_data[:, i:i + segment_length] 
        for i in range(0, eeg_data.shape[1] - segment_length + 1, step)
    ]

    # Compute correlation matrices
    corr_matrices = []
    for idx, segment in enumerate(split_segments):
        if segment.shape[1] < segment_length:
            continue  # Skip incomplete segments
        
        # Compute Pearson correlation
        corr_matrix = np.corrcoef(segment)
        corr_matrices.append(corr_matrix)

        if verbose:
            print(f"Computed correlation matrix {idx + 1}/{len(split_segments)}")

    # Optional: Visualization of correlation matrices
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
        verbose (bool): If True, prints progress.
        visualization (bool): If True, displays PLV matrices.
    
    Returns:
        list of numpy.ndarray: List of PLV matrices for each window.
    """
    step = int(sampling_rate * window * (1 - overlap))  # Step size for moving window
    segment_length = int(sampling_rate * window)

    # Split EEG data into overlapping windows
    split_segments = [
        eeg_data[:, i:i + segment_length] 
        for i in range(0, eeg_data.shape[1] - segment_length + 1, step)
    ]

    plv_matrices = []
    for idx, segment in enumerate(split_segments):
        if segment.shape[1] < segment_length:
            continue  # Skip incomplete segments
        
        # Compute Hilbert transform to obtain instantaneous phase
        analytic_signal = hilbert(segment, axis=1)
        phase_data = np.angle(analytic_signal)  # Extract phase information
        
        # Compute PLV matrix
        num_channels = phase_data.shape[0]
        plv_matrix = np.zeros((num_channels, num_channels))
        
        for ch1 in range(num_channels):
            for ch2 in range(num_channels):
                phase_diff = phase_data[ch1, :] - phase_data[ch2, :]
                plv_matrix[ch1, ch2] = np.abs(np.mean(np.exp(1j * phase_diff)))
        
        plv_matrices.append(plv_matrix)

        if verbose:
            print(f"Computed PLV matrix {idx + 1}/{len(split_segments)}")
    
    # Optional visualization
    if visualization and plv_matrices:
        avg_plv_matrix = np.mean(plv_matrices, axis=0)
        utils_visualization.draw_projection(avg_plv_matrix)
    
    return plv_matrices

from tqdm import tqdm  # 用于进度条显示
def compute_mi_matrices(eeg_data, sampling_rate, window=1, overlap=0, verbose=True, visualization=True):
    """
    Compute Mutual Information (MI) matrices for EEG data using a sliding window approach (optimized with parallelism).

    Parameters:
        eeg_data (numpy.ndarray): EEG data with shape (channels, time_samples).
        sampling_rate (int): Sampling rate of the EEG data in Hz.
        window (float): Window size in seconds for segmenting EEG data.
        overlap (float): Overlap fraction between consecutive windows (0 to 1).
        verbose (bool): If True, prints progress.
        visualization (bool): If True, displays MI matrices.

    Returns:
        list of numpy.ndarray: List of MI matrices for each window.
    """
    if verbose:
        print("Starting Mutual Information computation...")
    
    step = int(sampling_rate * window * (1 - overlap))  # Step size for moving window
    segment_length = int(sampling_rate * window)

    if verbose:
        print("Segmenting EEG data...")
    # Split EEG data into overlapping windows
    split_segments = [
        eeg_data[:, i:i + segment_length]
        for i in range(0, eeg_data.shape[1] - segment_length + 1, step)
    ]

    num_windows = len(split_segments)
    if verbose:
        print(f"Total segments: {num_windows}")

    def compute_mi_matrix(segment):
        """ Compute MI matrix for a single segment (Parallelizable). """
        num_channels = segment.shape[0]
        mi_matrix = np.zeros((num_channels, num_channels))

        def compute_mi(x, y):
            """ Fast mutual information computation using histogram method. """
            hist_2d, _, _ = np.histogram2d(x, y, bins=5)
            pxy = hist_2d / np.sum(hist_2d)
            px = np.sum(pxy, axis=1)
            py = np.sum(pxy, axis=0)
            px_py = np.outer(px, py)
            nonzero = pxy > 0  # Avoid log(0)
            return np.sum(pxy[nonzero] * np.log(pxy[nonzero] / px_py[nonzero]))
        
        # Parallel computation of MI matrix (only upper triangle)
        mi_values = joblib.Parallel(n_jobs=-1)(
            joblib.delayed(compute_mi)(segment[ch1], segment[ch2])
            for ch1 in range(num_channels) for ch2 in range(ch1 + 1, num_channels)
        )

        # Fill the matrix symmetrically
        idx = 0
        for ch1 in range(num_channels):
            for ch2 in range(ch1 + 1, num_channels):
                mi_matrix[ch1, ch2] = mi_matrix[ch2, ch1] = mi_values[idx]
                idx += 1

        np.fill_diagonal(mi_matrix, 1)  # Self-MI is 1
        return mi_matrix

    if verbose:
        print("Computing MI matrices...")

    # Compute MI matrices in parallel with progress tracking
    # mi_matrices = joblib.Parallel(n_jobs=8, verbose=10)(
    #     joblib.delayed(compute_mi_matrix)(segment) for segment in split_segments
    # )
    
    mi_matrices = []
    for segment in tqdm(split_segments, desc="Processing segments", disable=not verbose):
        mi_matrices.append(compute_mi_matrix(segment))
    
    if verbose:
        print(f"Computed {len(mi_matrices)} MI matrices.")

    # Optional visualization
    if visualization and mi_matrices:
        avg_mi_matrix = np.mean(mi_matrices, axis=0)
        utils_visualization.draw_projection(avg_mi_matrix)

    return mi_matrices

def compute_averaged_fcnetwork(feature, subjects=range(1, 16), experiments=range(1, 4), draw=True, save=False):
    # 初始化存储结果的列表
    cmdata_averages_dict = []

    # 用于累积频段的所有数据
    all_alpha_values = []
    all_beta_values = []
    all_gamma_values = []
    all_delta_values = []
    all_theta_values = []

    # 遍历 subject 和 experiment
    for subject in subjects:  # 假设 subjects 是整数
        for experiment in experiments:  # 假设 experiments 是整数
            identifier = f"sub{subject}ex{experiment}"
            print(identifier)
            
            cmdata_alpha = utils_feature_loading.read_fcs(dataset='seed', identifier=identifier, feature=feature,
                                                          band='alpha')
            cmdata_beta = utils_feature_loading.read_fcs(dataset='seed', identifier=identifier, feature=feature,
                                                         band='beta')
            cmdata_gamma = utils_feature_loading.read_fcs(dataset='seed', identifier=identifier, feature=feature,
                                                          band='gamma')
            cmdata_delta = utils_feature_loading.read_fcs(dataset='seed', identifier=identifier, feature=feature,
                                                          band='delta')
            cmdata_theta = utils_feature_loading.read_fcs(dataset='seed', identifier=identifier, feature=feature,
                                                          band='theta')

            # 计算平均值
            cmdata_alpha_averaged = np.mean(cmdata_alpha, axis=0)
            cmdata_beta_averaged = np.mean(cmdata_beta, axis=0)
            cmdata_gamma_averaged = np.mean(cmdata_gamma, axis=0)
            cmdata_delta_averaged = np.mean(cmdata_delta, axis=0)
            cmdata_theta_averaged = np.mean(cmdata_theta, axis=0)

            # 累积数据
            all_alpha_values.append(cmdata_alpha_averaged)
            all_beta_values.append(cmdata_beta_averaged)
            all_gamma_values.append(cmdata_gamma_averaged)
            all_delta_values.append(cmdata_delta_averaged)
            all_theta_values.append(cmdata_theta_averaged)

            # 合并同 subject 同 experiment 的数据
            cmdata_averages_dict.append({
                "subject": subject,
                "experiment": experiment,
                "averages": {
                    "alpha": cmdata_alpha_averaged,
                    "beta": cmdata_beta_averaged,
                    "gamma": cmdata_gamma_averaged,
                    "delta": cmdata_delta_averaged,
                    "theta": cmdata_theta_averaged
                }
            })

    # 计算整个数据集的全局平均值
    global_alpha_average = np.mean(all_alpha_values, axis=0)
    global_beta_average = np.mean(all_beta_values, axis=0)
    global_gamma_average = np.mean(all_gamma_values, axis=0)
    global_joint_average = np.mean(np.stack([global_alpha_average, global_beta_average, 
                                             global_gamma_average], axis=0), axis=0)
    
    global_delta_average = np.mean(all_delta_values, axis=0)
    global_theta_average = np.mean(all_theta_values, axis=0)
    global_joint_average = np.mean(np.stack([global_alpha_average, global_beta_average, 
                                             global_gamma_average, global_delta_average, 
                                             global_theta_average], axis=0), axis=0)

    fc_matrices = {'alpha': global_alpha_average, 
                   'beta': global_beta_average, 
                   'gamma': global_gamma_average,
                   'delta': global_delta_average,
                   'theta': global_theta_average,
                   'joint': global_joint_average
                   }
    
    if draw:
        # 输出结果
        utils_visualization.draw_projection(global_joint_average)

    if save:
        save_results('seed', 'functional connectivity\global_averaged', f'fc_global_averaged_{feature}_h5', fc_matrices)

        print("Results saved")

    return global_joint_average

def compute_averaged_fcnetwork_mat(feature, subjects=range(1, 16), experiments=range(1, 4), draw=True, save=False):
    # 初始化存储结果的列表
    cmdata_averages_dict = []

    # 用于累积频段的所有数据
    all_alpha_values, all_beta_values, all_gamma_values = [], [], []

    # 遍历 subject 和 experiment
    for subject in subjects:  # 假设 subjects 是整数
        for experiment in experiments:  # 假设 experiments 是整数
            identifier = f"sub{subject}ex{experiment}"
            print(identifier)
            
            features = utils_feature_loading.read_fcs_mat('seed', identifier, feature)
            cmdata_alpha = features['alpha']
            cmdata_beta = features['beta']
            cmdata_gamma = features['gamma']

            # 计算平均值
            cmdata_alpha_averaged = np.mean(cmdata_alpha, axis=0)
            cmdata_beta_averaged = np.mean(cmdata_beta, axis=0)
            cmdata_gamma_averaged = np.mean(cmdata_gamma, axis=0)

            # 累积数据
            all_alpha_values.append(cmdata_alpha_averaged)
            all_beta_values.append(cmdata_beta_averaged)
            all_gamma_values.append(cmdata_gamma_averaged)

            # 合并同 subject 同 experiment 的数据
            cmdata_averages_dict.append({
                "subject": subject,
                "experiment": experiment,
                "averages": {
                    "alpha": cmdata_alpha_averaged,
                    "beta": cmdata_beta_averaged,
                    "gamma": cmdata_gamma_averaged
                }
            })

    # 计算整个数据集的全局平均值
    global_alpha_average = np.mean(all_alpha_values, axis=0)
    global_beta_average = np.mean(all_beta_values, axis=0)
    global_gamma_average = np.mean(all_gamma_values, axis=0)
    global_joint_average = np.mean(np.stack([global_alpha_average, global_beta_average,
                                             global_gamma_average], axis=0), axis=0)

    global_joint_average = np.mean(np.stack([global_alpha_average, global_beta_average,
                                             global_gamma_average],  axis=0), axis=0)

    fc_matrices = {'alpha': global_alpha_average,
                   'beta': global_beta_average, 
                   'gamma': global_gamma_average,
                   'joint': global_joint_average
                   }
    
    if draw:
        # 输出结果
        utils_visualization.draw_projection(global_joint_average)

    if save:
        save_results('seed', 'global_averaged', f'fc_global_averaged_{feature}_mat', fc_matrices)

        print("Results saved")

    return global_joint_average
    
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
    filter_eeg_and_save_circle('seed', subject_range=range(1,16), experiment_range=range(1,4), verbose=True, save=True)
    
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
    # # mi_sample_dreamer = compute_mi_matrices(eeg_sample_dreamer, samplingrate=128)
    
    # %% Label Engineering
    labels_seed = utils_feature_loading.read_labels('seed')
    labels_dreamer = utils_feature_loading.read_labels('dreamer')
    labels_dreamer_ = generate_labels()
    
    # %% Interpolation
    
    # %% Feature Engineering; Computation circles
    fc_pcc_matrices_seed = fc_matrices_circle('SEED', feature='pcc', save=True, subject_range=range(1, 16), experiment_range=range(1, 4))
    # fc_plv_matrices_seed = fc_matrices_circle('SEED', feature='plv', save=False, subject_range=range(1, 2), experiment_range=range(1, 2))
    # fc_mi_matrices_seed = fc_matrices_circle('SEED', feature='mi', save=False, subject_range=range(1, 2), experiment_range=range(1, 2))

    # fc_pcc_matrices_dreamer = fc_matrices_circle('dreamer', feature='pcc', save=True, subject_range=range(1, 2))
    # fc_plv_matrices_dreamer = fc_matrices_circle('dreamer', feature='plv', save=True, subject_range=range(1, 2))
    # fc_mi_matrices_dreamer = fc_matrices_circle('dreamer', feature='mi', save=True, subject_range=range(1, 2))
    
    # %% Feature Engineering; Compute Average CM
    # global_joint_average = compute_averaged_fcnetwork(feature='pcc', subjects=range(1, 11), experiments=range(1, 4), save=True)
    
    # %% End program actions
    # utils.end_program_actions(play_sound=True, shutdown=False, countdown_seconds=120)