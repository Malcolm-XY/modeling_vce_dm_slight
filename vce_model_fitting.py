# -*- coding: utf-8 -*-
"""
Created on Wed Mar 26 17:30:25 2025

@author: usouu
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_filter
from scipy.optimize import differential_evolution

import feature_engineering
import vce_modeling
import cw_manager

# %% Normalize and prune CW
def prune_cw(cw, normalize_method='minmax', transform_method='boxcox'):
    cw = feature_engineering.normalize_matrix(cw, transform_method)
    cw = feature_engineering.normalize_matrix(cw, normalize_method)
    return cw

# %% Compute CM, get CW as Agent of GCM and RCM
from utils import utils_feature_loading, utils_visualization
def preprocessing_cm_global_averaged(cm_global_averaged, coordinates):
    # Global averaged connectivity matrix; For subsquent fitting computation
    cm_global_averaged = np.abs(cm_global_averaged)
    cm_global_averaged = feature_engineering.normalize_matrix(cm_global_averaged)
    
    # Rebuild CM; By removing bads and Gaussian smoothing
    param = {
    'method': 'zscore', 'threshold': 2.5,
    'kernel': 'gaussian',  # idw or 'gaussian'
    'sigma': 5.0,  # only used for gaussian
    'manual_bad_idx': []}
    
    cm_global_averaged = feature_engineering.rebuild_features(cm_global_averaged, coordinates, param, True)
    
    # 2D Gaussian Smooth CM
    # connectivity_matrix = gaussian_filter(connectivity_matrix, sigma=0.5)
    
    # Spatial Gaussian Smooth CM
    cm_global_averaged = feature_engineering.spatial_gaussian_smoothing_on_fc_matrix(cm_global_averaged, coordinates, 5, True)
    
    return cm_global_averaged

def prepare_target_and_inputs(feature='pcc', ranking_method='label_driven_mi', idxs_manual_remove=None):
    """
    Prepares smoothed channel weights, distance matrix, and global averaged connectivity matrix,
    with optional removal of specified bad channels.

    Parameters
    ----------
    feature : str
        Connectivity feature type (e.g., 'PCC').
    ranking_method : str
        Method for computing channel importance weights.
    idxs_manual_remove : list of int or None
        Indices of channels to manually remove from all matrices/vectors.

    Returns
    -------
    cw_target_smooth : np.ndarray of shape (n,)
    distance_matrix : np.ndarray of shape (n, n)
    cm_global_averaged : np.ndarray of shape (n, n)
    """
    # === 0. Electrodes; Remove specified channels
    electrodes = np.array(utils_feature_loading.read_distribution('seed')['channel'])
    electrodes = feature_engineering.remove_idx_manual(electrodes, idxs_manual_remove)
    
    # === 1. Target channel weight
    channel_weights = cw_manager.read_channel_weight_LD(identifier=ranking_method, sort=False)['ams']
    cw_target = prune_cw(channel_weights.to_numpy())
    # ==== 1.1 Remove specified channels
    cw_target = feature_engineering.remove_idx_manual(cw_target, idxs_manual_remove)
    # === 1.2 Coordinates and smoothing
    coordinates = utils_feature_loading.read_distribution('seed')
    coordinates = coordinates.drop(idxs_manual_remove)
    cw_target_smooth = feature_engineering.spatial_gaussian_smoothing_on_vector(cw_target, coordinates, 2.0)

    # === 2. Distance matrix
    _, distance_matrix = feature_engineering.compute_distance_matrix(dataset="seed", projection_params={"type": "3d"})
    # === 2.1 Remove specified channels
    distance_matrix = feature_engineering.remove_idx_manual(distance_matrix, idxs_manual_remove)
    # === 2.2 Normalization
    distance_matrix = feature_engineering.normalize_matrix(distance_matrix)

    # === 3. Connectivity matrix
    connectivity_matrix_global_joint_averaged = utils_feature_loading.read_fcs_global_average('seed', feature, 'joint', 'mat')['joint']
    
    # === 3.1 Remove specified channels
    cm_global_averaged = feature_engineering.remove_idx_manual(connectivity_matrix_global_joint_averaged, idxs_manual_remove)
    # === 3.2 Smoothing
    cm_global_averaged = preprocessing_cm_global_averaged(cm_global_averaged, coordinates)

    return electrodes, cw_target_smooth, distance_matrix, cm_global_averaged

# Here utilized VCE Model/FM=M(DM) Model
def compute_cw_fitting(method, params_dict, distance_matrix, connectivity_matrix, RCM='differ'):
    """
    Compute cw_fitting based on selected RCM method: differ, linear, or linear_ratio.
    """
    RCM = RCM.lower()

    # Step 1: Calculate FM
    factor_matrix = vce_modeling.compute_volume_conduction_factors_advanced_model(distance_matrix, method, params_dict)
    
    # *************************** here may should be revised 20250508
    factor_matrix = feature_engineering.normalize_matrix(factor_matrix)

    # Step 2: Calculate RCM
    cm, fm = connectivity_matrix, factor_matrix
    e = 1e-6  # Small value to prevent division by zero

    if RCM == 'differ':
        cm_recovered = cm - fm
    elif RCM == 'linear':
        scale_a = params_dict.get('scale_a', 1.0)
        cm_recovered = cm + scale_a * fm
    elif RCM == 'linear_ratio':
        scale_a = params_dict.get('scale_a', 1.0)
        scale_b = params_dict.get('scale_b', 1.0)
        cm_recovered = cm + scale_a * fm + scale_b * cm / (gaussian_filter(fm, sigma=1) + e)
    else:
        raise ValueError(f"Unsupported RCM mode: {RCM}")

    # Step 3: Normalize RCM
    cm_recovered = feature_engineering.normalize_matrix(cm_recovered)

    # Step 4: Compute CW
    global cw_fitting
    cw_fitting = np.mean(cm_recovered, axis=0)
    cw_fitting = prune_cw(cw_fitting)

    return cw_fitting

# %% Optimization
def optimize_and_store(method, loss_fn, bounds, param_keys, distance_matrix, connectivity_matrix, RCM='differ'):
    res = differential_evolution(loss_fn, bounds=bounds, strategy='best1bin', maxiter=1000)
    params = dict(zip(param_keys, res.x))
    
    result = {'params': params, 'loss': res.fun}
    cw_fitting = compute_cw_fitting(method, params, distance_matrix, connectivity_matrix, RCM)
    
    return result, cw_fitting

def loss_fn_template(method_name, param_dict_fn, cw_target, distance_matrix, connectivity_matrix, RCM):
    def loss_fn(params):
        loss = np.mean((compute_cw_fitting(method_name, param_dict_fn(params), distance_matrix, connectivity_matrix, RCM) - cw_target) ** 2)
        return loss
    return loss_fn

class FittingConfig:
    """
    Configuration for fitting models.
    Provides param_names, bounds, and automatic param_func.
    """
    
    @staticmethod
    def get_config(model_type: str, recovery_type: str):
        """
        Get the config dictionary based on model type and recovery type.
    
        Args:
            model_type (str): 'basic' or 'advanced'
            recovery_type (str): 'differ', 'linear', or 'linear_ratio'
    
        Returns:
            dict: Corresponding config dictionary
    
        Raises:
            ValueError: If input type is invalid
        """
        model_type = model_type.lower()
        recovery_type = recovery_type.lower()
    
        if model_type == 'basic' and recovery_type == 'differ':
            return FittingConfig.config_basic_model_differ_recovery
        elif model_type == 'advanced' and recovery_type == 'differ':
            return FittingConfig.config_advanced_model_differ_recovery
        elif model_type == 'basic' and recovery_type == 'linear':
            return FittingConfig.config_basic_model_linear_recovery
        elif model_type == 'advanced' and recovery_type == 'linear':
            return FittingConfig.config_advanced_model_linear_recovery
        elif model_type == 'basic' and recovery_type == 'linear_ratio':
            return FittingConfig.config_basic_model_linear_ratio_recovery
        elif model_type == 'advanced' and recovery_type == 'linear_ratio':
            return FittingConfig.config_advanced_model_linear_ratio_recovery
        else:
            raise ValueError(f"Invalid model_type '{model_type}' or recovery_type '{recovery_type}'")
    
    @staticmethod
    def make_param_func(param_names):
        """Auto-generate param_func based on param_names."""
        return lambda p: {name: p[i] for i, name in enumerate(param_names)}

    config_basic_model_differ_recovery = {
        'exponential': {
            'param_names': ['sigma'],
            'bounds': [(0.1, 20.0)],
        },
        'gaussian': {
            'param_names': ['sigma'],
            'bounds': [(0.1, 20.0)],
        },
        'inverse': {
            'param_names': ['sigma', 'alpha'],
            'bounds': [(0.1, 20.0), (0.1, 5.0)],
        },
        'powerlaw': {
            'param_names': ['alpha'],
            'bounds': [(0.1, 10.0)],
        },
        'rational_quadratic': {
            'param_names': ['sigma', 'alpha'],
            'bounds': [(0.1, 20.0), (0.1, 10.0)],
        },
        'generalized_gaussian': {
            'param_names': ['sigma', 'beta'],
            'bounds': [(0.1, 20.0), (0.1, 5.0)],
        },
        'sigmoid': {
            'param_names': ['mu', 'beta'],
            'bounds': [(0.1, 10.0), (0.1, 5.0)],
        },
    }

    config_advanced_model_differ_recovery = {
        'exponential': {
            'param_names': ['sigma', 'deviation', 'offset'],
            'bounds': [(0.1, 20.0), (-1.0, 1.0), (-1.0, 1.0)],
        },
        'gaussian': {
            'param_names': ['sigma', 'deviation', 'offset'],
            'bounds': [(0.1, 20.0), (-1.0, 1.0), (-1.0, 1.0)],
        },
        'inverse': {
            'param_names': ['sigma', 'alpha', 'deviation', 'offset'],
            'bounds': [(0.1, 20.0), (0.1, 5.0), (1e-6, 1.0), (-1.0, 1.0)],
        },
        'powerlaw': {
            'param_names': ['alpha', 'deviation', 'offset'],
            'bounds': [(0.1, 10.0), (1e-6, 1.0), (-1.0, 1.0)],
        },
        'rational_quadratic': {
            'param_names': ['sigma', 'alpha', 'deviation', 'offset'],
            'bounds': [(0.1, 20.0), (0.1, 10.0), (-1.0, 1.0), (-1.0, 1.0)],
        },
        'generalized_gaussian': {
            'param_names': ['sigma', 'beta', 'deviation', 'offset'],
            'bounds': [(0.1, 20.0), (0.1, 5.0), (1e-6, 1.0), (-1.0, 1.0)],
        },
        'sigmoid': {
            'param_names': ['mu', 'beta', 'deviation', 'offset'],
            'bounds': [(0.1, 10.0), (0.1, 5.0), (-1.0, 1.0), (-1.0, 1.0)],
        },
    }

    config_basic_model_linear_recovery = {
        'exponential': {
            'param_names': ['sigma', 'scale_a'],
            'bounds': [(0.1, 20.0), (-1.0, 1.0)],
        },
        'gaussian': {
            'param_names': ['sigma', 'scale_a'],
            'bounds': [(0.1, 20.0), (-1.0, 1.0)],
        },
        'inverse': {
            'param_names': ['sigma', 'alpha', 'scale_a'],
            'bounds': [(0.1, 20.0), (0.1, 5.0), (-1.0, 1.0)],
        },
        'powerlaw': {
            'param_names': ['alpha', 'scale_a'],
            'bounds': [(0.1, 10.0), (-1.0, 1.0)],
        },
        'rational_quadratic': {
            'param_names': ['sigma', 'alpha', 'scale_a'],
            'bounds': [(0.1, 20.0), (0.1, 10.0), (-1.0, 1.0)],
        },
        'generalized_gaussian': {
            'param_names': ['sigma', 'beta', 'scale_a'],
            'bounds': [(0.1, 20.0), (0.1, 5.0), (-1.0, 1.0)],
        },
        'sigmoid': {
            'param_names': ['mu', 'beta', 'scale_a'],
            'bounds': [(0.1, 10.0), (0.1, 5.0), (-1.0, 1.0)],
        },
    }

    config_advanced_model_linear_recovery = {
        'exponential': {
            'param_names': ['sigma', 'deviation', 'offset', 'scale_a'],
            'bounds': [(0.1, 20.0), (-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)],
        },
        'gaussian': {
            'param_names': ['sigma', 'deviation', 'offset', 'scale_a'],
            'bounds': [(0.1, 20.0), (-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)],
        },
        'inverse': {
            'param_names': ['sigma', 'alpha', 'deviation', 'offset', 'scale_a'],
            'bounds': [(0.1, 20.0), (0.1, 5.0), (1e-6, 1.0), (-1.0, 1.0), (-1.0, 1.0)],
        },
        'powerlaw': {
            'param_names': ['alpha', 'deviation', 'offset', 'scale_a'],
            'bounds': [(0.1, 10.0), (1e-6, 1.0), (-1.0, 1.0), (-1.0, 1.0)],
        },
        'rational_quadratic': {
            'param_names': ['sigma', 'alpha', 'deviation', 'offset', 'scale_a'],
            'bounds': [(0.1, 20.0), (0.1, 10.0), (-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)],
        },
        'generalized_gaussian': {
            'param_names': ['sigma', 'beta', 'deviation', 'offset', 'scale_a'],
            'bounds': [(0.1, 20.0), (0.1, 5.0), (1e-6, 1.0), (-1.0, 1.0), (-1.0, 1.0)],
        },
        'sigmoid': {
            'param_names': ['mu', 'beta', 'deviation', 'offset', 'scale_a'],
            'bounds': [(0.1, 10.0), (0.1, 5.0), (-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)],
        },
    }

    config_basic_model_linear_ratio_recovery = {
        'exponential': {
            'param_names': ['sigma', 'scale_a', 'scale_b'],
            'bounds': [(0.1, 20.0), (-1.0, 1.0), (0.01, 2.0)],
        },
        'gaussian': {
            'param_names': ['sigma', 'scale_a', 'scale_b'],
            'bounds': [(0.1, 20.0), (-1.0, 1.0), (0.01, 2.0)],
        },
        'inverse': {
            'param_names': ['sigma', 'alpha', 'scale_a', 'scale_b'],
            'bounds': [(0.1, 20.0), (0.1, 5.0), (-1.0, 1.0), (0.01, 2.0)],
        },
        'powerlaw': {
            'param_names': ['alpha', 'scale_a', 'scale_b'],
            'bounds': [(0.1, 10.0), (-1.0, 1.0), (0.01, 2.0)],
        },
        'rational_quadratic': {
            'param_names': ['sigma', 'alpha', 'scale_a', 'scale_b'],
            'bounds': [(0.1, 20.0), (0.1, 10.0), (-1.0, 1.0), (0.01, 2.0)],
        },
        'generalized_gaussian': {
            'param_names': ['sigma', 'beta', 'scale_a', 'scale_b'],
            'bounds': [(0.1, 20.0), (0.1, 5.0), (-1.0, 1.0), (0.01, 2.0)],
        },
        'sigmoid': {
            'param_names': ['mu', 'beta', 'scale_a', 'scale_b'],
            'bounds': [(0.1, 10.0), (0.1, 5.0), (-1.0, 1.0), (0.01, 2.0)],
        },
    }

    config_advanced_model_linear_ratio_recovery = {
        'exponential': {
            'param_names': ['sigma', 'deviation', 'offset', 'scale_a', 'scale_b'],
            'bounds': [(0.1, 20.0), (-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0), (0.01, 2.0)],
        },
        'gaussian': {
            'param_names': ['sigma', 'deviation', 'offset', 'scale_a', 'scale_b'],
            'bounds': [(0.1, 20.0), (-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0), (0.01, 2.0)],
        },
        'inverse': {
            'param_names': ['sigma', 'alpha', 'deviation', 'offset', 'scale_a', 'scale_b'],
            'bounds': [(0.1, 20.0), (0.1, 5.0), (1e-6, 1.0), (-1.0, 1.0), (-1.0, 1.0), (0.01, 2.0)],
        },
        'powerlaw': {
            'param_names': ['alpha', 'deviation', 'offset', 'scale_a', 'scale_b'],
            'bounds': [(0.1, 10.0), (1e-6, 1.0), (-1.0, 1.0), (-1.0, 1.0), (0.01, 2.0)],
        },
        'rational_quadratic': {
            'param_names': ['sigma', 'alpha', 'deviation', 'offset', 'scale_a', 'scale_b'],
            'bounds': [(0.1, 20.0), (0.1, 10.0), (-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0), (0.01, 2.0)],
        },
        'generalized_gaussian': {
            'param_names': ['sigma', 'beta', 'deviation', 'offset', 'scale_a', 'scale_b'],
            'bounds': [(0.1, 20.0), (0.1, 5.0), (1e-6, 1.0), (-1.0, 1.0), (-1.0, 1.0), (0.01, 2.0)],
        },
        'sigmoid': {
            'param_names': ['mu', 'beta', 'deviation', 'offset', 'scale_a', 'scale_b'],
            'bounds': [(0.1, 10.0), (0.1, 5.0), (-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0), (0.01, 2.0)],
        },
    }

def fitting_model(model_type='basic', recovery_type='differ', cw_target=None, distance_matrix=None, connectivity_matrix=None):
    """
    Perform model fitting across multiple methods.

    Args:
        model_type (str): 'basic' or 'advanced'
        recovery_type (str): 'differ', 'linear', 'linear_ratio'
        cw_target (np.ndarray): Target feature vector
        distance_matrix (np.ndarray): Distance matrix
        connectivity_matrix (np.ndarray): Connectivity matrix

    Returns:
        results (dict): Optimized parameters and losses
        cws_fitting (dict): Fitted CW vectors
    """

    results, cws_fitting = {}, {}

    # Load fitting configuration
    fitting_config = FittingConfig.get_config(model_type, recovery_type)

    for method, config in fitting_config.items():
        print(f"Fitting Method: {method}")

        param_names = config['param_names']
        bounds = config['bounds']
        param_func = FittingConfig.make_param_func(param_names)

        # Build loss function
        loss_fn = loss_fn_template(method, param_func, cw_target, distance_matrix, connectivity_matrix, RCM=recovery_type)

        # Optimize
        try:
            results[method], cws_fitting[method] = optimize_and_store(
                method,
                loss_fn,
                bounds,
                param_names,
                distance_matrix,
                connectivity_matrix,
                RCM=recovery_type
            )
        except Exception as e:
            print(f"[{method.upper()}] Optimization failed: {e}")
            results[method], cws_fitting[method] = None, None

    print("\n=== Fitting Results of All Models (Minimum MSE) ===")
    for method, result in results.items():
        if result is not None:
            print(f"[{method.upper()}] Best Parameters: {result['params']}, Minimum MSE: {result['loss']:.6f}")
        else:
            print(f"[{method.upper()}] Optimization Failed.")

    return results, cws_fitting

# %% Sort
def sort_ams(ams, labels, original_labels=None):
    dict_ams_original = pd.DataFrame({'labels': labels, 'ams': ams})
    
    dict_ams_sorted = dict_ams_original.sort_values(by='ams', ascending=False).reset_index()
            
    # idxs_in_original = []
    # for label in dict_ams_sorted['labels']:
    #     idx_in_original = list(original_labels).index(label)
    #     idxs_in_original.append(idx_in_original)
    
    dict_ams_summary = dict_ams_original.copy()
    # dict_ams_summary['idex_in_original'] = idxs_in_original
    
    dict_ams_summary = pd.concat([dict_ams_summary, dict_ams_sorted], axis=1)
    
    return dict_ams_summary

# %% Visualization
# scatter
from sklearn.metrics import mean_squared_error
def draw_scatter_comparison(x, A, B, pltlabels={'title':'title', 
                                                'label_x':'label_x', 'label_y':'label_y', 
                                                'label_A':'label_A', 'label_B':'label_B'}):
    # Compute MSE
    mse = mean_squared_error(A, B)
    
    # Labels
    title = pltlabels.get('title')
    label_x = pltlabels.get('label_x')
    label_y = pltlabels.get('label_y')
    label_A = pltlabels.get('label_A')
    label_B = pltlabels.get('label_B')
    
    # Plot
    plt.figure(figsize=(10, 4))
    plt.plot(x, A, label=label_A, linestyle='--', marker='o', color='black')
    plt.plot(x, B, label=label_B, marker='x', linestyle=':')
    plt.title(f"{title} - MSE: {mse:.4f}")
    plt.xlabel(label_x)
    plt.ylabel(label_y)
    plt.xticks(rotation=60)
    plt.tick_params(axis='x', labelsize=8)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def draw_scatter_multi_method(x, A, fittings_dict, pltlabels=None, save_path=None):
    """
    在同一张图中绘制目标通道权重与多个拟合结果的比较图。

    Args:
        x (array-like): 横轴标签（如电极名或编号）
        A: target (array-like): 目标通道权重
        fittings_dict (dict): {method_name: cw_fitting_array}
        pltlabels (dict): {'title': str, 'label_x': str, 'label_y': str, 'label_target': str}
        save_path (str or None): 若指定路径则保存图像（如 'figs/cw_comparison.pdf'）
    """
    # 默认标签
    if pltlabels is None:
        pltlabels = {'title': 'Comparison of Channel Weights across various Models',
                     'label_x': 'Electrodes', 'label_y': 'Channel Weight',
                     'label_A': 'target', 'label_B': 'label_B'}

    # 提取标签
    title = pltlabels.get('title', '')
    label_x = pltlabels.get('label_x', '')
    label_y = pltlabels.get('label_y', '')
    label_A = pltlabels.get('label_A', '')
    label_B = pltlabels.get('label_B', '')

    # 绘图
    plt.figure(figsize=(10, 4))
    plt.plot(x, A, label=label_A, linestyle='-', marker='o', color='black')

    # 绘制多个拟合曲线
    for method, B in fittings_dict.items():
        mse = mean_squared_error(A, B)
        label_B = f"{method} (MSE={mse:.4f})"
        plt.plot(x, B, label=label_B, linestyle='--', marker='x')  # 颜色自动分配

    # 图形设置
    plt.title(title)
    plt.xlabel(label_x)
    plt.ylabel(label_y)
    plt.xticks(rotation=60)
    plt.tick_params(axis='x', labelsize=8)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # 保存或显示
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"[INFO] 图像已保存到 {save_path}")
    else:
        plt.show()

def draw_scatter_subplots_vertical(x, A, fittings_dict, pltlabels=None, save_path=None):
    """
    绘制目标通道权重与多个拟合方法的对比子图（单列多行），适用于论文展示。

    Args:
        x (array-like): 横轴坐标（如电极标签）
        A: target (array-like): 目标通道权重
        fittings_dict (dict): {method_name: cw_fitting_array}
        pltlabels (dict): {'title': str, 'label_x': str, 'label_y': str, 'label_target': str}
        save_path (str or None): 若指定路径则保存图像（如 'figs/cw_subplot.pdf'）
    """
    if pltlabels is None:
        pltlabels = {'title': 'Comparison of Channel Weights across various Models',
                     'label_x': 'Electrodes', 'label_y': 'Channel Weight',
                     'label_A': 'target', 'label_B': 'label_B'}

    label_x = pltlabels.get('label_x', '')
    label_y = pltlabels.get('label_y', '')
    label_A = pltlabels.get('label_A', '')
    label_B = pltlabels.get('label_B', '')
    suptitle = pltlabels.get('title', '')

    methods = list(fittings_dict.keys())
    n_methods = len(methods)

    fig, axes = plt.subplots(nrows=n_methods, ncols=1, figsize=(10, 2.5 * n_methods), sharex=True)

    if n_methods == 1:
        axes = [axes]  # 保证可迭代性

    for ax, method in zip(axes, methods):
        B = fittings_dict[method]
        mse = mean_squared_error(A, B)

        ax.plot(x, A, label=label_A, linestyle='-', marker='o', color='black')
        label_B = f'CW of RCM; FM model: {method} (MSE={mse:.4f})'
        ax.plot(x, B, label=label_B, linestyle='--', marker='x')

        ax.set_ylabel(label_y)
        ax.grid(True)
        ax.legend(loc='best', fontsize=8)

    # 只设置最后一张图的 x label
    axes[-1].set_xlabel(label_x)
    axes[-1].tick_params(axis='x', labelrotation=60, labelsize=8)

    fig.suptitle(suptitle, fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"[INFO] 子图已保存到 {save_path}")
    else:
        plt.show()

# topography
import mne
def plot_cw_topomap(
    amps_df, label_col='labels', amp_col='ams',
    montage=None, distribution_df=None, normalize=True,
    title='Topomap'):
    """
    绘制 EEG 通道权重脑图。如果提供 distribution_df，则自动创建 montage。

    Args:
        amps_df (pd.DataFrame): 包含通道名和权重的 DataFrame。
        label_col (str): 通道名列名。
        amp_col (str): 权重列名。
        montage (mne.channels.DigMontage): 如果已有 montage，可直接传入。
        distribution_df (pd.DataFrame): 包含 'channel', 'x', 'y', 'z' 列，用于构建自定义 montage。
        title (str): 图标题。
        normalize (bool): 是否将 distribution_df 中的坐标归一化。
    """
    # Step 1: 从 distribution_df 创建 montage（若提供）
    if distribution_df is not None:
        required_cols = {'channel', 'x', 'y', 'z'}
        if not required_cols.issubset(distribution_df.columns):
            raise ValueError(f"distribution_df must contain columns: {required_cols}")
        ch_pos = {}
        for _, row in distribution_df.iterrows():
            pos = np.array([row['x'], row['y'], row['z']], dtype=np.float64)
            if normalize:
                norm = np.linalg.norm(pos)
                if norm > 0:
                    pos = pos / norm
            ch_pos[row['channel']] = pos
        montage = mne.channels.make_dig_montage(ch_pos=ch_pos, coord_frame='head')

    if montage is None:
        raise ValueError("必须提供 montage 或 distribution_df 参数之一。")

    # Step 2: 提取数据
    all_labels = amps_df[label_col].iloc[:, 0].values.tolist()
    all_amplitudes = amps_df[amp_col].iloc[:, 0].values
    amplitudes = np.array(all_amplitudes)

    # Step 3: 过滤无效通道
    available_labels = set(montage.ch_names)
    valid_indices, invalid_labels = [], []
    for i, lbl in enumerate(all_labels):
        if lbl in available_labels:
            valid_indices.append(i)
        else:
            invalid_labels.append(lbl)

    if len(valid_indices) == 0:
        print("[WARNING] 无可绘制通道。请检查通道名格式。")
        if invalid_labels:
            print("无效通道名如下：", invalid_labels)
        return

    if invalid_labels:
        print(f"[INFO] 以下通道未被绘制（未在 montage 中找到）: {invalid_labels}")

    used_labels = [all_labels[i] for i in valid_indices]
    used_amplitudes = amplitudes[valid_indices]

    # Step 4: 创建 evoked 对象
    info = mne.create_info(ch_names=used_labels, sfreq=1000, ch_types='eeg')
    evoked = mne.EvokedArray(used_amplitudes[:, np.newaxis], info)
    evoked.set_montage(montage)

    # Step 5: 绘图
    fig = evoked.plot_topomap(times=0, scalings=1, cmap='viridis', time_format='', show=False, sphere=(0., 0., 0., 1.1))

    fig.suptitle(title, fontsize=14)
    plt.show()

import math
def plot_joint_topomaps(
    amps_dict,  # dict[str, pd.DataFrame]
    label_col='labels', amp_col='ams',
    montage=None, distribution_df=None,
    normalize=True, title='Joint Topomap'
):
    """
    按每行两张图的方式绘制多个方法的 EEG 通道权重联合图。

    Args:
        amps_dict (dict): 例如 {'method1': df1, 'method2': df2, ...}，每个 df 包含通道名和权重。
        label_col (str): DataFrame 中通道名列名。
        amp_col (str): DataFrame 中权重列名。
        montage (mne.channels.DigMontage): 若已存在可重用 montage。
        distribution_df (pd.DataFrame): 若未提供 montage，可提供坐标 DataFrame 创建之。
        normalize (bool): 是否对通道坐标归一化。
        title (str): 整体图标题。
    """
    if montage is None:
        if distribution_df is None:
            raise ValueError("必须提供 montage 或 distribution_df 之一。")
        required_cols = {'channel', 'x', 'y', 'z'}
        if not required_cols.issubset(distribution_df.columns):
            raise ValueError(f"distribution_df 必须包含列: {required_cols}")
        ch_pos = {}
        for _, row in distribution_df.iterrows():
            pos = np.array([row['x'], row['y'], row['z']], dtype=np.float64)
            if normalize:
                norm = np.linalg.norm(pos)
                if norm > 0:
                    pos = pos / norm
            ch_pos[row['channel']] = pos
        montage = mne.channels.make_dig_montage(ch_pos=ch_pos, coord_frame='head')

    num_plots = len(amps_dict)
    num_cols = 2
    num_rows = math.ceil(num_plots / num_cols)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(8, 4 * num_rows))
    axes = axes.flatten()  # 保证一维数组形式

    for ax, (method, df) in zip(axes, amps_dict.items()):
        # 提取数据
        labels = df[label_col].iloc[:, 0].values.tolist()
        amps = df[amp_col].iloc[:, 0].values

        # 过滤无效通道
        available_labels = set(montage.ch_names)
        valid_indices = [i for i, l in enumerate(labels) if l in available_labels]
        if not valid_indices:
            print(f"[WARNING] {method}: 无有效通道")
            continue

        used_labels = [labels[i] for i in valid_indices]
        used_amps = amps[valid_indices]

        # 创建 evoked 对象
        info = mne.create_info(ch_names=used_labels, sfreq=1000, ch_types='eeg')
        evoked = mne.EvokedArray(used_amps[:, np.newaxis], info)
        evoked.set_montage(montage)

        # 绘图到指定子图
        mne.viz.plot_topomap(evoked.data[:, 0], evoked.info, axes=ax,
                             show=False, cmap='viridis', sphere=(0., 0., 0., 1.1))
        ax.set_title(method, fontsize=12)

    # 关闭多余子图
    for i in range(num_plots, len(axes)):
        fig.delaxes(axes[i])

    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.show()

# %% Save
import os
def save_fitting_results(results, save_dir='results', file_name='fitting_results.xlsx'):
    """
    Save fitting results (parameters and losses) into an Excel or TXT file.
    """
    os.makedirs(save_dir, exist_ok=True)
    results_path = os.path.join(save_dir, file_name)
    
    # Organize results into DataFrame
    data = []
    for method, result in results.items():
        if result is None:
            continue
        row = {'method': method.upper()}
        row.update(result['params'])
        row['loss'] = result['loss']
        data.append(row)
    
    df = pd.DataFrame(data)

    # Save
    if file_name.endswith('.xlsx'):
        df.to_excel(results_path, index=False)
    elif file_name.endswith('.txt'):
        df.to_csv(results_path, sep='\t', index=False)
    else:
        raise ValueError("Unsupported file extension. Use .xlsx or .txt")

    print(f"Fitting results saved to {results_path}")

def save_channel_weights(cws_fitting, save_dir='results', file_name='channel_weights.xlsx'):
    """
    将包含多个 DataFrame 的字典保存为一个 Excel 文件，不同的 sheet 存储不同的 DataFrame。

    Args:
        cws_fitting (dict): 键是 sheet 名，值是 DataFrame 或可以转换成 DataFrame 的数据结构。
        save_dir (str): 保存目录，默认为 'results'。
        file_name (str): 保存的文件名，默认为 'channel_weights.xlsx'。
    """

    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)
    
    # 组合完整路径
    save_path = os.path.join(save_dir, file_name)

    # 写入Excel
    with pd.ExcelWriter(save_path, engine='openpyxl') as writer:
        for sheet_name, data in cws_fitting.items():
            # 安全处理sheet名：截断长度，替换非法字符
            valid_sheet_name = sheet_name[:31].replace('/', '_').replace('\\', '_').replace('*', '_').replace('?', '_').replace(':', '_').replace('[', '_').replace(']', '_')

            # 如果data不是DataFrame，尝试转换
            if not isinstance(data, pd.DataFrame):
                data = pd.DataFrame(data)

            # 写入sheet
            data.to_excel(writer, sheet_name=valid_sheet_name, index=False)

    print(f"Channel weights successfully saved to {save_path}")

# %% Usage
if __name__ == '__main__':
    # Fittin target and DM
    channel_manual_remove = [57, 61] # or # channel_manual_remove = [57, 61, 58, 59, 60]
    electrodes, cw_target, distance_matrix, cm_global_averaged = prepare_target_and_inputs('pcc_10_15', 
                                                    'label_driven_mi_10_15', channel_manual_remove)
    
    # electrodes, cw_target, distance_matrix, cm_global_averaged = prepare_target_and_inputs('pcc', 
    #                                                 'label_driven_mi', channel_manual_remove)
    
    # %% Fitting
    fm_model, rcm_model = 'advanced', 'linear_ratio'
    results, cws_fitting = fitting_model(fm_model, rcm_model, cw_target, distance_matrix, cm_global_averaged)
    
    # %% Insert target cw (LDMI) and cm cw non modeled
    cw_non_modeled = np.mean(cm_global_averaged, axis=0)
    cw_non_modeled = feature_engineering.normalize_matrix(cw_non_modeled)
    
    cws_fitting = {'target': cw_target,'non_modeled': cw_non_modeled, **cws_fitting}
    
    # %% Sort ranks of channel weights based on fitted models
    # electrodes
    electrodes_original = np.array(utils_feature_loading.read_distribution('seed')['channel'])
    
    # fitted
    cws_fitted, cws_sorted = {}, {}
    for method, cw_fitted in cws_fitting.items():
        cw_fitted_temp = feature_engineering.insert_idx_manual(cws_fitting[method], channel_manual_remove, value=0)
        cws_fitted[method] = cw_fitted_temp
        cw_sorted_temp = sort_ams(cw_fitted_temp, electrodes_original, electrodes_original)
        cws_sorted[method] = cw_sorted_temp
    
    # %% Save
    path_currebt = os.getcwd()
    results_path = os.path.join(os.getcwd(), 'fitting_results')
    save_fitting_results(results, results_path, f'fitting_results({fm_model}_fm_{rcm_model}_rcm).xlsx')
    save_channel_weights(cws_sorted, results_path, f'channel_weights({fm_model}_fm_{rcm_model}_rcm).xlsx')
    
    # %% Validation of Fitting Comparison
    pltlabels = {'title':'Comparison of Fitted Channel Weights across various Models',
                 'label_x':'Electrodes', 'label_y':'Channel Weight', 
                 'label_A':'CW of target: LD MI', 'label_B':'CW of RCM; by Modeled FM'}
    
    # plot by list
    # pltlabels_non_modeled = pltlabels.copy()
    # pltlabels_non_modeled['title'] = 'Comparison of CWs; Before Modeling'
    # draw_scatter_comparison(electrodes, cw_target, cw_non_modeled, pltlabels_non_modeled)

    # for method, cw_fitting in cws_fitting.items():
    #     _pltlabels = pltlabels.copy()
    #     _pltlabels['title'] = f'Comparison of CWs; {method}'
    #     _pltlabels['label_B'] = 'CW_Recovered_CM_PCC(Fitted)'
    #     draw_scatter_comparison(electrodes, cw_target, cw_fitting, _pltlabels)
    
    # joint scatter
    draw_scatter_multi_method(electrodes, cw_target, cws_fitting, pltlabels)
    
    draw_scatter_subplots_vertical(electrodes, cw_target, cws_fitting, pltlabels)
    
    # %% Validation of Brain Topography
    # mne topography
    distribution = utils_feature_loading.read_distribution('seed')
    plot_joint_topomaps(amps_dict=cws_sorted, distribution_df=distribution, title="All Method Comparison")
    
    # %% Validation of Heatmap
    cws_fitting['cw_target'] = cw_target
    utils_visualization.draw_joint_heatmap_1d(cws_fitting)
    