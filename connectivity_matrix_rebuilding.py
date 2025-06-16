# -*- coding: utf-8 -*-
"""
Created on Fri May  2 23:50:20 2025

@author: 18307
"""

from vce_modeling import compute_volume_conduction_factors_basic_model as compute_fm_basic
from vce_modeling import compute_volume_conduction_factors_advanced_model as compute_fm_advanced
import vce_model_fitting

import numpy as np
import feature_engineering
def cm_rebuilding(cms, distance_matrix, params, model='exponential', model_fm='basic', model_rcm='differ',
                  fm_normalization=True, rcm_normalization=False):
    """
    重建功能连接矩阵（Reconstructed Connectivity Matrices, RCM）。

    参数：
        cms (np.ndarray): 原始功能连接矩阵，形状为 (N, H, W)
        distance_matrix (np.ndarray): 电极距离矩阵，形状为 (H, W)
        params (dict): 参数字典，包括 scale_a, scale_b 等
        model (str): 距离-因子建模方法
        model_fm (str): FM建模方式：basic 或 advanced
        model_rcm (str): RCM建模方式：differ, linear 或 linear_ratio
        normalize (bool): 是否进行归一化处理

    返回：
        cms_rebuilt (np.ndarray): 重建后的功能连接矩阵，形状为 (N, H, W)
    """

    # 参数验证
    supported_models = ['exponential', 'gaussian', 'inverse', 'generalized_gaussian', 'powerlaw', 'rational_quadratic', 'sigmoid']
    if model not in supported_models:
        raise ValueError(f"Unsupported model: {model}")
    if model_fm not in ['basic', 'advanced']:
        raise ValueError("model_fm must be 'basic' or 'advanced'")
    if model_rcm not in ['differ', 'linear', 'linear_ratio']:
        raise ValueError("model_rcm must be 'differ', 'linear' or 'linear_ratio'")

    scale_a = params.get('scale_a', 0)
    scale_b = params.get('scale_b', 0)

    # 计算距离衰减因子矩阵
    if model_fm == 'basic':
        factor_matrix = compute_fm_basic(distance_matrix, model, params)
    else:
        factor_matrix = compute_fm_advanced(distance_matrix, model, params)

    if fm_normalization:
        factor_matrix = feature_engineering.normalize_matrix(factor_matrix)

    # 重建
    if model_rcm == 'differ':
        cms_rebuilt = cms - factor_matrix
    elif model_rcm == 'linear':
        cms_rebuilt = cms + scale_a * factor_matrix
    elif model_rcm == 'linear_ratio':
        e = 1e-6
        smoothed_fm = vce_model_fitting.gaussian_filter(factor_matrix, sigma=1)
        cms_rebuilt = cms + scale_a * factor_matrix + scale_b * cms / (smoothed_fm + e)

    # 归一化（支持批处理）
    if rcm_normalization:
        cms_rebuilt = feature_engineering.normalize_matrix(np.abs(cms_rebuilt))

    return cms_rebuilt

def example_usage():
    import numpy as np
    import feature_engineering
    from utils import utils_feature_loading, utils_visualization
    _, dm = feature_engineering.compute_distance_matrix(dataset="seed", projection_params={"type": "3d"})
    dm = feature_engineering.normalize_matrix(dm)
    utils_visualization.draw_projection(dm, 'Distance Matrix')
    
    cms_sample = utils_feature_loading.read_fcs('seed', 'sub1ex1', 'pcc')
    cm_sample = cms_sample.get('alpha', '')
    utils_visualization.draw_projection(np.mean(cm_sample, axis=0), 'Connectivity Matrix Sample')

    params = {'sigma': 0.2}
    model, model_fm, model_rcm = 'exponential', 'basic', 'differ'
    
    rcm = cm_rebuilding(cm_sample, dm, params, model, model_fm, model_rcm)
    utils_visualization.draw_projection(np.mean(rcm, axis=0), 'Rebuilded Connectivity Matrix Sample')

if __name__ == '__main__':
    # example_usage()
    
    from utils import utils_visualization
    _, dm = feature_engineering.compute_distance_matrix(dataset="seed", projection_params={"type": "euclidean"})
    dm = feature_engineering.normalize_matrix(dm)
    utils_visualization.draw_projection(dm, 'Distance Matrix')
    
    _, dm = feature_engineering.compute_distance_matrix(dataset="seed", projection_params={"type": "3d"})
    dm = feature_engineering.normalize_matrix(dm)
    utils_visualization.draw_projection(dm, 'Distance Matrix')
    
    
    