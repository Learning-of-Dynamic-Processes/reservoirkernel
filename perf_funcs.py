# %% #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Hannah Lim, Lukas Gonon, Lyudmila Grigoryeva, and Juan-Pablo Ortega
"""
# %%
import os
print(os.getcwd())

import numpy as np

def calculate_mse(y_true, y_pred, mean=None, std=None):
    """
    Calculate Mean Squared Error (MSE) between true and predicted values.

    Parameters:
    - y_true: numpy array of true target values.
    - y_pred: numpy array of predicted target values.
    - mean: Mean of the target variable (optional).
    - std: Standard deviation of the target variable (optional).

    Returns:
    - mse: Mean Squared Error.
    """
    
    if mean is not None and std is not None:
        # Destandardize the data if required
        y_true = (y_true * std) + mean
        y_pred = (y_pred * std) + mean

    # Calculate MSE
    mse = np.mean((y_true - y_pred)**2)

    return mse

def calculate_nmse(y_true, y_pred):
    """
    Calculate Normalized Mean Squared Error (NMSE) between true and predicted values.

    Parameters:
    - y_true: numpy array of true target values.
    - y_pred: numpy array of predicted target values.

    Returns:
    - nmse: Normalized Mean Squared Error.
    """
    
    # Calculate MSE
    mse = np.mean((y_true - y_pred)**2, axis=0)
    # Calculate MSE
    factor = np.var((y_true), axis=0)
    nmse = np.mean(mse / factor)
    
    return nmse

EPSILON = 1e-10
def smape(actual: np.ndarray, predicted: np.ndarray):
    """
    Symmetric Mean Absolute Percentage Error
    Note: result is NOT multiplied by 100
    """
    return np.mean(2.0 * np.abs(actual - predicted) / ((np.abs(actual) + np.abs(predicted)) + EPSILON))