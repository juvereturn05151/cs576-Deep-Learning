"""
File Name:    math_utility.py
Author(s):    Ju-ve Chankasemporn
Copyright:    (c) 2025 DigiPen Institute of Technology. All rights reserved.
"""

import numpy as np

def mse_loss(y_pred, y_true):
    return np.mean((y_pred - y_true) ** 2)

def rmse(y_pred, y_true):
    return np.sqrt(mse_loss(y_pred, y_true))

def mse_backward(y_pred, y_true):
    N = y_true.shape[0]
    return (2.0 / N) * (y_pred - y_true)