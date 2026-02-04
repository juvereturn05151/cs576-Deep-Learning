"""
File Name:    layers.py
Author(s):    Ju-ve Chankasemporn
Copyright:    (c) 2025 DigiPen Institute of Technology. All rights reserved.
"""

import numpy as np

class Module:
    def __init__(self):
        self.W = None
        self.b = None

        self.O = None
        self.Z = None
        self.O_layer_plus_one_gradient = None

        self.weight_gradient = None
        self.bias_gradient = None

    def get_normal_formula_result(self, X):
        raise NotImplementedError

    def get_derived_formula_result(self, X):
        raise NotImplementedError

    def forward(self, X):
        self.O = X
        self.Z = X @ self.W + self.b
        return self.get_normal_formula_result(self.Z)

    def backward(self, O_layer_plus_one_gradient):
        self.O_layer_plus_one_gradient = O_layer_plus_one_gradient
        dZ = O_layer_plus_one_gradient * self.get_derived_formula_result(self.Z)
        return dZ @ self.W.T

    def update(self, lr):
        X = self.O
        dZ = self.O_layer_plus_one_gradient * self.get_derived_formula_result(self.Z)

        self.weight_gradient = X.T @ dZ
        self.bias_gradient = dZ

        self.W -= lr * self.weight_gradient
        self.b -= lr * self.bias_gradient

class Linear(Module):
    def __init__(self, in_dim, out_dim, seed=0):
        super().__init__()
        rng = np.random.default_rng(seed)
        # Xavier is fine for sigmoid
        self.W = rng.normal(0.0, np.sqrt(1.0 / in_dim), size=(in_dim, out_dim))
        self.b = np.zeros((1, out_dim))

    def get_normal_formula_result(self, X):
        return X

    def get_derived_formula_result(self, X):
        return 1

class ReLU(Module):
    def __init__(self, in_dim, out_dim, seed=0):
        super().__init__()
        rng = np.random.default_rng(seed)
        # He is fine for sigmoid
        self.W = rng.normal(0.0, np.sqrt(2.0 / in_dim), size=(in_dim, out_dim))
        self.b = np.zeros((1, out_dim))

    def get_normal_formula_result(self, X):
        return np.maximum(0, X)

    def get_derived_formula_result(self, X):
        return (X > 0).astype(X.dtype)

class Sigmoid(Module):
    def __init__(self, in_dim, out_dim, seed=0):
        super().__init__()
        rng = np.random.default_rng(seed)
        # Xavier is fine for sigmoid
        self.W = rng.normal(0.0, np.sqrt(1.0 / in_dim), size=(in_dim, out_dim))
        self.b = np.zeros((1, out_dim))

    def get_normal_formula_result(self, X):
        return 1.0 / (1.0 + np.exp(-X))

    def get_derived_formula_result(self, X):
        S = self.get_normal_formula_result(X)
        return S * (1.0 - S)

class Tanh(Module):
    def __init__(self, in_dim, out_dim, seed=0):
        super().__init__()
        rng = np.random.default_rng(seed)
        # Xavier init for tanh
        self.W = rng.normal(0.0, np.sqrt(1.0 / in_dim), size=(in_dim, out_dim))
        self.b = np.zeros((1, out_dim))

    def get_normal_formula_result(self, X):
        return np.tanh(X)

    def get_derived_formula_result(self, X):
        # derivative: 1 - tanh^2(x)
        T = np.tanh(X)
        return 1.0 - T * T

class LeakyReLU(Module):
    def __init__(self, in_dim, out_dim, seed=0, alpha=0.01):
        super().__init__()
        rng = np.random.default_rng(seed)
        # He
        self.W = rng.normal(0.0, np.sqrt(2.0 / in_dim), size=(in_dim, out_dim))
        self.b = np.zeros((1, out_dim))
        self.alpha = alpha

    def get_normal_formula_result(self, X):
        return np.where(X > 0, X, self.alpha * X)

    def get_derived_formula_result(self, X):
        return np.where(X > 0, 1.0, self.alpha).astype(X.dtype)


class Piecewise(Module):
    def __init__(self, in_dim, out_dim, seed=0, left=-1.0, right=1.0, slope_left=0.5, slope_right=1.0):
        super().__init__()
        rng = np.random.default_rng(seed)
        # Xavier
        self.W = rng.normal(0.0, np.sqrt(1.0 / in_dim), size=(in_dim, out_dim))
        self.b = np.zeros((1, out_dim))

        self.left = left
        self.right = right
        self.slope_left = slope_left
        self.slope_right = slope_right

    def get_normal_formula_result(self, X):
        mid = np.zeros_like(X)

        left_val = self.slope_left * (X - self.left)

        right_val = self.slope_right * (X - self.right)

        return np.where(X < self.left, left_val, np.where(X > self.right, right_val, mid))

    def get_derived_formula_result(self, X):
        return np.where(X < self.left, self.slope_left, np.where(X > self.right, self.slope_right, 0.0)).astype(X.dtype)

