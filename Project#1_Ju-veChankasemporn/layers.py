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
        dZ = self.O_layer_plus_one_gradient

        self.weight_gradient = X.T @ dZ
        self.bias_gradient = np.sum(dZ, axis=0, keepdims=True)

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
