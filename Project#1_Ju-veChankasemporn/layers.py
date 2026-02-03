"""
File Name:    layers.py
Author(s):    Ju-ve Chankasemporn
Copyright:    (c) 2025 DigiPen Institute of Technology. All rights reserved.
"""

import numpy as np


class Module:
    def forward(self, X):
        raise NotImplementedError

    def backward(self, dY):
        raise NotImplementedError

    def get_normal_formula_result(self, X):
        raise NotImplementedError

    def get_derived_formula_result(self, X):
        raise NotImplementedError

    def update(self, lr):
        pass


class Linear(Module):
    def __init__(self, in_dim, out_dim, seed=0):
        rng = np.random.default_rng(seed)
        # Xavier init for linear
        self.W = rng.normal(0.0, np.sqrt(1.0 / in_dim), size=(in_dim, out_dim))
        self.b = np.zeros((1, out_dim))

        self.O = None
        self.O_layer_plus_one_gradient = None
        self.weight_gradient = None
        self.bias_gradient = None

    def forward(self, X):
        self.O = X
        # Z = XW + b, identity activation => output is Z
        return self.get_normal_formula_result(X @ self.W + self.b)

    def get_normal_formula_result(self, X):
        # linear activation
        return X

    def get_derived_formula_result(self, X):
        # derivative of linear activation
        return 1

    def backward(self, O_layer_plus_one_gradient):
        self.O_layer_plus_one_gradient = O_layer_plus_one_gradient

        # For identity activation: dZ = dO * 1 = dO
        dZ = O_layer_plus_one_gradient * self.get_derived_formula_result(None)

        # Return gradient wrt input: dX = dZ @ W^T
        return dZ @ self.W.T

    def update(self, lr):
        # self.O is X (cached in forward)
        X = self.O
        dZ = self.O_layer_plus_one_gradient

        # dW = X^T @ dZ
        self.weight_gradient = X.T @ dZ

        # db = sum over batch (shape: (1, out_dim))
        self.bias_gradient = np.sum(dZ, axis=0, keepdims=True)

        self.W -= lr * self.weight_gradient
        self.b -= lr * self.bias_gradient


class ReLU(Module):
    def __init__(self, in_dim, out_dim, seed=0):
        rng = np.random.default_rng(seed)
        # He init for ReLU
        self.W = rng.normal(0.0, np.sqrt(2.0 / in_dim), size=(in_dim, out_dim))
        self.b = np.zeros((1, out_dim))

        self.O = None
        self.Z = None
        self.weight_gradient = None
        self.bias_gradient = None

    def forward(self, X):
        self.O = X
        # Pre-activation
        self.Z = X @ self.W + self.b
        # Activation
        return self.get_normal_formula_result(self.Z)

    def backward(self, O_layer_plus_one_gradient):
        # dZ = dO * relu'(Z)
        dZ = O_layer_plus_one_gradient * self.get_derived_formula_result(self.Z)

        # Gradients for params (keep where you originally compute them: in backward)
        X = self.O
        self.weight_gradient = X.T @ dZ
        self.bias_gradient = np.sum(dZ, axis=0, keepdims=True)

        # Gradient wrt input
        return dZ @ self.W.T

    def get_normal_formula_result(self, X):
        return np.maximum(0, X)

    def get_derived_formula_result(self, X):
        return (X > 0).astype(X.dtype)

    def update(self, lr):
        self.W -= lr * self.weight_gradient
        self.b -= lr * self.bias_gradient
