"""
File Name:    layer.py
Author(s):    Ju-ve Chankasemporn
Copyright:    (c) 2025 DigiPen Institute of Technology. All rights reserved.
"""

import numpy as np

class Layer:
    def __init__(self, in_dim, out_dim, seed=0):
        rng = np.random.default_rng(seed)
        # He init is good for ReLU layers
        self.W = rng.normal(0.0, np.sqrt(2.0 / in_dim), size=(in_dim, out_dim))
        self.b = np.zeros((1, out_dim))

        # cache for backward
        self.O = None
        self.weight_gradient = None
        self.bias_gradient = None

    def forward(self, X):
        # cache
        self.O = X
        #Z_X = W_X * O_X + b
        return X @ self.W + self.b

    def backward(self, O_layer_plus_one_gradient):
        O = self.O
        # (in_dim, out_dim)
        self.weight_gradient = O.T @ O_layer_plus_one_gradient
        # (1, out_dim)
        self.bias_gradient = np.sum(O_layer_plus_one_gradient, axis=0, keepdims=True)
        # (N, in_dim)
        # the derivative of the activation function is 1, so we only multiply O_layer_plus_one_gradient and weights
        O_gradient = O_layer_plus_one_gradient @ self.W.T
        return O_gradient

    def update(self, lr):
        self.W -= lr * self.weight_gradient
        self.b -= lr * self.bias_gradient