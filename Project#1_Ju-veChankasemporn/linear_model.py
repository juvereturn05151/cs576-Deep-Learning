"""
File Name:    linear_model.py
Author(s):    Ju-ve Chankasemporn
Copyright:    (c) 2025 DigiPen Institute of Technology. All rights reserved.
"""

import deep_learning_runner
import layer

class LinearModel:
    def __init__(self, input_layer_dim=2, hidden_layer_dim1=2, hidden_layer_dim2=2, output_layer_dim=1):
        self.hidden_layer1 = layer.Layer(input_layer_dim, hidden_layer_dim1, seed=1)
        self.hidden_layer2 = layer.Layer(hidden_layer_dim1, hidden_layer_dim2, seed=2)
        self.output_layer = layer.Layer(hidden_layer_dim2, output_layer_dim, seed=3)

    def forward(self, X):
        z1 = self.hidden_layer1.forward(X)   # linear activation => identity
        z2 = self.hidden_layer2.forward(z1)
        y  = self.output_layer.forward(z2)
        return y

    def backward(self, dY):
        d2 = self.output_layer.backward(dY)
        d1 = self.hidden_layer2.backward(d2)
        _  = self.hidden_layer1.backward(d1)

    def update(self, lr):
        self.hidden_layer1.update(lr)
        self.hidden_layer2.update(lr)
        self.output_layer.update(lr)
