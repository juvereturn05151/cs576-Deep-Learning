"""
File Name:    model.py
Author(s):    Ju-ve Chankasemporn
Copyright:    (c) 2025 DigiPen Institute of Technology. All rights reserved.
"""

import layers

class Model:
    def __init__(self, modules):
        self.modules = modules

    def forward(self, X):
        for m in self.modules:
            X = m.forward(X)
        return X

    def backward(self, dY):
        for m in reversed(self.modules):
            dY = m.backward(dY)
        return dY

    def update(self, lr):
        for m in self.modules:
            m.update(lr)
