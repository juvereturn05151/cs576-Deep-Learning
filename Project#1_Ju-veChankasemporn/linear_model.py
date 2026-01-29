"""
File Name:    linear_model.py
Author(s):    Ju-ve Chankasemporn
Copyright:    (c) 2025 DigiPen Institute of Technology. All rights reserved.
"""

import layer

class LinearModel:
    def __init__(self, input_dim=2, layer1=2, layer2=2, output_dim=1):
        self.layer1 = Layer(input_dim, layer1, seed=1)
