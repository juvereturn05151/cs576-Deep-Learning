"""
File Name:    main.py.py
Author(s):    Ju-ve Chankasemporn
Copyright:    (c) 2025 DigiPen Institute of Technology. All rights reserved.
"""

import model
import layers
import deep_learning_runner

def main():
    model_ReLU = model.Model([
        layers.ReLU(2, 2, seed=1),
        layers.ReLU(2, 2, seed=2),
        layers.ReLU(2, 1, seed=3),
    ])

    model_Sigmoid = model.Model([
        layers.Sigmoid(2, 2, seed=1),
        layers.Sigmoid(2, 2, seed=2),
        layers.Sigmoid(2, 1, seed=3),
    ])

    model_Tanh = model.Model([
        layers.Tanh(2, 2, seed=1),
        layers.Tanh(2, 2, seed=2),
        layers.Tanh(2, 1, seed=3),
    ])

    model_LeakyReLU = model.Model([
        layers.LeakyReLU(2, 2, seed=1),
        layers.LeakyReLU(2, 2, seed=2),
        layers.LeakyReLU(2, 1, seed=3),
    ])

    model_Piecewise = model.Model([
        layers.Piecewise(2, 2, seed=1),
        layers.Piecewise(2, 2, seed=2),
        layers.Piecewise(2, 1, seed=3),
    ])

    # Use Original Model
    runner = deep_learning_runner.DeepLearningRunner()
    runner.run()

    runner.replace_model(model_ReLU," ReLU")
    runner.run()

    runner.replace_model(model_Sigmoid," Sigmoid")
    runner.run()

    runner.replace_model(model_Tanh," TanH")
    runner.run()

    runner.replace_model(model_LeakyReLU," LeakyReLU")
    runner.run()

    runner.replace_model(model_Piecewise," Piecewise")
    runner.run()

if __name__ == '__main__':
    main()