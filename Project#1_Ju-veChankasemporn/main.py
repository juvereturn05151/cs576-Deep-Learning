"""
File Name:    main.py.py
Author(s):    Ju-ve Chankasemporn
Copyright:    (c) 2025 DigiPen Institute of Technology. All rights reserved.
"""

import model
import layers
import deep_learning_runner

def main():
    model2 = model.Model([
        layers.ReLU(2, 2, seed=1),
        layers.ReLU(2, 2, seed=2),
        layers.ReLU(2, 1, seed=3),
    ])

    runner = deep_learning_runner.DeepLearningRunner()
    runner.run()

    runner.replace_model(model2)
    runner.run()

if __name__ == '__main__':
    main()