"""
File Name:    dqn.py
Author(s):    Ju-ve Chankasemporn
Copyright:    (c) 2025 DigiPen Institute of Technology. All rights reserved.
"""

from torch import nn

class DQN(nn.Module):
    def __init__(self, input_size: int, output_size: int) -> None:
        super().__init__()

        # This is a simple Multi-Layer Perceptron (MLP)
        # Input → Hidden → Hidden → Output
        # Output = Q-values for each action
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),   # input → hidden
            nn.ReLU(),                    # non-linearity
            nn.Linear(128, 128),          # hidden → hidden
            nn.ReLU(),
            nn.Linear(128, output_size),  # hidden → Q-values
        )

    def forward(self, x):
        # Forward pass: compute Q-values
        return self.network(x)