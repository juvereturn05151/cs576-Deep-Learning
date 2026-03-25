"""
File Name:    dqn_wrapper.py
Author(s):    Ju-ve Chankasemporn
Copyright:    (c) 2025 DigiPen Institute of Technology. All rights reserved.
"""
from dataclasses import dataclass
from typing import Optional

from model.dqn_model import DQNModel

@dataclass
class DQNConfig:
    # Number of training episodes
    episodes: int = 250

    # Optional device override (cpu/cuda)
    device: Optional[str] = None


class DQNWrapper:
    def __init__(self, config: Optional[DQNConfig] = None) -> None:
        # Store configuration
        self.config = config or DQNConfig()

        # Model will be created later
        self.model: Optional[DQNModel] = None

        # Used to verify compatibility with environment
        self.grid_size: Optional[int] = None

        self.training_history: Optional[dict] = None

    def build_for_environment(self, environment) -> DQNModel:
        """
        Build DQN model using environment info
        """

        # Input = flattened grid
        input_size = environment.grid_size * environment.grid_size

        # Output = number of actions
        output_size = environment.action_count()

        self.model = DQNModel(input_size, output_size)

        # Optional device override
        if self.config.device is not None:
            self.model.device = self.config.device

        self.grid_size = environment.grid_size
        return self.model

    def train(self, environment):
        """
        Train the DQN agent
        """
        self.build_for_environment(environment)

        self.training_history = self.model.train(
            environment,
            episodes=self.config.episodes,
        )

        return self.training_history

    def act(self, state):
        """
        Get action from trained model
        """
        if self.model is None:
            raise RuntimeError("Model not trained yet")

        return self.model.act(state)