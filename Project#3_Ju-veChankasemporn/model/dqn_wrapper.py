"""
File Name:    dqn_wrapper.py
Author(s):    Ju-ve Chankasemporn
Copyright:    (c) 2025 DigiPen Institute of Technology. All rights reserved.
"""
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
    episodes: int = 250
    device: Optional[str] = None


class DQNWrapper:
    def __init__(self, config: Optional[DQNConfig] = None) -> None:
        self.config = config or DQNConfig()
        self.model: Optional[DQNModel] = None
        self.grid_size: Optional[int] = None
        self.training_history: Optional[dict] = None

    def build_for_environment(self, environment) -> DQNModel:
        state_size = environment.grid_size * environment.grid_size
        action_size = environment.action_count()

        self.model = DQNModel(
            state_size=state_size,
            action_size=action_size,
        )

        if self.config.device is not None:
            self.model.device = self.config.device

        self.grid_size = environment.grid_size
        return self.model

    def train(self, environment) -> dict:
        self.build_for_environment(environment)
        self.training_history = self.model.train(
            environment,
            episodes=self.config.episodes,
        )
        return self.training_history

    def act(self, state):
        if self.model is None:
            raise RuntimeError("DQNWrapper model has not been trained or built yet.")
        return self.model.act(state)

    def is_ready(self) -> bool:
        return self.model is not None

    def matches_environment(self, environment) -> bool:
        return self.is_ready() and self.grid_size == environment.grid_size