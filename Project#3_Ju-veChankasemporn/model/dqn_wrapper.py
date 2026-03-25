from dataclasses import dataclass
from typing import Optional

from model.dqn_model import DQNModel


@dataclass
class DQNConfig:
    # Number of training episodes
    episodes: int = 250

    # Optional device override ("cpu" or "cuda")
    device: Optional[str] = None


class DQNWrapper:
    def __init__(self, config: Optional[DQNConfig] = None) -> None:
        # Store user configuration
        self.config = config or DQNConfig()

        # Model will be created when training starts
        self.model: Optional[DQNModel] = None

        # Keep track of which grid size this model was built for
        self.grid_size: Optional[int] = None

        # Store training results
        self.training_history: Optional[dict] = None

    def build_for_environment(self, environment) -> DQNModel:
        """
        Build a DQN model using environment information.

        Input size:
            flattened grid state = grid_size * grid_size

        Output size:
            number of actions from the environment
            (now 4 movement actions: up, down, left, right)
        """
        input_size = environment.grid_size * environment.grid_size
        output_size = environment.action_count()

        self.model = DQNModel(input_size, output_size)

        # Optional manual device override
        if self.config.device is not None:
            self.model.device = self.config.device

        self.grid_size = environment.grid_size
        return self.model

    def train(self, environment):
        """Train the DQN agent on the given environment."""
        self.build_for_environment(environment)

        self.training_history = self.model.train(
            environment,
            episodes=self.config.episodes,
        )

        return self.training_history

    def act(self, state):
        """Get a greedy action from the trained model."""
        if self.model is None:
            raise RuntimeError("Model not trained yet")

        return self.model.act(state)