from torch import nn

class DQN(nn.Module):
    def __init__(self, input_size: int, output_size: int) -> None:
        super().__init__()

        # Simple MLP:
        # flattened state -> hidden -> hidden -> Q-values for each action
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_size),
        )

    def forward(self, x):
        return self.network(x)