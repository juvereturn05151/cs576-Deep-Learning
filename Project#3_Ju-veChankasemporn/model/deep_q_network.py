"""
Deep Q-network module adapted to work with the current vacuum environment.

This keeps the same general structure as Phil Tabor's version, but adds support
for flat vector observations from the current app/environment.
"""

import os
from typing import Sequence

import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class DeepQNetwork(nn.Module):
    def __init__(self, lr, n_actions, name, input_dims, chkpt_dir):
        super().__init__()
        self.checkpoint_dir = chkpt_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)

        self.input_dims = tuple(input_dims)
        self.n_actions = n_actions

        # The current VacuumEnvironment returns a flattened grid vector.
        # Use an MLP for 1D observations, while still supporting image-style
        # inputs if you later switch to them.
        self.is_vector_input = len(self.input_dims) == 1

        if self.is_vector_input:
            input_size = int(np.prod(self.input_dims))
            self.fc1 = nn.Linear(input_size, 256)
            self.fc2 = nn.Linear(256, 256)
            self.fc3 = nn.Linear(256, n_actions)
        else:
            self.conv1 = nn.Conv2d(self.input_dims[0], 32, 8, stride=4)
            self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
            self.conv3 = nn.Conv2d(64, 64, 3, stride=1)

            fc_input_dims = self.calculate_conv_output_dims(self.input_dims)
            self.fc1 = nn.Linear(fc_input_dims, 512)
            self.fc2 = nn.Linear(512, n_actions)

        self.optimizer = optim.RMSprop(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def calculate_conv_output_dims(self, input_dims: Sequence[int]) -> int:
        state = T.zeros(1, *input_dims)
        dims = self.conv1(state)
        dims = self.conv2(dims)
        dims = self.conv3(dims)
        return int(np.prod(dims.size()))

    def forward(self, state):
        if self.is_vector_input:
            if state.dim() > 2:
                state = state.view(state.size(0), -1)
            x = F.relu(self.fc1(state))
            x = F.relu(self.fc2(x))
            return self.fc3(x)

        conv1 = F.relu(self.conv1(state))
        conv2 = F.relu(self.conv2(conv1))
        conv3 = F.relu(self.conv3(conv2))
        conv_state = conv3.view(conv3.size()[0], -1)

        flat1 = F.relu(self.fc1(conv_state))
        actions = self.fc2(flat1)
        return actions

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file, map_location=self.device))


class DuelingDeepQNetwork(nn.Module):
    def __init__(self, lr, n_actions, name, input_dims, chkpt_dir):
        super().__init__()
        self.checkpoint_dir = chkpt_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)

        input_size = int(np.prod(input_dims))
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.V = nn.Linear(256, 1)
        self.A = nn.Linear(256, n_actions)

        self.optimizer = optim.RMSprop(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        if state.dim() > 2:
            state = state.view(state.size(0), -1)
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        V = self.V(x)
        A = self.A(x)
        return V, A

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file, map_location=self.device))
