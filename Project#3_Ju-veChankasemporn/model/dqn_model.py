"""
File Name:    dqn_model.py
Author(s):    Ju-ve Chankasemporn
Copyright:    (c) 2025 DigiPen Institute of Technology. All rights reserved.
"""

from collections import deque
import random

import numpy as np
import torch
from torch import nn
from torch.optim import Adam

from model.dqn import DQN


class DQNModel:
    def __init__(self, input_size: int, output_size: int) -> None:
        # input_size = flattened environment state size
        # output_size = number of available actions (now 4 movement actions)
        self.input_size = input_size
        self.action_size = output_size
        self.device = ("cuda" if torch.cuda.is_available() else "cpu")

        # Main network being trained
        self.policy_net = DQN(input_size, output_size).to(self.device)

        # Target network for stable TD targets
        self.target_net = DQN(input_size, output_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = Adam(self.policy_net.parameters(), lr=1e-3)
        self.loss_fn = nn.MSELoss()

        # Replay memory stores (state, action, reward, next_state, done)
        self.memory = deque(maxlen=5000)

        # DQN hyperparameters
        self.gamma = 0.9
        self.batch_size = 64
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.50
        self.target_sync_interval = 20

    def select_action(self, state: np.ndarray, greedy: bool = False) -> int:
        """
        Epsilon-greedy action selection:
        - Random action with probability epsilon
        - Best predicted Q action otherwise
        """
        if not greedy and random.random() < self.epsilon:
            return random.randrange(self.action_size)

        state_tensor = torch.tensor(
            state, dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        with torch.no_grad():
            q_values = self.policy_net(state_tensor)

        return int(torch.argmax(q_values, dim=1).item())

    def store_transition(self, state, action, reward, next_state, done):
        """Store one experience tuple in replay memory."""
        self.memory.append((state, action, reward, next_state, done))

    def train_step(self) -> float:
        """
        Perform one DQN update using a random mini-batch from replay memory.
        """
        if len(self.memory) < self.batch_size:
            return 0.0

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

        # Current predicted Q(s, a)
        current_q = self.policy_net(states).gather(1, actions)

        # Target y = r + gamma * max_a' Q_target(s', a')
        with torch.no_grad():
            max_next_q = self.target_net(next_states).max(dim=1, keepdim=True)[0]
            target_q = rewards + (1.0 - dones) * self.gamma * max_next_q

        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return float(loss.item())

    def train(self, environment, episodes: int = 250):
        """
        Main DQN training loop.
        """
        reward_history = []
        steps_history = []
        loss_history = []

        for episode in range(episodes):
            state = environment.reset()
            done = False

            episode_reward = 0.0
            episode_loss = 0.0
            updates = 0

            while not done:
                action = self.select_action(state)
                next_state, reward, done = environment.step(action)

                self.store_transition(state, action, reward, next_state, done)

                loss = self.train_step()
                if loss > 0:
                    episode_loss += loss
                    updates += 1

                state = next_state
                episode_reward += reward

            # Periodically copy policy network weights into target network
            if (episode + 1) % self.target_sync_interval == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

            # Decay exploration rate after each episode
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

            reward_history.append(episode_reward)
            steps_history.append(environment.steps_taken)
            loss_history.append(episode_loss / max(1, updates))

        return {
            "rewards": reward_history,
            "steps": steps_history,
            "losses": loss_history,
        }

    def act(self, state):
        """Use greedy action selection for evaluation."""
        return self.select_action(state, greedy=True)