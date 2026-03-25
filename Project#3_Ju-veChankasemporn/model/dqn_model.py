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
        # Basic config
        self.input_size = input_size
        self.action_size = output_size
        self.device = ("cuda" if torch.cuda.is_available() else "cpu")

        # Policy network (the one we train)
        self.policy_net = DQN(input_size, output_size).to(self.device)

        # Target network (stable version of policy_net)
        # Used to compute target Q values
        self.target_net = DQN(input_size, output_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Optimizer + loss
        self.optimizer = Adam(self.policy_net.parameters(), lr=1e-3)
        self.loss_fn = nn.MSELoss()

        # Replay buffer (stores experience)
        self.memory = deque(maxlen=5000)

        # DQN hyperparameters
        self.gamma = 0.99              # discount factor
        self.batch_size = 64
        self.epsilon = 1.0            # exploration rate
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.target_sync_interval = 20

    def select_action(self, state: np.ndarray, greedy: bool = False) -> int:
        """
        Choose action using epsilon-greedy:
        - random action with probability epsilon
        - best Q action otherwise
        """

        # Exploration (random action)
        if not greedy and random.random() < self.epsilon:
            return random.randrange(self.action_size)

        # Exploitation (use model)
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            q_values = self.policy_net(state_tensor)

        return int(torch.argmax(q_values, dim=1).item())

    def store_transition(self, state, action, reward, next_state, done):
        """
        Store experience in replay buffer
        (s, a, r, s', done)
        """
        self.memory.append((state, action, reward, next_state, done))

    def train_step(self) -> float:
        """
        Perform one DQN update using random batch from memory
        """

        # Not enough data yet
        if len(self.memory) < self.batch_size:
            return 0.0

        # Sample random batch
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert to tensors
        states = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

        # Q(s,a)
        current_q = self.policy_net(states).gather(1, actions)

        # max_a' Q_target(s', a')
        with torch.no_grad():
            max_next_q = self.target_net(next_states).max(dim=1, keepdim=True)[0]

        # Bellman equation:
        # y = r + gamma * max Q(s',a')
        target_q = rewards + (1.0 - dones) * self.gamma * max_next_q

        # Loss = (Q - target)^2
        loss = self.loss_fn(current_q, target_q)

        # Backprop
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return float(loss.item())

    def train(self, environment, episodes: int = 250):
        """
        Main training loop
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
                # Choose action
                action = self.select_action(state)

                # Step environment
                next_state, reward, done = environment.step(action)

                # Store transition
                self.store_transition(state, action, reward, next_state, done)

                # Train
                loss = self.train_step()

                if loss > 0:
                    episode_loss += loss
                    updates += 1

                state = next_state
                episode_reward += reward

            # Sync target network
            if (episode + 1) % self.target_sync_interval == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

            # Decay epsilon
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
        # Pure greedy action (used for testing)
        return self.select_action(state, greedy=True)