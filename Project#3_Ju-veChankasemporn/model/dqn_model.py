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
    def __init__(self, state_size: int, action_size: int) -> None:
        self.state_size = state_size
        self.action_size = action_size
        self.device = ("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_net = DQN(state_size, action_size).to(self.device)
        self.target_net = DQN(state_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = Adam(self.policy_net.parameters(), lr=1e-3)
        self.loss_fn = nn.MSELoss()
        self.memory = deque(maxlen=5000)

        self.gamma = 0.99
        self.batch_size = 64
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.target_sync_interval = 20

    def select_action(self, state: np.ndarray, greedy: bool = False) -> int:
        if not greedy and random.random() < self.epsilon:
            return random.randrange(self.action_size)

        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
        return int(torch.argmax(q_values, dim=1).item())

    def store_transition(self, state: np.ndarray, action: int,reward: float, next_state: np.ndarray,done: bool,) -> None:
        self.memory.append((state, action, reward, next_state, done))

    def train_step(self) -> float:
        if len(self.memory) < self.batch_size:
            return 0.0

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states_tensor = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
        actions_tensor = torch.tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states_tensor = torch.tensor(np.array(next_states), dtype=torch.float32, device=self.device)
        dones_tensor = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

        current_q = self.policy_net(states_tensor).gather(1, actions_tensor)
        with torch.no_grad():
            max_next_q = self.target_net(next_states_tensor).max(dim=1, keepdim=True)[0]
            target_q = rewards_tensor + (1.0 - dones_tensor) * self.gamma * max_next_q

        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return float(loss.item())

    def train(self, environment, episodes: int = 250) -> dict:
        reward_history: list[float] = []
        steps_history: list[int] = []
        loss_history: list[float] = []

        for episode in range(episodes):
            state = environment.reset()
            done = False
            episode_reward = 0.0
            episode_loss_total = 0.0
            loss_updates = 0

            while not done:
                action = self.select_action(state)
                next_state, reward, done = environment.step(action)
                self.store_transition(state, action, reward, next_state, done)
                loss = self.train_step()

                if loss > 0.0:
                    episode_loss_total += loss
                    loss_updates += 1

                state = next_state
                episode_reward += reward

            if (episode + 1) % self.target_sync_interval == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            reward_history.append(episode_reward)
            steps_history.append(environment.steps_taken)
            loss_history.append(episode_loss_total / max(1, loss_updates))

        return {
            "episodes": episodes, "rewards": reward_history, "steps": steps_history, "losses": loss_history, "final_epsilon": self.epsilon,
        }

    def act(self, state: np.ndarray) -> int:
        return self.select_action(state, greedy=True)