"""
File Name:    agents.py
Author(s):    Ju-ve Chankasemporn
Copyright:    (c) 2025 DigiPen Institute of Technology. All rights reserved.
"""

import numpy as np
import torch as T

from model.deep_q_network import DeepQNetwork
from model.replay_memory import ReplayBuffer

class Agent:
    def __init__(self, gamma, epsilon, lr, n_actions, input_dims, mem_size, batch_size, eps_min=0.01, eps_dec=0.05, replace=1000,algo=None,env_name=None,chkpt_dir='saved_models',
    ):
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.action_space = [i for i in range(n_actions)]
        self.learn_step_counter = 0
        self.batch_size = batch_size
        self.replace_target_cnt = replace
        self.algo = algo or 'DQNAgent'
        self.env_name = env_name or 'VacuumEnvironment'
        self.chkpt_dir = chkpt_dir

        self.memory = ReplayBuffer(mem_size, input_dims, n_actions)

    def store_transition(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)

    def choose_action(self, observation, evaluate=False):
        raise NotImplementedError

    def replace_target_network(self):
        if self.learn_step_counter % self.replace_target_cnt == 0:
            self.q_next.load_state_dict(self.q_eval.state_dict())

    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min

    def sample_memory(self):
        state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)

        states = T.tensor(state, dtype=T.float32).to(self.q_eval.device)
        rewards = T.tensor(reward, dtype=T.float32).to(self.q_eval.device)
        dones = T.tensor(done, dtype=T.bool).to(self.q_eval.device)
        actions = T.tensor(action, dtype=T.long).to(self.q_eval.device)
        states_ = T.tensor(new_state, dtype=T.float32).to(self.q_eval.device)

        return states, actions, rewards, states_, dones

    def learn(self):
        raise NotImplementedError

    def save_models(self):
        self.q_eval.save_checkpoint()
        self.q_next.save_checkpoint()

    def load_models(self):
        self.q_eval.load_checkpoint()
        self.q_next.load_checkpoint()


class DQNAgent(Agent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.q_eval = DeepQNetwork(
            self.lr,
            self.n_actions,
            input_dims=self.input_dims,
            name=self.env_name + '_' + self.algo + '_q_eval',
            chkpt_dir=self.chkpt_dir,
        )
        self.q_next = DeepQNetwork(
            self.lr,
            self.n_actions,
            input_dims=self.input_dims,
            name=self.env_name + '_' + self.algo + '_q_next',
            chkpt_dir=self.chkpt_dir,
        )

    def choose_action(self, observation, evaluate=False):
        if (not evaluate) and np.random.random() < self.epsilon:
            return int(np.random.choice(self.action_space))

        state = T.tensor(np.array([observation]), dtype=T.float32).to(self.q_eval.device)
        actions = self.q_eval.forward(state)
        action = T.argmax(actions).item()
        return int(action)

    def learn(self):
        if self.memory.mem_counter < self.batch_size:
            return 0.0

        self.q_eval.optimizer.zero_grad()
        self.replace_target_network()

        states, actions, rewards, next_states, dones = self.sample_memory()
        indices = np.arange(self.batch_size)

        q_pred = self.q_eval.forward(states)[indices, actions]
        q_next = self.q_next.forward(next_states).max(dim=1)[0]
        q_next[dones] = 0.0
        q_target = rewards + self.gamma * q_next

        loss = self.q_eval.loss(q_target, q_pred).to(self.q_eval.device)
        loss.backward()
        self.q_eval.optimizer.step()
        self.learn_step_counter += 1
        self.decrement_epsilon()
        return float(loss.item())