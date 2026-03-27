"""
File Name:    dqn_wrapper.py
Author(s):    Ju-ve Chankasemporn
Copyright:    (c) 2025 DigiPen Institute of Technology. All rights reserved.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from model.agents import DQNAgent
import matplotlib.pyplot as plt

@dataclass
class DQNConfig:
    episodes: int = 500
    gamma: float = 0.9
    lr: float = 0.001
    epsilon: float = 1.0
    epsilon_min: float = 0.05
    epsilon_decay: float = 0.995
    mem_size: int = 10000
    batch_size: int = 64
    target_replace: int = 100
    checkpoint_dir: str = 'saved_models'


class DQNWrapper:
    def __init__(self) -> None:
        self.config = DQNConfig()
        self.agent = None
        self.grid_size: Optional[int] = None
        self.training_history: Optional[dict] = None

    def build_for_environment(self, environment):
        input_dims = (environment.grid_size * environment.grid_size,)
        n_actions = environment.action_count()

        agent_cls = DQNAgent
        self.agent = agent_cls(
            gamma=self.config.gamma,
            epsilon=self.config.epsilon,
            lr=self.config.lr,
            n_actions=n_actions,
            input_dims=input_dims,
            mem_size=self.config.mem_size,
            batch_size=self.config.batch_size,
            eps_min=self.config.epsilon_min,
            eps_dec=self.config.epsilon_decay,
            replace=self.config.target_replace,
            env_name=f'vacuum_{environment.grid_size}x{environment.grid_size}',
            chkpt_dir=self.config.checkpoint_dir,
        )
        self.grid_size = environment.grid_size
        return self.agent

    def get_checkpoint_paths(self, environment):
        checkpoint_dir = Path(self.config.checkpoint_dir)
        base_name = f'vacuum_{environment.grid_size}x{environment.grid_size}_DQNAgent'
        return {
            'q_eval': checkpoint_dir / f'{base_name}_q_eval',
            'q_next': checkpoint_dir / f'{base_name}_q_next',
        }

    def has_saved_model(self, environment) -> bool:
        paths = self.get_checkpoint_paths(environment)
        return paths['q_eval'].exists() and paths['q_next'].exists()

    def load_model(self, environment) -> bool:
        if not self.has_saved_model(environment):
            return False

        agent = self.build_for_environment(environment)
        agent.load_models()
        agent.epsilon = agent.eps_min
        self.training_history = self.training_history or {
            'rewards': [],
            'steps': [],
            'losses': [],
            'epsilons': [],
        }
        return True

    def train(self, environment):
        agent = self.build_for_environment(environment)

        reward_history = []
        steps_history = []
        loss_history = []
        epsilon_history = []

        for _ in range(self.config.episodes):
            state = environment.reset()
            done = False
            episode_reward = 0.0
            episode_loss = 0.0
            updates = 0

            while not done:
                action = agent.choose_action(state)
                next_state, reward, done = environment.step(action)
                agent.store_transition(state, action, reward, next_state, done)

                loss = agent.learn()
                if loss > 0.0:
                    episode_loss += loss
                    updates += 1

                state = next_state
                episode_reward += reward

            reward_history.append(episode_reward)
            steps_history.append(environment.steps_taken)
            loss_history.append(episode_loss / max(1, updates))
            epsilon_history.append(agent.epsilon)

        agent.save_models()

        self.training_history = {
            'rewards': reward_history,
            'steps': steps_history,
            'losses': loss_history,
            'epsilons': epsilon_history,
        }

        self.save_training_plots()

        return self.training_history

    def act(self, state):
        if self.agent is None:
            raise RuntimeError('Model not trained yet')
        return self.agent.choose_action(state, evaluate=True)

    def save_training_plots(self):
        if not self.training_history or self.grid_size is None:
            return

        rewards = self.training_history['rewards']
        steps = self.training_history['steps']

        if not rewards or not steps:
            return

        plot_dir = Path("plot_path")
        plot_dir.mkdir(parents=True, exist_ok=True)

        episodes = list(range(1, len(rewards) + 1))
        prefix = f"vacuum_{self.grid_size}x{self.grid_size}"

        #reward plot ---
        plt.figure(figsize=(10, 5))
        plt.plot(episodes, rewards)
        plt.xlabel("Learning Epochs")
        plt.ylabel("Reward")
        plt.title(f"Reward vs Learning Epochs ({prefix})")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(plot_dir / f"{prefix}_reward.png")
        plt.close()

        #steps plot ---
        plt.figure(figsize=(10, 5))
        plt.plot(episodes, steps)
        plt.xlabel("Learning Epochs")
        plt.ylabel("Steps to Complete Task")
        plt.title(f"Steps vs Learning Epochs ({prefix})")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(plot_dir / f"{prefix}_steps.png")
        plt.close()

        #cmbined plot ---
        plt.figure(figsize=(10, 5))
        plt.plot(episodes, rewards, label="Reward")
        plt.plot(episodes, steps, label="Steps")
        plt.xlabel("Learning Epochs")
        plt.title(f"Reward & Steps ({prefix})")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(plot_dir / f"{prefix}_combined.png")
        plt.close()