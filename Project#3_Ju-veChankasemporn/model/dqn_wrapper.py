from dataclasses import dataclass
from typing import Optional

from model.agents import DQNAgent


@dataclass
class DQNConfig:
    episodes: int = 500
    gamma: float = 0.9
    lr: float = 0.1
    epsilon: float = 1.0
    epsilon_min: float = 0.05
    epsilon_decay: float = 0.5
    mem_size: int = 10000
    batch_size: int = 64
    target_replace: int = 100
    checkpoint_dir: str = 'tmp/dqn'


class DQNWrapper:
    def __init__(self, config: Optional[DQNConfig] = None) -> None:
        self.config = config or DQNConfig()
        self.agent = None
        self.grid_size: Optional[int] = None
        self.training_history: Optional[dict] = None

    def build_for_environment(self, environment):
        input_dims = (environment.grid_size * environment.grid_size,)
        n_actions = environment.action_count()

        agent_cls =DQNAgent
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

        self.training_history = {
            'rewards': reward_history,
            'steps': steps_history,
            'losses': loss_history,
            'epsilons': epsilon_history,
        }
        return self.training_history

    def act(self, state):
        if self.agent is None:
            raise RuntimeError('Model not trained yet')
        return self.agent.choose_action(state, evaluate=True)
