from __future__ import annotations

import csv
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt

from dqn_agent import DQNAgent, DQNConfig
from VacuumEnvironment import NUM_ACTIONS, RewardConfig, VacuumEnvironment


def moving_average(values: List[float], window: int = 25) -> List[float]:
    if not values:
        return []
    out = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        out.append(sum(values[start : i + 1]) / (i - start + 1))
    return out


def default_dirt_count(rows: int, cols: int) -> int:
    if rows == 5 and cols == 5:
        return 5
    if rows == 10 and cols == 10:
        return 8
    return 10


def evaluate_agent(
    agent: DQNAgent,
    rows: int,
    cols: int,
    dirt_count: int,
    episodes: int = 20,
    fixed_dirty_positions: Optional[List[Tuple[int, int]]] = None,
) -> Dict[str, float]:
    rewards = []
    steps_list = []
    success = 0
    for _ in range(episodes):
        env = VacuumEnvironment(rows=rows, cols=cols, dirt_count=dirt_count)
        state = env.reset(fixed_dirty_positions=fixed_dirty_positions)
        done = False
        total_reward = 0.0
        while not done:
            action = agent.select_action(state, epsilon=0.0)
            next_state, reward, done, info = env.step(action)
            state = next_state
            total_reward += reward
        rewards.append(total_reward)
        steps_list.append(env.steps)
        if info.get("all_clean", False):
            success += 1
    return {
        "avg_reward": mean(rewards),
        "avg_steps": mean(steps_list),
        "success_rate": success / episodes,
    }


def plot_training(metrics: Dict[str, List[float]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 5))
    plt.plot(metrics["episode_rewards"], label="Episode Reward")
    plt.plot(moving_average(metrics["episode_rewards"], 25), label="Reward MA(25)")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Reward vs. Learning Epochs")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "reward_vs_epochs.png")
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(metrics["episode_steps"], label="Steps")
    plt.plot(moving_average(metrics["episode_steps"], 25), label="Steps MA(25)")
    plt.xlabel("Episode")
    plt.ylabel("Steps")
    plt.title("Steps Needed to Finish Each Episode")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "steps_vs_epochs.png")
    plt.close()


def save_metrics_csv(metrics: Dict[str, List[float]], output_path: Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "reward", "steps", "epsilon", "loss"])
        for i, (r, s, e, l) in enumerate(
            zip(
                metrics["episode_rewards"],
                metrics["episode_steps"],
                metrics["epsilons"],
                metrics["episode_losses"],
            ),
            start=1,
        ):
            writer.writerow([i, r, s, e, l])


def train_dqn(
    rows: int = 5,
    cols: int = 5,
    dirt_count: Optional[int] = None,
    episodes: int = 400,
    random_configurations: bool = True,
    fixed_dirty_positions: Optional[List[Tuple[int, int]]] = None,
    output_dir: str = "outputs_5x5",
) -> Tuple[DQNAgent, Dict[str, List[float]]]:
    dirt_count = dirt_count if dirt_count is not None else default_dirt_count(rows, cols)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    env = VacuumEnvironment(
        rows=rows,
        cols=cols,
        dirt_count=dirt_count,
        reward_config=RewardConfig(
            step_penalty=-0.2,
            bump_penalty=-1.0,
            useless_suck_penalty=-2.0,
            clean_reward=8.0,
            finish_reward=25.0,
            progress_reward_scale=0.3,
        ),
    )
    state_size = env.get_state_size()
    agent = DQNAgent(state_size, NUM_ACTIONS, DQNConfig())

    epsilon = agent.config.epsilon_start
    metrics = {
        "episode_rewards": [],
        "episode_steps": [],
        "epsilons": [],
        "episode_losses": [],
    }

    for episode in range(1, episodes + 1):
        if random_configurations:
            state = env.reset()
        else:
            state = env.reset(fixed_dirty_positions=fixed_dirty_positions)

        done = False
        total_reward = 0.0
        losses = []
        while not done:
            action = agent.select_action(state, epsilon)
            next_state, reward, done, info = env.step(action)
            agent.store(state, action, reward, next_state, done)
            loss = agent.train_step()
            if loss is not None:
                losses.append(loss)
            state = next_state
            total_reward += reward

        epsilon = max(agent.config.epsilon_end, epsilon * agent.config.epsilon_decay)
        avg_loss = float(sum(losses) / len(losses)) if losses else 0.0
        metrics["episode_rewards"].append(total_reward)
        metrics["episode_steps"].append(env.steps)
        metrics["epsilons"].append(epsilon)
        metrics["episode_losses"].append(avg_loss)

        if episode % 25 == 0 or episode == 1:
            print(
                f"Episode {episode:4d}/{episodes} | reward={total_reward:7.2f} | "
                f"steps={env.steps:4d} | epsilon={epsilon:.3f} | "
                f"loss={avg_loss:.4f} | success={info.get('all_clean', False)}"
            )

    agent.save(str(output_path / "dqn_vacuum.pt"))
    save_metrics_csv(metrics, output_path / "training_metrics.csv")
    plot_training(metrics, output_path)
    return agent, metrics


def run_experiments() -> None:
    experiments = [
        {"rows": 5, "cols": 5, "dirt_count": 5, "episodes": 400, "output_dir": "outputs_5x5"},
        {"rows": 10, "cols": 10, "dirt_count": 8, "episodes": 700, "output_dir": "outputs_10x10"},
        {"rows": 15, "cols": 15, "dirt_count": 10, "episodes": 1000, "output_dir": "outputs_15x15"},
    ]

    summary_rows = []
    for exp in experiments:
        print(f"\n=== Training {exp['rows']}x{exp['cols']} with dirt_count={exp['dirt_count']} ===")
        agent, _ = train_dqn(**exp)
        eval_result = evaluate_agent(
            agent,
            rows=exp["rows"],
            cols=exp["cols"],
            dirt_count=exp["dirt_count"],
            episodes=30,
        )
        summary = {**exp, **eval_result}
        summary_rows.append(summary)
        print(
            f"Eval -> avg_reward={eval_result['avg_reward']:.2f}, "
            f"avg_steps={eval_result['avg_steps']:.2f}, success_rate={eval_result['success_rate']:.2%}"
        )

    summary_path = Path("experiment_summary.csv")
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["rows", "cols", "dirt_count", "episodes", "output_dir", "avg_reward", "avg_steps", "success_rate"],
        )
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"\nSaved experiment summary to {summary_path.resolve()}")


if __name__ == "__main__":
    run_experiments()
