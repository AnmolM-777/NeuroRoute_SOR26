"""
NeuroRoute RL Training CLI Script.

Trains Q-Learning or PyTorch Deep Q-Network (DQN) routing agents in the
Gymnasium NetworkRoutingEnv environment over N episodes. Saves trained policy
checkpoint to disk.
"""

import random
import sys
import click
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from neuroroute.ai.agent import DQNAgent, QLearningAgent
from neuroroute.ai.env import NetworkRoutingEnv

console = Console()


def train_qlearning(
    episodes: int,
    num_nodes: int,
    lr: float,
    gamma: float,
    epsilon_decay: float,
    save_path: str,
) -> None:
    from neuroroute.network.topology import TopologyManager
    import os

    topo = TopologyManager()
    if os.path.exists("configs/square-topology.json"):
        # set custom training topology here
        # topo.load_topology("configs/my-custom-topology.json")
        # env = NetworkRoutingEnv(num_nodes=len(topo.get_all_nodes()), topology_graph=topo)
        topo.load_topology("configs/square-topology.json")
    num_nodes = len(topo.get_all_nodes()) if topo else num_nodes

    env = NetworkRoutingEnv(num_nodes=num_nodes, topology_graph=topo)
    agent = QLearningAgent(
        num_states=num_nodes,
        num_actions=num_nodes,
        learning_rate=lr,
        discount_factor=gamma,
        epsilon=1.0,
        epsilon_decay=epsilon_decay,
        min_epsilon=0.01,
    )

    console.print(f"[bold cyan]Starting Q-Learning Training ({episodes} episodes, {num_nodes} nodes)...[/bold cyan]")

    successful_episodes = 0
    total_rewards = []
    total_steps = []

    for episode in range(1, episodes + 1):
        # Pick random source and destination for each episode
        src, dst = random.sample(range(num_nodes), 2)
        obs, info = env.reset(options={"current_node": src, "destination_node": dst})

        done = False
        ep_reward = 0.0
        steps = 0

        while not done:
            # Use compact (current_node, destination) state for tabular Q-learning
            # This gives N*(N-1) unique states — fully learnable for small networks
            state = (env.current_node, dst)
            action_mask = env.get_action_mask(env.current_node)
            action = agent.choose_action(state, valid_actions=action_mask)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            next_state = (env.current_node, dst)
            agent.update(state, action, reward, next_state, done)
            obs = next_obs
            ep_reward += reward
            steps += 1

        if terminated and env.current_node == dst:
            successful_episodes += 1

        total_rewards.append(ep_reward)
        total_steps.append(steps)

        if episode % max(1, episodes // 5) == 0:
            avg_rew = np.mean(total_rewards[-50:])
            avg_st = np.mean(total_steps[-50:])
            succ_rate = (successful_episodes / episode) * 100
            console.print(
                f"  Episode {episode:4d}/{episodes} | "
                f"Epsilon: {agent.epsilon:.3f} | "
                f"Success Rate: {succ_rate:5.1f}% | "
                f"Avg Reward: {avg_rew:6.2f} | "
                f"Avg Steps: {avg_st:4.1f}"
            )

    # Save trained Q-Table
    agent.save_q_table(save_path)
    console.print(f"[bold green]✔ Q-Table saved to '{save_path}'[/bold green]")

    # Evaluation Phase (epsilon = 0.0, pure exploitation)
    eval_episodes = num_nodes * (num_nodes - 1) * 2  # test all src->dst pairs multiple times
    console.print(f"\n[bold yellow]Evaluating Trained Agent ({eval_episodes} episodes, Exploitation Mode)...[/bold yellow]")
    agent.epsilon = 0.0
    eval_successes = 0
    eval_steps = []

    for episode in range(eval_episodes):
        src, dst = random.sample(range(num_nodes), 2)
        obs, info = env.reset(options={"current_node": src, "destination_node": dst})
        done = False
        steps = 0
        while not done and steps < 20:
            state = (env.current_node, dst)
            mask = env.get_action_mask(env.current_node)
            action = agent.choose_action(state, valid_actions=mask)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            steps += 1

        if terminated and env.current_node == dst:
            eval_successes += 1
            eval_steps.append(steps)

    eval_rate = (eval_successes / eval_episodes) * 100
    avg_eval_steps = np.mean(eval_steps) if eval_steps else 0.0

    table = Table(title="Q-Learning Evaluation Summary", expand=False)
    table.add_column("Metric", style="bold cyan")
    table.add_column("Value", style="bold green", justify="right")
    table.add_row("Evaluation Delivery Rate", f"{eval_rate:.1f}%")
    table.add_row("Avg Hops per Delivery", f"{avg_eval_steps:.2f}")
    table.add_row("Final Epsilon", f"{agent.epsilon:.3f}")
    table.add_row("Q-Table Size", f"{len(agent.q_table)} states")
    console.print(table)


def train_dqn(
    episodes: int,
    num_nodes: int,
    lr: float,
    gamma: float,
    epsilon_decay: float,
    batch_size: int,
    save_path: str,
) -> None:
    """Train Deep Q-Network (DQN) Agent."""
    from neuroroute.network.topology import TopologyManager
    import os

    topo = TopologyManager()
    if os.path.exists("configs/square-topology.json"):
        # set custom training topology here
        # topo.load_topology("configs/my-custom-topology.json")
        topo.load_topology("configs/square-topology.json")
    num_nodes = len(topo.get_all_nodes()) if topo.get_all_nodes() else num_nodes

    env = NetworkRoutingEnv(num_nodes=num_nodes, topology_graph=topo)
    obs_dim = 2 * num_nodes + 1
    agent = DQNAgent(
        state_dim=obs_dim,
        action_dim=num_nodes,
        lr=lr,
        gamma=gamma,
        epsilon=1.0,
        epsilon_decay=epsilon_decay,
        min_epsilon=0.01,
        batch_size=batch_size,
    )

    console.print(f"[bold cyan]Starting DQN Training ({episodes} episodes, {num_nodes} nodes)...[/bold cyan]")

    successful_episodes = 0
    total_rewards = []

    for episode in range(1, episodes + 1):
        src = np.random.randint(0, num_nodes - 1)
        dst = num_nodes - 1
        obs, info = env.reset(options={"current_node": src, "destination_node": dst})

        done = False
        ep_reward = 0.0

        while not done:
            action_mask = env.get_action_mask(env.current_node)
            action = agent.choose_action(obs, valid_actions=action_mask)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            agent.replay_buffer.push(obs, action, reward, next_obs, done)
            agent.update()

            obs = next_obs
            ep_reward += reward

        if terminated and env.current_node == dst:
            successful_episodes += 1

        total_rewards.append(ep_reward)

        if episode % max(1, episodes // 5) == 0:
            succ_rate = (successful_episodes / episode) * 100
            avg_rew = np.mean(total_rewards[-50:])
            console.print(
                f"  Episode {episode:4d}/{episodes} | "
                f"Epsilon: {agent.epsilon:.3f} | "
                f"Success Rate: {succ_rate:5.1f}% | "
                f"Avg Reward: {avg_rew:6.2f}"
            )

    agent.save_model(save_path)
    console.print(f"[bold green]✔ PyTorch DQN Model saved to '{save_path}'[/bold green]")


@click.command()
@click.option(
    "--agent-type",
    "-a",
    type=click.Choice(["qlearning", "dqn"]),
    default="qlearning",
    help="RL algorithm to train.",
)
@click.option("--episodes", "-e", default=5000, help="Number of RL training episodes.")
@click.option("--nodes", "-n", default=4, help="Number of nodes in network graph.")
@click.option("--lr", default=0.1, help="Learning rate.")
@click.option("--gamma", default=0.95, help="Discount factor.")
@click.option("--epsilon-decay", default=0.96, help="Epsilon decay multiplier per episode.")
@click.option("--batch-size", default=32, help="Batch size for DQN training.")
@click.option(
    "--save-path",
    "-s",
    default="q_table.json",
    help="File path to save the trained model / Q-table checkpoint.",
)
def main(
    agent_type: str,
    episodes: int,
    nodes: int,
    lr: float,
    gamma: float,
    epsilon_decay: float,
    batch_size: int,
    save_path: str,
) -> None:
    """NeuroRoute AI Training Command Line Tool."""
    console.print(
        Panel.fit(
            f"[bold green]NeuroRoute AI Trainer[/bold green]\n"
            f"Algorithm: [bold yellow]{agent_type.upper()}[/bold yellow] | "
            f"Episodes: [bold cyan]{episodes}[/bold cyan] | "
            f"Nodes: [bold cyan]{nodes}[/bold cyan]",
            style="bold white on blue",
        )
    )

    if agent_type == "qlearning":
        train_qlearning(episodes, nodes, lr, gamma, epsilon_decay, save_path)
    else:
        if save_path == "q_table.json":
            save_path = "dqn_model.pt"
        train_dqn(episodes, nodes, lr, gamma, epsilon_decay, batch_size, save_path)


if __name__ == "__main__":
    main()
