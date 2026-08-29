"""
Multi-Node Routing Integration Test Suite for NeuroRoute.
Simulates multi-node packet traversal across a 4-node diamond topology.
"""

import pytest
import numpy as np
from neuroroute.ai.env import NetworkRoutingEnv
from neuroroute.ai.agent import QLearningAgent, DQNAgent


@pytest.fixture
def diamond_topology():
    """
    Diamond network topology:
    Node 0 (Source A) -> Node 1 (B, 2ms) or Node 2 (C, 25ms)
    Node 1 (B) -> Node 3 (Destination D, 3ms)  => Total path latency: 5ms
    Node 2 (C) -> Node 3 (Destination D, 25ms) => Total path latency: 50ms
    """
    return {
        0: {1: {"latency": 2.0}, 2: {"latency": 25.0}},
        1: {3: {"latency": 3.0}},
        2: {3: {"latency": 25.0}},
        3: {},
    }


def test_packet_traversal_diamond_topology_qlearning(diamond_topology):
    """
    Integration Test: Train Q-Learning agent on diamond topology for 80 episodes,
    then evaluate performance over 20 episodes.

    Asserts:
    1. 100% delivery success rate.
    2. Average hop count <= 2.
    3. Agent learns to prefer shorter path (A -> B -> D) over (A -> C -> D).
    """
    env = NetworkRoutingEnv(num_nodes=4, topology_graph=diamond_topology)
    agent = QLearningAgent(
        num_states=10,
        num_actions=4,
        learning_rate=0.2,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_decay=0.95,
        min_epsilon=0.01,
    )

    # RL Training Phase (80 episodes)
    for episode in range(80):
        obs, info = env.reset(options={"current_node": 0, "destination_node": 3})
        done = False
        while not done:
            action_mask = env.get_action_mask(env.current_node)
            action = agent.choose_action(obs, valid_actions=action_mask)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            agent.update(obs, action, reward, next_obs, done)
            obs = next_obs

    # Evaluation Phase (20 episodes, pure exploitation)
    agent.epsilon = 0.0
    successful_deliveries = 0
    total_hops = 0
    path_counts = {"A->B->D": 0, "A->C->D": 0, "Other": 0}

    eval_episodes = 20
    for episode in range(eval_episodes):
        obs, info = env.reset(options={"current_node": 0, "destination_node": 3})
        done = False
        hops = 0
        nodes_visited = [env.current_node]

        while not done and hops < 10:
            action_mask = env.get_action_mask(env.current_node)
            action = agent.choose_action(obs, valid_actions=action_mask)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            hops += 1
            nodes_visited.append(env.current_node)
            obs = next_obs

        if terminated and env.current_node == 3:
            successful_deliveries += 1
            total_hops += hops

        if nodes_visited == [0, 1, 3]:
            path_counts["A->B->D"] += 1
        elif nodes_visited == [0, 2, 3]:
            path_counts["A->C->D"] += 1
        else:
            path_counts["Other"] += 1

    env.close()

    # Assertions
    delivery_rate = successful_deliveries / eval_episodes
    avg_hops = total_hops / max(1, successful_deliveries)

    assert delivery_rate == 1.0, f"Delivery rate {delivery_rate * 100}% is less than 100%"
    assert avg_hops <= 2.0, f"Average hop count {avg_hops} exceeds 2.0"
    assert path_counts["A->B->D"] > path_counts["A->C->D"], (
        f"Agent did not favor lower latency path A->B->D. Path choices: {path_counts}"
    )


def test_packet_traversal_diamond_topology_dqn(diamond_topology):
    """
    Integration Test: Train DQNAgent on diamond topology and evaluate performance.
    """
    env = NetworkRoutingEnv(num_nodes=4, topology_graph=diamond_topology)
    obs_dim = 3 * 4 + 1
    agent = DQNAgent(
        state_dim=obs_dim,
        action_dim=4,
        lr=0.01,
        gamma=0.95,
        epsilon=1.0,
        epsilon_decay=0.96,
        min_epsilon=0.01,
        batch_size=16,
    )

    # RL Training Phase (60 episodes)
    for episode in range(60):
        obs, info = env.reset(options={"current_node": 0, "destination_node": 3})
        done = False
        while not done:
            action_mask = env.get_action_mask(env.current_node)
            action = agent.choose_action(obs, valid_actions=action_mask)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            agent.replay_buffer.push(obs, action, reward, next_obs, done)
            agent.update()
            obs = next_obs

    # Evaluation Phase
    agent.epsilon = 0.0
    successful_deliveries = 0
    total_hops = 0

    eval_episodes = 10
    for episode in range(eval_episodes):
        obs, info = env.reset(options={"current_node": 0, "destination_node": 3})
        done = False
        hops = 0

        while not done and hops < 10:
            action_mask = env.get_action_mask(env.current_node)
            action = agent.choose_action(obs, valid_actions=action_mask)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            hops += 1
            obs = next_obs

        if terminated and env.current_node == 3:
            successful_deliveries += 1
            total_hops += hops

    env.close()

    delivery_rate = successful_deliveries / eval_episodes
    avg_hops = total_hops / max(1, successful_deliveries)

    assert delivery_rate == 1.0
    assert avg_hops <= 2.0
