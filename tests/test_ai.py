"""
Comprehensive unit tests for neuroroute.ai module: NetworkRoutingEnv, QLearningAgent, and DQNAgent.
"""

import os
import random
import numpy as np
import pytest
import torch
import gymnasium as gym

from neuroroute.ai.env import NetworkRoutingEnv
from neuroroute.ai.agent import QLearningAgent, DQNAgent, DQNModel, ReplayBuffer
from neuroroute.ai import (
    NetworkRoutingEnv as ExportedEnv,
    QLearningAgent as ExportedQL,
    DQNAgent as ExportedDQN,
    DQNModel as ExportedModel,
    ReplayBuffer as ExportedBuffer,
)


def test_imports_and_exports():
    """Verify classes are exported properly from neuroroute.ai."""
    assert ExportedEnv is NetworkRoutingEnv
    assert ExportedQL is QLearningAgent
    assert ExportedDQN is DQNAgent
    assert ExportedModel is DQNModel
    assert ExportedBuffer is ReplayBuffer


def test_env_initialization_and_spaces():
    """Test NetworkRoutingEnv creation, action space, observation space, and reset behavior."""
    num_nodes = 5
    env = NetworkRoutingEnv(num_nodes=num_nodes, max_queue_capacity=100)

    # Action space check
    assert isinstance(env.action_space, gym.spaces.Discrete)
    assert env.action_space.n == num_nodes

    # Observation space check
    # Observation space: 3 * num_nodes + 1 = 16
    obs_dim = 3 * num_nodes + 1
    assert isinstance(env.observation_space, gym.spaces.Box)
    assert env.observation_space.shape == (obs_dim,)
    assert env.observation_space.dtype == np.float32

    # Reset check (Gymnasium API)
    obs, info = env.reset(seed=42)
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (obs_dim,)
    assert np.all(obs >= env.observation_space.low)
    assert np.all(obs <= env.observation_space.high)

    # Diagnostic info check
    assert "current_node" in info
    assert "destination_node" in info
    assert "dropped_packets" in info
    assert "successful_deliveries" in info
    assert "action_mask" in info
    assert len(info["action_mask"]) == num_nodes

    env.close()


def test_env_step_valid_action_and_reward():
    """Test valid hop stepping and destination reach reward."""
    env = NetworkRoutingEnv(num_nodes=5, max_queue_capacity=100)
    
    # Force state to node 0 -> node 1
    obs, info = env.reset(options={"current_node": 0, "destination_node": 1})
    assert info["current_node"] == 0
    assert info["destination_node"] == 1

    # Valid step to neighbor 1
    obs, reward, terminated, truncated, info = env.step(1)

    assert isinstance(obs, np.ndarray)
    assert reward == pytest.approx(98.0)  # max(10, 100 - 2*1) = 98.0, step-aware delivery
    assert terminated is True
    assert truncated is False
    assert info["successful_deliveries"] == 1
    assert info["current_node"] == 1

    env.close()


def test_env_invalid_action_penalty():
    """Test execution of invalid actions demonstrating heavy penalty and state retention."""
    topology = {
        0: {1: {"latency": 10.0}},
        1: {0: {"latency": 10.0}, 2: {"latency": 15.0}},
        2: {1: {"latency": 15.0}},
        3: {},
        4: {},
    }
    env = NetworkRoutingEnv(num_nodes=5, topology_graph=topology)
    
    obs, info = env.reset(options={"current_node": 0, "destination_node": 2})
    assert env.current_node == 0

    # Action 3 is invalid from node 0
    obs, reward, terminated, truncated, info = env.step(3)

    assert reward == -100.0
    assert terminated is False
    assert env.dropped_packets == 1
    assert info["current_node"] == 0
    assert "error" in info

    env.close()


def test_env_render():
    """Test environment rendering in ansi and human modes."""
    env_ansi = NetworkRoutingEnv(num_nodes=4, render_mode="ansi")
    env_ansi.reset()
    output = env_ansi.render()
    assert isinstance(output, str)
    assert "Node:" in output

    env_human = NetworkRoutingEnv(num_nodes=4, render_mode="human")
    env_human.reset()
    output_human = env_human.render()
    assert isinstance(output_human, str)

    env_ansi.close()
    env_human.close()


def test_agent_initialization_and_quantization():
    """Test QLearningAgent initialization and state quantization."""
    agent = QLearningAgent(
        num_states=10,
        num_actions=5,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_decay=0.99,
        min_epsilon=0.05,
    )

    assert agent.num_states == 10
    assert agent.num_actions == 5
    assert agent.learning_rate == 0.1
    assert agent.discount_factor == 0.95
    assert agent.epsilon == 1.0

    state_arr = np.array([0.23, 1.0, 4.0, 15.2], dtype=np.float32)
    q_tuple = agent.quantize_state(state_arr)
    assert isinstance(q_tuple, tuple)
    assert len(q_tuple) == len(state_arr)

    dict_state = {"current": 0, "dest": 4}
    q_dict_tuple = agent.quantize_state(dict_state)
    assert isinstance(q_dict_tuple, tuple)


def test_agent_choose_action_with_masking():
    """Test epsilon-greedy action selection and action masking."""
    agent = QLearningAgent(num_states=10, num_actions=5, epsilon=0.0)

    state = (0,)
    q_values = np.array([1.0, 10.0, 3.0, 0.0, 8.0])
    agent.q_table[state] = q_values

    action_unmasked = agent.choose_action(state)
    assert action_unmasked == 1

    action_masked_list = agent.choose_action(state, valid_actions=[0, 3, 4])
    assert action_masked_list == 4

    bool_mask = np.array([True, False, False, False, False])
    action_masked_bool = agent.choose_action(state, valid_actions=bool_mask)
    assert action_masked_bool == 0


def test_agent_q_table_bellman_update():
    """Test Bellman equation Q-table updates and epsilon decay."""
    agent = QLearningAgent(
        num_states=10,
        num_actions=4,
        learning_rate=0.1,
        discount_factor=0.9,
        epsilon=1.0,
        epsilon_decay=0.9,
        min_epsilon=0.01,
    )

    state = (0,)
    next_state = (1,)
    action = 2
    reward = 10.0

    agent.q_table[next_state] = np.array([20.0, 5.0, 0.0, 0.0])
    assert agent.q_table[state][action] == 0.0

    td_error = agent.update(state, action, reward, next_state, done=False)

    assert pytest.approx(td_error, abs=1e-5) == 28.0
    assert pytest.approx(agent.q_table[state][action], abs=1e-5) == 2.8
    assert pytest.approx(agent.epsilon, abs=1e-5) == 0.9

    td_error_term = agent.update(state, action, reward=5.0, next_state=next_state, done=True)
    assert pytest.approx(agent.q_table[state][action], abs=1e-5) == 3.02


def test_agent_save_and_load_q_table(tmp_path):
    """Test serializing and deserializing Q-table to/from JSON."""
    agent_save = QLearningAgent(num_states=10, num_actions=4)
    agent_save.q_table[(0, 1)] = np.array([1.5, 2.5, 3.5, 4.5])
    agent_save.q_table[(2, 3)] = np.array([0.1, 0.2, 0.3, 0.4])

    json_path = os.path.join(tmp_path, "q_table.json")
    agent_save.save_q_table(json_path)

    assert os.path.exists(json_path)

    agent_load = QLearningAgent(num_states=10, num_actions=4)
    agent_load.load_q_table(json_path)

    assert (0, 1) in agent_load.q_table
    assert (2, 3) in agent_load.q_table
    np.testing.assert_allclose(agent_load.q_table[(0, 1)], np.array([1.5, 2.5, 3.5, 4.5]))
    np.testing.assert_allclose(agent_load.q_table[(2, 3)], np.array([0.1, 0.2, 0.3, 0.4]))


# --- DQNAgent and ReplayBuffer Unit Tests ---

def test_replay_buffer_push_and_sample():
    """Test ReplayBuffer capacity, pushing, and tensor conversion shapes."""
    buffer = ReplayBuffer(capacity=100)
    state_dim = 16

    for i in range(50):
        s = np.random.randn(state_dim).astype(np.float32)
        a = i % 5
        r = float(i)
        ns = np.random.randn(state_dim).astype(np.float32)
        d = (i % 10 == 0)
        buffer.push(s, a, r, ns, d)

    assert len(buffer) == 50

    batch_size = 16
    states, actions, rewards, next_states, dones = buffer.sample(batch_size)

    assert isinstance(states, torch.Tensor)
    assert states.shape == (batch_size, state_dim)
    assert isinstance(actions, torch.Tensor)
    assert actions.shape == (batch_size,)
    assert isinstance(rewards, torch.Tensor)
    assert rewards.shape == (batch_size,)
    assert isinstance(next_states, torch.Tensor)
    assert next_states.shape == (batch_size, state_dim)
    assert isinstance(dones, torch.Tensor)
    assert dones.shape == (batch_size,)


def test_dqn_agent_action_selection_and_masking():
    """Test DQNAgent action selection with torch.no_grad context and valid action mask."""
    state_dim = 16
    action_dim = 5
    agent = DQNAgent(state_dim=state_dim, action_dim=action_dim, epsilon=0.0)

    obs = np.zeros(state_dim, dtype=np.float32)

    # Choose action without mask
    action = agent.choose_action(obs)
    assert 0 <= action < action_dim

    # Choose action with mask [0, 2, 4]
    valid_list = [0, 2, 4]
    action_masked = agent.choose_action(obs, valid_actions=valid_list)
    assert action_masked in valid_list

    # Choose action with boolean mask [False, True, False, False, False] -> must select index 1
    bool_mask = np.array([False, True, False, False, False])
    action_bool = agent.choose_action(obs, valid_actions=bool_mask)
    assert action_bool == 1


def test_dqn_agent_update():
    """Test DQNAgent update step computes loss and decays epsilon."""
    state_dim = 4
    action_dim = 2
    batch_size = 8
    agent = DQNAgent(state_dim=state_dim, action_dim=action_dim, batch_size=batch_size, epsilon=1.0)

    # Buffer < batch_size should return None
    assert agent.update() is None

    # Fill replay buffer
    for _ in range(20):
        s = np.random.randn(state_dim).astype(np.float32)
        a = random.randint(0, action_dim - 1)
        r = random.random()
        ns = np.random.randn(state_dim).astype(np.float32)
        d = False
        agent.replay_buffer.push(s, a, r, ns, d)

    loss = agent.update()
    assert isinstance(loss, float)
    assert loss >= 0.0
    assert agent.epsilon < 1.0


def test_dqn_agent_save_and_load_model(tmp_path):
    """Test saving and loading PyTorch weights (.pth)."""
    state_dim = 6
    action_dim = 3
    agent_save = DQNAgent(state_dim=state_dim, action_dim=action_dim)

    model_path = os.path.join(tmp_path, "dqn_weights.pth")
    agent_save.save_model(model_path)
    assert os.path.exists(model_path)

    agent_load = DQNAgent(state_dim=state_dim, action_dim=action_dim)
    agent_load.load_model(model_path)

    # Verify weights match
    for p1, p2 in zip(agent_save.policy_net.parameters(), agent_load.policy_net.parameters()):
        np.testing.assert_allclose(p1.detach().numpy(), p2.detach().numpy())
