import pytest
import numpy as np
import torch
import os

from rl_sdn_controller.ai.models import DQN, DuelingDQN
from rl_sdn_controller.ai.agent import DQNAgent, ReplayBuffer
from rl_sdn_controller.ai.policy_exporter import export_policy_to_onnx


def test_dqn_model_forward():
    state_dim = 16
    action_dim = 4
    model = DQN(state_dim, action_dim, hidden_dims=[64, 64])
    x = torch.randn(2, state_dim)
    q_vals = model(x)
    assert q_vals.shape == (2, action_dim)


def test_dueling_dqn_forward():
    state_dim = 16
    action_dim = 4
    model = DuelingDQN(state_dim, action_dim, hidden_dims=[64, 64])
    x = torch.randn(2, state_dim)
    q_vals = model(x)
    assert q_vals.shape == (2, action_dim)


def test_replay_buffer():
    buf = ReplayBuffer(capacity=10)
    s = np.zeros(4, dtype=np.float32)
    a = 1
    r = 1.0
    ns = np.ones(4, dtype=np.float32)
    d = False

    buf.push(s, a, r, ns, d)
    assert len(buf) == 1

    states, actions, rewards, next_states, dones = buf.sample(1)
    assert states.shape == (1, 4)
    assert actions[0] == 1
    assert rewards[0] == 1.0


def test_agent_action_selection_and_update():
    state_dim = 8
    action_dim = 3
    config = {
        "learning_rate": 0.001,
        "gamma": 0.99,
        "epsilon_start": 0.5,
        "epsilon_end": 0.05,
        "epsilon_decay": 0.9,
        "batch_size": 4,
        "target_update_freq": 2,
        "network_hidden": [32, 32]
    }
    agent = DQNAgent(state_dim, action_dim, config)
    
    state = np.random.rand(state_dim).astype(np.float32)
    action = agent.select_action(state, evaluate=False)
    assert 0 <= action < action_dim

    # Fill replay buffer to trigger update
    for _ in range(10):
        s = np.random.rand(state_dim).astype(np.float32)
        ns = np.random.rand(state_dim).astype(np.float32)
        agent.memory.push(s, 0, 1.0, ns, False)

    loss = agent.update()
    assert loss >= 0.0


def test_onnx_export(tmp_path):
    model = DQN(state_dim=8, action_dim=2)
    onnx_file = str(tmp_path / "test_model.onnx")
    res = export_policy_to_onnx(model, state_dim=8, output_path=onnx_file)
    assert os.path.exists(res)
