"""
Agent module for Network Routing.
Provides Tabular Q-Learning Agent (QLearningAgent), PyTorch Deep Q-Network Agent (DQNAgent, DQNModel, ReplayBuffer),
TorchScript/inference optimization, batch execution, and pure NumPy fast path execution.
"""

import collections
import json
import os
import random
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class QLearningAgent:
    """
    Tabular Q-Learning agent with continuous state quantization, action masking,
    and JSON serialization support.
    """

    def __init__(
        self,
        num_states: int,
        num_actions: int,
        learning_rate: float = 0.1,
        discount_factor: float = 0.95,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        min_epsilon: float = 0.01,
    ) -> None:
        """
        Initialize Q-Learning agent hyperparameters and Q-table.
        """
        self.num_states = num_states
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

        self.q_table: Dict[Tuple, np.ndarray] = defaultdict(
            lambda: np.zeros(self.num_actions, dtype=np.float64)
        )

    def quantize_state(self, state: Any) -> Tuple:
        """
        Convert continuous observation array or structure into discrete tuple representation.
        """
        if isinstance(state, tuple):
            return state
        if isinstance(state, (int, np.integer)):
            return (int(state),)
        if isinstance(state, dict):
            return tuple(sorted((k, self.quantize_state(v)) for k, v in state.items()))
        if isinstance(state, (list, np.ndarray)):
            state_arr = np.asarray(state, dtype=np.float64)
            quantized = []
            for val in state_arr.flat:
                if val.is_integer():
                    quantized.append(int(val))
                else:
                    if 0.0 <= val <= 1.0:
                        binned = int(np.clip(val * self.num_states, 0, self.num_states - 1))
                    else:
                        binned = int(np.clip(val, 0, 1000))
                    quantized.append(binned)
            return tuple(quantized)
        return (str(state),)

    def get_q_values(self, state: Any) -> np.ndarray:
        """
        Get Q-values vector for a given state.
        """
        q_key = self.quantize_state(state)
        return self.q_table[q_key]

    def choose_action(
        self,
        state: Any,
        valid_actions: Optional[Union[List[int], np.ndarray, Set[int], Tuple[int, ...]]] = None,
    ) -> int:
        """
        Choose action using epsilon-greedy policy with optional action masking.
        """
        q_key = self.quantize_state(state)

        if valid_actions is not None:
            if isinstance(valid_actions, np.ndarray) and valid_actions.dtype == bool:
                valid_indices = np.where(valid_actions)[0]
            elif isinstance(valid_actions, (list, set, tuple, np.ndarray)):
                valid_indices = np.array(list(valid_actions), dtype=int)
            else:
                valid_indices = np.arange(self.num_actions)
        else:
            valid_indices = np.arange(self.num_actions)

        if len(valid_indices) == 0:
            valid_indices = np.arange(self.num_actions)

        if np.random.random() < self.epsilon:
            return int(np.random.choice(valid_indices))
        else:
            q_values = self.q_table[q_key]
            valid_q = q_values[valid_indices]
            max_q = np.max(valid_q)
            best_mask = np.isclose(valid_q, max_q)
            best_actions = valid_indices[best_mask]
            return int(np.random.choice(best_actions))

    def update(
        self,
        state: Any,
        action: int,
        reward: float,
        next_state: Any,
        done: bool,
    ) -> float:
        """
        Update Q-table value using the standard Bellman equation and decay epsilon.
        """
        s_key = self.quantize_state(state)
        s_prime_key = self.quantize_state(next_state)

        q_s = self.q_table[s_key]
        q_s_prime = self.q_table[s_prime_key]

        if done:
            target = reward
        else:
            target = reward + self.discount_factor * np.max(q_s_prime)

        td_error = target - q_s[action]
        q_s[action] += self.learning_rate * td_error

        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

        return float(td_error)

    def save_q_table(self, filepath: str) -> None:
        """
        Serialize Q-table dictionary to JSON file format.
        """
        serialized = {}
        for key, q_vals in self.q_table.items():
            key_str = json.dumps(list(key))
            serialized[key_str] = q_vals.tolist()

        dir_name = os.path.dirname(os.path.abspath(filepath))
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(serialized, f, indent=2)

    def load_q_table(self, filepath: str) -> None:
        """
        Deserialize Q-table dictionary from JSON file format.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Q-table file not found: {filepath}")

        with open(filepath, "r", encoding="utf-8") as f:
            serialized = json.load(f)

        self.q_table.clear()
        for key_str, q_vals_list in serialized.items():
            key_tuple = tuple(json.loads(key_str))
            self.q_table[key_tuple] = np.array(q_vals_list, dtype=np.float64)


class ReplayBuffer:
    """
    Experience Replay Buffer storing transition tuples using fixed-size deque.
    """

    def __init__(self, capacity: int = 10000) -> None:
        """
        Initialize replay buffer with maximum capacity.
        """
        self.capacity = capacity
        self.buffer = collections.deque(maxlen=capacity)

    def push(
        self,
        state: Any,
        action: int,
        reward: float,
        next_state: Any,
        done: bool,
    ) -> None:
        """
        Append experience tuple to buffer.
        """
        self.buffer.append((state, action, reward, next_state, done))

    def sample(
        self, batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample random minibatch of experiences converted to PyTorch FloatTensors.
        """
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states_t = torch.tensor(np.array(states, dtype=np.float32), dtype=torch.float32)
        actions_t = torch.tensor(actions, dtype=torch.long)
        rewards_t = torch.tensor(rewards, dtype=torch.float32)
        next_states_t = torch.tensor(np.array(next_states, dtype=np.float32), dtype=torch.float32)
        dones_t = torch.tensor(dones, dtype=torch.float32)

        return states_t, actions_t, rewards_t, next_states_t, dones_t

    def __len__(self) -> int:
        """
        Get current size of replay buffer.
        """
        return len(self.buffer)


class DQNModel(nn.Module):
    """
    Lightweight PyTorch MLP for mapping state vector to action Q-values.
    Sub-millisecond CPU execution architecture.
    """

    def __init__(self, state_dim: int, action_dim: int) -> None:
        """
        Initialize MLP network layers.
        """
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass computing Q-values vector for input batch x.
        """
        return self.net(x)


class NumPyFastPath:
    """
    Pure NumPy BLAS execution engine for PyTorch policy network weights.
    Bypasses autograd and PyTorch framework overhead completely for microsecond latency.
    """

    def __init__(self, w1: np.ndarray, b1: np.ndarray, w2: np.ndarray, b2: np.ndarray, w3: np.ndarray, b3: np.ndarray) -> None:
        self.w1 = w1.T.copy()
        self.b1 = b1.copy()
        self.w2 = w2.T.copy()
        self.b2 = b2.copy()
        self.w3 = w3.T.copy()
        self.b3 = b3.copy()

    def predict_q_values(self, states: Union[np.ndarray, List[float]]) -> np.ndarray:
        """
        Execute forward pass using BLAS matrix multiplication.
        """
        x = np.asarray(states, dtype=np.float32)
        single = (x.ndim == 1)
        if single:
            x = x[np.newaxis, :]

        h1 = np.maximum(0.0, x @ self.w1 + self.b1)
        h2 = np.maximum(0.0, h1 @ self.w2 + self.b2)
        q = h2 @ self.w3 + self.b3

        return q[0] if single else q

    def choose_action(
        self,
        state: np.ndarray,
        valid_actions: Optional[Union[List[int], np.ndarray, Set[int], Tuple[int, ...]]] = None,
    ) -> int:
        """
        Select action using pure NumPy Q-values calculation.
        """
        q_vals = self.predict_q_values(state)
        action_dim = len(q_vals)

        if valid_actions is not None:
            if isinstance(valid_actions, np.ndarray) and valid_actions.dtype == bool:
                valid_indices = np.where(valid_actions)[0]
            elif isinstance(valid_actions, (list, set, tuple, np.ndarray)):
                valid_indices = np.array(list(valid_actions), dtype=int)
            else:
                valid_indices = np.arange(action_dim)
        else:
            valid_indices = np.arange(action_dim)

        if len(valid_indices) == 0:
            valid_indices = np.arange(action_dim)

        valid_q = q_vals[valid_indices]
        max_q = np.max(valid_q)
        best_mask = np.isclose(valid_q, max_q)
        best_actions = valid_indices[best_mask]
        return int(np.random.choice(best_actions))


class DQNAgent:
    """
    Deep Q-Network (DQN) Agent for continuous observation spaces and discrete action selection.
    Supports TorchScript tracing (`optimize_for_inference`), batch inference (`choose_action_batch`),
    and NumPy fast path execution (`export_to_numpy_fastpath`).
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 0.001,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        min_epsilon: float = 0.01,
        buffer_capacity: int = 10000,
        batch_size: int = 32,
    ) -> None:
        """
        Initialize DQNAgent neural networks, target network, optimizer, and replay buffer.
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.batch_size = batch_size

        self.policy_net = DQNModel(state_dim, action_dim)
        self.target_net = DQNModel(state_dim, action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.traced_policy_net: Optional[Any] = None
        self.is_optimized: bool = False

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.criterion = nn.SmoothL1Loss()
        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity)

    def optimize_for_inference(self, dummy_input: Optional[torch.Tensor] = None) -> None:
        """
        JIT trace policy network for zero-overhead inference execution using TorchScript.
        """
        self.policy_net.eval()
        if dummy_input is None:
            dummy_input = torch.zeros((1, self.state_dim), dtype=torch.float32)

        try:
            traced = torch.jit.trace(self.policy_net, dummy_input)
            self.traced_policy_net = torch.jit.optimize_for_inference(traced)
        except Exception:
            self.traced_policy_net = torch.jit.trace(self.policy_net, dummy_input)

        self.is_optimized = True

    def choose_action(
        self,
        state: Any,
        valid_actions: Optional[Union[List[int], np.ndarray, Set[int], Tuple[int, ...]]] = None,
    ) -> int:
        """
        Select action using epsilon-greedy policy with optional action masking.
        Inference is wrapped in torch.inference_mode() for minimal framework overhead.
        """
        if valid_actions is not None:
            if isinstance(valid_actions, np.ndarray) and valid_actions.dtype == bool:
                valid_indices = np.where(valid_actions)[0]
            elif isinstance(valid_actions, (list, set, tuple, np.ndarray)):
                valid_indices = np.array(list(valid_actions), dtype=int)
            else:
                valid_indices = np.arange(self.action_dim)
        else:
            valid_indices = np.arange(self.action_dim)

        if len(valid_indices) == 0:
            valid_indices = np.arange(self.action_dim)

        if np.random.random() < self.epsilon:
            return int(np.random.choice(valid_indices))

        state_arr = np.asarray(state, dtype=np.float32)
        if state_arr.ndim == 1:
            state_t = torch.from_numpy(state_arr).unsqueeze(0)
        else:
            state_t = torch.from_numpy(state_arr)

        model_to_use = self.traced_policy_net if self.traced_policy_net is not None else self.policy_net

        with torch.inference_mode():
            q_values = model_to_use(state_t).squeeze(0).cpu().numpy()

        valid_q = q_values[valid_indices]
        max_q = np.max(valid_q)
        best_mask = np.isclose(valid_q, max_q)
        best_actions = valid_indices[best_mask]

        return int(np.random.choice(best_actions))

    def choose_action_batch(
        self,
        states: Union[np.ndarray, torch.Tensor],
        valid_action_masks: Optional[Union[np.ndarray, List[Any]]] = None,
    ) -> np.ndarray:
        """
        Execute vector batch inference in a single forward pass for concurrent packet decisions.

        Args:
            states: Array or Tensor of shape [N, state_dim].
            valid_action_masks: Boolean array or list of valid action masks per sample.

        Returns:
            Numpy array of shape (N,) containing selected action index per packet.
        """
        if isinstance(states, torch.Tensor):
            states_t = states.float()
            N = states_t.shape[0]
        else:
            states_arr = np.asarray(states, dtype=np.float32)
            if states_arr.ndim == 1:
                states_arr = states_arr[np.newaxis, :]
            N = states_arr.shape[0]
            states_t = torch.from_numpy(states_arr)

        model_to_use = self.traced_policy_net if self.traced_policy_net is not None else self.policy_net

        with torch.inference_mode():
            q_values_batch = model_to_use(states_t).cpu().numpy()

        selected_actions = np.zeros(N, dtype=int)

        for i in range(N):
            q_vals = q_values_batch[i]
            mask = valid_action_masks[i] if valid_action_masks is not None else None

            if mask is not None:
                if isinstance(mask, np.ndarray) and mask.dtype == bool:
                    valid_indices = np.where(mask)[0]
                elif isinstance(mask, (list, tuple, set, np.ndarray)):
                    valid_indices = np.array(list(mask), dtype=int)
                else:
                    valid_indices = np.arange(self.action_dim)
            else:
                valid_indices = np.arange(self.action_dim)

            if len(valid_indices) == 0:
                valid_indices = np.arange(self.action_dim)

            if np.random.random() < self.epsilon:
                selected_actions[i] = int(np.random.choice(valid_indices))
            else:
                valid_q = q_vals[valid_indices]
                max_q = np.max(valid_q)
                best_mask = np.isclose(valid_q, max_q)
                best_actions = valid_indices[best_mask]
                selected_actions[i] = int(np.random.choice(best_actions))

        return selected_actions

    def export_to_numpy_fastpath(self) -> NumPyFastPath:
        """
        Export policy network weights as pure NumPy arrays for zero-autograd microsecond BLAS execution.
        """
        params = list(self.policy_net.net.parameters())
        w1 = params[0].detach().cpu().numpy()
        b1 = params[1].detach().cpu().numpy()
        w2 = params[2].detach().cpu().numpy()
        b2 = params[3].detach().cpu().numpy()
        w3 = params[4].detach().cpu().numpy()
        b3 = params[5].detach().cpu().numpy()

        return NumPyFastPath(w1, b1, w2, b2, w3, b3)

    def update(self) -> Optional[float]:
        """
        Sample minibatch from ReplayBuffer, compute Smooth L1 (Huber) loss against target network,
        perform backpropagation, update parameters, and decay epsilon.
        """
        if len(self.replay_buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        state_action_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_state_values = self.target_net(next_states).max(dim=1)[0]
            expected_state_action_values = rewards + (1.0 - dones) * self.gamma * next_state_values

        loss = self.criterion(state_action_values, expected_state_action_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

        if self.is_optimized:
            self.traced_policy_net = None
            self.is_optimized = False

        return float(loss.item())

    def update_target_network(self) -> None:
        """
        Copy parameters from policy_net to target_net.
        """
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save_model(self, filepath: str) -> None:
        """
        Save PyTorch policy network state dictionary to file.
        """
        dir_name = os.path.dirname(os.path.abspath(filepath))
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        checkpoint = {
            "policy_net_state_dict": self.policy_net.state_dict(),
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "epsilon": self.epsilon,
            "gamma": self.gamma,
        }
        torch.save(checkpoint, filepath)

    def load_model(self, filepath: str) -> None:
        """
        Load PyTorch policy network state dictionary from file and update target network.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        checkpoint = torch.load(filepath, weights_only=False)
        # Support both old (raw state_dict) and new (checkpoint dict) formats
        if isinstance(checkpoint, dict) and "policy_net_state_dict" in checkpoint:
            self.policy_net.load_state_dict(checkpoint["policy_net_state_dict"])
            self.epsilon = checkpoint.get("epsilon", 0.0)
        else:
            # Legacy format: raw state dict
            self.policy_net.load_state_dict(checkpoint)
        self.update_target_network()
        self.traced_policy_net = None
        self.is_optimized = False
