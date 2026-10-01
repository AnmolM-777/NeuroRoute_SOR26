import random
from collections import deque
from typing import Tuple, List, Dict, Any, Optional

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

from rl_sdn_controller.ai.models import DQN, DuelingDQN


class SumTree:
    """
    Binary SumTree data structure for O(log N) prioritized experience sampling and updates.
    """
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float32)
        self.data = [None] * capacity
        self.write_idx = 0
        self.size = 0

    def _propagate(self, idx: int, change: float):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx: int, s: float) -> int:
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total_priority(self) -> float:
        return float(self.tree[0])

    def add(self, priority: float, data: Any):
        tree_idx = self.write_idx + self.capacity - 1
        self.data[self.write_idx] = data
        self.update(tree_idx, priority)

        self.write_idx = (self.write_idx + 1) % self.capacity
        if self.size < self.capacity:
            self.size += 1

    def update(self, tree_idx: int, priority: float):
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        self._propagate(tree_idx, change)

    def get(self, s: float) -> Tuple[int, float, Any]:
        tree_idx = self._retrieve(0, s)
        data_idx = tree_idx - self.capacity + 1
        return tree_idx, self.tree[tree_idx], self.data[data_idx]

    def __len__(self):
        return self.size


class ReplayBuffer:
    """Standard Experience Replay Buffer for uniform transition storage."""
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32)
        )

    def __len__(self):
        return len(self.buffer)


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay (PER) Buffer (DeepMind / OpenAI standard).
    Samples transitions proportionally to their Temporal Difference (TD) error magnitude.
    """
    def __init__(self, capacity: int = 10000, alpha: float = 0.6, beta_start: float = 0.4, beta_frames: int = 100000, eps: float = 1e-5):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta_start
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.eps = eps
        self.max_priority = 1.0
        self.frame = 0

    def push(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        # New transitions enter with maximal priority to ensure they get sampled at least once
        priority = (self.max_priority + self.eps) ** self.alpha
        self.tree.add(priority, (state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        states, actions, rewards, next_states, dones = [], [], [], [], []
        indices = []
        priorities = []

        total_p = self.tree.total_priority()
        segment = total_p / batch_size

        self.frame += 1
        # Anneal beta towards 1.0
        self.beta = min(1.0, self.beta_start + self.frame * (1.0 - self.beta_start) / max(1, self.beta_frames))

        min_prob = 1e-8
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            idx, priority, data = self.tree.get(s)
            while data is None:
                s = random.uniform(0, total_p)
                idx, priority, data = self.tree.get(s)

            priorities.append(priority)
            indices.append(idx)
            states.append(data[0])
            actions.append(data[1])
            rewards.append(data[2])
            next_states.append(data[3])
            dones.append(data[4])

        sampling_probabilities = np.array(priorities, dtype=np.float32) / (total_p + 1e-8)
        sampling_probabilities = np.maximum(sampling_probabilities, min_prob)
        is_weights = np.power(len(self.tree) * sampling_probabilities, -self.beta)
        is_weights /= (is_weights.max() + 1e-8) # Normalize weights

        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
            np.array(indices, dtype=np.int32),
            np.array(is_weights, dtype=np.float32)
        )

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        for idx, error in zip(indices, td_errors):
            error = abs(float(error)) + self.eps
            self.max_priority = max(self.max_priority, error)
            priority = (error) ** self.alpha
            self.tree.update(int(idx), priority)

    def __len__(self):
        return len(self.tree)


class DQNAgent:
    """
    Deep Q-Network Agent for SDN Routing Control.
    Supports:
      - Standard DQN and Dueling DQN Architectures
      - Double DQN (DDQN) decorrelated value estimation
      - Prioritized Experience Replay (PER) with Importance Sampling
      - Soft Target Updates (Polyak tau) and Hard Periodic Updates
    """
    def __init__(self, state_dim: int, action_dim: int, config: Dict[str, Any]):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config

        self.lr = config.get("learning_rate", 0.0005)
        self.gamma = config.get("gamma", 0.99)
        self.epsilon = config.get("epsilon_start", 1.0)
        self.epsilon_end = config.get("epsilon_end", 0.05)
        self.epsilon_decay = config.get("epsilon_decay", 0.995)
        self.batch_size = config.get("batch_size", 64)
        self.target_update_freq = config.get("target_update_freq", 10)
        self.tau = config.get("tau", 0.001) # Soft update parameter
        self.double_dqn = config.get("double_dqn", True)
        self.use_per = config.get("use_per", config.get("prioritized_replay", True))

        hidden_dims = config.get("network_hidden", config.get("hidden_dims", [128, 128]))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize online and target Q-networks
        use_dueling = config.get("use_dueling", True)
        if use_dueling:
            self.policy_net = DuelingDQN(state_dim, action_dim, hidden_dims).to(self.device)
            self.target_net = DuelingDQN(state_dim, action_dim, hidden_dims).to(self.device)
        else:
            self.policy_net = DQN(state_dim, action_dim, hidden_dims).to(self.device)
            self.target_net = DQN(state_dim, action_dim, hidden_dims).to(self.device)

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        
        capacity = config.get("buffer_capacity", 10000)
        if self.use_per:
            self.memory = PrioritizedReplayBuffer(
                capacity=capacity,
                alpha=config.get("per_alpha", 0.6),
                beta_start=config.get("per_beta_start", 0.4),
                beta_frames=config.get("per_beta_frames", 100000)
            )
        else:
            self.memory = ReplayBuffer(capacity=capacity)

        self.train_step_count = 0

    def select_action(self, state: np.ndarray, evaluate: bool = False) -> int:
        """Epsilon-greedy action selection."""
        if not evaluate and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        self.policy_net.eval()
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
        self.policy_net.train()
        return torch.argmax(q_values, dim=1).item()

    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        """Returns predicted Q-values across all discrete actions."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        self.policy_net.eval()
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
        self.policy_net.train()
        return q_values.cpu().numpy().flatten()

    def update(self) -> float:
        """Performs one step of Q-learning policy optimization."""
        if len(self.memory) < self.batch_size:
            return 0.0

        if self.use_per:
            states, actions, rewards, next_states, dones, indices, is_weights = self.memory.sample(self.batch_size)
            weights_t = torch.FloatTensor(is_weights).unsqueeze(1).to(self.device)
        else:
            states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
            weights_t = torch.ones((self.batch_size, 1), dtype=torch.float32, device=self.device)
            indices = None

        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards_t = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # Compute Q(s, a)
        current_q = self.policy_net(states_t).gather(1, actions_t)

        # Target Q calculation
        with torch.no_grad():
            if self.double_dqn:
                # Double DQN: Policy net selects best action, Target net evaluates its value
                next_actions = self.policy_net(next_states_t).argmax(dim=1, keepdim=True)
                max_next_q = self.target_net(next_states_t).gather(1, next_actions)
            else:
                # Standard DQN
                max_next_q = self.target_net(next_states_t).max(dim=1, keepdim=True)[0]

            target_q = rewards_t + (1.0 - dones_t) * self.gamma * max_next_q

        # Compute TD error and loss
        td_errors = (current_q - target_q).abs().detach().cpu().numpy().flatten()

        # Weighted Smooth L1 / Huber Loss
        elementwise_loss = F.smooth_l1_loss(current_q, target_q, reduction='none')
        loss = (weights_t * elementwise_loss).mean()

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        # Update PER priorities with latest TD errors
        if self.use_per and indices is not None:
            self.memory.update_priorities(indices, td_errors)

        self.train_step_count += 1

        # Target Network Update
        if self.tau > 0:
            # Soft target update (Polyak averaging)
            for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
                target_param.data.copy_(self.tau * policy_param.data + (1.0 - self.tau) * target_param.data)
        elif self.train_step_count % self.target_update_freq == 0:
            # Periodic hard copy (fallback when tau=0)
            self.target_net.load_state_dict(self.policy_net.state_dict())

        # NOTE: Epsilon decay is applied once per EPISODE via decay_epsilon()
        # called by the controller — NOT per gradient step.

        return float(loss.item())


    def decay_epsilon(self):
        """
        Applies one step of epsilon decay — called ONCE per training episode.
        This ensures epsilon schedule is tied to episodes, not gradient update steps,
        giving predictable exploration: e.g., 0.95^20 ≈ 0.36 after 20 episodes.
        """
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

