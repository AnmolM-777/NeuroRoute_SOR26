import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Any, Tuple

from rl_sdn_controller.network.topology import NetworkTopology
from rl_sdn_controller.network.state_manager import StateManager
from rl_sdn_controller.network.routing_engine import RLRoutingEngine
from rl_sdn_controller.sdn_api.routing_table_api import RoutingTableAPI
from rl_sdn_controller.data_plane.simulator import DataPlaneSimulator


class SDNEnv(gym.Env):
    """
    Gymnasium Custom Environment for SDN Routing Policy Optimization.
    State: Aggregated telemetry vector (link utilization %, queue depth, drop rate, latency).
    Action: Discrete index representing path assignment policy for flows.
    Reward: R = alpha * throughput - beta * drop_rate - gamma * latency.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, topology_path: str, traffic_configs: list, rl_config: Dict[str, Any], chaos_config: Dict[str, Any] = None):
        super(SDNEnv, self).__init__()

        self.topology = topology_path if isinstance(topology_path, NetworkTopology) else NetworkTopology(topology_path)
        self.routing_table_api = RoutingTableAPI()

        self.simulator = DataPlaneSimulator(self.topology, self.routing_table_api, traffic_configs, chaos_config=chaos_config)
        
        self.flow_ids = list(self.simulator.generators.keys())
        self.flow_src_dst = self.simulator.flow_src_dst
        
        self.rl_routing_engine = RLRoutingEngine(
            self.topology,
            self.routing_table_api,
            self.flow_ids,
            self.flow_src_dst
        )

        self.state_manager = StateManager(self.topology.get_link_keys())
        self.state_dim = self.state_manager.state_dim

        # Calculate discrete action space dimension based on combinations of flow candidate paths
        num_actions = 1
        for f_id in self.flow_ids:
            num_actions *= len(self.rl_routing_engine.candidate_paths[f_id])

        self.action_dim = num_actions
        self.action_space = spaces.Discrete(self.action_dim)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(self.state_dim,), dtype=np.float32)

        # Config parameters
        ctrl_cfg = rl_config.get("control_plane", {})
        self.update_interval_sec = ctrl_cfg.get("update_interval_ms", 100) / 1000.0
        self.max_steps = ctrl_cfg.get("max_simulation_steps", 600)

        rw_cfg = rl_config.get("reward_function", {})
        self.w_throughput = rw_cfg.get("throughput_weight", 1.0)
        self.w_drop = rw_cfg.get("drop_rate_weight", -100.0)
        self.w_latency = rw_cfg.get("latency_weight", -0.1)

        self.current_step = 0

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        self.current_step = 0
        self.routing_table_api.clear_rules()
        self.simulator.reset()

        # Apply default initial routing rules
        self.rl_routing_engine.apply_action(0)
        # Warmup simulation step (100ms)
        self.simulator.step(self.update_interval_sec)

        telemetry = self.simulator.stats_provider.collect_window_telemetry(self.update_interval_sec)
        obs = self.state_manager.get_observation_vector(telemetry)
        return obs, {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self.current_step += 1

        # 1. Update Layer 2 Routing Tables based on RL Action
        self.rl_routing_engine.apply_action(action)

        # Reset telemetry window statistics before running interval
        for link_q in self.simulator.links.values():
            link_q.reset_window_stats()

        # 2. Run Data Plane Simulator for update_interval_sec
        # Execute in micro-substeps (e.g., 10 sub-steps of 10ms) for realistic queueing
        substeps = 10
        sub_dt = self.update_interval_sec / float(substeps)
        for _ in range(substeps):
            self.simulator.step(sub_dt)

        # 3. Collect window telemetry
        telemetry = self.simulator.stats_provider.collect_window_telemetry(self.update_interval_sec)
        obs = self.state_manager.get_observation_vector(telemetry)

        # 4. Compute Reward R
        total_tx_mbps = sum((st.tx_bytes * 8.0 / 1_000_000.0) / self.update_interval_sec for st in telemetry.values())
        total_tx_p = sum(st.tx_packets for st in telemetry.values())
        total_drop_p = sum(st.dropped_packets for st in telemetry.values())
        total_arrived = total_tx_p + total_drop_p
        overall_drop_pct = (total_drop_p / total_arrived * 100.0) if total_arrived > 0 else 0.0

        # Packet-count-weighted average latency (physically correct: weight by traffic volume)
        total_lat_weighted = sum(st.avg_latency_ms * st.tx_packets for st in telemetry.values() if st.avg_latency_ms > 0)
        total_lat_packets = sum(st.tx_packets for st in telemetry.values() if st.avg_latency_ms > 0)
        avg_lat_ms = total_lat_weighted / total_lat_packets if total_lat_packets > 0 else 0.0

        # Normalized metric components in [0, 1]
        norm_tp = total_tx_mbps / 30.0
        norm_drop = overall_drop_pct / 100.0
        # Latency normalized over 80ms ceiling:
        # Dual-path wire delay: 16-22ms -> norm_lat ≈ 0.20-0.28 (small penalty)
        # Bufferbloat / congestion delay: 50-80ms -> norm_lat ≈ 0.62-1.0 (heavy penalty)
        norm_lat = min(1.0, avg_lat_ms / 80.0)

        # Pareto Multi-Objective Reward (equal weight on speed AND reliability):
        # +1.5 * Throughput   — reward for bits delivered
        # -10.0 * Drop Rate   — heavy penalty for packet loss
        # -10.0 * Latency     — equal penalty for high-latency paths (DQN prefers fast paths)
        reward = (1.5 * norm_tp) - (10.0 * norm_drop) - (10.0 * norm_lat)

        terminated = self.current_step >= self.max_steps
        truncated = False

        chaos_metrics = self.simulator.chaos_engine.get_metrics() if self.simulator.chaos_engine else {}

        info = {
            "step": self.current_step,
            "sim_time_sec": self.simulator.current_time,
            "total_throughput_mbps": total_tx_mbps,
            "avg_drop_pct": overall_drop_pct,
            "avg_latency_ms": avg_lat_ms,
            "telemetry": telemetry,
            "chaos_metrics": chaos_metrics
        }

        return obs, float(reward), terminated, truncated, info

    def inject_link_failure(self, src: str, dst: str, duration_sec: float = 2.0):
        """Manually triggers a link failure for fault-tolerance evaluation."""
        if self.simulator.chaos_engine:
            self.simulator.chaos_engine.fail_link(src, dst, duration_sec, self.simulator.current_time)

    def heal_link(self, src: str, dst: str):
        """Manually recovers a failed link."""
        if self.simulator.chaos_engine:
            self.simulator.chaos_engine.recover_link(src, dst, self.simulator.current_time)

