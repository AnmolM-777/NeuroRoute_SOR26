import logging
from typing import Dict, List, Tuple, Any, Optional
import numpy as np

from rl_sdn_controller.ai.agent import DQNAgent
from rl_sdn_controller.ai.env import SDNEnv
from rl_sdn_controller.network.topology import NetworkTopology
from rl_sdn_controller.sdn_api.stats_provider import LinkStats

logger = logging.getLogger(__name__)


class NetworkRegion:
    """
    Represents a localized network partition (e.g. Data Center Pod or Regional Network).
    """
    def __init__(self, region_id: str, nodes: List[str], link_keys: List[Tuple[str, str]], flow_ids: List[str]):
        self.region_id = region_id
        self.nodes = set(nodes)
        self.link_keys = link_keys
        self.flow_ids = flow_ids


class RegionalAgent:
    """
    Autonomous Regional RL Agent.
    Manages local routing decisions, queue balancing, and link health within its assigned network region.
    Communicates summary telemetry and boundary crossing requests to the Global Coordinator.
    """
    def __init__(self, region: NetworkRegion, state_dim: int, action_dim: int, agent_config: Dict[str, Any]):
        self.region = region
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.agent = DQNAgent(state_dim, action_dim, agent_config)

    def extract_local_state(self, full_telemetry: Dict[Tuple[str, str], LinkStats]) -> np.ndarray:
        """Extracts normalized telemetry for links belonging to this region."""
        obs = []
        for key in self.region.link_keys:
            if key in full_telemetry:
                st = full_telemetry[key]
                util = st.utilization_pct / 100.0
                q_depth = st.queue_depth / float(st.max_queue_packets) if st.max_queue_packets > 0 else 0.0
                drop = st.drop_rate_pct / 100.0
                lat = min(1.0, st.avg_latency_ms / 50.0)
                is_up = 1.0 if st.is_up else 0.0
            else:
                util, q_depth, drop, lat, is_up = 0.0, 0.0, 0.0, 0.0, 0.0
            obs.extend([util, q_depth, drop, lat, is_up])

        if not obs:
            obs = [0.0] * self.state_dim
        # Pad or slice to match expected local state_dim
        if len(obs) < self.state_dim:
            obs.extend([0.0] * (self.state_dim - len(obs)))
        return np.array(obs[:self.state_dim], dtype=np.float32)

    def compute_local_summary(self, full_telemetry: Dict[Tuple[str, str], LinkStats]) -> Dict[str, float]:
        """Computes condensed telemetry summary to report up to the Global Coordinator."""
        utils = []
        drops = []
        lats = []
        for key in self.region.link_keys:
            if key in full_telemetry:
                st = full_telemetry[key]
                utils.append(st.utilization_pct)
                drops.append(st.drop_rate_pct)
                if st.avg_latency_ms > 0:
                    lats.append(st.avg_latency_ms)

        return {
            "avg_utilization": float(np.mean(utils)) if utils else 0.0,
            "avg_drop_rate": float(np.mean(drops)) if drops else 0.0,
            "avg_latency": float(np.mean(lats)) if lats else 0.0
        }

    def select_action(self, local_state: np.ndarray, evaluate: bool = False) -> int:
        return self.agent.select_action(local_state, evaluate=evaluate)

    def update(self) -> float:
        return self.agent.update()


class GlobalCoordinatorAgent:
    """
    Global Coordinator Agent (Top-Level Controller).
    Performs inter-region routing, global load balancing, and aggregates regional health metrics.
    """
    def __init__(self, num_regions: int, global_link_keys: List[Tuple[str, str]], action_dim: int, agent_config: Dict[str, Any]):
        self.num_regions = num_regions
        self.global_link_keys = global_link_keys
        # Summary state: 3 metrics per region + 5 features per inter-region boundary link
        self.state_dim = max(5, (num_regions * 3) + (len(global_link_keys) * 5))
        self.action_dim = max(1, action_dim)
        self.agent = DQNAgent(self.state_dim, self.action_dim, agent_config)

    def extract_global_state(self, regional_summaries: List[Dict[str, float]], full_telemetry: Dict[Tuple[str, str], LinkStats]) -> np.ndarray:
        obs = []
        # 1. Regional health summaries
        for s in regional_summaries:
            obs.append(s.get("avg_utilization", 0.0) / 100.0)
            obs.append(s.get("avg_drop_rate", 0.0) / 100.0)
            obs.append(min(1.0, s.get("avg_latency", 0.0) / 50.0))

        # 2. Inter-region boundary links telemetry
        for key in self.global_link_keys:
            if key in full_telemetry:
                st = full_telemetry[key]
                util = st.utilization_pct / 100.0
                q_depth = st.queue_depth / float(st.max_queue_packets) if st.max_queue_packets > 0 else 0.0
                drop = st.drop_rate_pct / 100.0
                lat = min(1.0, st.avg_latency_ms / 50.0)
                is_up = 1.0 if st.is_up else 0.0
            else:
                util, q_depth, drop, lat, is_up = 0.0, 0.0, 0.0, 0.0, 0.0
            obs.extend([util, q_depth, drop, lat, is_up])

        if len(obs) < self.state_dim:
            obs.extend([0.0] * (self.state_dim - len(obs)))
        return np.array(obs[:self.state_dim], dtype=np.float32)

    def select_action(self, global_state: np.ndarray, evaluate: bool = False) -> int:
        return self.agent.select_action(global_state, evaluate=evaluate)

    def update(self) -> float:
        return self.agent.update()


class HierarchicalSDNController:
    """
    Hierarchical Multi-Agent SDN Controller Orchestrator.
    Partitions topology into regional sub-domains with independent Regional Agents
    coordinated by a Global Coordinator.
    """
    def __init__(self, topology_path: str, traffic_configs: list, rl_config: Dict[str, Any], chaos_config: Dict[str, Any] = None, num_regions: int = 2):
        self.rl_config = rl_config
        self.env = SDNEnv(topology_path, traffic_configs, rl_config, chaos_config=chaos_config)
        self.topology = self.env.topology
        self.num_regions = num_regions

        self.regions, self.boundary_links = self._partition_topology(num_regions)
        agent_cfg = rl_config.get("agent", {})

        # Initialize Regional Agents
        self.regional_agents: List[RegionalAgent] = []
        for reg in self.regions:
            local_state_dim = max(5, len(reg.link_keys) * 5)
            # Each regional agent controls a subset of candidate choices
            local_action_dim = max(1, self.env.action_dim // len(self.regions)) if self.env.action_dim > len(self.regions) else self.env.action_dim
            r_agent = RegionalAgent(reg, local_state_dim, local_action_dim, agent_cfg)
            self.regional_agents.append(r_agent)

        # Initialize Global Coordinator
        self.coordinator = GlobalCoordinatorAgent(
            num_regions=len(self.regions),
            global_link_keys=self.boundary_links,
            action_dim=self.env.action_dim,
            agent_config=agent_cfg
        )

        self.global_reward_weight = rl_config.get("hierarchical", {}).get("global_reward_weight", 0.3)

    def _partition_topology(self, num_regions: int) -> Tuple[List[NetworkRegion], List[Tuple[str, str]]]:
        """Partitions topology graph into regional clusters and boundary links."""
        nodes = list(self.topology.nodes_info.keys())
        all_links = self.topology.get_link_keys()

        # Partition nodes evenly across regions
        chunks = np.array_split(nodes, num_regions)
        regions: List[NetworkRegion] = []
        assigned_links = set()

        for idx, chunk in enumerate(chunks):
            r_nodes = list(chunk)
            r_node_set = set(r_nodes)
            r_links = [k for k in all_links if k[0] in r_node_set and k[1] in r_node_set]
            assigned_links.update(r_links)
            # Flow IDs associated with this region
            r_flows = [f_id for f_id, (src, dst) in self.env.flow_src_dst.items() if src in r_node_set or dst in r_node_set]
            regions.append(NetworkRegion(f"region_{idx+1}", r_nodes, r_links, r_flows))

        boundary_links = [k for k in all_links if k not in assigned_links]
        return regions, boundary_links

    def train_episodes(self, num_episodes: int = 10, verbose: bool = True) -> List[Dict[str, Any]]:
        """Trains Hierarchical Multi-Agent system over multiple episodes."""
        history = []

        for ep in range(1, num_episodes + 1):
            obs, _ = self.env.reset()
            telemetry = self.env.simulator.stats_provider.collect_window_telemetry(self.env.update_interval_sec)

            total_reward = 0.0
            episode_losses = []
            episode_throughput = []
            episode_drops = []
            episode_latencies = []

            done = False
            while not done:
                # 1. Regional agents collect local observations and summaries
                local_states = [r_agent.extract_local_state(telemetry) for r_agent in self.regional_agents]
                regional_summaries = [r_agent.compute_local_summary(telemetry) for r_agent in self.regional_agents]

                # 2. Coordinator collects global state and selects coordination action
                global_state = self.coordinator.extract_global_state(regional_summaries, telemetry)
                coord_action = self.coordinator.select_action(global_state, evaluate=False)

                # 3. Regional agents select local routing actions
                regional_actions = [r_agent.select_action(local_states[i], evaluate=False) for i, r_agent in enumerate(self.regional_agents)]

                # Combined action execution on SDN Environment
                env_action = coord_action % self.env.action_dim
                next_obs, global_reward, terminated, truncated, info = self.env.step(env_action)
                done = terminated or truncated
                telemetry = info.get("telemetry", {})

                # 4. Next states
                next_local_states = [r_agent.extract_local_state(telemetry) for r_agent in self.regional_agents]
                next_regional_summaries = [r_agent.compute_local_summary(telemetry) for r_agent in self.regional_agents]
                next_global_state = self.coordinator.extract_global_state(next_regional_summaries, telemetry)

                # 5. Distributed Reward Signals: Local reward + Coordinator feedback
                self.coordinator.agent.memory.push(global_state, coord_action, global_reward, next_global_state, done)
                coord_loss = self.coordinator.update()
                if coord_loss > 0:
                    episode_losses.append(coord_loss)

                for i, r_agent in enumerate(self.regional_agents):
                    # Local reward calculation based on regional drop and latency
                    sum_i = regional_summaries[i]
                    local_reward = - (sum_i.get("avg_drop_rate", 0.0) * 0.1) - (sum_i.get("avg_latency", 0.0) * 0.01)
                    combined_reward = (1.0 - self.global_reward_weight) * local_reward + self.global_reward_weight * global_reward

                    r_agent.agent.memory.push(local_states[i], regional_actions[i], combined_reward, next_local_states[i], done)
                    r_loss = r_agent.update()
                    if r_loss > 0:
                        episode_losses.append(r_loss)

                total_reward += global_reward
                episode_throughput.append(info["total_throughput_mbps"])
                episode_drops.append(info["avg_drop_pct"])
                episode_latencies.append(info["avg_latency_ms"])

            avg_loss = float(sum(episode_losses) / len(episode_losses)) if episode_losses else 0.0
            avg_tp = float(sum(episode_throughput) / len(episode_throughput)) if episode_throughput else 0.0
            avg_drop = float(sum(episode_drops) / len(episode_drops)) if episode_drops else 0.0
            avg_lat = float(sum(episode_latencies) / len(episode_latencies)) if episode_latencies else 0.0

            stats = {
                "episode": ep,
                "total_reward": total_reward,
                "avg_loss": avg_loss,
                "avg_throughput_mbps": avg_tp,
                "avg_drop_pct": avg_drop,
                "avg_latency_ms": avg_lat,
                "epsilon": self.coordinator.agent.epsilon
            }
            history.append(stats)

            if verbose:
                logger.info(f"[Hierarchical RL] Ep {ep}/{num_episodes} | Reward: {total_reward:.2f} | TP: {avg_tp:.1f} Mbps | Drop: {avg_drop:.2f}% | Lat: {avg_lat:.2f} ms")

        return history
