import os
import yaml
import tempfile
import logging
from typing import Dict, List, Tuple, Any, Optional, Callable
import numpy as np

from rl_sdn_controller.control_plane.controller import SDNController
from rl_sdn_controller.ai.env import SDNEnv
from rl_sdn_controller.ai.agent import DQNAgent

logger = logging.getLogger(__name__)


class CurriculumStage:
    """
    Defines a curriculum learning training stage with specific difficulty settings.
    """
    def __init__(
        self,
        stage_id: int,
        name: str,
        episodes: int,
        traffic_rate_pps: int,
        packet_size_bytes: int = 1000,
        link_failure_prob: float = 0.0,
        chaos_enabled: bool = False,
        topology_yaml_content: Optional[str] = None
    ):
        self.stage_id = stage_id
        self.name = name
        self.episodes = episodes
        self.traffic_rate_pps = traffic_rate_pps
        self.packet_size_bytes = packet_size_bytes
        self.link_failure_prob = link_failure_prob
        self.chaos_enabled = chaos_enabled
        self.topology_yaml_content = topology_yaml_content


class CurriculumScheduler:
    """
    Curriculum Learning Scheduler for Progressive Difficulty RL Training.
    Transits across stages:
      - Stage 1 (Easy): 4-node network, light traffic (50 pps), no link failures.
      - Stage 2 (Moderate): 6-node diamond network, medium traffic (200 pps), 2% link failures.
      - Stage 3 (Complex / Enterprise): Multi-path mesh, heavy traffic (500+ pps), 5% link failures.
    """
    def __init__(self, stages: Optional[List[CurriculumStage]] = None):
        if stages is None:
            self.stages = self._default_stages()
        else:
            self.stages = stages
        self.current_stage_idx = 0
        self.stage_transition_log: List[Dict[str, Any]] = []

    def _default_stages(self) -> List[CurriculumStage]:
        # Stage 1: Simple 4-node linear network
        topo_stage_1 = """
nodes:
  h1: {type: host, ip: "10.0.0.1"}
  h2: {type: host, ip: "10.0.0.2"}
  r1: {type: router}
  r2: {type: router}
links:
  - {src: h1, dst: r1, capacity: 1000, latency: 5000, max_queue_packets: 100}
  - {src: r1, dst: h1, capacity: 1000, latency: 5000, max_queue_packets: 100}
  - {src: r1, dst: r2, capacity: 50, latency: 5000, max_queue_packets: 100}
  - {src: r2, dst: r1, capacity: 50, latency: 5000, max_queue_packets: 100}
  - {src: r2, dst: h2, capacity: 1000, latency: 5000, max_queue_packets: 100}
  - {src: h2, dst: r2, capacity: 1000, latency: 5000, max_queue_packets: 100}
"""
        # Stage 2: 6-node dual-path topology
        topo_stage_2 = """
nodes:
  h1: {type: host, ip: "10.0.0.1"}
  h2: {type: host, ip: "10.0.0.2"}
  r1: {type: router}
  r2: {type: router}
  r3: {type: router}
  r4: {type: router}
links:
  - {src: h1, dst: r1, capacity: 1000, latency: 5000, max_queue_packets: 100}
  - {src: r1, dst: h1, capacity: 1000, latency: 5000, max_queue_packets: 100}
  - {src: r1, dst: r2, capacity: 25, latency: 5000, max_queue_packets: 100}
  - {src: r2, dst: r4, capacity: 25, latency: 5000, max_queue_packets: 100}
  - {src: r1, dst: r3, capacity: 20, latency: 5100, max_queue_packets: 100}
  - {src: r3, dst: r4, capacity: 20, latency: 5100, max_queue_packets: 100}
  - {src: r4, dst: h2, capacity: 1000, latency: 5000, max_queue_packets: 100}
  - {src: h2, dst: r4, capacity: 1000, latency: 5000, max_queue_packets: 100}
"""
        return [
            CurriculumStage(
                stage_id=1,
                name="Stage 1: Basic Routing (4-node, 50 pps, no failures)",
                episodes=5,
                traffic_rate_pps=50,
                link_failure_prob=0.0,
                chaos_enabled=False,
                topology_yaml_content=topo_stage_1
            ),
            CurriculumStage(
                stage_id=2,
                name="Stage 2: Load Balancing (6-node, 200 pps, 2% link failure)",
                episodes=5,
                traffic_rate_pps=200,
                link_failure_prob=0.02,
                chaos_enabled=True,
                topology_yaml_content=topo_stage_2
            ),
            CurriculumStage(
                stage_id=3,
                name="Stage 3: Enterprise Complex (Full topology, 500 pps, 5% link failure)",
                episodes=10,
                traffic_rate_pps=500,
                link_failure_prob=0.05,
                chaos_enabled=True,
                topology_yaml_content=None # Uses base configs/topology.yaml
            )
        ]

    def get_stage_for_episode(self, episode_num: int) -> Tuple[CurriculumStage, bool]:
        """
        Returns the appropriate CurriculumStage for the given episode number,
        along with a boolean flag indicating if a stage transition just occurred.
        """
        accumulated_episodes = 0
        for idx, stage in enumerate(self.stages):
            accumulated_episodes += stage.episodes
            if episode_num <= accumulated_episodes:
                is_transition = (idx != self.current_stage_idx)
                if is_transition:
                    self.current_stage_idx = idx
                    log_entry = {
                        "from_stage": self.current_stage_idx,
                        "to_stage": idx + 1,
                        "episode": episode_num,
                        "stage_name": stage.name
                    }
                    self.stage_transition_log.append(log_entry)
                    logger.info(f"🚀 [CURRICULUM TRANSITION] Episode {episode_num} transitioned to {stage.name}")
                return stage, is_transition

        # Default to final stage
        return self.stages[-1], False


class CurriculumTrainer:
    """
    Coordinates training execution across progressive curriculum stages.
    Transfers weights from previous stage model to accelerate convergence.
    """
    def __init__(self, rl_config: Dict[str, Any], base_topology_path: str = "configs/topology.yaml"):
        self.rl_config = rl_config
        self.base_topology_path = base_topology_path
        self.scheduler = CurriculumScheduler()

    def train_curriculum(self, verbose: bool = True) -> List[Dict[str, Any]]:
        """Executes the complete multi-stage curriculum learning pipeline."""
        full_history = []
        overall_ep = 0

        # We will reuse model parameters across stages for progressive transfer
        shared_weights = None

        for stage_idx, stage in enumerate(self.scheduler.stages):
            if verbose:
                logger.info(f"\n🎓 Starting {stage.name} ({stage.episodes} episodes)")

            # Prepare topology file
            if stage.topology_yaml_content:
                tmp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
                tmp_file.write(stage.topology_yaml_content)
                tmp_file.flush()
                top_path = tmp_file.name
            else:
                top_path = self.base_topology_path

            # Prepare traffic configuration for current stage
            traffic_configs = [
                {
                    "id": f"stage_{stage.stage_id}_flow1",
                    "src": "h1",
                    "dst": "h2",
                    "pattern": "poisson" if stage.stage_id == 1 else "bursty",
                    "rate_pps": stage.traffic_rate_pps,
                    "packet_size_bytes": stage.packet_size_bytes,
                    "duration_sec": 60.0
                }
            ]

            # Prepare chaos config
            chaos_config = {
                "chaos": {
                    "enabled": stage.chaos_enabled,
                    "link_failure_probability": stage.link_failure_prob,
                    "link_repair_time_sec": 2.0,
                    "jitter_std_us": 20.0,
                    "random_drop_rate": 0.01
                }
            }

            controller = SDNController(
                topology_path=top_path,
                traffic_configs=traffic_configs,
                rl_config=self.rl_config,
                chaos_config=chaos_config
            )

            # If previous stage trained weights exist with compatible shape, load them
            if shared_weights is not None:
                try:
                    controller.agent.policy_net.load_state_dict(shared_weights, strict=False)
                    controller.agent.target_net.load_state_dict(shared_weights, strict=False)
                except Exception as e:
                    logger.debug(f"Weight transfer partial adaptation for stage {stage.stage_id}: {e}")

            # Train on this stage
            stage_history = controller.train_episodes(num_episodes=stage.episodes, verbose=verbose)
            
            for ep_stat in stage_history:
                overall_ep += 1
                ep_stat["overall_episode"] = overall_ep
                ep_stat["stage_id"] = stage.stage_id
                ep_stat["stage_name"] = stage.name
                full_history.append(ep_stat)

            # Store policy weights for subsequent stage
            shared_weights = controller.agent.policy_net.state_dict()

            # Clean up temp file
            if stage.topology_yaml_content and os.path.exists(top_path):
                try:
                    os.remove(top_path)
                except Exception:
                    pass

        return full_history
