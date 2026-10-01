import time
import logging
from typing import Dict, Any, List, Optional, Tuple, Callable
import numpy as np

from rl_sdn_controller.ai.env import SDNEnv
from rl_sdn_controller.ai.agent import DQNAgent
from rl_sdn_controller.cli.dashboard import MetricsTracker, DashboardPlotter

logger = logging.getLogger(__name__)


class SDNController:
    """
    Main Orchestration Controller for RL-SDN System.
    Executes RL evaluation and training cycle every 10-100ms:
    1. Collect Telemetry
    2. RL Agent Forward Pass (Select Action)
    3. Update Routing Tables
    4. Compute Reward & Store Experience (PER)
    5. Perform DDQN Optimization with Soft Updates
    6. Record telemetry in MetricsTracker
    """
    def __init__(self, topology_path: str, traffic_configs: list, rl_config: Dict[str, Any], chaos_config: Dict[str, Any] = None):
        self.rl_config = rl_config
        self.env = SDNEnv(topology_path, traffic_configs, rl_config, chaos_config=chaos_config)
        self.state_dim = self.env.state_dim
        self.action_dim = self.env.action_dim
        self.agent = DQNAgent(self.state_dim, self.action_dim, rl_config.get("agent", {}))
        self.metrics_tracker = MetricsTracker(window_size=rl_config.get("dashboard", {}).get("window_size", 5))

    def train_episodes(
        self,
        num_episodes: int = 10,
        verbose: bool = True,
        progress_callback: Optional[Callable[[int, int, Dict[str, Any], float], None]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Trains RL Agent over multiple simulation episodes.
        Returns metrics history for each episode.
        """
        history = []

        for ep in range(1, num_episodes + 1):
            state, _ = self.env.reset()
            total_reward = 0.0
            episode_losses = []
            episode_throughput = []
            episode_drops = []
            episode_latencies = []
            q_vals_all = []

            done = False
            last_telemetry = {}
            while not done:
                # 1. Action selection
                action = self.agent.select_action(state, evaluate=False)
                if hasattr(self.agent, "get_q_values"):
                    q_vals = self.agent.get_q_values(state)
                    q_vals_all.append(q_vals)

                # 2. Environment step (updates Layer 2 rules, advances simulation, collects telemetry)
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                last_telemetry = info.get("telemetry", {})

                # 3. Store experience tuple in replay memory
                self.agent.memory.push(state, action, reward, next_state, done)

                # 4. Perform Q-learning optimization step
                loss = self.agent.update()
                if loss > 0:
                    episode_losses.append(loss)

                state = next_state
                total_reward += reward
                if progress_callback is not None:
                    progress_callback(ep, self.env.current_step, info, reward)

                episode_throughput.append(info["total_throughput_mbps"])
                episode_drops.append(info["avg_drop_pct"])
                episode_latencies.append(info["avg_latency_ms"])

            avg_loss = float(sum(episode_losses) / len(episode_losses)) if episode_losses else 0.0
            avg_tp = float(sum(episode_throughput) / len(episode_throughput)) if episode_throughput else 0.0
            avg_drop = float(sum(episode_drops) / len(episode_drops)) if episode_drops else 0.0
            valid_lat = [l for l in episode_latencies if not np.isnan(l) and l > 0]
            avg_lat = float(np.mean(valid_lat)) if valid_lat else 0.0
            p99_lat = float(np.percentile(valid_lat, 99)) if valid_lat else 0.0

            avg_q = float(np.mean([np.mean(q) for q in q_vals_all])) if q_vals_all else 0.0
            max_q = float(np.mean([np.max(q) for q in q_vals_all])) if q_vals_all else 0.0

            link_utils = {k: st.utilization_pct for k, st in last_telemetry.items()} if last_telemetry else None

            # Record in metrics tracker
            self.metrics_tracker.record_episode(
                episode=ep,
                reward=total_reward,
                throughput_mbps=avg_tp,
                drop_rate_pct=avg_drop,
                avg_latency_ms=avg_lat,
                p99_latency_ms=p99_lat,
                avg_q_val=avg_q,
                max_q_val=max_q,
                loss=avg_loss,
                link_utils=link_utils
            )

            ep_stats = {
                "episode": ep,
                "total_reward": total_reward,
                "avg_loss": avg_loss,
                "avg_throughput_mbps": avg_tp,
                "avg_drop_pct": avg_drop,
                "avg_latency_ms": avg_lat,
                "p99_latency_ms": p99_lat,
                "epsilon": self.agent.epsilon
            }
            history.append(ep_stats)

            # Decay epsilon once per episode (not per gradient step)
            self.agent.decay_epsilon()

            if verbose and (ep % 1 == 0 or ep == num_episodes):
                logger.info(
                    f"Episode {ep}/{num_episodes} | Reward: {total_reward:.2f} | "
                    f"Throughput: {avg_tp:.1f} Mbps | Drop: {avg_drop:.2f}% | Latency: {avg_lat:.2f} ms | Epsilon: {self.agent.epsilon:.3f}"
                )

        return history

    def evaluate(
        self,
        max_steps: int = 600,
        progress_callback: Optional[Callable[[int, int, Dict[str, Any], float], None]] = None,
    ) -> Tuple[Dict[str, float], Dict[Any, Any]]:
        """Evaluates current policy in deterministic mode."""
        self.env.max_steps = max_steps
        state, _ = self.env.reset()
        tp_list, drop_list, lat_list = [], [], []
        telemetry_history = []
        done = False

        while not done:
            action = self.agent.select_action(state, evaluate=True)
            state, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            if progress_callback is not None:
                progress_callback(1, self.env.current_step, info, reward)
            tp_list.append(info["total_throughput_mbps"])
            drop_list.append(info["avg_drop_pct"])
            lat_list.append(info["avg_latency_ms"])
            if "telemetry" in info:
                telemetry_history.append(info["telemetry"])

        valid_lat = [l for l in lat_list if not np.isnan(l) and l > 0]
        metrics = {
            "throughput_mbps": float(np.nanmean(tp_list)) if tp_list else 0.0,
            "drop_rate_pct": float(np.nanmean(drop_list)) if drop_list else 0.0,
            "avg_latency_ms": float(np.mean(valid_lat)) if valid_lat else 0.0,
            "p99_latency_ms": float(np.percentile(valid_lat, 99)) if valid_lat else 0.0
        }
        return metrics, telemetry_history[-1] if telemetry_history else {}

    def save_dashboard_plots(self, output_dir: str = "plots", filename: str = "training_dashboard.png") -> str:
        """Exports diagnostic graphs using DashboardPlotter."""
        plotter = DashboardPlotter(self.metrics_tracker, output_dir=output_dir)
        return plotter.save_plots(filename=filename)

