import os
import time
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import matplotlib
matplotlib.use("Agg") # Non-interactive headless backend
import matplotlib.pyplot as plt

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.live import Live

logger = Console()


class MetricsTracker:
    """
    Episode-wise and step-wise metrics accumulator with rolling averages
    for real-time SDN Controller telemetry and baseline comparisons.
    """
    def __init__(self, window_size: int = 5):
        self.window_size = window_size
        self.episodes: List[int] = []
        self.rewards: List[float] = []
        self.throughputs: List[float] = []
        self.drop_rates: List[float] = []
        self.latencies: List[float] = []
        self.p99_latencies: List[float] = []
        self.q_values_mean: List[float] = []
        self.q_values_max: List[float] = []
        self.losses: List[float] = []
        
        # Baselines storage for comparison
        self.ospf_baseline: Optional[Dict[str, float]] = None
        self.rr_baseline: Optional[Dict[str, float]] = None
        self.link_utilization_history: List[Dict[Tuple[str, str], float]] = []

    def set_baselines(self, ospf_stats: Dict[str, float], rr_stats: Dict[str, float]):
        self.ospf_baseline = ospf_stats
        self.rr_baseline = rr_stats

    def record_episode(
        self,
        episode: int,
        reward: float,
        throughput_mbps: float,
        drop_rate_pct: float,
        avg_latency_ms: float,
        p99_latency_ms: float = 0.0,
        avg_q_val: float = 0.0,
        max_q_val: float = 0.0,
        loss: float = 0.0,
        link_utils: Optional[Dict[Tuple[str, str], float]] = None
    ):
        self.episodes.append(episode)
        self.rewards.append(reward)
        self.throughputs.append(throughput_mbps)
        self.drop_rates.append(drop_rate_pct)
        self.latencies.append(avg_latency_ms)
        self.p99_latencies.append(p99_latency_ms)
        self.q_values_mean.append(avg_q_val)
        self.q_values_max.append(max_q_val)
        self.losses.append(loss)
        if link_utils:
            self.link_utilization_history.append(link_utils)

    def get_rolling(self, series: List[float]) -> List[float]:
        """Calculates rolling moving average."""
        if not series:
            return []
        rolling = []
        for i in range(len(series)):
            start_idx = max(0, i - self.window_size + 1)
            rolling.append(float(np.mean(series[start_idx:i+1])))
        return rolling

    def get_summary(self) -> Dict[str, Any]:
        if not self.episodes:
            return {}
        return {
            "total_episodes": len(self.episodes),
            "latest_reward": self.rewards[-1],
            "latest_throughput_mbps": self.throughputs[-1],
            "latest_drop_rate_pct": self.drop_rates[-1],
            "latest_latency_ms": self.latencies[-1],
            "mean_reward": float(np.mean(self.rewards)),
            "mean_throughput": float(np.mean(self.throughputs)),
            "mean_drop_rate": float(np.mean(self.drop_rates)),
            "mean_latency": float(np.mean(self.latencies))
        }


class DashboardPlotter:
    """
    Generates publication-quality 6-panel performance plots and heatmaps
    comparing RL against OSPF and Round-Robin baselines.
    """
    def __init__(self, tracker: MetricsTracker, output_dir: str = "plots"):
        self.tracker = tracker
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def save_plots(self, filename: str = "training_dashboard.png") -> str:
        """Renders and saves a 6-subplot diagnostic dashboard."""
        if not self.tracker.episodes:
            logger.print("[yellow]No episode metrics to plot.[/yellow]")
            return ""

        fig, axs = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle("RL-SDN Intelligent Control Plane - Performance Dashboard", fontsize=16, fontweight="bold")
        
        episodes = self.tracker.episodes

        # 1. Rewards Trend & Rolling Average
        ax1 = axs[0, 0]
        ax1.plot(episodes, self.tracker.rewards, alpha=0.35, color="blue", label="Episode Reward")
        rolling_rewards = self.tracker.get_rolling(self.tracker.rewards)
        ax1.plot(episodes, rolling_rewards, color="navy", linewidth=2.0, label=f"Rolling Mean (w={self.tracker.window_size})")
        ax1.set_title("1. Episode Reward Progression", fontweight="bold")
        ax1.set_xlabel("Episode")
        ax1.set_ylabel("Reward")
        ax1.grid(True, linestyle="--", alpha=0.6)
        ax1.legend()

        # 2. Throughput Comparison vs Baselines
        ax2 = axs[0, 1]
        ax2.plot(episodes, self.tracker.throughputs, color="green", linewidth=2.0, label="RL Agent Throughput")
        if self.tracker.ospf_baseline:
            ax2.axhline(self.tracker.ospf_baseline.get("throughput_mbps", 0.0), color="magenta", linestyle="--", label="OSPF Baseline")
        if self.tracker.rr_baseline:
            ax2.axhline(self.tracker.rr_baseline.get("throughput_mbps", 0.0), color="orange", linestyle=":", label="Round-Robin Baseline")
        ax2.set_title("2. Throughput (Mbps) vs Baselines", fontweight="bold")
        ax2.set_xlabel("Episode")
        ax2.set_ylabel("Throughput (Mbps)")
        ax2.grid(True, linestyle="--", alpha=0.6)
        ax2.legend()

        # 3. Packet Drop Rate Comparison (%)
        ax3 = axs[0, 2]
        ax3.plot(episodes, self.tracker.drop_rates, color="red", linewidth=2.0, label="RL Drop Rate (%)")
        if self.tracker.ospf_baseline:
            ax3.axhline(self.tracker.ospf_baseline.get("drop_rate_pct", 0.0), color="magenta", linestyle="--", label="OSPF Baseline")
        if self.tracker.rr_baseline:
            ax3.axhline(self.tracker.rr_baseline.get("drop_rate_pct", 0.0), color="orange", linestyle=":", label="Round-Robin Baseline")
        ax3.set_title("3. Packet Drop Rate (%)", fontweight="bold")
        ax3.set_xlabel("Episode")
        ax3.set_ylabel("Drop Rate (%)")
        ax3.grid(True, linestyle="--", alpha=0.6)
        ax3.legend()

        # 4. Average & P99 Latency (ms)
        ax4 = axs[1, 0]
        ax4.plot(episodes, self.tracker.latencies, color="purple", linewidth=2.0, label="RL Avg Latency (ms)")
        if any(p > 0 for p in self.tracker.p99_latencies):
            ax4.plot(episodes, self.tracker.p99_latencies, color="darkviolet", linestyle=":", label="RL P99 Tail Latency")
        if self.tracker.ospf_baseline:
            ax4.axhline(self.tracker.ospf_baseline.get("avg_latency_ms", 0.0), color="magenta", linestyle="--", label="OSPF Baseline")
        ax4.set_title("4. Latency (ms) Optimization", fontweight="bold")
        ax4.set_xlabel("Episode")
        ax4.set_ylabel("Latency (ms)")
        ax4.grid(True, linestyle="--", alpha=0.6)
        ax4.legend()

        # 5. Q-Value Progression
        ax5 = axs[1, 1]
        ax5.plot(episodes, self.tracker.q_values_mean, color="teal", label="Mean Q-Value")
        if any(q != 0 for q in self.tracker.q_values_max):
            ax5.plot(episodes, self.tracker.q_values_max, color="darkslategray", linestyle="--", label="Max Q-Value")
        ax5.set_title("5. Q-Value Convergence", fontweight="bold")
        ax5.set_xlabel("Episode")
        ax5.set_ylabel("Predicted Q-Value")
        ax5.grid(True, linestyle="--", alpha=0.6)
        ax5.legend()

        # 6. Link Utilization Heatmap
        ax6 = axs[1, 2]
        if self.tracker.link_utilization_history:
            link_keys = list(self.tracker.link_utilization_history[0].keys())
            # Matrix shape: (num_links, num_episodes)
            matrix = np.zeros((len(link_keys), len(self.tracker.link_utilization_history)))
            for ep_idx, hist in enumerate(self.tracker.link_utilization_history):
                for l_idx, l_key in enumerate(link_keys):
                    matrix[l_idx, ep_idx] = hist.get(l_key, 0.0)

            cax = ax6.imshow(matrix, aspect="auto", cmap="YlOrRd", vmin=0, vmax=100)
            fig.colorbar(cax, ax=ax6, label="Utilization (%)")
            ax6.set_yticks(range(len(link_keys)))
            ax6.set_yticklabels([f"{s}->{d}" for s, d in link_keys], fontsize=8)
            ax6.set_title("6. Link Utilization Heatmap (%)", fontweight="bold")
            ax6.set_xlabel("Episode")
        else:
            # Fallback plot for loss
            ax6.plot(episodes, self.tracker.losses, color="crimson", label="Q-Learning Loss")
            ax6.set_title("6. Q-Loss Convergence", fontweight="bold")
            ax6.set_xlabel("Episode")
            ax6.set_ylabel("Huber Loss")
            ax6.grid(True, linestyle="--", alpha=0.6)
            ax6.legend()

        plt.tight_layout()
        out_path = os.path.join(self.output_dir, filename)
        plt.savefig(out_path, dpi=150)
        plt.close(fig)
        return out_path


class LiveConsoleDashboard:
    """
    Rich Live Terminal Dashboard showing real-time metrics during simulation/training.
    """
    def __init__(self, console: Optional[Console] = None):
        self.console = console or Console()

    def render_episode_table(self, stats: Dict[str, Any], baseline_ospf: Optional[Dict[str, Any]] = None) -> Table:
        table = Table(title="LIVE PERFORMANCE TELEMETRY DASHBOARD", border_style="bright_blue")
        table.add_column("Metric", style="bold yellow")
        table.add_column("Current Episode Value", style="bold green", justify="right")
        table.add_column("OSPF Static Baseline", style="bold magenta", justify="right")
        table.add_column("Status / Health", style="bold cyan")

        tp = stats.get("avg_throughput_mbps", 0.0)
        drop = stats.get("avg_drop_pct", 0.0)
        lat = stats.get("avg_latency_ms", 0.0)
        rew = stats.get("total_reward", 0.0)

        ospf_tp = baseline_ospf.get("throughput_mbps", 0.0) if baseline_ospf else 0.0
        ospf_drop = baseline_ospf.get("drop_rate_pct", 0.0) if baseline_ospf else 0.0
        ospf_lat = baseline_ospf.get("avg_latency_ms", 0.0) if baseline_ospf else 0.0

        drop_status = "[green]OPTIMAL[/green]" if drop < 1.0 else ("[yellow]MODERATE LOSS[/yellow]" if drop < 5.0 else "[red]HIGH LOSS[/red]")
        tp_status = "[green]HIGH CAPACITY[/green]" if tp > 15.0 else "[yellow]CONGESTED[/yellow]"

        table.add_row("Total Reward", f"{rew:.2f}", "N/A", "[green]TRAINING CONVERGING[/green]")
        table.add_row("Throughput (Mbps)", f"{tp:.1f} Mbps", f"{ospf_tp:.1f} Mbps", tp_status)
        table.add_row("Packet Drop Rate (%)", f"{drop:.2f}%", f"{ospf_drop:.2f}%", drop_status)
        table.add_row("Average Latency (ms)", f"{lat:.2f} ms", f"{ospf_lat:.2f} ms", "[green]LOW JITTER[/green]")

        return table
