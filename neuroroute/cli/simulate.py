from __future__ import annotations

import asyncio
import logging
import os
import random
import sys
import time
from typing import Any, Dict, List, Optional

import numpy as np

import click
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table
from rich.traceback import install as install_rich_traceback

from neuroroute.ai.agent import DQNAgent, QLearningAgent
from neuroroute.ai.env import NetworkRoutingEnv
from neuroroute.network.algorithms import Dijkstras, Random, RoundRobin
from neuroroute.network.topology import TopologyManager
from neuroroute.router.plane import Packet, SimRouterNode

__version__ = "0.1.0"
DEFAULT_TOPOLOGY_PATH = "configs/square-topology.json"

console = Console()

_VERBOSITY_LEVELS = {
    0: logging.WARNING,
    1: logging.INFO,
    2: logging.DEBUG,
}


def configure_logging(verbosity: int) -> logging.Logger:
    level = _VERBOSITY_LEVELS.get(verbosity, logging.DEBUG)

    install_rich_traceback(console=console, show_locals=verbosity >= 2)

    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[
            RichHandler(
                console=console,
                rich_tracebacks=True,
                show_time=True,
                show_path=verbosity >= 2,
                markup=True,
            )
        ],
        force=True,
    )

    logger = logging.getLogger("neuroroute")
    logger.setLevel(level)
    return logger


def print_banner(topology: str, steps: int, strategy: str) -> None:
    body = (
        f"[bold cyan]Topology[/bold cyan]: {topology}\n"
        f"[bold cyan]Steps[/bold cyan]:    {steps}\n"
        f"[bold cyan]Strategy[/bold cyan]: {strategy}"
    )
    console.print(
        Panel(
            body,
            title="[bold green]NeuroRoute Simulator[/bold green]",
            subtitle=f"v{__version__}",
            expand=False,
        )
    )


def _get_congestion_bucket(env: NetworkRoutingEnv, node_idx: int) -> int:
    """Discretize queue depth at a node into 3 congestion levels (0=low, 1=med, 2=high)."""
    ratio = env.queue_depths[node_idx] / float(env.max_queue_capacity)
    if ratio < 0.33:
        return 0
    elif ratio < 0.66:
        return 1
    else:
        return 2


class QLearningStrategy:
    """Wrapper strategy around QLearningAgent and NetworkRoutingEnv."""

    def __init__(self, topo: TopologyManager, continuous_learning: bool = False) -> None:
        self.topo = topo
        self.nodes = sorted(topo.get_all_nodes())
        num_nodes = len(self.nodes)
        self.node_to_idx = {node: i for i, node in enumerate(self.nodes)}
        self.continuous_learning = continuous_learning
        self.agent = QLearningAgent(
            num_states=num_nodes,
            num_actions=num_nodes,
            learning_rate=0.3,
            epsilon=0.15 if continuous_learning else 0.1,
        )
        if os.path.exists("q_table.json"):
            try:
                self.agent.load_q_table("q_table.json")
                if not continuous_learning:
                    self.agent.epsilon = 0.0
                else:
                    self.agent.epsilon = 0.1
                    self.agent.min_epsilon = 0.05
                    self.agent.epsilon_decay = 0.9999
            except Exception:
                pass

        self.env = NetworkRoutingEnv(
            num_nodes=num_nodes,
            topology_graph=topo,
        )

        # Pre-compute per-source-node action masks (topology is static)
        self._mask_cache: Dict[int, np.ndarray] = {}
        for node_name in self.nodes:
            idx = self.node_to_idx[node_name]
            self._mask_cache[idx] = self.env.get_action_mask(idx)

        # Router nodes reference for live queue sync
        self.router_nodes: Optional[Dict[str, Any]] = None
        # Pending transition state for continuous learning feedback
        self._pending: Dict[str, Any] = {}

    def set_router_nodes(self, router_nodes: Dict[str, Any]) -> None:
        self.router_nodes = router_nodes

    def get_next_hop(self, current: str, destination: str,
                     prev_hop: Optional[str] = None) -> Optional[str]:
        if current == destination:
            return current

        if current not in self.node_to_idx or destination not in self.node_to_idx:
            return None

        curr_idx = self.node_to_idx[current]
        dest_idx = self.node_to_idx[destination]

        # Sync live queue depths from data plane
        if self.router_nodes:
            for name, idx in self.node_to_idx.items():
                if name in self.router_nodes and hasattr(self.router_nodes[name], "queue_length"):
                    self.env.queue_depths[idx] = float(self.router_nodes[name].queue_length)

        congestion = _get_congestion_bucket(self.env, curr_idx)
        state = (curr_idx, dest_idx, congestion)
        action_mask = self._mask_cache[curr_idx].copy()

        # Anti-bounce: mask out the node we just came from
        if prev_hop and prev_hop in self.node_to_idx:
            prev_idx = self.node_to_idx[prev_hop]
            # Only mask if there are other valid actions
            if action_mask.sum() > 1:
                action_mask[prev_idx] = False

        action_idx = self.agent.choose_action(state, valid_actions=action_mask)

        if 0 <= action_idx < len(self.nodes) and action_mask[action_idx]:
            chosen_hop = self.nodes[action_idx]
            if self.continuous_learning:
                self._pending[f"{current}:{destination}"] = {
                    "state": state,
                    "action": action_idx,
                    "dest_idx": dest_idx,
                    "current": current,
                }
            return chosen_hop

        # Fallback: never random — pick the valid action with best Q-value
        valid_indices = np.where(action_mask)[0]
        if len(valid_indices) > 0:
            q_values = self.agent.get_q_values(state)
            best_idx = valid_indices[np.argmax(q_values[valid_indices])]
            return self.nodes[best_idx]
        return None

    def record_outcome(self, current: str, destination: str, next_hop: str,
                       delivered: bool, dropped: bool) -> None:
        """Record the outcome of a forwarding decision and update Q-table.

        Uses dynamic Pareto-optimal rewards based on actual link metrics.
        """
        if not self.continuous_learning:
            return

        key = f"{current}:{destination}"
        pending = self._pending.pop(key, None)
        if pending is None:
            return

        state = pending["state"]
        action = pending["action"]
        dest_idx = pending["dest_idx"]

        if delivered:
            reward = 50.0
        elif dropped:
            reward = -50.0
        else:
            # Dynamic intermediate reward based on link quality
            reward = self._compute_hop_reward(current, next_hop)

        next_idx = self.node_to_idx.get(next_hop, action)
        next_congestion = _get_congestion_bucket(self.env, next_idx)
        next_state = (next_idx, dest_idx, next_congestion)
        done = delivered or dropped
        self.agent.update(state, action, reward, next_state, done)

    def _compute_hop_reward(self, current: str, next_hop: str) -> float:
        """Compute Pareto-optimal intermediate hop reward from live topology metrics."""
        try:
            metrics = self.topo.get_link_metrics(current, next_hop)
            latency = float(metrics.get("latency", 10.0))
            bandwidth = float(metrics.get("bandwidth", 1000.0))
        except (KeyError, AttributeError):
            latency = 10.0
            bandwidth = 1000.0

        # Get next hop queue congestion
        queue_ratio = 0.0
        if next_hop in self.node_to_idx and self.router_nodes and next_hop in self.router_nodes:
            node = self.router_nodes[next_hop]
            if hasattr(node, "queue_length") and hasattr(node, "buffer_size"):
                queue_ratio = node.queue_length / max(node.buffer_size, 1)

        # Normalize
        norm_latency = min(latency / 20.0, 1.0)  # 20ms as reference max
        norm_bandwidth = min(bandwidth / 2000.0, 1.0)  # 2000 as reference max

        # Pareto reward: good hops → small penalty, bad hops → large penalty
        return -(1.0 + 3.0 * norm_latency + 3.0 * queue_ratio - 2.0 * norm_bandwidth)

    def save_learned_weights(self) -> None:
        """Save updated Q-table after continuous learning session."""
        if self.continuous_learning:
            self.agent.save_q_table("q_table.json")


class DQNStrategy:
    """Wrapper strategy around DQNAgent and NetworkRoutingEnv."""

    def __init__(self, topo: TopologyManager, continuous_learning: bool = False) -> None:
        self.topo = topo
        self.nodes = sorted(topo.get_all_nodes())
        num_nodes = len(self.nodes)
        self.node_to_idx = {node: i for i, node in enumerate(self.nodes)}
        self.continuous_learning = continuous_learning

        obs_dim = 3 * num_nodes + 1  # queues + dest + latencies + bandwidths
        self.agent = DQNAgent(
            state_dim=obs_dim,
            action_dim=num_nodes,
        )
        if os.path.exists("dqn_model.pt"):
            try:
                self.agent.load_model("dqn_model.pt")
                if not continuous_learning:
                    self.agent.epsilon = 0.0
                else:
                    self.agent.epsilon = 0.1
                    self.agent.min_epsilon = 0.05
                    self.agent.epsilon_decay = 0.9999
            except Exception:
                pass
        # Optimize model for zero-overhead inference in the data plane
        self.agent.optimize_for_inference()
        self.fast_net = self.agent.export_to_numpy_fastpath()
        self.env = NetworkRoutingEnv(
            num_nodes=num_nodes,
            topology_graph=topo,
        )
        self.router_nodes = None
        self._pending: Dict[str, Any] = {}
        self._update_counter: int = 0
        self._target_update_freq: int = 50

    def set_router_nodes(self, router_nodes: Dict[str, Any]) -> None:
        self.router_nodes = [router_nodes[n] for n in self.nodes]

    def get_next_hop(self, current: str, destination: str,
                     prev_hop: Optional[str] = None) -> Optional[str]:
        if current == destination:
            return current

        if current not in self.node_to_idx or destination not in self.node_to_idx:
            return None

        curr_idx = self.node_to_idx[current]
        dest_idx = self.node_to_idx[destination]

        obs, _ = self.env.reset(options={
            "current_node": curr_idx,
            "destination_node": dest_idx,
            "router_nodes": self.router_nodes
        })

        action_mask = self.env.get_action_mask(curr_idx)

        # Anti-bounce: mask out the node we just came from
        if prev_hop and prev_hop in self.node_to_idx:
            prev_idx = self.node_to_idx[prev_hop]
            if action_mask.sum() > 1:
                action_mask[prev_idx] = False

        # Fast-path NumPy Inference (non-blocking)
        q_vals = self.fast_net.predict_q_values(obs)
        valid_indices = np.where(action_mask)[0]
        
        if len(valid_indices) > 0:
            if np.random.random() < self.agent.epsilon:
                action_idx = int(np.random.choice(valid_indices))
            else:
                valid_q = q_vals[valid_indices]
                best_actions = valid_indices[np.isclose(valid_q, np.max(valid_q))]
                action_idx = int(np.random.choice(best_actions))
        else:
            action_idx = int(np.argmax(q_vals))

        if 0 <= action_idx < len(self.nodes) and (len(valid_indices) == 0 or action_mask[action_idx]):
            chosen_hop = self.nodes[action_idx]
            if self.continuous_learning:
                self._pending[f"{current}:{destination}"] = {
                    "obs": obs,
                    "action": action_idx,
                    "dest_idx": dest_idx,
                    "current": current,
                }
            return chosen_hop

        return None

    def record_outcome(self, current: str, destination: str, next_hop: str,
                       delivered: bool, dropped: bool) -> None:
        """Record the outcome and update DQN with dynamic Pareto rewards."""
        if not self.continuous_learning:
            return

        key = f"{current}:{destination}"
        pending = self._pending.pop(key, None)
        if pending is None:
            return

        obs = pending["obs"]
        action = pending["action"]
        dest_idx = pending["dest_idx"]

        if delivered:
            reward = 50.0
        elif dropped:
            reward = -50.0
        else:
            # Dynamic intermediate reward based on link quality
            reward = self._compute_hop_reward(pending["current"], next_hop)

        # Build next observation
        next_hop_idx = self.node_to_idx.get(next_hop, action)
        next_obs, _ = self.env.reset(options={
            "current_node": next_hop_idx,
            "destination_node": dest_idx,
            "router_nodes": self.router_nodes,
        })
        done = delivered or dropped

        self.agent.replay_buffer.push(obs, action, reward, next_obs, done)
        self.agent.update()

        self._update_counter += 1
        if self._update_counter % self._target_update_freq == 0:
            self.agent.update_target_network()
            # Update fastpath weights during continuous learning
            self.fast_net = self.agent.export_to_numpy_fastpath()

    def _compute_hop_reward(self, current: str, next_hop: str) -> float:
        """Compute Pareto-optimal intermediate hop reward from live topology metrics."""
        try:
            metrics = self.topo.get_link_metrics(current, next_hop)
            latency = float(metrics.get("latency", 10.0))
            bandwidth = float(metrics.get("bandwidth", 1000.0))
        except (KeyError, AttributeError):
            latency = 10.0
            bandwidth = 1000.0

        # Get next hop queue congestion from live router nodes
        queue_ratio = 0.0
        next_idx = self.node_to_idx.get(next_hop)
        if next_idx is not None and self.router_nodes:
            node = self.router_nodes[next_idx]
            if hasattr(node, "queue_length") and hasattr(node, "buffer_size"):
                queue_ratio = node.queue_length / max(node.buffer_size, 1)

        # Normalize
        norm_latency = min(latency / 20.0, 1.0)
        norm_bandwidth = min(bandwidth / 2000.0, 1.0)

        # Pareto reward: good hops → small penalty, bad hops → large penalty
        return -(1.0 + 3.0 * norm_latency + 3.0 * queue_ratio - 2.0 * norm_bandwidth)

    def save_learned_weights(self) -> None:
        """Save updated DQN model after continuous learning session."""
        if self.continuous_learning:
            self.agent.save_model("dqn_model.pt")



def get_strategy(strategy_name: str, topo: TopologyManager, continuous_learning: bool = False) -> Any:
    name = strategy_name.lower().replace("-", "").replace("_", "")
    if name in ("static", "dijkstra", "dijkstras"):
        return Dijkstras(topo)
    elif name in ("roundrobin", "rr"):
        return RoundRobin(topo)
    elif name == "random":
        return Random(topo)
    elif name in ("qlearning", "qlearningagent", "ql"):
        return QLearningStrategy(topo, continuous_learning=continuous_learning)
    elif name == "dqn":
        return DQNStrategy(topo, continuous_learning=continuous_learning)
    else:
        raise ValueError(f"Unknown routing strategy: {strategy_name}")


async def _generate_packets(
    nodes: List[str],
    router_nodes: Dict[str, SimRouterNode],
    total_steps: int,
    stats: Dict[str, Any],
    logger: logging.Logger,
    stop_event: asyncio.Event,
) -> None:
    if len(nodes) < 2:
        logger.warning("Network has fewer than 2 nodes. Packet generation skipped.")
        return

    for step in range(total_steps):
        if stop_event.is_set():
            break

        src, dst = random.sample(nodes, 2)
        packet = Packet(
            packet_id="",
            source=src,
            destination=dst,
            payload=f"sim-packet-{step}".encode("utf-8"),
        )

        logger.debug(
            f"Step {step+1}/{total_steps}: Generating packet {packet.packet_id[:8]} ({src} -> {dst})"
        )

        enqueued = await router_nodes[src].enqueue(packet)
        if enqueued:
            stats["packets_generated"] += 1
        else:
            stats["packets_dropped"] += 1
            logger.debug(f"Buffer full at source node {src}. Packet dropped.")

        await asyncio.sleep(0.01)


async def run_simulation(
    topology_path: str,
    steps: int,
    strategy_name: str,
    logger: logging.Logger,
    use_tui: bool = False,
    enable_chaos: bool = False,
    continuous_learning: bool = False,
) -> Dict[str, Any]:
    logger.info("Loading topology from '%s'...", topology_path)
    topo = TopologyManager()
    topo.load_topology(topology_path)

    nodes = topo.get_all_nodes()
    if not nodes:
        raise ValueError(f"No nodes found in topology file {topology_path}")

    logger.info("Initializing %d router nodes...", len(nodes))
    strategy = get_strategy(strategy_name, topo, continuous_learning=continuous_learning)
    if continuous_learning:
        logger.info("Continuous Learning ENABLED: Model weights will update during simulation.")

    router_nodes: Dict[str, SimRouterNode] = {
        node_id: SimRouterNode(node_id, topo, strategy) for node_id in nodes
    }

    for node in router_nodes.values():
        node.set_peers(router_nodes)

    # Push initial precomputed routes into all router node caches
    topo.recompute_routes()

    # Give adaptive strategies access to live router node state
    if hasattr(strategy, "set_router_nodes"):
        strategy.set_router_nodes(router_nodes)

    stats: Dict[str, Any] = {
        "packets_generated": 0,
        "packets_delivered": 0,
        "packets_dropped": 0,
        "total_latency": 0.0,
        "start_time": time.time(),
        "end_time": 0.0,
    }

    stop_event = asyncio.Event()

    node_tasks = [
        asyncio.create_task(node.run(stop_event, stats))
        for node in router_nodes.values()
    ]

    chaos_scheduler = None
    chaos_task = None
    if enable_chaos:
        from neuroroute.network.chaos import ChaosScheduler
        logger.info("Chaos Engineering ENABLED: Injecting random link failures and latency spikes.")
        chaos_scheduler = ChaosScheduler(topo)
        chaos_task = asyncio.create_task(chaos_scheduler.start(interval_seconds=0.069, duration_seconds=steps * 0.05))

    tui_task = None
    if use_tui:
        from neuroroute.cli.tui import TUIState, run_live_tui
        links = []
        for src, neighbors in topo.graph.items():
            for dst, metrics in neighbors.items():
                links.append(
                    (src, dst, float(metrics.get("latency", 0.0)), float(metrics.get("bandwidth", 0.0)))
                )

        tui_state = TUIState(
            nodes=nodes,
            links=links,
            current_strategy=strategy_name,
            router_nodes=router_nodes,
            topology_manager=topo,
            stats_ref=stats,
        )
        tui_task = asyncio.create_task(run_live_tui(tui_state, stop_event, refresh_rate=0.1))

    logger.info(
        "Starting packet generation loop (%d steps, strategy: %s)...",
        steps,
        strategy_name,
    )
    await _generate_packets(nodes, router_nodes, steps, stats, logger, stop_event)

    logger.info("Packet generation complete. Draining remaining network queues...")
    await asyncio.sleep(0.5)

    stop_event.set()
    if chaos_scheduler:
        chaos_scheduler.stop()
    await asyncio.gather(*node_tasks, return_exceptions=True)
    if chaos_task:
        await asyncio.gather(chaos_task, return_exceptions=True)

    # Save updated weights if continuous learning was active
    if continuous_learning and hasattr(strategy, "save_learned_weights"):
        strategy.save_learned_weights()
        logger.info("Continuous learning weights saved.")
    if tui_task:
        await asyncio.gather(tui_task, return_exceptions=True)

    stats["end_time"] = time.time()
    return stats


def display_summary(stats: Dict[str, Any], strategy: str, topology: str) -> None:
    runtime = max(0.001, stats["end_time"] - stats["start_time"])
    delivered = stats["packets_delivered"]
    avg_latency = (
        (stats["total_latency"] / delivered) * 1000.0 if delivered > 0 else 0.0
    )

    table = Table(
        title="Simulation Statistics Summary",
        title_style="bold green",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Value", style="bold white", justify="right")

    table.add_row("Strategy", strategy)
    table.add_row("Topology", topology)
    table.add_row("Packets Generated", str(stats["packets_generated"]))
    table.add_row("Packets Delivered", str(stats["packets_delivered"]))
    table.add_row("Packets Dropped", str(stats["packets_dropped"]))
    table.add_row("Average Latency", f"{avg_latency:.2f} ms")
    table.add_row("Total Runtime", f"{runtime:.2f} s")

    console.print()
    console.print(table)


@click.command(name="simulate")
@click.option(
    "-t",
    "--topology",
    "topology",
    type=click.Path(exists=True),
    default=DEFAULT_TOPOLOGY_PATH,
    show_default=True,
    help="Path to the network topology JSON/YAML file.",
)
@click.option(
    "-s",
    "--steps",
    "steps",
    type=int,
    default=100,
    show_default=True,
    help="Total number of simulation steps to run.",
)
@click.option(
    "-r",
    "--strategy",
    "strategy",
    type=click.Choice(["static", "round-robin", "random", "qlearning", "dqn"], case_sensitive=False),
    default="static",
    show_default=True,
    help="Routing strategy/algorithm to execute.",
)
@click.option(
    "--tui",
    "use_tui",
    is_flag=True,
    default=False,
    help="Run live TUI dashboard during simulation.",
)
@click.option(
    "--chaos",
    "enable_chaos",
    is_flag=True,
    default=False,
    help="Inject random link failures and latency spikes during simulation.",
)
@click.option(
    "--learn",
    "continuous_learning",
    is_flag=True,
    default=False,
    help="Enable continuous learning: model updates weights during simulation (qlearning/dqn only).",
)
@click.option(
    "-v",
    "--verbose",
    "verbose",
    count=True,
    help="Increase logging verbosity. Use -v for INFO, -vv for DEBUG.",
)
@click.version_option(version=__version__, prog_name="neuroroute-simulate")
def main(topology: str, steps: int, strategy: str, use_tui: bool, enable_chaos: bool, continuous_learning: bool, verbose: int) -> None:
    logger = configure_logging(verbose)
    if not use_tui:
        print_banner(topology, steps, strategy)

    stats: Optional[Dict[str, Any]] = None
    start_time = time.time()

    try:
        stats = asyncio.run(run_simulation(topology, steps, strategy, logger, use_tui=use_tui, enable_chaos=enable_chaos, continuous_learning=continuous_learning))
    except KeyboardInterrupt:
        logger.warning("\n[bold yellow]Simulation interrupted by user (Ctrl+C). Cleaning up...[/bold yellow]")
        if stats is None:
            stats = {
                "packets_generated": 0,
                "packets_delivered": 0,
                "packets_dropped": 0,
                "total_latency": 0.0,
                "start_time": start_time,
                "end_time": time.time(),
            }
        else:
            stats["end_time"] = time.time()
    except Exception as e:
        logger.exception("Simulation failed unexpectedly: %s", e)
        sys.exit(1)

    display_summary(stats, strategy, topology)
    logger.info("Done.")


if __name__ == "__main__":
    main()