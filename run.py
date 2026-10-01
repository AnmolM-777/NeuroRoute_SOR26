#!/usr/bin/env python3
"""
RL-SDN Interactive Terminal User Interface (TUI).
Production-grade dashboard and execution environment supporting all 10 architectural enhancements:
  1. Hierarchical Multi-Agent Coordination
  2. Prioritized Experience Replay (PER)
  3. Dueling DQN Architecture
  4. Link Failure Detection & Recovery (MTTR)
  5. Multi-Traffic Types (VoIP, IoT, Bursty, Bulk, Poisson)
  6. Progressive Difficulty Curriculum Learning
  7. Double DQN with Soft Updates
  8. Real-Time Performance Dashboard & Plots
  9. Transfer Learning & Topology Adaptation
 10. Policy Distillation for Edge Switches
"""
import sys
import os
import tempfile
import copy
import yaml
import logging
import pytest
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.table import Table

from rl_sdn_controller.control_plane.controller import SDNController
from rl_sdn_controller.ai.hierarchical_agent import HierarchicalSDNController
from rl_sdn_controller.ai.curriculum import CurriculumTrainer
from rl_sdn_controller.ai.transfer import TransferLearningManager
from rl_sdn_controller.ai.distillation import PolicyDistiller
from rl_sdn_controller.ai.env import SDNEnv
from rl_sdn_controller.ai.tabular_q_learning import TabularQLearningAgent
from rl_sdn_controller.network.topology import NetworkTopology
from rl_sdn_controller.network.topology_factory import build_router_topology, MIN_ROUTERS, MAX_ROUTERS
from rl_sdn_controller.sdn_api.routing_table_api import RoutingTableAPI
from rl_sdn_controller.data_plane.simulator import DataPlaneSimulator
from rl_sdn_controller.network.routing_engine import RLRoutingEngine, OSPFRoutingEngine, RoundRobinRoutingEngine
from rl_sdn_controller.network.state_manager import StateManager
from rl_sdn_controller.cli.live_tui import LiveTelemetryDashboard
from rl_sdn_controller.sdn_api.stats_provider import LinkStats

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
console = Console()


def controller_progress(dashboard: LiveTelemetryDashboard):
    # Adapt controller step telemetry to the live dashboard.
    def update(episode: int, step: int, info: Dict[str, Any], reward: float):
        dashboard.update(
            info["telemetry"],
            sim_time_sec=info["sim_time_sec"],
            episode=episode,
            step=step,
            throughput_mbps=info["total_throughput_mbps"],
            drop_rate_pct=info["avg_drop_pct"],
            avg_latency_ms=info["avg_latency_ms"],
            reward=reward,
        )
    return update


def print_result_table(policy: str, metrics: Dict[str, float]) -> None:
    table = Table(title=f"FINAL SIMULATION RESULTS - {policy}", border_style="bright_blue")
    table.add_column("Metric", style="cyan")
    table.add_column("Result", justify="right", style="bold white")
    table.add_row("Link throughput", f"{metrics['throughput_mbps']:.2f} Mb/s")
    table.add_row("Packet drop rate", f"{metrics['drop_rate_pct']:.2f}%")
    table.add_row("Mean link latency", f"{metrics['avg_latency_ms']:.2f} ms")
    table.add_row("P99 of window mean latency", f"{metrics['p99_latency_ms']:.2f} ms")
    console.print(table)




def packet_weighted_latency(telem):
    """Compute packet-count-weighted average latency (physically correct metric)."""
    total_lat_w = sum(st.avg_latency_ms * st.tx_packets for st in telem.values() if st.avg_latency_ms > 0)
    total_pkt = sum(st.tx_packets for st in telem.values() if st.avg_latency_ms > 0)
    return total_lat_w / total_pkt if total_pkt > 0 else 0.0


def load_configs(chaos_enabled: bool = True, topology_path: str = "configs/topology.yaml"):
    top_path = topology_path
    traffic_path = "configs/traffic_profiles.yaml"
    rl_path = "configs/rl_config.yaml"
    chaos_path = "configs/chaos_config.yaml" if chaos_enabled else None

    with open(traffic_path, "r") as f:
        traffic_configs = yaml.safe_load(f).get("flows", [])
    with open(rl_path, "r") as f:
        rl_config = yaml.safe_load(f)
    chaos_config = None
    if chaos_path and os.path.exists(chaos_path):
        with open(chaos_path, "r") as f:
            chaos_config = yaml.safe_load(f)

    return top_path, traffic_configs, rl_config, chaos_config


def aggregate_telemetry(history):
    if not history:
        return {}
    aggregated = {}
    link_keys = history[0].keys()
    for key in link_keys:
        sample = history[0][key]
        total_tx_bytes = sum(h[key].tx_bytes for h in history if key in h)
        total_tx_p = sum(h[key].tx_packets for h in history if key in h)
        total_drop_p = sum(h[key].dropped_packets for h in history if key in h)
        total_arrived = total_tx_p + total_drop_p
        drop_rate = (total_drop_p / total_arrived * 100.0) if total_arrived > 0 else 0.0

        avg_util = float(np.mean([h[key].utilization_pct for h in history if key in h]))
        valid_lat = [h[key].avg_latency_ms for h in history if key in h and h[key].avg_latency_ms > 0]
        avg_lat = float(np.mean(valid_lat)) if valid_lat else 0.0
        avg_q = int(np.mean([h[key].queue_depth for h in history if key in h]))

        aggregated[key] = LinkStats(
            src=sample.src,
            dst=sample.dst,
            capacity_mbps=sample.capacity_mbps,
            tx_bytes=total_tx_bytes,
            tx_packets=total_tx_p,
            dropped_bytes=total_drop_p * 1000,
            dropped_packets=total_drop_p,
            queue_depth=avg_q,
            max_queue_packets=sample.max_queue_packets,
            utilization_pct=avg_util,
            drop_rate_pct=drop_rate,
            avg_latency_ms=avg_lat
        )
    return aggregated


def evaluate_rl_agent(use_dueling: bool, episodes: int, chaos: bool, save_plots: bool = False, topology_path: str = "configs/topology.yaml"):
    top_path, traffic_configs, rl_config, chaos_config = load_configs(chaos, topology_path)
    model_config = copy.deepcopy(rl_config)
    model_config["agent"]["use_dueling"] = use_dueling

    model_name = "Dueling DQN" if use_dueling else "Standard DQN"
    console.print(f"\n[bold green]Training {model_name} for {episodes} episodes...[/bold green]")
    controller = SDNController(top_path, traffic_configs, model_config, chaos_config=chaos_config)
    with LiveTelemetryDashboard(console) as dashboard:
        dashboard.set_phase("Training", model_name, total_episodes=episodes, total_steps=controller.env.max_steps)
        controller.train_episodes(num_episodes=episodes, verbose=False, progress_callback=controller_progress(dashboard))
        dashboard.set_phase("Policy evaluation", model_name, total_episodes=1, total_steps=600)
        metrics, last_telem = controller.evaluate(max_steps=600, progress_callback=controller_progress(dashboard))

    console.print(f"[bold green]Finished evaluating {model_name} policy.[/bold green]")
    if save_plots:
        ospf_m = evaluate_ospf(chaos=chaos, topology_path=topology_path)
        rr_m = evaluate_round_robin(chaos=chaos, topology_path=topology_path)
        controller.metrics_tracker.set_baselines(ospf_m, rr_m)
        plot_path = controller.save_dashboard_plots(output_dir="plots", filename=f"{model_name.lower().replace(' ', '_')}_dashboard.png")
        console.print(f"[bold green]Diagnostic plot saved to: {plot_path}[/bold green]")

    return metrics, last_telem, controller


def evaluate_ospf(chaos: bool, topology_path: str = "configs/topology.yaml"):
    top_path, traffic_configs, rl_config, chaos_config = load_configs(chaos, topology_path)
    topo = NetworkTopology(top_path)
    rt_api = RoutingTableAPI()
    sim = DataPlaneSimulator(topo, rt_api, traffic_configs, chaos_config=chaos_config)
    flow_ids = list(sim.generators.keys())
    engine = OSPFRoutingEngine(topo, rt_api)
    engine.update_routes(flow_ids, sim.flow_src_dst)

    tp_list, drop_list, lat_list = [], [], []
    with LiveTelemetryDashboard(console) as dashboard:
        dashboard.set_phase("Simulation", "Static OSPF", total_steps=600)
        for step in range(600):
            for lq in sim.links.values():
                lq.reset_window_stats()
            for _ in range(10):
                sim.step(0.01)
            telem = sim.stats_provider.collect_window_telemetry(0.1)
            tp = sum((st.tx_bytes * 8.0 / 1_000_000.0) / 0.1 for st in telem.values())
            dr = float(np.mean([st.drop_rate_pct for st in telem.values()])) if telem else 0.0
            lat = packet_weighted_latency(telem)
            tp_list.append(tp)
            drop_list.append(dr)
            lat_list.append(lat)
            dashboard.update(
                telem,
                sim_time_sec=sim.current_time,
                step=step + 1,
                throughput_mbps=tp,
                drop_rate_pct=dr,
                avg_latency_ms=lat,
            )

    valid_lat = [l for l in lat_list if not np.isnan(l) and l > 0]
    return {
        "throughput_mbps": float(np.nanmean(tp_list)) if tp_list else 0.0,
        "drop_rate_pct": float(np.nanmean(drop_list)) if drop_list else 0.0,
        "avg_latency_ms": float(np.mean(valid_lat)) if valid_lat else 0.0,
        "p99_latency_ms": float(np.percentile(valid_lat, 99)) if valid_lat else 0.0
    }


def evaluate_round_robin(chaos: bool, topology_path: str = "configs/topology.yaml"):
    top_path, traffic_configs, rl_config, chaos_config = load_configs(chaos, topology_path)
    topo = NetworkTopology(top_path)
    rt_api = RoutingTableAPI()
    sim = DataPlaneSimulator(topo, rt_api, traffic_configs, chaos_config=chaos_config)
    flow_ids = list(sim.generators.keys())
    engine = RoundRobinRoutingEngine(topo, rt_api)

    tp_list, drop_list, lat_list = [], [], []
    with LiveTelemetryDashboard(console) as dashboard:
        dashboard.set_phase("Simulation", "Round Robin", total_steps=600)
        for step in range(600):
            engine.update_routes(flow_ids, sim.flow_src_dst)
            for lq in sim.links.values():
                lq.reset_window_stats()
            for _ in range(10):
                sim.step(0.01)
            telem = sim.stats_provider.collect_window_telemetry(0.1)
            tp = sum((st.tx_bytes * 8.0 / 1_000_000.0) / 0.1 for st in telem.values())
            dr = float(np.mean([st.drop_rate_pct for st in telem.values()])) if telem else 0.0
            lat = packet_weighted_latency(telem)
            tp_list.append(tp)
            drop_list.append(dr)
            lat_list.append(lat)
            dashboard.update(
                telem,
                sim_time_sec=sim.current_time,
                step=step + 1,
                throughput_mbps=tp,
                drop_rate_pct=dr,
                avg_latency_ms=lat,
            )

    valid_lat = [l for l in lat_list if not np.isnan(l) and l > 0]
    return {
        "throughput_mbps": float(np.nanmean(tp_list)) if tp_list else 0.0,
        "drop_rate_pct": float(np.nanmean(drop_list)) if drop_list else 0.0,
        "avg_latency_ms": float(np.mean(valid_lat)) if valid_lat else 0.0,
        "p99_latency_ms": float(np.percentile(valid_lat, 99)) if valid_lat else 0.0
    }


def evaluate_q_learning(episodes: int, chaos: bool, topology_path: str = "configs/topology.yaml"):
    """Train and evaluate Q-learning over the same 600 windows as the DQN policies."""
    top_path, traffic_configs, rl_config, chaos_config = load_configs(chaos, topology_path)
    env = SDNEnv(top_path, traffic_configs, rl_config, chaos_config=chaos_config)
    agent_cfg = rl_config["agent"]
    agent = TabularQLearningAgent(env.action_dim, gamma=agent_cfg["gamma"])
    epsilon = agent_cfg["epsilon_start"]

    console.print(f"\n[bold green]Training Q-learning for {episodes} episodes...[/bold green]")
    with LiveTelemetryDashboard(console) as dashboard:
        dashboard.set_phase("Training", "Q-learning", total_episodes=episodes, total_steps=env.max_steps)
        progress = controller_progress(dashboard)
        for episode in range(1, episodes + 1):
            state, _ = env.reset()
            done = False
            while not done:
                action = agent.select_action(state, epsilon=epsilon)
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                agent.update(state, action, reward, next_state, done)
                state = next_state
                progress(episode, env.current_step, info, reward)
            epsilon = max(agent_cfg["epsilon_end"], epsilon * agent_cfg["epsilon_decay"])

        dashboard.set_phase("Policy evaluation", "Q-learning", total_episodes=1, total_steps=600)
        env.max_steps = 600
        state, _ = env.reset()
        throughput, drops, latencies = [], [], []
        done = False
        while not done:
            action = agent.select_action(state, evaluate=True)
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            progress(1, env.current_step, info, reward)
            throughput.append(info["total_throughput_mbps"])
            drops.append(info["avg_drop_pct"])
            latencies.append(info["avg_latency_ms"])

    valid_latencies = [value for value in latencies if not np.isnan(value) and value > 0]
    return {
        "throughput_mbps": float(np.nanmean(throughput)) if throughput else 0.0,
        "drop_rate_pct": float(np.nanmean(drops)) if drops else 0.0,
        "avg_latency_ms": float(np.mean(valid_latencies)) if valid_latencies else 0.0,
        "p99_latency_ms": float(np.percentile(valid_latencies, 99)) if valid_latencies else 0.0,
    }


def run_full_comparison(episodes: int, chaos: bool, topology_path: str = "configs/topology.yaml"):
    console.print("\n[bold cyan]RUNNING FULL MULTI-ALGORITHM BENCHMARK...[/bold cyan]")
    dueling_m, _, ctrl = evaluate_rl_agent(use_dueling=True, episodes=episodes, chaos=chaos, save_plots=False, topology_path=topology_path)
    standard_m, _, _ = evaluate_rl_agent(use_dueling=False, episodes=episodes, chaos=chaos, topology_path=topology_path)
    ospf_m = evaluate_ospf(chaos=chaos, topology_path=topology_path)
    q_m = evaluate_q_learning(episodes=episodes, chaos=chaos, topology_path=topology_path)

    table = Table(title=f"BENCHMARK COMPARISON ({'CHAOS ACTIVE' if chaos else 'NORMAL NETWORK'})")
    table.add_column("Metric", style="bold yellow")
    table.add_column("Dueling DQN (Proposed)", style="bold green")
    table.add_column("Standard DQN", style="bold cyan")
    table.add_column("Static OSPF", style="bold magenta")
    table.add_column("Q-learning", style="bold blue")

    table.add_row(
        "Throughput (Mbps)",
        f"{dueling_m['throughput_mbps']:.1f}",
        f"{standard_m['throughput_mbps']:.1f}",
        f"{ospf_m['throughput_mbps']:.1f}",
        f"{q_m['throughput_mbps']:.1f}"
    )
    table.add_row(
        "Packet Drop Rate (%)",
        f"{dueling_m['drop_rate_pct']:.2f}%",
        f"{standard_m['drop_rate_pct']:.2f}%",
        f"{ospf_m['drop_rate_pct']:.2f}%",
        f"{q_m['drop_rate_pct']:.2f}%"
    )
    table.add_row(
        "Average Latency (ms)",
        f"{dueling_m['avg_latency_ms']:.2f} ms",
        f"{standard_m['avg_latency_ms']:.2f} ms",
        f"{ospf_m['avg_latency_ms']:.2f} ms",
        f"{q_m['avg_latency_ms']:.2f} ms"
    )
    table.add_row(
        "P99 of window mean latency (ms)",
        f"{dueling_m['p99_latency_ms']:.2f} ms",
        f"{standard_m['p99_latency_ms']:.2f} ms",
        f"{ospf_m['p99_latency_ms']:.2f} ms",
        f"{q_m['p99_latency_ms']:.2f} ms"
    )

    console.print(table)
    ctrl.metrics_tracker.set_baselines(ospf_m, None)
    plot_path = ctrl.save_dashboard_plots(output_dir='plots', filename='dueling_dqn_dashboard.png')
    console.print(f'Diagnostic plot saved to: {plot_path}')


from rl_sdn_controller.network.dynamic_scenario import generate_random_production_scenario


def evaluate_model_on_packets(
    model_type: str,
    trained_controller: Optional[SDNController],
    target_packets: int,
    chaos: bool,
    custom_scenario: Optional[Tuple[NetworkTopology, List[Dict[str, Any]], Dict[str, Any]]] = None,
    topology_path: str = "configs/topology.yaml",
    dashboard: Optional[LiveTelemetryDashboard] = None,
) -> Dict[str, Any]:
    """
    Evaluates a specific model (Dueling DQN, Standard DQN, OSPF, Q-learning, or Round-Robin)
    on an exact target number of packets under a specific or randomized network scenario.
    """
    if dashboard is None:
        with LiveTelemetryDashboard(console) as live_dashboard:
            return evaluate_model_on_packets(
                model_type, trained_controller, target_packets, chaos,
                custom_scenario=custom_scenario, topology_path=topology_path,
                dashboard=live_dashboard,
            )

    if custom_scenario is not None:
        topo, traffic_configs, chaos_config = custom_scenario
        if not chaos:
            chaos_config = None
    else:
        top_path, traffic_configs, rl_config, chaos_config = load_configs(chaos, topology_path)
        topo = NetworkTopology(top_path)

    rt_api = RoutingTableAPI()
    sim = DataPlaneSimulator(topo, rt_api, traffic_configs, chaos_config=chaos_config)
    flow_ids = list(sim.generators.keys())

    if model_type == "ospf":
        engine = OSPFRoutingEngine(topo, rt_api)
        engine.update_routes(flow_ids, sim.flow_src_dst)
    elif model_type == "round_robin":
        engine = RoundRobinRoutingEngine(topo, rt_api)
    elif model_type in ["dueling_dqn", "standard_dqn", "q_learning"]:
        engine = RLRoutingEngine(topo, rt_api, flow_ids, sim.flow_src_dst)
        state_mgr = StateManager(topo)
        engine.apply_action(0)

    # Run packet processing loop — collect TRUE end-to-end latency per delivered packet
    total_tx_bytes = 0
    total_tx_pkts = 0
    total_drop_pkts = 0
    e2e_latencies_ms = []   # True end-to-end latency per delivered packet

    sim.reset()
    dashboard.set_phase("Packet evaluation", model_type.replace("_", " ").title())
    if model_type == "ospf":
        engine.update_routes(flow_ids, sim.flow_src_dst)
    elif model_type in ["dueling_dqn", "standard_dqn", "q_learning"]:
        engine.apply_action(0)

    step_dt = 0.01          # 10ms micro-step
    rl_step_interval = 0.1  # 100ms RL control interval
    time_since_rl = 0.0
    last_telem = {}

    while total_tx_pkts + total_drop_pkts < target_packets and sim.current_time < 300.0:

        # Reset window stats so this step's telemetry is fresh
        for lq in sim.links.values():
            lq.reset_window_stats()

        sim.step(step_dt)

        # Collect fresh telemetry AFTER the step so it reflects current link state
        step_telem = sim.stats_provider.collect_window_telemetry(step_dt)

        # RL / routing decisions based on CURRENT post-step link state
        if model_type == "round_robin":
            engine.update_routes(flow_ids, sim.flow_src_dst)
        elif model_type in ["dueling_dqn", "standard_dqn", "q_learning"]:
            time_since_rl += step_dt
            if time_since_rl >= rl_step_interval:
                time_since_rl = 0.0
                obs = state_mgr.get_observation_vector(step_telem)  # Fresh post-step obs
                action = trained_controller.agent.select_action(obs, evaluate=True)
                engine.apply_action(action)

        # Collect TRUE end-to-end latency from packets delivered this step
        for pkt in sim.delivered_packets:
            total_tx_bytes += pkt.size_bytes
            total_tx_pkts += 1
            if pkt.e2e_latency_ms > 0:
                e2e_latencies_ms.append(pkt.e2e_latency_ms)

        # Use simulator's precise per-step drop counter (tracks each enqueue() failure)
        total_drop_pkts += sim.step_dropped_packets

        last_telem = step_telem
        step_tx_mbps = sum(st.tx_bytes * 8.0 / 1_000_000.0 / step_dt for st in step_telem.values())
        step_tx = sum(st.tx_packets for st in step_telem.values())
        step_drop = sum(st.dropped_packets for st in step_telem.values())
        step_drop_pct = (step_drop / (step_tx + step_drop) * 100.0) if step_tx + step_drop else 0.0
        dashboard.update(
            step_telem,
            sim_time_sec=sim.current_time,
            step=round(sim.current_time / step_dt),
            throughput_mbps=step_tx_mbps,
            drop_rate_pct=step_drop_pct,
            avg_latency_ms=packet_weighted_latency(step_telem),
            delivered_packets=total_tx_pkts,
            dropped_packets=total_drop_pkts,
        )

    total_arrived = total_tx_pkts + total_drop_pkts
    drop_pct = (total_drop_pkts / total_arrived * 100.0) if total_arrived > 0 else 0.0
    throughput = (total_tx_bytes * 8.0 / 1_000_000.0) / max(0.001, sim.current_time)
    avg_lat = float(np.mean(e2e_latencies_ms)) if e2e_latencies_ms else 0.0
    p99_lat = float(np.percentile(e2e_latencies_ms, 99)) if e2e_latencies_ms else 0.0

    return {
        "target_packets": target_packets,
        "total_packets_sent": total_arrived,
        "delivered_packets": total_tx_pkts,
        "dropped_packets": total_drop_pkts,
        "drop_rate_pct": drop_pct,
        "throughput_mbps": throughput,
        "avg_latency_ms": avg_lat,
        "p99_latency_ms": p99_lat,
        "sim_time_sec": sim.current_time,
        "telemetry": last_telem
    }




def run_custom_train_and_packet_eval(topology_path: str = "configs/topology.yaml"):
    console.print("\n[bold cyan]CUSTOM WORKFLOW: TRAIN ON N CYCLES -> EVALUATE ON M PACKETS[/bold cyan]\n")
    
    # 1. Manually select train cycles
    train_cycles_str = Prompt.ask("Enter number of training cycles (episodes)", default="20")
    train_cycles = int(train_cycles_str)

    # 2. Manually select test packets
    test_packets_str = Prompt.ask("Enter number of test packets to evaluate against all models", default="5000")
    test_packets = int(test_packets_str)

    # 3. Chaos toggle
    chaos = Confirm.ask("Enable Network Chaos Engine (flapping links, BER drops, delay jitter)?", default=True)

    # 4. Randomized Real-World Scenario toggle
    random_scenario = (topology_path == "configs/topology.yaml" and
                       Confirm.ask("Generate Randomized Real-World Production Scenario (asymmetric link capacities & mixed traffic)?", default=False))

    # Generate or load scenario
    if random_scenario:
        scenario_tuple = generate_random_production_scenario()
        console.print("[dim italic]Generated fresh randomized real-world production topology & traffic mix.[/dim italic]")
    else:
        scenario_tuple = None

    # Phase 1: Training Dueling DQN
    console.print(f"\n[bold green][PHASE 1] Training Proposed Dueling DQN for {train_cycles} cycles...[/bold green]")
    top_path, traffic_configs, rl_config, chaos_config = load_configs(chaos, topology_path)
    dueling_cfg = copy.deepcopy(rl_config)
    dueling_cfg["agent"]["use_dueling"] = True
    
    # Train controller
    if scenario_tuple:
        # Train on dynamic scenario
        topo_obj, flows, ch_cfg = scenario_tuple
        import tempfile
        # Write temporary topology yaml for controller initialization
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tf:
            yaml.dump(topo_obj.topology_dict, tf)
            temp_top_path = tf.name
        dueling_controller = SDNController(temp_top_path, flows, dueling_cfg, chaos_config=ch_cfg if chaos else None)
    else:
        dueling_controller = SDNController(top_path, traffic_configs, dueling_cfg, chaos_config=chaos_config)

    with LiveTelemetryDashboard(console) as dashboard:
        dashboard.set_phase("Training", "Dueling DQN", total_episodes=train_cycles, total_steps=dueling_controller.env.max_steps)
        dueling_controller.train_episodes(
            num_episodes=train_cycles, verbose=False,
            progress_callback=controller_progress(dashboard),
        )

    # Train Standard DQN
    console.print(f"\n[bold green]Training Baseline Standard DQN for {train_cycles} cycles...[/bold green]")
    std_cfg = copy.deepcopy(rl_config)
    std_cfg["agent"]["use_dueling"] = False
    if scenario_tuple:
        std_controller = SDNController(temp_top_path, flows, std_cfg, chaos_config=ch_cfg if chaos else None)
    else:
        std_controller = SDNController(top_path, traffic_configs, std_cfg, chaos_config=chaos_config)
    with LiveTelemetryDashboard(console) as dashboard:
        dashboard.set_phase("Training", "Standard DQN", total_episodes=train_cycles, total_steps=std_controller.env.max_steps)
        std_controller.train_episodes(
            num_episodes=train_cycles, verbose=False,
            progress_callback=controller_progress(dashboard),
        )

    # Train a tabular Q-learning baseline on the same scenario and episode count.
    q_env = SDNEnv(
        scenario_tuple[0] if scenario_tuple else top_path,
        scenario_tuple[1] if scenario_tuple else traffic_configs,
        rl_config,
        chaos_config=(scenario_tuple[2] if chaos else None) if scenario_tuple else chaos_config,
    )
    q_agent = TabularQLearningAgent(
        q_env.action_dim, gamma=rl_config["agent"]["gamma"]
    )
    epsilon = rl_config["agent"]["epsilon_start"]
    console.print(f"\n[bold green]Training Q-learning for {train_cycles} cycles...[/bold green]")
    with LiveTelemetryDashboard(console) as dashboard:
        dashboard.set_phase("Training", "Q-learning", total_episodes=train_cycles, total_steps=q_env.max_steps)
        for episode in range(1, train_cycles + 1):
            state, _ = q_env.reset()
            done = False
            while not done:
                action = q_agent.select_action(state, epsilon=epsilon)
                next_state, reward, terminated, truncated, info = q_env.step(action)
                done = terminated or truncated
                q_agent.update(state, action, reward, next_state, done)
                state = next_state
                controller_progress(dashboard)(episode, q_env.current_step, info, reward)
            epsilon = max(
                rl_config["agent"]["epsilon_end"],
                epsilon * rl_config["agent"]["epsilon_decay"],
            )

    # Clean up temp file if needed
    if scenario_tuple and os.path.exists(temp_top_path):
        os.remove(temp_top_path)

    q_controller = SimpleNamespace(agent=q_agent)

    # Phase 2: Evaluation on exact M packets
    console.print(f"\n[bold cyan][PHASE 2] Evaluating all 4 models on {test_packets:,} test packets...[/bold cyan]")
    dueling_res = evaluate_model_on_packets("dueling_dqn", dueling_controller, test_packets, chaos, custom_scenario=scenario_tuple, topology_path=topology_path)
    std_res = evaluate_model_on_packets("standard_dqn", std_controller, test_packets, chaos, custom_scenario=scenario_tuple, topology_path=topology_path)
    ospf_res = evaluate_model_on_packets("ospf", None, test_packets, chaos, custom_scenario=scenario_tuple, topology_path=topology_path)
    q_res = evaluate_model_on_packets("q_learning", q_controller, test_packets, chaos, custom_scenario=scenario_tuple, topology_path=topology_path)


    # Phase 3: Display Comparison Results
    table = Table(title=f"\nPERFORMANCE COMPARISON ON {test_packets:,} TEST PACKETS ({'CHAOS ACTIVE' if chaos else 'NORMAL'})", border_style="bright_blue")
    table.add_column("Evaluation Metric", style="bold yellow")
    table.add_column("Dueling DQN (Proposed)", style="bold green", justify="right")
    table.add_column("Standard DQN", style="bold cyan", justify="right")
    table.add_column("Static OSPF Baseline", style="bold magenta", justify="right")
    table.add_column("Q-learning Baseline", style="bold blue", justify="right")

    table.add_row(
        "Packets Sent",
        f"{dueling_res['total_packets_sent']:,}",
        f"{std_res['total_packets_sent']:,}",
        f"{ospf_res['total_packets_sent']:,}",
        f"{q_res['total_packets_sent']:,}"
    )
    table.add_row(
        "Packets Delivered",
        f"{dueling_res['delivered_packets']:,}",
        f"{std_res['delivered_packets']:,}",
        f"{ospf_res['delivered_packets']:,}",
        f"{q_res['delivered_packets']:,}"
    )
    table.add_row(
        "Packets Dropped",
        f"{dueling_res['dropped_packets']:,}",
        f"{std_res['dropped_packets']:,}",
        f"{ospf_res['dropped_packets']:,}",
        f"{q_res['dropped_packets']:,}"
    )
    table.add_row(
        "Packet Drop Rate (%)",
        f"{dueling_res['drop_rate_pct']:.2f}%",
        f"{std_res['drop_rate_pct']:.2f}%",
        f"{ospf_res['drop_rate_pct']:.2f}%",
        f"{q_res['drop_rate_pct']:.2f}%"
    )
    table.add_row(
        "Throughput (Mbps)",
        f"{dueling_res['throughput_mbps']:.2f} Mbps",
        f"{std_res['throughput_mbps']:.2f} Mbps",
        f"{ospf_res['throughput_mbps']:.2f} Mbps",
        f"{q_res['throughput_mbps']:.2f} Mbps"
    )
    table.add_row(
        "Average Latency (ms)",
        f"{dueling_res['avg_latency_ms']:.2f} ms",
        f"{std_res['avg_latency_ms']:.2f} ms",
        f"{ospf_res['avg_latency_ms']:.2f} ms",
        f"{q_res['avg_latency_ms']:.2f} ms"
    )
    table.add_row(
        "Tail Latency P99 (ms)",
        f"{dueling_res['p99_latency_ms']:.2f} ms",
        f"{std_res['p99_latency_ms']:.2f} ms",
        f"{ospf_res['p99_latency_ms']:.2f} ms",
        f"{q_res['p99_latency_ms']:.2f} ms"
    )

    console.print(table)

    # Save diagnostic plots
    dueling_controller.metrics_tracker.set_baselines(ospf_res, None)
    plot_file = dueling_controller.save_dashboard_plots(output_dir="plots", filename=f"eval_{test_packets}_packets.png")
    console.print(f"\n[bold green]Performance diagnostic plots saved to: [underline]{plot_file}[/underline][/bold green]")


TOPOLOGY_PRESETS = {
    4: "configs/topology.yaml",
    8: "configs/topology_8.yaml",
    14: "configs/topology_14.yaml",
    22: "configs/topology_22.yaml",
}


def choose_topology():
    while True:
        raw_count = Prompt.ask("Router count", default="4")
        try:
            router_count = int(raw_count)
        except ValueError:
            router_count = None
        if router_count is not None and MIN_ROUTERS <= router_count <= MAX_ROUTERS:
            break
        console.print(f"[red]Enter a whole number between {MIN_ROUTERS} and {MAX_ROUTERS}.[/red]")

    topology = TOPOLOGY_PRESETS.get(router_count)
    if topology is None:
        topology = yaml.safe_dump(build_router_topology(router_count), sort_keys=False)
    console.print(f"[green]Using {router_count}-router topology[/green]")
    return topology


def main():
    os.system("clear" if os.name == "posix" else "cls")
    
    banner = Panel.fit(
        "[bold cyan]RL-SDN AUTOMATED CONTROL PLANE & PACKET SIMULATOR TUI[/bold cyan]\n"
        "[dim]Routing policy training and network performance comparison[/dim]",
        border_style="bright_blue"
    )
    console.print(banner)

    console.print("\n[bold yellow]Select Mode to Run:[/bold yellow]")
    console.print("  [1] [bold green]Train on N Cycles -> Test on M Packets (Custom Comparison)[/bold green]")
    console.print("  [2] [bold cyan]Full Comparison Benchmark[/bold cyan] (Dueling DQN vs Standard DQN vs OSPF vs Q-learning)")
    console.print("  [3] [bold white]Dueling DQN Agent Only[/bold white]")
    console.print("  [4] [bold magenta]Standard DQN Agent Only[/bold magenta]")
    console.print("  [5] [bold blue]Static OSPF (Dijkstra Shortest Path)[/bold blue]")
    console.print("  [6] [bold yellow]Round-Robin (ECMP Load Balancer)[/bold yellow]")
    console.print("  [7] [bold cyan]Run Automated PyTest Suite[/bold cyan]")
    console.print("  [8] Exit")

    visible_choice = Prompt.ask("Choose option", choices=[str(number) for number in range(1, 9)], default="1")
    choice = {"3": "7", "4": "8", "5": "9", "6": "10", "7": "12", "8": "13"}.get(visible_choice, visible_choice)
    if choice == "13":
        console.print("[yellow]Exiting RL-SDN TUI. Goodbye![/yellow]")
        return

    if choice == "12":
        console.print("\n[bold cyan]Running Automated Test Suite...[/bold cyan]\n")
        with tempfile.TemporaryDirectory(prefix="neuroroute-tests-", dir=os.path.dirname(os.path.abspath(__file__))) as temp_root:
            result = pytest.main(["tests/", "-v", "--basetemp", os.path.join(temp_root, "pytest")])
        raise SystemExit(result)

    topology_path = choose_topology()

    if choice == "1":
        run_custom_train_and_packet_eval(topology_path=topology_path)
        return

    if choice == "4":
        run_curriculum_demo()
        return

    if choice == "5":
        run_distillation_demo()
        return

    if choice == "6":
        run_transfer_learning_demo()
        return

    chaos = Confirm.ask("Enable Network Chaos Engine (flapping links, BER drops, delay jitter)?", default=True)
    episodes_str = Prompt.ask("Number of training / evaluation episodes", default="15")
    episodes = int(episodes_str)

    if choice == "2":
        run_full_comparison(episodes=episodes, chaos=chaos, topology_path=topology_path)

    elif choice == "3":
        run_hierarchical_demo(episodes=episodes, chaos=chaos)

    elif choice == "7":
        m, _, _ = evaluate_rl_agent(use_dueling=True, episodes=episodes, chaos=chaos, save_plots=True, topology_path=topology_path)
        print_result_table("Dueling DQN", m)

    elif choice == "8":
        m, _, _ = evaluate_rl_agent(use_dueling=False, episodes=episodes, chaos=chaos, save_plots=True, topology_path=topology_path)
        print_result_table("Standard DQN", m)

    elif choice == "9":
        m = evaluate_ospf(chaos=chaos, topology_path=topology_path)
        print_result_table("Static OSPF", m)

    elif choice == "10":
        m = evaluate_round_robin(chaos=chaos, topology_path=topology_path)
        print_result_table("Round Robin", m)

    elif choice == "11":
        onnx_file = Prompt.ask("Enter output ONNX filename", default="model.onnx")
        top_path, traffic_configs, rl_config, chaos_config = load_configs(chaos, topology_path)
        controller = SDNController(top_path, traffic_configs, rl_config, chaos_config=chaos_config)
        with LiveTelemetryDashboard(console) as dashboard:
            dashboard.set_phase("Training", "Dueling DQN", total_episodes=episodes, total_steps=controller.env.max_steps)
            controller.train_episodes(
                num_episodes=episodes, verbose=False,
                progress_callback=controller_progress(dashboard),
            )
        from rl_sdn_controller.ai.policy_exporter import export_policy_to_onnx
        export_policy_to_onnx(controller.agent.policy_net, controller.state_dim, onnx_file)
        console.print(f"[bold green]Model exported to {onnx_file}[/bold green]")



if __name__ == "__main__":
    main()

