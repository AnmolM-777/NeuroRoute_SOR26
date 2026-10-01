import sys
import os
import copy
import yaml
import logging
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rich.console import Console
from rich.table import Table

from rl_sdn_controller.control_plane.controller import SDNController
from rl_sdn_controller.network.topology import NetworkTopology
from rl_sdn_controller.sdn_api.routing_table_api import RoutingTableAPI
from rl_sdn_controller.data_plane.simulator import DataPlaneSimulator
from rl_sdn_controller.network.routing_engine import OSPFRoutingEngine, RoundRobinRoutingEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("compare_models")
console = Console()



def packet_weighted_latency(telem):
    """Compute packet-count-weighted average latency (physically correct metric)."""
    total_lat_w = sum(st.avg_latency_ms * st.tx_packets for st in telem.values() if st.avg_latency_ms > 0)
    total_pkt = sum(st.tx_packets for st in telem.values() if st.avg_latency_ms > 0)
    return total_lat_w / total_pkt if total_pkt > 0 else 0.0


def evaluate_rl_model(use_dueling: bool, episodes: int, chaos: bool = True):
    """Trains and evaluates an RL agent with the specified model architecture."""
    top_path = "configs/topology.yaml"
    traffic_path = "configs/traffic_profiles.yaml"
    rl_path = "configs/rl_config.yaml"
    chaos_path = "configs/chaos_config.yaml" if chaos else None

    with open(traffic_path, "r") as f:
        traffic_configs = yaml.safe_load(f).get("flows", [])
    with open(rl_path, "r") as f:
        rl_config = yaml.safe_load(f)
    
    # Toggle dueling flag in config
    model_config = copy.deepcopy(rl_config)
    model_config["agent"]["use_dueling"] = use_dueling

    chaos_config = None
    if chaos_path:
        with open(chaos_path, "r") as f:
            chaos_config = yaml.safe_load(f)

    model_name = "Dueling DQN" if use_dueling else "Standard DQN"
    logger.info(f"Training and Evaluating {model_name}...")

    controller = SDNController(top_path, traffic_configs, model_config, chaos_config=chaos_config)
    controller.train_episodes(num_episodes=episodes, verbose=False)

    # Evaluation phase
    controller.env.max_steps = 600
    state, _ = controller.env.reset()
    tp_list, drop_list, lat_list = [], [], []
    done = False

    while not done:
        action = controller.agent.select_action(state, evaluate=True)
        state, reward, terminated, truncated, info = controller.env.step(action)
        done = terminated or truncated
        tp_list.append(info["total_throughput_mbps"])
        drop_list.append(info["avg_drop_pct"])
        lat_list.append(info["avg_latency_ms"])

    valid_lat = [l for l in lat_list if not np.isnan(l) and l > 0]
    return {
        "throughput_mbps": float(np.nanmean(tp_list)) if tp_list else 0.0,
        "drop_rate_pct": float(np.nanmean(drop_list)) if drop_list else 0.0,
        "avg_latency_ms": float(np.mean(valid_lat)) if valid_lat else 0.0,
        "p99_latency_ms": float(np.percentile(valid_lat, 99)) if valid_lat else 0.0
    }


def evaluate_baselines(chaos: bool = True):
    """Evaluates Static OSPF and Round-Robin baselines."""
    top_path = "configs/topology.yaml"
    traffic_path = "configs/traffic_profiles.yaml"
    chaos_path = "configs/chaos_config.yaml" if chaos else None

    with open(traffic_path, "r") as f:
        traffic_configs = yaml.safe_load(f).get("flows", [])
    chaos_config = None
    if chaos_path:
        with open(chaos_path, "r") as f:
            chaos_config = yaml.safe_load(f)

    topo = NetworkTopology(top_path)
    
    # 1. OSPF
    rt_api_ospf = RoutingTableAPI()
    sim_ospf = DataPlaneSimulator(topo, rt_api_ospf, traffic_configs, chaos_config=chaos_config)
    flow_ids = list(sim_ospf.generators.keys())
    ospf_engine = OSPFRoutingEngine(topo, rt_api_ospf)
    ospf_engine.update_routes(flow_ids, sim_ospf.flow_src_dst)

    ospf_tp, ospf_drops, ospf_lat = [], [], []
    for _ in range(600):
        for lq in sim_ospf.links.values():
            lq.reset_window_stats()
        for _ in range(10):
            sim_ospf.step(0.01)
        telem = sim_ospf.stats_provider.collect_window_telemetry(0.1)
        tp = sum((st.tx_bytes * 8.0 / 1_000_000.0) / 0.1 for st in telem.values())
        dr = float(np.mean([st.drop_rate_pct for st in telem.values()])) if telem else 0.0
        lat = packet_weighted_latency(telem)
        ospf_tp.append(tp)
        ospf_drops.append(dr)
        ospf_lat.append(lat)

    # 2. Round-Robin
    rt_api_rr = RoutingTableAPI()
    sim_rr = DataPlaneSimulator(topo, rt_api_rr, traffic_configs, chaos_config=chaos_config)
    rr_engine = RoundRobinRoutingEngine(topo, rt_api_rr)
    rr_tp, rr_drops, rr_lat = [], [], []
    for _ in range(600):
        rr_engine.update_routes(flow_ids, sim_rr.flow_src_dst)
        for lq in sim_rr.links.values():
            lq.reset_window_stats()
        for _ in range(10):
            sim_rr.step(0.01)
        telem = sim_rr.stats_provider.collect_window_telemetry(0.1)
        tp = sum((st.tx_bytes * 8.0 / 1_000_000.0) / 0.1 for st in telem.values())
        dr = float(np.mean([st.drop_rate_pct for st in telem.values()])) if telem else 0.0
        lat = packet_weighted_latency(telem)
        rr_tp.append(tp)
        rr_drops.append(dr)
        rr_lat.append(lat)

    def calc_metrics(tp_list, drop_list, lat_list):
        valid = [l for l in lat_list if not np.isnan(l) and l > 0]
        return {
            "throughput_mbps": float(np.nanmean(tp_list)) if tp_list else 0.0,
            "drop_rate_pct": float(np.nanmean(drop_list)) if drop_list else 0.0,
            "avg_latency_ms": float(np.mean(valid)) if valid else 0.0,
            "p99_latency_ms": float(np.percentile(valid, 99)) if valid else 0.0
        }

    return calc_metrics(ospf_tp, ospf_drops, ospf_lat), calc_metrics(rr_tp, rr_drops, rr_lat)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Compare RL Model Architectures & Baselines")
    parser.add_argument("--episodes", type=int, default=15, help="Number of training episodes")
    parser.add_argument("--chaos", action="store_true", help="Activate network chaos engine")
    args = parser.parse_args()

    console.print("\n[bold cyan]====== MODEL ARCHITECTURE COMPARISON BENCHMARK ======[/bold cyan]\n")

    dueling_results = evaluate_rl_model(use_dueling=True, episodes=args.episodes, chaos=args.chaos)
    standard_results = evaluate_rl_model(use_dueling=False, episodes=args.episodes, chaos=args.chaos)
    ospf_results, rr_results = evaluate_baselines(chaos=args.chaos)

    table = Table(title="🤖 MODEL COMPARISON: Dueling DQN vs Standard DQN vs OSPF vs Round-Robin")
    table.add_column("Metric", style="bold yellow")
    table.add_column("Dueling DQN", style="bold green")
    table.add_column("Standard DQN", style="bold cyan")
    table.add_column("Static OSPF", style="magenta")
    table.add_column("Round-Robin", style="blue")

    table.add_row(
        "Throughput (Mbps)",
        f"{dueling_results['throughput_mbps']:.1f}",
        f"{standard_results['throughput_mbps']:.1f}",
        f"{ospf_results['throughput_mbps']:.1f}",
        f"{rr_results['throughput_mbps']:.1f}"
    )
    table.add_row(
        "Packet Drop Rate (%)",
        f"{dueling_results['drop_rate_pct']:.2f}%",
        f"{standard_results['drop_rate_pct']:.2f}%",
        f"{ospf_results['drop_rate_pct']:.2f}%",
        f"{rr_results['drop_rate_pct']:.2f}%"
    )
    table.add_row(
        "Average Latency (ms)",
        f"{dueling_results['avg_latency_ms']:.2f} ms",
        f"{standard_results['avg_latency_ms']:.2f} ms",
        f"{ospf_results['avg_latency_ms']:.2f} ms",
        f"{rr_results['avg_latency_ms']:.2f} ms"
    )
    table.add_row(
        "Tail Latency P99 (ms)",
        f"{dueling_results['p99_latency_ms']:.2f} ms",
        f"{standard_results['p99_latency_ms']:.2f} ms",
        f"{ospf_results['p99_latency_ms']:.2f} ms",
        f"{rr_results['p99_latency_ms']:.2f} ms"
    )

    console.print(table)


if __name__ == "__main__":
    main()
