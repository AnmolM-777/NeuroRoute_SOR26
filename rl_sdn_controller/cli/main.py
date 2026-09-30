import argparse
import sys
import yaml
import logging
import numpy as np

from rl_sdn_controller.control_plane.controller import SDNController
from rl_sdn_controller.ai.policy_exporter import export_policy_to_onnx, export_policy_to_torchscript
from rl_sdn_controller.cli.visualizer import TerminalVisualizer
from rl_sdn_controller.network.topology import NetworkTopology
from rl_sdn_controller.sdn_api.routing_table_api import RoutingTableAPI
from rl_sdn_controller.data_plane.simulator import DataPlaneSimulator
from rl_sdn_controller.network.routing_engine import OSPFRoutingEngine, RoundRobinRoutingEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("rl_sdn_controller")


def load_configs(top_path: str, traffic_path: str, rl_path: str, chaos_path: str = None):
    with open(traffic_path, 'r') as f:
        traffic_data = yaml.safe_load(f)
    with open(rl_path, 'r') as f:
        rl_data = yaml.safe_load(f)
    chaos_data = None
    if chaos_path:
        with open(chaos_path, 'r') as f:
            chaos_data = yaml.safe_load(f)
    return top_path, traffic_data.get("flows", []), rl_data, chaos_data


def run_train(args):
    chaos_path = args.chaos_config if getattr(args, 'chaos', False) else None
    top_path, traffic_configs, rl_config, chaos_config = load_configs(args.topology, args.traffic, args.rl_config, chaos_path)
    controller = SDNController(top_path, traffic_configs, rl_config, chaos_config=chaos_config)
    
    viz = TerminalVisualizer()
    title = "RL AGENT TRAINING LOOP (WITH CHAOS)" if chaos_config else "RL AGENT TRAINING LOOP"
    viz.print_simulation_header(title)
    
    logger.info(f"Starting training for {args.episodes} episodes...")
    history = controller.train_episodes(num_episodes=args.episodes, verbose=True)

    if args.export_onnx:
        export_policy_to_onnx(controller.agent.policy_net, controller.env.state_dim, args.export_onnx)


def run_benchmark(args):
    chaos_path = args.chaos_config if getattr(args, 'chaos', False) else None
    top_path, traffic_configs, rl_config, chaos_config = load_configs(args.topology, args.traffic, args.rl_config, chaos_path)
    viz = TerminalVisualizer()
    title = "RUNNING RL vs OSPF vs ROUND-ROBIN BENCHMARK (NETWORK CHAOS ACTIVE)" if chaos_config else "RUNNING RL vs OSPF vs ROUND-ROBIN BENCHMARK"
    viz.print_simulation_header(title)

    # 1. Evaluate RL Agent
    logger.info("Evaluating RL Agent policy...")
    rl_controller = SDNController(top_path, traffic_configs, rl_config, chaos_config=chaos_config)
    # Quick pre-train for 20 episodes to align policy
    rl_controller.train_episodes(num_episodes=args.episodes, verbose=False)
    
    # Evaluate RL without exploration
    state, _ = rl_controller.env.reset()
    rl_tp, rl_drops, rl_latencies = [], [], []
    rl_telemetry_history = []
    done = False
    while not done:
        action = rl_controller.agent.select_action(state, evaluate=True)
        state, reward, terminated, truncated, info = rl_controller.env.step(action)
        done = terminated or truncated
        rl_tp.append(info["total_throughput_mbps"])
        rl_drops.append(info["avg_drop_pct"])
        rl_latencies.append(info["avg_latency_ms"])
        if "telemetry" in info:
            rl_telemetry_history.append(info["telemetry"])

    def calc_metrics(tp_list, drop_list, lat_list):
        valid_lat = [l for l in lat_list if not np.isnan(l) and l > 0]
        return {
            "throughput_mbps": float(np.nanmean(tp_list)) if tp_list else 0.0,
            "drop_rate_pct": float(np.nanmean(drop_list)) if drop_list else 0.0,
            "avg_latency_ms": float(np.mean(valid_lat)) if valid_lat else 0.0,
            "p99_latency_ms": float(np.percentile(valid_lat, 99)) if valid_lat else 0.0
        }

    rl_metrics = calc_metrics(rl_tp, rl_drops, rl_latencies)

    # 2. Evaluate Static OSPF
    logger.info("Evaluating Static OSPF Baseline...")
    topo = NetworkTopology(top_path)
    rt_api_ospf = RoutingTableAPI()
    sim_ospf = DataPlaneSimulator(topo, rt_api_ospf, traffic_configs, chaos_config=chaos_config)
    flow_ids = list(sim_ospf.generators.keys())
    ospf_engine = OSPFRoutingEngine(topo, rt_api_ospf)
    ospf_engine.update_routes(flow_ids, sim_ospf.flow_src_dst)

    ospf_tp, ospf_drops, ospf_latencies = [], [], []
    steps = rl_config.get("control_plane", {}).get("max_simulation_steps", 600)
    for _ in range(steps):
        for lq in sim_ospf.links.values():
            lq.reset_window_stats()
        for _ in range(10):
            sim_ospf.step(0.01)
        telem = sim_ospf.stats_provider.collect_window_telemetry(0.1)
        
        tp = sum((st.tx_bytes * 8.0 / 1_000_000.0) / 0.1 for st in telem.values())
        dr = float(np.mean([st.drop_rate_pct for st in telem.values()])) if telem else 0.0
        valid_l = [st.avg_latency_ms for st in telem.values() if st.avg_latency_ms > 0]
        lat = float(np.mean(valid_l)) if valid_l else 0.0
        
        ospf_tp.append(tp)
        ospf_drops.append(dr)
        ospf_latencies.append(lat)

    ospf_metrics = calc_metrics(ospf_tp, ospf_drops, ospf_latencies)

    # 3. Evaluate Round-Robin
    logger.info("Evaluating Round-Robin Baseline...")
    rt_api_rr = RoutingTableAPI()
    sim_rr = DataPlaneSimulator(topo, rt_api_rr, traffic_configs, chaos_config=chaos_config)
    rr_engine = RoundRobinRoutingEngine(topo, rt_api_rr)

    rr_tp, rr_drops, rr_latencies = [], [], []
    for _ in range(steps):
        rr_engine.update_routes(flow_ids, sim_rr.flow_src_dst)
        for lq in sim_rr.links.values():
            lq.reset_window_stats()
        for _ in range(10):
            sim_rr.step(0.01)
        telem = sim_rr.stats_provider.collect_window_telemetry(0.1)

        tp = sum((st.tx_bytes * 8.0 / 1_000_000.0) / 0.1 for st in telem.values())
        dr = float(np.mean([st.drop_rate_pct for st in telem.values()])) if telem else 0.0
        valid_l = [st.avg_latency_ms for st in telem.values() if st.avg_latency_ms > 0]
        lat = float(np.mean(valid_l)) if valid_l else 0.0

        rr_tp.append(tp)
        rr_drops.append(dr)
        rr_latencies.append(lat)

    rr_metrics = calc_metrics(rr_tp, rr_drops, rr_latencies)

    # Aggregate telemetry over all steps for full episode summary table
    from rl_sdn_controller.sdn_api.stats_provider import LinkStats
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

    aggregated_rl_telemetry = aggregate_telemetry(rl_telemetry_history)

    # Display summary
    viz.print_benchmark_table(rl_metrics, ospf_metrics, rr_metrics)
    if aggregated_rl_telemetry:
        viz.print_link_telemetry_table(aggregated_rl_telemetry)


def run_export(args):
    top_path, traffic_configs, rl_config, _ = load_configs(args.topology, args.traffic, args.rl_config)
    controller = SDNController(top_path, traffic_configs, rl_config)
    onnx_path = export_policy_to_onnx(controller.agent.policy_net, controller.env.state_dim, args.output)
    print(f"Exported model successfully to {onnx_path}")


def main():
    parser = argparse.ArgumentParser(description="RL-SDN Three-Layer Controller CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Train parser
    train_parser = subparsers.add_parser("train", help="Train RL Agent")
    train_parser.add_argument("--topology", default="configs/topology.yaml", help="Path to topology.yaml")
    train_parser.add_argument("--traffic", default="configs/traffic_profiles.yaml", help="Path to traffic_profiles.yaml")
    train_parser.add_argument("--rl-config", default="configs/rl_config.yaml", help="Path to rl_config.yaml")
    train_parser.add_argument("--chaos-config", default="configs/chaos_config.yaml", help="Path to chaos_config.yaml")
    train_parser.add_argument("--chaos", action="store_true", help="Enable network chaos during training")
    train_parser.add_argument("--episodes", type=int, default=15, help="Number of training episodes")
    train_parser.add_argument("--export-onnx", default="model.onnx", help="Path to export ONNX model")

    # Benchmark parser
    bench_parser = subparsers.add_parser("benchmark", help="Run RL vs OSPF vs Round-Robin Benchmark")
    bench_parser.add_argument("--topology", default="configs/topology.yaml", help="Path to topology.yaml")
    bench_parser.add_argument("--traffic", default="configs/traffic_profiles.yaml", help="Path to traffic_profiles.yaml")
    bench_parser.add_argument("--rl-config", default="configs/rl_config.yaml", help="Path to rl_config.yaml")
    bench_parser.add_argument("--chaos-config", default="configs/chaos_config.yaml", help="Path to chaos_config.yaml")
    bench_parser.add_argument("--chaos", action="store_true", help="Enable network chaos during benchmark")
    bench_parser.add_argument("--episodes", type=int, default=15, help="RL training episodes before evaluation")

    # Export parser
    export_parser = subparsers.add_parser("export", help="Export PyTorch model to ONNX")
    export_parser.add_argument("--topology", default="configs/topology.yaml")
    export_parser.add_argument("--traffic", default="configs/traffic_profiles.yaml")
    export_parser.add_argument("--rl-config", default="configs/rl_config.yaml")
    export_parser.add_argument("--output", default="model.onnx", help="Output path for ONNX file")

    args = parser.parse_args()

    if args.command == "train":
        run_train(args)
    elif args.command == "benchmark":
        run_benchmark(args)
    elif args.command == "export":
        run_export(args)


if __name__ == "__main__":
    main()
