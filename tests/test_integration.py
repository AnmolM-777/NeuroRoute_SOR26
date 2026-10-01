import pytest
import os
import yaml

from rl_sdn_controller.control_plane.controller import SDNController


def test_full_controller_training_integration(tmp_path):
    topo_file = str(tmp_path / "topology.yaml")
    with open(topo_file, "w") as f:
        f.write("""
nodes:
  h1: {type: host}
  h2: {type: host}
  r1: {type: router}
  r2: {type: router}
links:
  - {src: h1, dst: r1, capacity: 500, latency: 10, max_queue_packets: 100}
  - {src: r1, dst: r2, capacity: 500, latency: 10, max_queue_packets: 100}
  - {src: r2, dst: h2, capacity: 500, latency: 10, max_queue_packets: 100}
""")

    traffic_configs = [{
        "id": "flow_test",
        "src": "h1",
        "dst": "h2",
        "pattern": "poisson",
        "rate_pps": 100,
        "packet_size_bytes": 800,
        "duration_sec": 5.0
    }]

    rl_config = {
        "agent": {
            "learning_rate": 0.001,
            "gamma": 0.99,
            "epsilon_start": 0.5,
            "epsilon_end": 0.05,
            "epsilon_decay": 0.9,
            "buffer_capacity": 1000,
            "batch_size": 8,
            "target_update_freq": 2,
            "network_hidden": [32, 32]
        },
        "control_plane": {
            "update_interval_ms": 100,
            "telemetry_window_ms": 100,
            "max_simulation_steps": 5
        },
        "reward_function": {
            "throughput_weight": 1.0,
            "drop_rate_weight": -100.0,
            "latency_weight": -0.1
        }
    }

    controller = SDNController(topo_file, traffic_configs, rl_config)
    history = controller.train_episodes(num_episodes=2, verbose=False)

    assert len(history) == 2
    assert "total_reward" in history[0]
    assert "avg_throughput_mbps" in history[0]
