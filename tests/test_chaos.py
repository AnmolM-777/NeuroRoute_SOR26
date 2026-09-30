import pytest
import numpy as np

from rl_sdn_controller.data_plane.chaos import ChaosEngine
from rl_sdn_controller.data_plane.packet import Packet, LinkQueue
from rl_sdn_controller.data_plane.traffic_gen import TrafficFlowGenerator


def test_chaos_engine_link_flapping():
    chaos_config = {
        "chaos": {
            "enabled": True,
            "link_failure_probability": 1.0, # Force link failure for test
            "link_repair_time_sec": 1.0,
            "jitter_std_us": 0.0,
            "random_drop_rate": 0.0
        }
    }
    chaos = ChaosEngine(chaos_config)
    link_key = ("r1", "r2")
    chaos.register_links([link_key])

    assert chaos.is_link_up("r1", "r2") is True
    # Trigger state update
    chaos.update(current_time=0.1, delta_time=1.0)
    assert chaos.is_link_up("r1", "r2") is False

    # Advance time beyond repair_time_sec
    chaos.update(current_time=1.5, delta_time=1.0)
    assert chaos.is_link_up("r1", "r2") is True


def test_chaos_delay_jitter_and_random_drops():
    chaos_config = {
        "chaos": {
            "enabled": True,
            "jitter_std_us": 1000.0,
            "latency_spike_probability": 0.0,
            "random_drop_rate": 1.0 # Force drop
        }
    }
    chaos = ChaosEngine(chaos_config)
    assert chaos.should_drop_packet() is True

    base_lat = 0.010 # 10ms
    sampled_lat = chaos.sample_delay(base_lat)
    assert sampled_lat > 0.0


def test_pareto_traffic_generator():
    cfg = {
        "id": "flow_pareto",
        "src": "h1",
        "dst": "h2",
        "pattern": "pareto",
        "rate_pps": 100,
        "packet_size_bytes": 1000,
        "duration_sec": 5.0
    }
    gen = TrafficFlowGenerator(cfg)
    pkts = gen.generate_packets(current_time=0.0, delta_time=0.1)
    assert len(pkts) > 0
