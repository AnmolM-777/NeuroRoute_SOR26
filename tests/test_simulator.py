import pytest
import numpy as np

from rl_sdn_controller.data_plane.packet import Packet, LinkQueue
from rl_sdn_controller.data_plane.traffic_gen import TrafficFlowGenerator
from rl_sdn_controller.network.topology import NetworkTopology
from rl_sdn_controller.sdn_api.routing_table_api import RoutingTableAPI
from rl_sdn_controller.data_plane.simulator import DataPlaneSimulator


def test_link_queue_overflow():
    link_q = LinkQueue(src="r1", dst="r2", capacity_mbps=10, latency_us=50, max_queue_packets=2)
    pkt1 = Packet(flow_id="f1", src="h1", dst="h2", size_bytes=1000, creation_time=0.0)
    pkt2 = Packet(flow_id="f1", src="h1", dst="h2", size_bytes=1000, creation_time=0.0)
    pkt3 = Packet(flow_id="f1", src="h1", dst="h2", size_bytes=1000, creation_time=0.0)

    assert link_q.enqueue(pkt1) is True
    assert link_q.enqueue(pkt2) is True
    # Third packet overflows max_queue_packets limit
    assert link_q.enqueue(pkt3) is False
    assert link_q.total_dropped_packets == 1


def test_traffic_generator_poisson():
    cfg = {
        "id": "f1",
        "src": "h1",
        "dst": "h2",
        "pattern": "poisson",
        "rate_pps": 100,
        "packet_size_bytes": 1000,
        "duration_sec": 10.0
    }
    gen = TrafficFlowGenerator(cfg)
    pkts = gen.generate_packets(current_time=0.0, delta_time=0.1)
    # At 100 pps for 0.1s, expect roughly ~10 packets
    assert len(pkts) > 0


def test_data_plane_simulator(tmp_path):
    topo_file = str(tmp_path / "topology.yaml")
    with open(topo_file, "w") as f:
        f.write("""
nodes:
  h1: {type: host}
  h2: {type: host}
  r1: {type: router}
links:
  - {src: h1, dst: r1, capacity: 1000, latency: 10, max_queue_packets: 100}
  - {src: r1, dst: h2, capacity: 1000, latency: 10, max_queue_packets: 100}
""")

    traffic_cfg = [{
        "id": "f1",
        "src": "h1",
        "dst": "h2",
        "pattern": "poisson",
        "rate_pps": 50,
        "packet_size_bytes": 500,
        "duration_sec": 5.0
    }]

    topo = NetworkTopology(topo_file)
    rt_api = RoutingTableAPI()
    sim = DataPlaneSimulator(topo, rt_api, traffic_cfg)

    # Install routing path rule
    rt_api.install_flow_rule("f1", ["h1", "r1", "h2"])

    # Run simulation steps
    for _ in range(10):
        sim.step(0.01)

    telem = sim.stats_provider.collect_window_telemetry(0.1)
    assert ("h1", "r1") in telem
    assert ("r1", "h2") in telem
