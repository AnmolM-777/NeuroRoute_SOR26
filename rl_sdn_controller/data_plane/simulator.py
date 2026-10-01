import asyncio
import logging
from typing import Dict, List, Tuple, Any, Optional
from rl_sdn_controller.network.topology import NetworkTopology
from rl_sdn_controller.sdn_api.routing_table_api import RoutingTableAPI
from rl_sdn_controller.sdn_api.stats_provider import StatsProvider
from rl_sdn_controller.data_plane.packet import Packet, LinkQueue
from rl_sdn_controller.data_plane.traffic_gen import TrafficFlowGenerator

from rl_sdn_controller.data_plane.chaos import ChaosEngine

logger = logging.getLogger(__name__)


class DataPlaneSimulator:
    """
    Simulation Mode Data Plane Packet Forwarding Engine (Layer 3).
    Asynchronous packet scheduling and hop-by-hop packet processing.
    Supports stochastic network chaos (link flapping, jitter, BER drops).
    """
    def __init__(self, topology: NetworkTopology, routing_table_api: RoutingTableAPI, traffic_configs: List[Dict[str, Any]], chaos_config: Dict[str, Any] = None):
        self.topology = topology
        self.routing_table_api = routing_table_api
        
        self.chaos_engine = ChaosEngine(chaos_config) if chaos_config else None
        if self.chaos_engine:
            self.chaos_engine.register_links(topology.get_link_keys())

        # Initialize link queues
        self.links: Dict[Tuple[str, str], LinkQueue] = {}
        for (src, dst), info in topology.links_info.items():
            self.links[(src, dst)] = LinkQueue(
                src=src,
                dst=dst,
                capacity_mbps=info["capacity"],
                latency_us=info["latency"],
                max_queue_packets=info["max_queue_packets"],
                chaos_engine=self.chaos_engine
            )

        # Initialize traffic generators
        self.generators: Dict[str, TrafficFlowGenerator] = {}
        self.flow_src_dst: Dict[str, Tuple[str, str]] = {}
        for cfg in traffic_configs:
            gen = TrafficFlowGenerator(cfg)
            self.generators[gen.flow_id] = gen
            self.flow_src_dst[gen.flow_id] = (gen.src, gen.dst)

        self.stats_provider = StatsProvider(self.links)
        self.current_time = 0.0

        # Transit buffer for packets currently in transit across links
        self.in_flight_packets: List[Packet] = []
        # Fully delivered packets (reached final destination host) — cleared each step
        self.delivered_packets: List[Packet] = []
        # Packets dropped this step (at any hop) — cleared each step
        self.step_dropped_packets: int = 0

    def reset(self):
        self.current_time = 0.0
        self.in_flight_packets.clear()
        self.delivered_packets.clear()
        self.step_dropped_packets = 0
        if self.chaos_engine:
            self.chaos_engine.reset()
        for link_q in self.links.values():
            link_q.queue.clear()
            link_q.reset_window_stats()
        for gen in self.generators.values():
            gen.reset()

    def step(self, delta_time: float):
        """
        Advances virtual clock by delta_time seconds.
        Performs packet generation, queueing, forwarding lookup, link transmission, and chaos updates.
        """
        if self.chaos_engine:
            self.chaos_engine.update(self.current_time, delta_time)
        start_step_time = self.current_time
        self.current_time += delta_time

        # 1. Generate new packets from traffic flows
        new_packets: List[Packet] = []
        for gen in self.generators.values():
            pkts = gen.generate_packets(start_step_time, delta_time)
            for pkt in pkts:
                # Lookup end-to-end path in SDN Routing Table API
                path = self.routing_table_api.get_flow_path(pkt.flow_id)
                if not path:
                    # Fallback to shortest path if no rule installed
                    candidate_paths = self.topology.get_candidate_paths(pkt.src, pkt.dst, k=1)
                    path = candidate_paths[0] if candidate_paths else [pkt.src, pkt.dst]
                pkt.hops = list(path)
                pkt.current_hop_idx = 0
                new_packets.append(pkt)

        # 2. Process newly generated packets onto their first hop queue
        self.step_dropped_packets = 0  # Reset per-step drop counter
        for pkt in new_packets:
            if len(pkt.hops) > 1:
                first_hop = (pkt.hops[0], pkt.hops[1])
                if first_hop in self.links:
                    dropped = not self.links[first_hop].enqueue(pkt, self.current_time)
                    if dropped:
                        self.step_dropped_packets += 1

        # 3. Process existing in-flight packets reaching next hop
        still_in_flight: List[Packet] = []
        for pkt in self.in_flight_packets:
            if pkt.transmission_time and self.current_time >= pkt.transmission_time:
                pkt.current_hop_idx += 1
                if pkt.current_hop_idx < len(pkt.hops) - 1:
                    next_src = pkt.hops[pkt.current_hop_idx]
                    next_dst = pkt.hops[pkt.current_hop_idx + 1]
                    next_hop = (next_src, next_dst)
                    if next_hop in self.links:
                        dropped = not self.links[next_hop].enqueue(pkt, self.current_time)
                        if dropped:
                            self.step_dropped_packets += 1
                    else:
                        self.step_dropped_packets += 1  # No route available
                else:
                    # Packet reached final destination host
                    pkt.arrival_time = self.current_time
            else:
                still_in_flight.append(pkt)

        self.in_flight_packets = still_in_flight

        # 4. Transmit packets from all link queues based on link capacity
        self.delivered_packets.clear()  # Fresh per-step delivery list
        for (src, dst), link_q in self.links.items():
            delivered_pkts = link_q.process_transmissions(self.current_time, delta_time)
            for pkt in delivered_pkts:
                # Check if this packet has just crossed its LAST hop (arrived at destination)
                if pkt.current_hop_idx >= len(pkt.hops) - 2:
                    pkt.arrival_time = self.current_time
                    self.delivered_packets.append(pkt)
                else:
                    self.in_flight_packets.append(pkt)

    async def run_async(self, duration_sec: float, step_interval_sec: float = 0.01):
        """Asynchronous execution loop for real-time simulation runs."""
        elapsed = 0.0
        while elapsed < duration_sec:
            self.step(step_interval_sec)
            elapsed += step_interval_sec
            await asyncio.sleep(0.001) # Yield control to event loop
