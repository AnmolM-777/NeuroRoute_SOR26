import time
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Packet:
    flow_id: str
    src: str
    dst: str
    size_bytes: int
    creation_time: float      # Virtual timestamp in seconds when packet was born
    seq_id: int = 0
    hops: List[str] = field(default_factory=list)
    current_hop_idx: int = 0
    arrival_time: Optional[float] = None    # Time packet arrived at current link queue
    transmission_time: Optional[float] = None
    e2e_latency_ms: float = 0.0             # Accumulated end-to-end latency across ALL hops (ms)


class LinkQueue:
    """
    Simulates a network link queue with bandwidth capacity limits,
    propagation delay, FIFO buffer limits, and telemetry tracking.
    Supports optional ChaosEngine for link flapping, jitter, and BER drops.
    """
    def __init__(self, src: str, dst: str, capacity_mbps: float, latency_us: float, max_queue_packets: int = 100, chaos_engine=None):
        self.src = src
        self.dst = dst
        self.capacity_mbps = capacity_mbps
        self.latency_us = latency_us
        self.latency_sec = latency_us / 1_000_000.0
        self.max_queue_packets = max_queue_packets
        self.chaos_engine = chaos_engine

        self.queue: List[Packet] = []
        
        # Telemetry metrics (resettable per monitoring window)
        self.total_tx_bytes: int = 0
        self.total_tx_packets: int = 0
        self.total_dropped_bytes: int = 0
        self.total_dropped_packets: int = 0
        self.latencies_sec: List[float] = []

    @property
    def queue_depth(self) -> int:
        return len(self.queue)

    @property
    def is_full(self) -> bool:
        return len(self.queue) >= self.max_queue_packets

    @property
    def is_up(self) -> bool:
        """Returns True if the link is operational (not currently failed by chaos engine)."""
        if self.chaos_engine:
            return self.chaos_engine.is_link_up(self.src, self.dst)
        return True

    def enqueue(self, packet: Packet, current_time: float = 0.0) -> bool:
        """Enqueues packet if capacity allows and link is up, otherwise drops it."""
        if self.chaos_engine and not self.chaos_engine.is_link_up(self.src, self.dst):
            self.total_dropped_packets += 1
            self.total_dropped_bytes += packet.size_bytes
            return False

        if self.is_full:
            self.total_dropped_packets += 1
            self.total_dropped_bytes += packet.size_bytes
            return False

        if self.chaos_engine and self.chaos_engine.should_drop_packet():
            self.total_dropped_packets += 1
            self.total_dropped_bytes += packet.size_bytes
            return False

        packet.arrival_time = current_time  # Time entered THIS link queue
        self.queue.append(packet)
        return True

    def process_transmissions(self, current_time: float, delta_time: float) -> List[Packet]:
        """
        Transmits packets from the queue based on link bandwidth capacity.
        Returns a list of packets that successfully crossed the link in this step.
        """
        delivered: List[Packet] = []
        if not self.queue or delta_time <= 0:
            return delivered

        if self.chaos_engine and not self.chaos_engine.is_link_up(self.src, self.dst):
            return delivered

        # Max bytes that can be transmitted in this time interval based on Mbps capacity
        bytes_capacity = (self.capacity_mbps * 1_000_000 / 8.0) * delta_time

        bytes_processed = 0.0
        while self.queue and bytes_processed < bytes_capacity:
            pkt = self.queue[0]
            if bytes_processed + pkt.size_bytes <= bytes_capacity or bytes_processed == 0:
                pkt = self.queue.pop(0)
                bytes_processed += pkt.size_bytes

                lat_sec = self.chaos_engine.sample_delay(self.latency_sec) if self.chaos_engine else self.latency_sec

                # Per-hop latency = queuing wait time on THIS link + propagation delay
                arr_t = pkt.arrival_time if pkt.arrival_time is not None else current_time
                queueing_delay = max(0.0, current_time - arr_t)
                hop_latency_sec = queueing_delay + lat_sec

                self.latencies_sec.append(hop_latency_sec)
                self.total_tx_bytes += pkt.size_bytes
                self.total_tx_packets += 1

                # Accumulate into end-to-end latency tracker on the packet itself
                pkt.e2e_latency_ms += hop_latency_sec * 1000.0

                pkt.transmission_time = current_time + lat_sec
                delivered.append(pkt)
            else:
                break

        return delivered

    def reset_window_stats(self):
        """Resets telemetry metrics for the next observation window."""
        self.total_tx_bytes = 0
        self.total_tx_packets = 0
        self.total_dropped_bytes = 0
        self.total_dropped_packets = 0
        self.latencies_sec.clear()
