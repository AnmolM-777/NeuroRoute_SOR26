from dataclasses import dataclass
from typing import Dict, List, Tuple, Any
import numpy as np


@dataclass
class LinkStats:
    src: str
    dst: str
    capacity_mbps: float
    tx_bytes: int
    tx_packets: int
    dropped_bytes: int
    dropped_packets: int
    queue_depth: int
    max_queue_packets: int
    utilization_pct: float
    drop_rate_pct: float
    avg_latency_ms: float
    is_up: bool = True


class StatsProvider:
    """
    Unified Statistics API (Layer 2 -> Layer 1 Telemetry Interface).
    Pulls per-link metrics from simulated LinkQueues or OpenFlow switches.
    """
    def __init__(self, links_dict: Dict[Tuple[str, str], Any]):
        self.links = links_dict

    def collect_window_telemetry(self, window_duration_sec: float) -> Dict[Tuple[str, str], LinkStats]:
        """
        Calculates link utilization %, drop %, queue depth, and latency
        over the monitoring window.
        """
        stats_dict = {}
        for (src, dst), link_q in self.links.items():
            tx_b = link_q.total_tx_bytes
            tx_p = link_q.total_tx_packets
            drop_b = link_q.total_dropped_bytes
            drop_p = link_q.total_dropped_packets
            q_depth = link_q.queue_depth
            max_q = link_q.max_queue_packets
            is_up = getattr(link_q, 'is_up', True)

            # Utilization % = (transmitted bits / capacity bits) * 100
            cap_bits = (link_q.capacity_mbps * 1_000_000.0) * window_duration_sec
            tx_bits = tx_b * 8.0
            utilization = (tx_bits / cap_bits * 100.0) if cap_bits > 0 else 0.0
            utilization = min(100.0, utilization)

            # Drop rate %
            total_arrived = tx_p + drop_p
            drop_rate = (drop_p / total_arrived * 100.0) if total_arrived > 0 else 0.0

            # Avg Latency ms
            avg_lat_ms = (float(np.nanmean(link_q.latencies_sec)) * 1000.0) if link_q.latencies_sec else 0.0
            if np.isnan(avg_lat_ms):
                avg_lat_ms = 0.0

            stats_dict[(src, dst)] = LinkStats(
                src=src,
                dst=dst,
                capacity_mbps=link_q.capacity_mbps,
                tx_bytes=tx_b,
                tx_packets=tx_p,
                dropped_bytes=drop_b,
                dropped_packets=drop_p,
                queue_depth=q_depth,
                max_queue_packets=max_q,
                utilization_pct=utilization,
                drop_rate_pct=drop_rate,
                avg_latency_ms=avg_lat_ms,
                is_up=is_up
            )

        return stats_dict
