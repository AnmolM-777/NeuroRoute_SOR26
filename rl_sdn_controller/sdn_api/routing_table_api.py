import threading
from typing import Dict, List, Optional


class RoutingTableAPI:
    """
    SDN Abstraction Layer (Layer 2) Routing Table Manager.
    Exposes unified forwarding table API for both simulated data plane
    and real hardware controllers (OpenFlow / P4).
    """
    def __init__(self):
        self._lock = threading.Lock()
        # Mapping: flow_id -> list of node IDs forming the end-to-end path
        self._flow_paths: Dict[str, List[str]] = {}
        # Mapping: (node_id, dst_host) -> egress_next_hop_node
        self._node_next_hops: Dict[tuple, str] = {}

    def install_flow_rule(self, flow_id: str, path: List[str]):
        """
        Installs an end-to-end flow routing rule atomically.
        :param flow_id: Identifier for the flow
        :param path: Full node sequence e.g., ['h1', 'r1', 'r2', 'r4', 'h2']
        """
        with self._lock:
            self._flow_paths[flow_id] = list(path)
            # Derive hop-by-hop forwarding rules
            for i in range(len(path) - 1):
                node = path[i]
                next_hop = path[i + 1]
                dst_host = path[-1]
                self._node_next_hops[(node, dst_host)] = next_hop

    def get_flow_path(self, flow_id: str) -> Optional[List[str]]:
        with self._lock:
            return self._flow_paths.get(flow_id)

    def get_next_hop(self, current_node: str, dst_host: str) -> Optional[str]:
        with self._lock:
            return self._node_next_hops.get((current_node, dst_host))

    def clear_rules(self):
        with self._lock:
            self._flow_paths.clear()
            self._node_next_hops.clear()

    def get_all_rules(self) -> Dict[str, List[str]]:
        with self._lock:
            return dict(self._flow_paths)
