import yaml
import networkx as nx
from typing import Dict, List, Tuple, Any, Optional


class NetworkTopology:
    """
    Network Topology representation and path finder using NetworkX.
    Parses nodes and links from YAML configuration files.
    """
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.graph = nx.DiGraph()
        self.nodes_info: Dict[str, Dict[str, Any]] = {}
        self.links_info: Dict[Tuple[str, str], Dict[str, Any]] = {}

        self._load_topology()

    def _load_topology(self):
        if isinstance(self.config_path, dict):
            data = self.config_path
        elif isinstance(self.config_path, str) and ("\n" in self.config_path or "nodes:" in self.config_path):
            data = yaml.safe_load(self.config_path)
        else:
            with open(self.config_path, 'r') as f:
                data = yaml.safe_load(f)

        self.topology_dict = data
        self.nodes_info = data.get("nodes", {})

        for node_id, attrs in self.nodes_info.items():
            self.graph.add_node(node_id, **attrs)

        for link in data.get("links", []):
            src = link["src"]
            dst = link["dst"]
            cap = float(link.get("capacity", 1000.0))
            lat = float(link.get("latency", 100.0))
            max_q = int(link.get("max_queue_packets", 100))

            self.links_info[(src, dst)] = {
                "capacity": cap,
                "latency": lat,
                "max_queue_packets": max_q
            }
            # Add edge with weight as latency for shortest path computation
            self.graph.add_edge(src, dst, weight=lat, capacity=cap, max_queue_packets=max_q)

    def get_hosts(self) -> List[str]:
        return [node for node, attrs in self.nodes_info.items() if attrs.get("type") == "host"]

    def get_routers(self) -> List[str]:
        return [node for node, attrs in self.nodes_info.items() if attrs.get("type") == "router"]

    def get_candidate_paths(self, src: str, dst: str, k: int = 3) -> List[List[str]]:
        """Finds up to k simple paths between src and dst sorted by latency/weight."""
        paths = []
        try:
            generator = nx.shortest_simple_paths(self.graph, src, dst, weight="weight")
            for _ in range(k):
                paths.append(next(generator))
        except (nx.NetworkXNoPath, StopIteration):
            pass
        return paths

    def get_link_keys(self) -> List[Tuple[str, str]]:
        return list(self.links_info.keys())
