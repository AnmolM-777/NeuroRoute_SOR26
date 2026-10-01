from typing import Dict, List, Tuple
from rl_sdn_controller.network.topology import NetworkTopology
from rl_sdn_controller.sdn_api.routing_table_api import RoutingTableAPI


class OSPFRoutingEngine:
    """
    Static OSPF Baseline Routing Engine.
    Uses Dijkstra shortest path algorithm based on physical link latencies.
    """
    def __init__(self, topology: NetworkTopology, routing_table_api: RoutingTableAPI):
        self.topology = topology
        self.routing_table_api = routing_table_api

    def update_routes(self, flow_ids: List[str], flow_src_dst: Dict[str, Tuple[str, str]]):
        for flow_id in flow_ids:
            src, dst = flow_src_dst[flow_id]
            paths = self.topology.get_candidate_paths(src, dst, k=1)
            if paths:
                self.routing_table_api.install_flow_rule(flow_id, paths[0])


class RoundRobinRoutingEngine:
    """
    Static Round-Robin (ECMP / Load Balancing) Baseline Routing Engine.
    Alternates paths across available candidate routes for each flow invocation.
    """
    def __init__(self, topology: NetworkTopology, routing_table_api: RoutingTableAPI):
        self.topology = topology
        self.routing_table_api = routing_table_api
        self.flow_path_counters: Dict[str, int] = {}

    def update_routes(self, flow_ids: List[str], flow_src_dst: Dict[str, Tuple[str, str]]):
        for flow_id in flow_ids:
            src, dst = flow_src_dst[flow_id]
            paths = self.topology.get_candidate_paths(src, dst, k=3)
            if paths:
                idx = self.flow_path_counters.get(flow_id, 0) % len(paths)
                selected_path = paths[idx]
                self.flow_path_counters[flow_id] = idx + 1
                self.routing_table_api.install_flow_rule(flow_id, selected_path)


class RLRoutingEngine:
    """
    RL Dynamic Routing Engine.
    Translates RL action indices into path choices across candidate network paths.
    """
    def __init__(self, topology: NetworkTopology, routing_table_api: RoutingTableAPI, flow_ids: List[str], flow_src_dst: Dict[str, Tuple[str, str]]):
        self.topology = topology
        self.routing_table_api = routing_table_api
        self.flow_ids = flow_ids
        self.flow_src_dst = flow_src_dst

        # Precompute candidate paths for each flow
        self.candidate_paths: Dict[str, List[List[str]]] = {}
        for flow_id in self.flow_ids:
            src, dst = self.flow_src_dst[flow_id]
            paths = self.topology.get_candidate_paths(src, dst, k=3)
            self.candidate_paths[flow_id] = paths if paths else [[src, dst]]

    def apply_action(self, action: int):
        """
        Maps discrete action index to path selection combinations across flows.
        For example: for 2 primary flows each with 2 candidate paths,
        action 0 -> (Path0, Path0), action 1 -> (Path0, Path1), action 2 -> (Path1, Path0), action 3 -> (Path1, Path1).
        """
        temp_action = action
        for flow_id in self.flow_ids:
            paths = self.candidate_paths[flow_id]
            path_idx = temp_action % len(paths)
            temp_action //= len(paths)

            selected_path = paths[path_idx]
            self.routing_table_api.install_flow_rule(flow_id, selected_path)
