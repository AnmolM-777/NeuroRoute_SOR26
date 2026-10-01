import pytest
from rl_sdn_controller.network.topology import NetworkTopology
from rl_sdn_controller.network.topology_factory import build_router_topology
from rl_sdn_controller.sdn_api.routing_table_api import RoutingTableAPI
from rl_sdn_controller.network.routing_engine import OSPFRoutingEngine, RoundRobinRoutingEngine, RLRoutingEngine


def test_routing_table_api():
    rt_api = RoutingTableAPI()
    rt_api.install_flow_rule("f1", ["h1", "r1", "r2", "h2"])
    
    assert rt_api.get_flow_path("f1") == ["h1", "r1", "r2", "h2"]
    assert rt_api.get_next_hop("r1", "h2") == "r2"
    assert rt_api.get_next_hop("r2", "h2") == "h2"


def test_ospf_routing(tmp_path):
    topo_file = str(tmp_path / "topology.yaml")
    with open(topo_file, "w") as f:
        f.write("""
nodes:
  h1: {type: host}
  h2: {type: host}
  r1: {type: router}
  r2: {type: router}
links:
  - {src: h1, dst: r1, capacity: 100, latency: 10}
  - {src: r1, dst: r2, capacity: 100, latency: 20}
  - {src: r2, dst: h2, capacity: 100, latency: 10}
""")
    topo = NetworkTopology(topo_file)
    rt_api = RoutingTableAPI()
    ospf = OSPFRoutingEngine(topo, rt_api)
    
    flow_src_dst = {"f1": ("h1", "h2")}
    ospf.update_routes(["f1"], flow_src_dst)

    path = rt_api.get_flow_path("f1")
    assert path == ["h1", "r1", "r2", "h2"]


def test_round_robin_routing(tmp_path):
    topo_file = str(tmp_path / "topology.yaml")
    with open(topo_file, "w") as f:
        f.write("""
nodes:
  h1: {type: host}
  h2: {type: host}
  r1: {type: router}
  r2: {type: router}
  r3: {type: router}
links:
  - {src: h1, dst: r1, capacity: 100, latency: 10}
  - {src: r1, dst: r2, capacity: 100, latency: 10}
  - {src: r2, dst: h2, capacity: 100, latency: 10}
  - {src: r1, dst: r3, capacity: 100, latency: 15}
  - {src: r3, dst: h2, capacity: 100, latency: 15}
""")
    topo = NetworkTopology(topo_file)
    rt_api = RoutingTableAPI()
    rr = RoundRobinRoutingEngine(topo, rt_api)
    
    flow_src_dst = {"f1": ("h1", "h2")}
    rr.update_routes(["f1"], flow_src_dst)
    path1 = rt_api.get_flow_path("f1")

    rr.update_routes(["f1"], flow_src_dst)
    path2 = rt_api.get_flow_path("f1")

    assert path1 != path2


@pytest.mark.parametrize("router_count", [5, 9, 23])
def test_generated_topology_routes_all_routers_both_directions(router_count):
    topo = NetworkTopology(build_router_topology(router_count))
    forward = topo.get_candidate_paths("h1", "h2")
    reverse = topo.get_candidate_paths("h2", "h1")

    assert len(topo.get_routers()) == router_count
    assert len(forward) == len(reverse) == 3
    assert {router for path in forward for router in path if router.startswith("r")} == set(topo.get_routers())
    assert all(path[0] == "h1" and path[-1] == "h2" for path in forward)
    assert all(path[0] == "h2" and path[-1] == "h1" for path in reverse)
