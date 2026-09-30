"""Build a connected SDN topology for an arbitrary supported router count."""

from typing import Any, Dict


MIN_ROUTERS = 4
MAX_ROUTERS = 64


def build_router_topology(router_count: int) -> Dict[str, Any]:
    """Create up to three bidirectional core paths between the two hosts."""
    if not MIN_ROUTERS <= router_count <= MAX_ROUTERS:
        raise ValueError(f"Router count must be between {MIN_ROUTERS} and {MAX_ROUTERS}")

    nodes = {
        "h1": {"type": "host", "ip": "10.0.0.1"},
        "h2": {"type": "host", "ip": "10.0.0.2"},
    }
    nodes.update({f"r{index}": {"type": "router"} for index in range(1, router_count + 1)})
    links = []

    def add_bidirectional(src: str, dst: str, capacity: int, latency: int) -> None:
        for start, end in ((src, dst), (dst, src)):
            links.append({
                "src": start, "dst": end, "capacity": capacity,
                "latency": latency, "max_queue_packets": 100,
            })

    ingress, egress = "r1", f"r{router_count}"
    add_bidirectional("h1", ingress, 1000, 5000)
    add_bidirectional(egress, "h2", 1000, 5000)

    path_count = min(3, router_count - 2)
    branches = [[] for _ in range(path_count)]
    for index, router in enumerate(range(2, router_count)):
        branches[index % path_count].append(f"r{router}")

    for index, branch in enumerate(branches):
        route = [ingress, *branch, egress]
        capacity = 15 + index * 5
        latency = round((10000 + index * 200) / (len(route) - 1))
        for src, dst in zip(route, route[1:]):
            add_bidirectional(src, dst, capacity, latency)

    return {"nodes": nodes, "links": links}

