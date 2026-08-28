import heapq
import json
from typing import Any, List, Optional


class TopologyManager:

  def __init__(self) -> None:
    self.graph: dict[str, dict[str, dict[str, Any]]] = {}
    # Precomputed routing tables: {source: {destination: next_hop}}
    self._routing_tables: dict[str, dict[str, Optional[str]]] = {}
    # Registered router nodes that receive route pushes
    self._registered_nodes: List[Any] = []

  # ---- Node Registration (Observer Pattern) ----------------------------

  def register_node(self, node: Any) -> None:
    """Register a router node to receive routing table pushes."""
    if node not in self._registered_nodes:
      self._registered_nodes.append(node)

  def unregister_node(self, node: Any) -> None:
    """Unregister a router node from receiving routing table pushes."""
    try:
      self._registered_nodes.remove(node)
    except ValueError:
      pass

  # ---- Centralized Dijkstra Route Computation --------------------------

  def _dijkstra_next_hops(self, source: str) -> dict[str, Optional[str]]:
    """Run Dijkstra from a single source and return {destination: next_hop}."""
    if source not in self.graph:
      return {}

    distances: dict[str, float] = {source: 0.0}
    previous: dict[str, Optional[str]] = {source: None}
    pq: list[tuple[float, str]] = [(0.0, source)]
    visited: set[str] = set()

    while pq:
      current_dist, current_node = heapq.heappop(pq)

      if current_node in visited:
        continue
      visited.add(current_node)

      neighbors = self.graph.get(current_node, {})
      for neighbor, metrics in neighbors.items():
        if not metrics.get("active", True):
          continue

        latency = metrics.get("latency", float("inf"))
        new_dist = current_dist + latency

        if new_dist < distances.get(neighbor, float("inf")):
          distances[neighbor] = new_dist
          previous[neighbor] = current_node
          heapq.heappush(pq, (new_dist, neighbor))

    # Build next_hop table by tracing back from each destination to source
    next_hops: dict[str, Optional[str]] = {}
    for dest in self.graph:
      if dest == source:
        next_hops[dest] = dest
        continue
      if dest not in previous:
        next_hops[dest] = None
        continue
      # Trace back from dest to find the first hop after source
      curr = dest
      while previous.get(curr) is not None and previous[curr] != source:
        curr = previous[curr]
      if previous.get(curr) == source:
        next_hops[dest] = curr
      else:
        next_hops[dest] = None

    return next_hops

  def recompute_routes(self) -> None:
    """Recompute all-pairs shortest-path next-hop tables and push to nodes.

    Called automatically whenever the topology graph is mutated.
    When no nodes are registered, the tables are still computed for
    direct lookup via get_next_hop().
    """
    self._routing_tables.clear()
    for source in self.graph:
      self._routing_tables[source] = self._dijkstra_next_hops(source)
    self._push_routes_to_nodes()

  def _push_routes_to_nodes(self) -> None:
    """Push precomputed routes into each registered router node's cache."""
    for node in self._registered_nodes:
      node_id = getattr(node, "node_id", None)
      if node_id and node_id in self._routing_tables:
        routes = self._routing_tables[node_id]
        if hasattr(node, "receive_routes"):
          node.receive_routes(routes)

  def get_next_hop(self, source: str, destination: str) -> Optional[str]:
    """Look up the precomputed next hop from source toward destination.

    Returns None if no route exists or tables haven't been computed.
    """
    source_table = self._routing_tables.get(source)
    if source_table is None:
      return None
    return source_table.get(destination)

  # ---- Topology Loading ------------------------------------------------

  def load_topology(self, filepath: str) -> None:
    """Loads network topology from a JSON file."""
    with open(filepath, "r") as f:
      data = json.load(f)

    self.graph.clear()
    for node in data.get("nodes", []):
      if node not in self.graph:
        self.graph[node] = {}

    for link in data.get("links", []):
      src = link["from"]
      dst = link["to"]
      latency = float(link.get("latency", 1.0))
      bandwidth = float(link.get("bandwidth", 100.0))
      # Use _add_link_no_recompute to avoid N recomputes during loading
      self._add_link_no_recompute(src, dst, latency=latency, bandwidth=bandwidth)

    # Single recompute after all links are loaded
    self.recompute_routes()

  # ---- Link Management -------------------------------------------------

  def _add_link_no_recompute(
      self,
      start: str,
      end: str,
      latency: float = 1.0,
      bandwidth: float = 100.0,
      active: bool = True,
  ) -> None:
    """Internal: adds a link without triggering route recomputation."""
    if start not in self.graph:
      self.graph[start] = {}
    if end not in self.graph:
      self.graph[end] = {}

    self.graph[start][end] = {
        "latency": latency,
        "base_latency": latency,
        "bandwidth": bandwidth,
        "active": active,
    }

  def add_link(
      self,
      start: str,
      end: str,
      latency: float = 1.0,
      bandwidth: float = 100.0,
      active: bool = True,
  ) -> None:
    """Adds or updates a directed link with active status tracking."""
    self._add_link_no_recompute(start, end, latency=latency, bandwidth=bandwidth, active=active)
    self.recompute_routes()

  def update_link(
      self,
      start: str,
      end: str,
      latency: Optional[float] = None,
      bandwidth: Optional[float] = None,
      active: Optional[bool] = None,
  ) -> None:
    """Updates specific link metrics dynamically."""
    if start in self.graph and end in self.graph[start]:
      link = self.graph[start][end]
      if latency is not None:
        link["latency"] = latency
        link["base_latency"] = latency
      if bandwidth is not None:
        link["bandwidth"] = bandwidth
      if active is not None:
        link["active"] = active
      self.recompute_routes()

  def set_link_status(self, start: str, end: str, active: bool) -> bool:
    """Enables or disables a specific link."""
    if start in self.graph and end in self.graph[start]:
      self.graph[start][end]["active"] = active
      self.recompute_routes()
      return True
    return False

  def apply_latency_spike(
      self, start: str, end: str, chaos_factor: float
  ) -> bool:
    """Multiplies link latency by a chaos factor."""
    if start in self.graph and end in self.graph[start]:
      link = self.graph[start][end]
      link["latency"] = link["base_latency"] * chaos_factor
      self.recompute_routes()
      return True
    return False

  def restore_link_latency(self, start: str, end: str) -> bool:
    """Restores link latency back to its base value."""
    if start in self.graph and end in self.graph[start]:
      link = self.graph[start][end]
      link["latency"] = link["base_latency"]
      self.recompute_routes()
      return True
    return False

  # ---- Query Methods ---------------------------------------------------

  def get_neighbours(self, node: str) -> list[str]:
    """Returns adjacent neighbor nodes connected via active links."""
    if node not in self.graph:
      return []
    return [
        neighbor
        for neighbor, metrics in self.graph[node].items()
        if metrics.get("active", True)
    ]

  def get_all_links(self) -> list[tuple[str, str]]:
    """Returns list of all directed link tuples (src, dst)."""
    links = []
    for src, neighbors in self.graph.items():
      for dst in neighbors:
        links.append((src, dst))
    return links

  def get_all_nodes(self) -> list[str]:
    """Returns list of all node IDs present in the topology."""
    return list(self.graph.keys())

  def is_connected(self, start: str, end: str) -> bool:
    """Returns True if an active directed link exists from start to end."""
    if start in self.graph and end in self.graph[start]:
      return bool(self.graph[start][end].get("active", True))
    return False

  def get_link_metrics(self, start: str, end: str) -> dict:
    """Returns the full metrics dict for the directed link from start to end.

    Raises KeyError if the link does not exist.
    """
    if start in self.graph and end in self.graph[start]:
      return self.graph[start][end]
    raise KeyError(f"No link from {start!r} to {end!r}")