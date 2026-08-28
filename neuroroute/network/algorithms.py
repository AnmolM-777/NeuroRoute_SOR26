import heapq
import random
from typing import Optional

from neuroroute.network.topology import TopologyManager

class DijkstraRouter:
  """Computes shortest latency paths using Dijkstra's algorithm."""

  def __init__(self, topology: TopologyManager) -> None:
    self.topology = topology

  def get_shortest_path(self, source: str, target: str) -> list[str]:
    """Calculates the minimum latency node path from source to target."""
    if (
        source not in self.topology.graph
        or target not in self.topology.graph
    ):
      return []

    if source == target:
      return [source]

    distances: dict[str, float] = {source: 0.0}
    previous: dict[str, Optional[str]] = {source: None}
    pq: list[tuple[float, str]] = [(0.0, source)]
    visited: set[str] = set()

    while pq:
      current_dist, current_node = heapq.heappop(pq)

      if current_node in visited:
        continue
      visited.add(current_node)

      if current_node == target:
        break

      neighbors = self.topology.graph.get(current_node, {})
      for neighbor, metrics in neighbors.items():
        # Skip down/broken links
        if not metrics.get("active", True):
          continue

        latency = metrics.get("latency", float("inf"))
        new_dist = current_dist + latency

        if new_dist < distances.get(neighbor, float("inf")):
          distances[neighbor] = new_dist
          previous[neighbor] = current_node
          heapq.heappush(pq, (new_dist, neighbor))

    # Reconstruct path
    path: list[str] = []
    curr: Optional[str] = target
    while curr is not None:
      path.append(curr)
      curr = previous.get(curr)

    path.reverse()
    return path if (path and path[0] == source) else []

  def get_next_hop(self, current: str, destination: str) -> Optional[str]:
    """Returns immediate next-hop node along the active shortest path.

    Uses precomputed routing tables from TopologyManager when available
    (O(1) lookup). Falls back to live Dijkstra computation otherwise
    (backward compatibility for standalone usage without registered nodes).
    """
    if current == destination:
      return current

    # Fast path: use precomputed tables from centralized TopologyManager
    precomputed = self.topology.get_next_hop(current, destination)
    if precomputed is not None:
      return precomputed

    # Fallback: live compute (for tests and standalone usage)
    path = self.get_shortest_path(current, destination)
    return path[1] if len(path) > 1 else None


class RoundRobinRouter:

  def __init__(self, topology: TopologyManager) -> None:
    self.topology = topology
    self._indices: dict[str, int] = {}

  def get_next_hop(self, current: str, destination: str) -> Optional[str]:
    if current == destination:
      return current

    neighbors = self.topology.get_neighbours(current)
    if not neighbors:
      return None

    idx = self._indices.get(current, 0)
    next_hop = neighbors[idx % len(neighbors)]
    self._indices[current] = idx + 1
    return next_hop


class RandomRouter:

  def __init__(self, topology: TopologyManager) -> None:
    self.topology = topology

  def get_next_hop(self, current: str, destination: str) -> Optional[str]:
    if current == destination:
      return current

    neighbors = self.topology.get_neighbours(current)
    if not neighbors:
      return None

    return random.choice(neighbors)


# Backward-compatible aliases (tests import these names)
Dijkstras = DijkstraRouter
RoundRobin = RoundRobinRouter
Random = RandomRouter