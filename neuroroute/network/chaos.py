import asyncio
import random
from typing import Optional
from neuroroute.network.topology import TopologyManager


class ChaosScheduler:
  """Injects dynamic network faults and latency fluctuations into a topology."""

  def __init__(
      self,
      topology: TopologyManager,
      spike_factor_range: tuple[float, float] = (2.0, 10.0),
  ) -> None:
    self.topology = topology
    self.spike_factor_range = spike_factor_range
    self._running = False
    self._task: Optional[asyncio.Task] = None

  def fail_link(self, src: str, dst: str) -> bool:
    """Disables a specific link."""
    return self.topology.set_link_status(src, dst, active=False)

  def restore_link(self, src: str, dst: str) -> bool:
    """Re-enables a specific link and restores its latency."""
    res = self.topology.set_link_status(src, dst, active=True)
    self.topology.restore_link_latency(src, dst)
    return res

  def spike_latency(
      self, src: str, dst: str, chaos_factor: Optional[float] = None
  ) -> bool:
    """Multiplies link latency by a chaos factor."""
    if chaos_factor is None:
      chaos_factor = random.uniform(*self.spike_factor_range)
    return self.topology.apply_latency_spike(src, dst, chaos_factor)

  def node_dropout(self, node: str) -> None:
    """Disables all incoming and outgoing links for a specific node."""
    if node in self.topology.graph:
      for dst in list(self.topology.graph[node].keys()):
        self.fail_link(node, dst)
      for src in self.topology.graph:
        if node in self.topology.graph[src]:
          self.fail_link(src, node)

  def node_restore(self, node: str) -> None:
    """Restores all incoming and outgoing links for a specific node."""
    if node in self.topology.graph:
      for dst in list(self.topology.graph[node].keys()):
        self.restore_link(node, dst)
      for src in self.topology.graph:
        if node in self.topology.graph[src]:
          self.restore_link(src, node)
.

  def trigger_random_event(self) -> str:
    """Selects a random active link and applies a random chaos event."""
    links = self.topology.get_all_links()
    if not links:
      return "no_links"

    src, dst = random.choice(links)
    event_type = random.choice(["failure", "latency_spike", "restoration"])

    if event_type == "failure":
      self.fail_link(src, dst)
      return f"link_failed:{src}->{dst}"
    elif event_type == "latency_spike":
      factor = random.uniform(*self.spike_factor_range)
      self.spike_latency(src, dst, factor)
      return f"latency_spiked:{src}->{dst}:x{factor:.1f}"
    else:
      self.restore_link(src, dst)
      return f"link_restored:{src}->{dst}"

  async def start(
      self, interval_seconds: float, duration_seconds: Optional[float] = None
  ) -> None:
    """Starts async background loop periodically injecting chaos events."""
    self._running = True
    elapsed = 0.0

    while self._running:
      await asyncio.sleep(interval_seconds)
      self.trigger_random_event()
      elapsed += interval_seconds

      if duration_seconds and elapsed >= duration_seconds:
        break

  def stop(self) -> None:
    """Stops the chaos scheduler loop."""
    self._running = False
    if self._task and not self._task.done():
      self._task.cancel()