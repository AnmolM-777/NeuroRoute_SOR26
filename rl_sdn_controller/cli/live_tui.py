"""Rich dashboard for live telemetry from the current SDN simulator."""

from __future__ import annotations

import logging
import time
from collections import deque
from typing import Any, Dict, Mapping, Optional, Tuple

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


LinkKey = Tuple[str, str]


class LiveTelemetryDashboard:
    """Render real link statistics while a training or evaluation loop runs."""

    def __init__(self, console: Optional[Console] = None, refresh_interval: float = 0.12):
        self.console = console or Console()
        self.refresh_interval = refresh_interval
        self.phase = "Starting"
        self.policy = ""
        self.episode = 0
        self.total_episodes = 0
        self.step = 0
        self.total_steps = 0
        self.sim_time_sec = 0.0
        self.telemetry: Dict[LinkKey, Any] = {}
        self.throughput_mbps = 0.0
        self.drop_rate_pct = 0.0
        self.avg_latency_ms = 0.0
        self.reward: Optional[float] = None
        self.delivered_packets: Optional[int] = None
        self.dropped_packets: Optional[int] = None
        self.latency_history = deque(maxlen=36)
        self.events = deque(maxlen=30)
        self._link_status: Dict[LinkKey, bool] = {}
        self._last_refresh = 0.0
        self._live: Optional[Live] = None
        self._chaos_logger = logging.getLogger("rl_sdn_controller.data_plane.chaos")
        self._old_chaos_level: Optional[int] = None

    def __enter__(self) -> "LiveTelemetryDashboard":
        self._old_chaos_level = self._chaos_logger.level
        # Link transitions are shown in the event panel instead of interrupting Live.
        self._chaos_logger.setLevel(logging.ERROR)
        self._live = Live(
            self.render(),
            console=self.console,
            screen=self.console.is_terminal,
            auto_refresh=False,
            transient=self.console.is_terminal,
        )
        self._live.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        try:
            if self._live is not None:
                self._refresh(force=True)
                self._live.__exit__(exc_type, exc_value, traceback)
        finally:
            if self._old_chaos_level is not None:
                self._chaos_logger.setLevel(self._old_chaos_level)
            self._live = None

    def set_phase(self, phase: str, policy: str = "", total_episodes: int = 0, total_steps: int = 0) -> None:
        self.phase = phase
        self.policy = policy
        self.episode = 0
        self.total_episodes = total_episodes
        self.step = 0
        self.total_steps = total_steps
        self.sim_time_sec = 0.0
        self.telemetry = {}
        self.throughput_mbps = 0.0
        self.drop_rate_pct = 0.0
        self.avg_latency_ms = 0.0
        self.reward = None
        self.delivered_packets = None
        self.dropped_packets = None
        self.latency_history.clear()
        self._link_status.clear()
        self.events.clear()
        self.events.append(f"{phase}: {policy}" if policy else phase)
        self._refresh(force=True)

    def update(
        self,
        telemetry: Mapping[LinkKey, Any],
        *,
        sim_time_sec: float,
        episode: int = 0,
        step: int = 0,
        throughput_mbps: Optional[float] = None,
        drop_rate_pct: Optional[float] = None,
        avg_latency_ms: Optional[float] = None,
        reward: Optional[float] = None,
        delivered_packets: Optional[int] = None,
        dropped_packets: Optional[int] = None,
    ) -> None:
        if episode and episode != self.episode:
            self._link_status.clear()
            self.events.append(f"Episode {episode} started")
        self.telemetry = dict(telemetry)
        self.sim_time_sec = sim_time_sec
        self.episode = episode
        self.step = step
        if throughput_mbps is not None:
            self.throughput_mbps = throughput_mbps
        if drop_rate_pct is not None:
            self.drop_rate_pct = drop_rate_pct
        if avg_latency_ms is not None:
            self.avg_latency_ms = avg_latency_ms
            self.latency_history.append(avg_latency_ms)
        self.reward = reward
        self.delivered_packets = delivered_packets
        self.dropped_packets = dropped_packets

        for key, stats in self.telemetry.items():
            is_up = bool(stats.is_up)
            previous = self._link_status.get(key)
            if (previous is not None and previous != is_up) or (previous is None and not is_up):
                event = "RECOVERED" if is_up else "FAILED"
                self.events.append(f"t={sim_time_sec:.2f}s  {key[0]} → {key[1]} {event}")
            self._link_status[key] = is_up

        self._refresh()

    def _refresh(self, force: bool = False) -> None:
        if self._live is None:
            return
        now = time.monotonic()
        if force or now - self._last_refresh >= self.refresh_interval:
            self._live.update(self.render(), refresh=True)
            self._last_refresh = now

    def render(self) -> Layout:
        width = self.console.width
        height = self.console.height
        row_limit = max(4, min(12, height - 15))

        layout = Layout()
        layout.split_column(Layout(self._header(), size=3), Layout(name="body"))
        if width >= 105 and height >= 22:
            layout["body"].split_row(Layout(name="left"), Layout(name="right"))
            layout["left"].split_column(
                Layout(self._links_panel(row_limit), ratio=2),
                Layout(self._metrics_panel(), size=9),
            )
            layout["right"].split_column(
                Layout(self._queues_panel(row_limit), ratio=2),
                Layout(self._events_panel(), size=7),
            )
        else:
            # Combine status and queues so an 80-column terminal still shows live rows.
            compact_limit = max(3, min(12, height - 20))
            layout["body"].split_column(
                Layout(self._metrics_panel(), size=8),
                Layout(self._compact_links_panel(compact_limit), ratio=1),
                Layout(self._events_panel(), size=5),
            )
        return layout

    def _header(self) -> Panel:
        progress = f"Episode {self.episode}/{self.total_episodes}" if self.total_episodes else self.phase
        if self.total_steps:
            progress += f"  ·  Step {self.step}/{self.total_steps}"
        title = f"NeuroRoute  ·  {self.phase}  ·  {self.policy}"
        return Panel(f"[bold cyan]{title}[/bold cyan]    [white]{progress}[/white]    [dim]simulation {self.sim_time_sec:.2f}s[/dim]", border_style="bright_blue")

    def _ordered_links(self):
        return sorted(
            self.telemetry.items(),
            key=lambda item: (
                item[1].is_up,
                -item[1].queue_depth / max(1, item[1].max_queue_packets),
                -item[1].utilization_pct,
                item[0],
            ),
        )

    def _links_panel(self, limit: int) -> Panel:
        table = Table(expand=True, box=None, header_style="bold cyan")
        table.add_column("Directed link")
        table.add_column("Capacity", justify="right")
        table.add_column("Util.", justify="right")
        table.add_column("Latency", justify="right")
        table.add_column("Status", justify="right")
        for (src, dst), stats in self._ordered_links()[:limit]:
            status = "[green]UP[/green]" if stats.is_up else "[bold red]DOWN[/bold red]"
            color = "red" if stats.utilization_pct >= 85 else "yellow" if stats.utilization_pct >= 60 else "green"
            table.add_row(
                f"{src} → {dst}",
                f"{stats.capacity_mbps:g} Mb/s",
                f"[{color}]{stats.utilization_pct:.0f}%[/{color}]",
                f"{stats.avg_latency_ms:.1f} ms",
                status,
            )
        hidden = max(0, len(self.telemetry) - limit)
        if hidden:
            table.add_row(f"[dim]+ {hidden} more links[/dim]", "", "", "", "")
        return Panel(table, title="Topology and link status", border_style="cyan")

    def _compact_links_panel(self, limit: int) -> Panel:
        table = Table(expand=True, box=None, header_style="bold cyan")
        table.add_column("Directed link")
        table.add_column("Status")
        table.add_column("Queue", justify="right")
        table.add_column("Util.", justify="right")
        table.add_column("Drops", justify="right")
        for (src, dst), stats in self._ordered_links()[:limit]:
            status = "[green]UP[/green]" if stats.is_up else "[bold red]DOWN[/bold red]"
            table.add_row(
                f"{src}->{dst}", status,
                f"{stats.queue_depth}/{stats.max_queue_packets}",
                f"{stats.utilization_pct:.0f}%",
                str(stats.dropped_packets),
            )
        hidden = max(0, len(self.telemetry) - limit)
        if hidden:
            table.add_row(f"[dim]+ {hidden} more links[/dim]", "", "", "", "")
        return Panel(table, title="Link status and queues", border_style="cyan")

    def _queues_panel(self, limit: int) -> Panel:
        table = Table(expand=True, box=None, header_style="bold yellow")
        table.add_column("Link queue")
        table.add_column("Fill")
        table.add_column("Depth", justify="right")
        table.add_column("Drops", justify="right")
        for (src, dst), stats in self._ordered_links()[:limit]:
            fraction = min(1.0, stats.queue_depth / max(1, stats.max_queue_packets))
            filled = round(fraction * 10)
            color = "red" if fraction >= 0.85 else "yellow" if fraction >= 0.60 else "green"
            bar = f"[{color}]{'█' * filled}{'░' * (10 - filled)}[/{color}]"
            table.add_row(f"{src} → {dst}", bar, f"{stats.queue_depth}/{stats.max_queue_packets}", str(stats.dropped_packets))
        hidden = max(0, len(self.telemetry) - limit)
        if hidden:
            table.add_row(f"[dim]+ {hidden} more links[/dim]", "", "", "")
        return Panel(table, title="Link queue depth and window drops", border_style="yellow")

    def _metrics_panel(self) -> Panel:
        table = Table.grid(expand=True)
        table.add_column(style="cyan")
        table.add_column(justify="right")
        table.add_row("Link transmissions", f"{self.throughput_mbps:.2f} Mb/s")
        table.add_row("Window drop rate", f"{self.drop_rate_pct:.2f}%")
        table.add_row("Mean link latency", f"{self.avg_latency_ms:.2f} ms")
        if self.reward is not None:
            table.add_row("Step reward", f"{self.reward:.2f}")
        if self.delivered_packets is not None:
            table.add_row("Packets delivered", f"{self.delivered_packets:,}")
        if self.dropped_packets is not None:
            table.add_row("Packets dropped", f"{self.dropped_packets:,}")
        if self.latency_history:
            values = list(self.latency_history)
            scale = max(max(values), 0.001)
            blocks = "▁▂▃▄▅▆▇█"
            sparkline = "".join(blocks[min(7, int(value / scale * 7))] for value in values)
            table.add_row("Latency trend", sparkline)
        return Panel(table, title="Live telemetry", border_style="green")

    def _events_panel(self) -> Panel:
        lines = list(self.events)[-5:] or ["Waiting for telemetry"]
        return Panel(Text("\n".join(lines)), title="Events", border_style="magenta")
