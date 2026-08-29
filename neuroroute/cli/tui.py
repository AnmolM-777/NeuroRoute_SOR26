from __future__ import annotations

import asyncio
import random
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import click
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

__version__ = "0.1.0"

console = Console()

AVAILABLE_STRATEGIES = ["static", "round-robin", "random", "qlearning"]


# --------------------------------------------------------------------------
# Non-blocking keypress helper
# --------------------------------------------------------------------------

def check_keypress() -> Optional[str]:
    """Non-blocking keyboard check for interactive strategy toggling.

    Works on Windows (msvcrt) and macOS/Linux (select + tty/termios).
    Returns the lower-cased character pressed, or None if no key is waiting.
    """
    try:
        # Windows path
        import msvcrt
        if msvcrt.kbhit():
            ch = msvcrt.getch()
            try:
                return ch.decode("utf-8").lower()
            except Exception:
                return None
    except ImportError:
        # macOS / Linux path: non-blocking stdin read via select
        import select
        import sys
        import tty
        import termios
        fd = sys.stdin.fileno()
        try:
            old_settings = termios.tcgetattr(fd)
            try:
                tty.setraw(fd)
                ready, _, _ = select.select([sys.stdin], [], [], 0)
                if ready:
                    ch = sys.stdin.read(1)
                    return ch.lower() if ch else None
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        except Exception:
            pass
    return None


# --------------------------------------------------------------------------
# TUI Dashboard State
# --------------------------------------------------------------------------

@dataclass
class TUIState:
    """Maintains real-time telemetry state for the TUI dashboard."""

    nodes: List[str] = field(default_factory=lambda: ["nodeA", "nodeB", "nodeC", "nodeD"])
    links: List[Tuple[str, str, float, float]] = field(
        default_factory=lambda: [
            ("nodeA", "nodeB", 40.0, 100.0),
            ("nodeA", "nodeC", 50.0, 120.0),
            ("nodeB", "nodeC", 20.0, 80.0),
            ("nodeB", "nodeD", 30.0, 150.0),
            ("nodeC", "nodeD", 20.0, 100.0),
        ]
    )
    queue_depths: Dict[str, float] = field(
        default_factory=lambda: {"nodeA": 15.0, "nodeB": 42.0, "nodeC": 8.0, "nodeD": 65.0}
    )
    queue_capacities: Dict[str, int] = field(
        default_factory=lambda: {"nodeA": 100, "nodeB": 100, "nodeC": 100, "nodeD": 100}
    )
    packets_generated: int = 0
    packets_delivered: int = 0
    packets_dropped: int = 0
    total_latency_ms: float = 0.0
    throughput_pps: float = 0.0
    current_strategy: str = "static"
    start_time: float = field(default_factory=time.time)
    event_logs: List[Tuple[str, str]] = field(default_factory=list)
    latency_history: List[float] = field(default_factory=list)

    # Live simulation object bindings
    router_nodes: Optional[Dict[str, Any]] = None
    topology_manager: Optional[Any] = None
    stats_ref: Optional[Dict[str, Any]] = None

    def add_log(self, message: str) -> None:
        timestamp = time.strftime("%H:%M:%S")
        self.event_logs.append((timestamp, message))
        if len(self.event_logs) > 50:
            self.event_logs.pop(0)

    @property
    def delivery_rate(self) -> float:
        if self.packets_generated == 0:
            return 100.0
        return (self.packets_delivered / self.packets_generated) * 100.0

    @property
    def average_latency(self) -> float:
        if self.packets_delivered == 0:
            return 0.0
        return self.total_latency_ms / self.packets_delivered

    @property
    def runtime(self) -> float:
        return max(0.1, time.time() - self.start_time)

    def update_from_simulation(self) -> None:
        """Safely fetch live telemetry from router nodes and simulation stats."""
        if self.router_nodes:
            for node_id, node in self.router_nodes.items():
                self.queue_depths[node_id] = float(node.queue_length)
                self.queue_capacities[node_id] = int(node.buffer_size)

        if self.stats_ref:
            self.packets_generated = self.stats_ref.get("packets_generated", self.packets_generated)
            self.packets_delivered = self.stats_ref.get("packets_delivered", self.packets_delivered)
            self.packets_dropped = self.stats_ref.get("packets_dropped", self.packets_dropped)
            raw_lat = self.stats_ref.get("total_latency", 0.0)
            # Store in ms
            self.total_latency_ms = raw_lat * 1000.0 if raw_lat < 1000.0 else raw_lat

        if self.topology_manager:
            new_links = []
            for src, neighbors in self.topology_manager.graph.items():
                for dst, metrics in neighbors.items():
                    lat = float(metrics.get("latency", 0.0))
                    bw = float(metrics.get("bandwidth", 0.0))
                    active = metrics.get("active", True)
                    new_links.append((src, dst, lat, bw, active))
            self.links = new_links

        self.throughput_pps = self.packets_delivered / self.runtime

        avg_lat = self.average_latency
        self.latency_history.append(avg_lat)
        if len(self.latency_history) > 30:
            self.latency_history.pop(0)


def cycle_strategy(state: TUIState) -> str:
    """Cycle active strategy dynamically during runtime."""
    curr = state.current_strategy.lower().replace("-", "").replace("_", "")
    idx = 0
    for i, strat in enumerate(AVAILABLE_STRATEGIES):
        if strat.replace("-", "") == curr:
            idx = i
            break

    next_idx = (idx + 1) % len(AVAILABLE_STRATEGIES)
    new_strategy_name = AVAILABLE_STRATEGIES[next_idx]
    state.current_strategy = new_strategy_name

    if state.router_nodes and state.topology_manager:
        from neuroroute.cli.simulate import get_strategy
        new_strat_obj = get_strategy(new_strategy_name, state.topology_manager)
        for node in state.router_nodes.values():
            node.strategy = new_strat_obj
            # Clear cached routes so the new strategy takes effect immediately
            if hasattr(node, "_route_cache"):
                node._route_cache.clear()

    state.add_log(f"[bold yellow]Switched routing strategy to: {new_strategy_name.upper()}[/bold yellow]")
    return new_strategy_name


# --------------------------------------------------------------------------
# Render Component Functions
# --------------------------------------------------------------------------

def make_header_panel(state: Optional[TUIState] = None) -> Panel:
    """Render top header banner with strategy & live status."""
    strategy_str = state.current_strategy.upper() if state else "STATIC"
    grid = Table.grid(expand=True)
    grid.add_column(justify="left", ratio=1)
    grid.add_column(justify="right", ratio=1)
    grid.add_row(
        f"[bold green]NeuroRoute Simulator Dashboard[/bold green] | Strategy: [bold yellow]{strategy_str}[/bold yellow]",
        f"[bold cyan]v{__version__}[/bold cyan] | [bold green]LIVE (Press 'S' to toggle)[/bold green]",
    )
    return Panel(grid, style="bold white on blue")


def make_topology_panel(state: TUIState) -> Panel:
    """Render Network Topology panel with color-coded link latencies."""
    table = Table(expand=True, box=None, show_header=True, header_style="bold magenta")
    table.add_column("From", style="cyan")
    table.add_column("To", style="cyan")
    table.add_column("Latency", justify="right")
    table.add_column("Bandwidth", justify="right", style="green")
    table.add_column("Status", justify="center")

    for link_data in state.links:
        if len(link_data) == 5:
            src, dst, lat, bw, active = link_data
        else:
            src, dst, lat, bw = link_data
            active = True

        if not active:
            lat_str = "[bold red]--[/bold red]"
            status_str = "[bold red blink]DOWN[/bold red blink]"
            bw_str = "[bold red]--[/bold red]"
            table.add_row(src, dst, lat_str, bw_str, status_str)
            continue

        bw_str = f"{bw:.0f} Mbps"
        if lat < 30.0:
            lat_str = f"[bold green]{lat:.1f} ms[/bold green]"
            status_str = "[bold green]OPTIMAL[/bold green]"
        elif lat < 70.0:
            lat_str = f"[bold yellow]{lat:.1f} ms[/bold yellow]"
            status_str = "[bold yellow]DEGRADED[/bold yellow]"
        else:
            lat_str = f"[bold red]{lat:.1f} ms[/bold red]"
            status_str = "[bold red]CRITICAL[/bold red]"

        table.add_row(src, dst, lat_str, bw_str, status_str)

    return Panel(table, title="[bold green]Network Topology & Link Status[/bold green]", border_style="cyan")


def make_metrics_panel(state: TUIState) -> Panel:
    """Render Performance Metrics panel."""
    table = Table(expand=True, box=None, show_header=False)
    table.add_column("Metric", style="bold cyan")
    table.add_column("Value", justify="right", style="bold white")

    table.add_row("Throughput", f"{state.throughput_pps:.1f} pkts/sec")
    table.add_row("Delivery Rate", f"{state.delivery_rate:.1f}%")
    table.add_row("Packets Generated", str(state.packets_generated))
    table.add_row("Packets Delivered", str(state.packets_delivered))
    table.add_row("Packets Dropped", f"[bold red]{state.packets_dropped}[/bold red]")
    table.add_row("Average Latency", f"{state.average_latency:.2f} ms")
    table.add_row("Elapsed Time", f"{state.runtime:.1f} s")

    return Panel(table, title="[bold green]Performance Metrics[/bold green]", border_style="green")


def make_queue_panel(state: TUIState) -> Panel:
    """Render Queue Buffers fill level panel."""
    table = Table(expand=True, box=None, show_header=False)
    table.add_column("Node", style="bold cyan", width=8)
    table.add_column("Buffer Fill", ratio=1)
    table.add_column("Pct", justify="right", width=6)

    for node in state.nodes:
        depth = state.queue_depths.get(node, 0.0)
        cap = state.queue_capacities.get(node, 100)
        pct = min(100.0, max(0.0, (depth / float(cap)) * 100.0))

        bar_len = 15
        filled_len = int((pct / 100.0) * bar_len)
        empty_len = bar_len - filled_len

        if pct < 60:
            color = "green"
        elif pct < 85:
            color = "yellow"
        else:
            color = "red"

        bar_str = f"[{color}]" + "█" * filled_len + "░" * empty_len + f"[/{color}]"
        table.add_row(node, bar_str, f"[{color}]{pct:.0f}%[/{color}]")

    return Panel(table, title="[bold green]Queue Buffers[/bold green]", border_style="yellow")


def make_latency_chart_panel(state: TUIState) -> Panel:
    """Render rolling average delivery latency sparkline chart."""
    history = state.latency_history
    if not history or max(history) == 0:
        chart_str = "[dim]No latency trend data recorded...[/dim]"
    else:
        blocks = [" ", "▂", "▃", "▄", "▅", "▆", "▇", "█"]
        max_val = max(history)
        min_val = min(history)
        val_range = max(0.001, max_val - min_val)

        bar_chars = []
        for val in history:
            norm = (val - min_val) / val_range
            idx = min(len(blocks) - 1, int(norm * (len(blocks) - 1)))
            if val > 60:
                color = "red"
            elif val > 30:
                color = "yellow"
            else:
                color = "green"
            bar_chars.append(f"[{color}]{blocks[idx]}[/{color}]")

        chart_str = "".join(bar_chars) + f" [bold white]{state.average_latency:.1f} ms[/bold white]"

    return Panel(Text.from_markup(chart_str), title="[bold green]Delivery Latency Trend (Sparkline)[/bold green]", border_style="blue")


def make_event_log_panel(state: TUIState, max_entries: int = 8) -> Panel:
    """Render recent Event Log panel."""
    logs_to_show = state.event_logs[-max_entries:] if state.event_logs else [("", "System initialized.")]
    lines = []
    for ts, msg in logs_to_show:
        if ts:
            lines.append(f"[dim]{ts}[/dim] [white]{msg}[/white]")
        else:
            lines.append(f"[white]{msg}[/white]")

    log_text = Text.from_markup("\n".join(lines))
    return Panel(log_text, title="[bold green]Event Log[/bold green]", border_style="magenta")


# --------------------------------------------------------------------------
# Dashboard Layout Construction
# --------------------------------------------------------------------------

def make_dashboard_layout(state: TUIState, console_height: int = 24) -> Layout:
    """Construct top-level multi-panel dashboard layout."""
    layout = Layout()

    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="body", ratio=1),
    )

    layout["body"].split_row(
        Layout(name="left", ratio=1),
        Layout(name="right", ratio=1),
    )

    layout["left"].split_column(
        Layout(name="topology", ratio=1),
        Layout(name="metrics", ratio=1),
    )

    layout["right"].split_column(
        Layout(name="queues", ratio=1),
        Layout(name="chart", size=3),
        Layout(name="logs", ratio=1),
    )

    layout["header"].update(make_header_panel(state))
    layout["topology"].update(make_topology_panel(state))
    layout["metrics"].update(make_metrics_panel(state))
    layout["queues"].update(make_queue_panel(state))
    layout["chart"].update(make_latency_chart_panel(state))

    max_logs = max(3, (console_height - 15) // 3)
    layout["logs"].update(make_event_log_panel(state, max_entries=max_logs))

    return layout


# --------------------------------------------------------------------------
# Telemetry Loop & Live Execution
# --------------------------------------------------------------------------

async def run_live_tui(
    state: TUIState, stop_event: asyncio.Event, refresh_rate: float = 0.1
) -> None:
    """
    Decoupled TUI rendering loop running at fixed framerate (10 FPS).
    Reads metrics safely from routing tasks and updates Live display.
    """
    with Live(
        make_dashboard_layout(state, console.height),
        console=console,
        refresh_per_second=int(1 / refresh_rate),
        screen=True,
    ) as live:
        while not stop_event.is_set():
            # Check keypress for interactive strategy toggle ('s' key)
            key = check_keypress()
            if key == "s":
                cycle_strategy(state)

            state.update_from_simulation()
            layout = make_dashboard_layout(state, console.height)
            live.update(layout)
            await asyncio.sleep(refresh_rate)


async def mock_telemetry_loop(
    state: TUIState, refresh_rate: float = 0.25, duration: Optional[int] = None
) -> None:
    """Simulates dynamic network telemetry updates for standalone demonstration."""
    state.add_log("Simulator TUI started.")
    state.add_log("Loaded topology: 4 nodes, 5 bidirectional links.")

    start_time = time.time()
    step_count = 0

    while True:
        if duration and (time.time() - start_time) >= duration:
            break

        step_count += 1
        new_pkts = random.randint(3, 8)
        delivered_pkts = random.randint(2, new_pkts)
        dropped_pkts = new_pkts - delivered_pkts if random.random() < 0.15 else 0

        state.packets_generated += new_pkts
        state.packets_delivered += delivered_pkts
        state.packets_dropped += dropped_pkts
        state.total_latency_ms += delivered_pkts * random.uniform(15.0, 45.0)
        state.throughput_pps = state.packets_delivered / state.runtime

        for node in state.nodes:
            current = state.queue_depths[node]
            delta = random.uniform(-10.0, 12.0)
            state.queue_depths[node] = min(95.0, max(5.0, current + delta))

        state.latency_history.append(state.average_latency)
        if len(state.latency_history) > 30:
            state.latency_history.pop(0)

        if step_count % 3 == 0:
            src = random.choice(state.nodes)
            dst = random.choice([n for n in state.nodes if n != src])
            state.add_log(f"Routed batch {new_pkts} pkts ({src} -> {dst})")

        if dropped_pkts > 0:
            congested_node = random.choice(state.nodes)
            state.add_log(f"[bold red]Buffer overflow at {congested_node}! Dropped {dropped_pkts} pkts.[/bold red]")

        await asyncio.sleep(refresh_rate)


async def run_tui(refresh_rate: float = 0.25, duration: Optional[int] = None) -> None:
    """Run standalone mock TUI dashboard."""
    state = TUIState()
    telemetry_task = asyncio.create_task(
        mock_telemetry_loop(state, refresh_rate=refresh_rate, duration=duration)
    )

    try:
        with Live(
            make_dashboard_layout(state, console.height),
            console=console,
            refresh_per_second=int(1 / refresh_rate),
            screen=True,
        ) as live:
            while not telemetry_task.done():
                key = check_keypress()
                if key == "s":
                    cycle_strategy(state)
                layout = make_dashboard_layout(state, console.height)
                live.update(layout)
                await asyncio.sleep(refresh_rate)
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        telemetry_task.cancel()
        console.print("[bold yellow]TUI Dashboard exited cleanly.[/bold yellow]")


# --------------------------------------------------------------------------
# CLI Entrypoint
# --------------------------------------------------------------------------

@click.command(name="tui")
@click.option(
    "-r",
    "--refresh-rate",
    "refresh_rate",
    type=float,
    default=0.25,
    show_default=True,
    help="Dashboard refresh interval in seconds.",
)
@click.option(
    "-d",
    "--duration",
    "duration",
    type=int,
    default=None,
    help="Duration in seconds to run dashboard (default: run until Ctrl+C).",
)
def main(refresh_rate: float, duration: Optional[int]) -> None:
    """Launch the interactive NeuroRoute TUI dashboard."""
    try:
        asyncio.run(run_tui(refresh_rate=refresh_rate, duration=duration))
    except KeyboardInterrupt:
        console.print("[bold yellow]\nDashboard stopped by user.[/bold yellow]")


if __name__ == "__main__":
    main()
