from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.live import Live
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from typing import Dict, Any, List


class TerminalVisualizer:
    """
    Rich Terminal Visualizer for real-time SDN Controller metrics,
    per-link telemetry, and baseline comparison benchmarks.
    """
    def __init__(self):
        self.console = Console()

    def print_simulation_header(self, title: str = "RL-SDN CONTROLLER SIMULATION"):
        panel = Panel(
            f"[bold cyan]{title}[/bold cyan]\n"
            "[dim]Three-Layer RL Control Plane (100ms cycle) | Data Plane Asyncio Packet Engine[/dim]",
            border_style="green",
            expand=False
        )
        self.console.print(panel)

    def print_benchmark_table(self, rl_stats: Dict[str, Any], ospf_stats: Dict[str, Any], rr_stats: Dict[str, Any]):
        """Prints comparative table for RL vs OSPF vs Round-Robin baselines."""
        table = Table(title="PERFORMANCE BENCHMARK COMPARISON", header_style="bold magenta")
        
        table.add_column("Metric", style="bold white", justify="left")
        table.add_column("RL Agent (Proposed)", style="bold green", justify="right")
        table.add_column("Static OSPF (Baseline)", style="bold yellow", justify="right")
        table.add_column("Round-Robin (Baseline)", style="bold cyan", justify="right")

        table.add_row(
            "Throughput (Mbps)",
            f"{rl_stats.get('throughput_mbps', 0.0):.1f}",
            f"{ospf_stats.get('throughput_mbps', 0.0):.1f}",
            f"{rr_stats.get('throughput_mbps', 0.0):.1f}"
        )
        table.add_row(
            "Packet Drop Rate (%)",
            f"{rl_stats.get('drop_rate_pct', 0.0):.2f}%",
            f"{ospf_stats.get('drop_rate_pct', 0.0):.2f}%",
            f"{rr_stats.get('drop_rate_pct', 0.0):.2f}%"
        )
        table.add_row(
            "Average Latency (ms)",
            f"{rl_stats.get('avg_latency_ms', 0.0):.2f} ms",
            f"{ospf_stats.get('avg_latency_ms', 0.0):.2f} ms",
            f"{rr_stats.get('avg_latency_ms', 0.0):.2f} ms"
        )
        table.add_row(
            "Tail Latency P99 (ms)",
            f"{rl_stats.get('p99_latency_ms', 0.0):.2f} ms",
            f"{ospf_stats.get('p99_latency_ms', 0.0):.2f} ms",
            f"{rr_stats.get('p99_latency_ms', 0.0):.2f} ms"
        )

        self.console.print("\n")
        self.console.print(table)

    def print_link_telemetry_table(self, telemetry: Dict[Any, Any]):
        """Prints detailed per-link telemetry statistics."""
        table = Table(title="PER-LINK TELEMETRY METRICS", header_style="bold blue")
        
        table.add_column("Link (Src -> Dst)", style="bold white")
        table.add_column("Capacity", justify="right")
        table.add_column("Utilization (%)", justify="right")
        table.add_column("Queue Depth", justify="right")
        table.add_column("Drop Rate (%)", justify="right")
        table.add_column("Avg Latency (ms)", justify="right")

        for (src, dst), stats in telemetry.items():
            util_color = "red" if stats.utilization_pct > 85 else ("yellow" if stats.utilization_pct > 60 else "green")
            drop_color = "red" if stats.drop_rate_pct > 0.5 else "green"

            table.add_row(
                f"{src} -> {dst}",
                f"{stats.capacity_mbps:.0f} Mbps",
                f"[{util_color}]{stats.utilization_pct:.1f}%[/{util_color}]",
                f"{stats.queue_depth} / {stats.max_queue_packets}",
                f"[{drop_color}]{stats.drop_rate_pct:.2f}%[/{drop_color}]",
                f"{stats.avg_latency_ms:.2f} ms"
            )

        self.console.print(table)
