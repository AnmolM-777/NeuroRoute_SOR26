import asyncio
import logging
from typing import Dict, Tuple
from rl_sdn_controller.sdn_api.stats_provider import StatsProvider, LinkStats

logger = logging.getLogger(__name__)


class TelemetryCollector:
    """
    Control Plane Telemetry Collector (Layer 1).
    Polls statistics from Layer 2 SDN StatsProvider every telemetry window.
    """
    def __init__(self, stats_provider: StatsProvider, window_duration_sec: float = 0.1):
        self.stats_provider = stats_provider
        self.window_duration_sec = window_duration_sec

    def collect(self) -> Dict[Tuple[str, str], LinkStats]:
        """Synchronous telemetry polling."""
        return self.stats_provider.collect_window_telemetry(self.window_duration_sec)

    async def poll_loop(self, callback, interval_sec: float = 0.1):
        """Asynchronous background telemetry polling loop."""
        while True:
            telemetry = self.collect()
            if callback:
                callback(telemetry)
            await asyncio.sleep(interval_sec)
