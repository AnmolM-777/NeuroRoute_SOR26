import random
import math
from typing import List, Dict, Any, Optional
from rl_sdn_controller.data_plane.packet import Packet


class BaseTrafficGenerator:
    """Base class for synthetic traffic flow generators."""
    def __init__(self, flow_config: Dict[str, Any]):
        self.flow_id = flow_config.get("id", flow_config.get("flow_id", "flow_default"))
        self.src = flow_config["src"]
        self.dst = flow_config["dst"]

        self.pattern = flow_config.get("pattern", flow_config.get("type", "poisson"))
        self.rate_pps = flow_config.get("rate_pps", 100)
        self.packet_size_bytes = flow_config.get("packet_size_bytes", 1000)
        self.duration_sec = flow_config.get("duration_sec", 60.0)
        self.seq_counter = 0
        self.next_arrival_time = 0.0

    def reset(self):
        self.seq_counter = 0
        self.next_arrival_time = 0.0

    def generate_packets(self, current_time: float, delta_time: float) -> List[Packet]:
        raise NotImplementedError


class PoissonTrafficGenerator(BaseTrafficGenerator):
    """Generates standard Poisson process arrivals (exponential interarrival times)."""
    def _sample_interval(self) -> float:
        if self.rate_pps <= 0:
            return 1.0
        return random.expovariate(self.rate_pps)

    def generate_packets(self, current_time: float, delta_time: float) -> List[Packet]:
        packets: List[Packet] = []
        if current_time > self.duration_sec:
            return packets

        end_time = current_time + delta_time
        if self.next_arrival_time < current_time:
            self.next_arrival_time = current_time + self._sample_interval()

        while self.next_arrival_time < end_time:
            self.seq_counter += 1
            pkt = Packet(
                flow_id=self.flow_id,
                src=self.src,
                dst=self.dst,
                size_bytes=self.packet_size_bytes,
                creation_time=self.next_arrival_time,
                seq_id=self.seq_counter
            )
            packets.append(pkt)
            self.next_arrival_time += self._sample_interval()

        return packets


class BurstyTrafficGenerator(BaseTrafficGenerator):
    """Generates ON/OFF bursty traffic typical of video streaming or HTTP chunk downloads."""
    def __init__(self, flow_config: Dict[str, Any]):
        super().__init__(flow_config)
        self.burst_on_sec = flow_config.get("burst_on_ms", 500) / 1000.0
        self.burst_off_sec = flow_config.get("burst_off_ms", 1000) / 1000.0
        self.is_burst_on = True
        self.last_burst_switch_time = 0.0

    def reset(self):
        super().reset()
        self.is_burst_on = True
        self.last_burst_switch_time = 0.0

    def generate_packets(self, current_time: float, delta_time: float) -> List[Packet]:
        packets: List[Packet] = []
        if current_time > self.duration_sec:
            return packets

        # Update ON/OFF burst state
        time_in_state = current_time - self.last_burst_switch_time
        if self.is_burst_on and time_in_state >= self.burst_on_sec:
            self.is_burst_on = False
            self.last_burst_switch_time = current_time
        elif not self.is_burst_on and time_in_state >= self.burst_off_sec:
            self.is_burst_on = True
            self.last_burst_switch_time = current_time

        if not self.is_burst_on:
            return packets

        end_time = current_time + delta_time
        interval = (1.0 / self.rate_pps) if self.rate_pps > 0 else 1.0
        if self.next_arrival_time < current_time:
            self.next_arrival_time = current_time + interval

        while self.next_arrival_time < end_time:
            self.seq_counter += 1
            pkt = Packet(
                flow_id=self.flow_id,
                src=self.src,
                dst=self.dst,
                size_bytes=self.packet_size_bytes,
                creation_time=self.next_arrival_time,
                seq_id=self.seq_counter
            )
            packets.append(pkt)
            self.next_arrival_time += interval

        return packets


class RealtimeTrafficGenerator(BaseTrafficGenerator):
    """
    Generates low-latency realtime traffic (VoIP audio / high-frequency telemetry / trading).
    Characterized by strict periodic arrivals (e.g., 20ms frames) and small payload sizes.
    """
    def __init__(self, flow_config: Dict[str, Any]):
        super().__init__(flow_config)
        self.interval_sec = flow_config.get("interval_ms", 20.0) / 1000.0
        if "packet_size_bytes" not in flow_config:
            self.packet_size_bytes = 200 # Standard VoIP G.711 / Opus frame

    def generate_packets(self, current_time: float, delta_time: float) -> List[Packet]:
        packets: List[Packet] = []
        if current_time > self.duration_sec:
            return packets

        end_time = current_time + delta_time
        if self.next_arrival_time < current_time:
            self.next_arrival_time = current_time + self.interval_sec

        while self.next_arrival_time < end_time:
            self.seq_counter += 1
            # Add small microsecond jitter typical of OS audio subsystem
            jitter = random.gauss(0.0, 0.0005)
            pkt_time = max(current_time, self.next_arrival_time + jitter)

            pkt = Packet(
                flow_id=self.flow_id,
                src=self.src,
                dst=self.dst,
                size_bytes=self.packet_size_bytes,
                creation_time=pkt_time,
                seq_id=self.seq_counter
            )
            packets.append(pkt)
            self.next_arrival_time += self.interval_sec

        return packets


class BulkTransferGenerator(BaseTrafficGenerator):
    """
    Generates high-throughput bulk transfer traffic (FTP / Database backup / Large file transfer).
    Sends dense packet trains (e.g. 1500 byte MTU packets at line rate).
    """
    def __init__(self, flow_config: Dict[str, Any]):
        super().__init__(flow_config)
        if "packet_size_bytes" not in flow_config:
            self.packet_size_bytes = 1500 # Ethernet MTU standard
        if "rate_pps" not in flow_config:
            self.rate_pps = 2000 # ~24 Mbps

    def generate_packets(self, current_time: float, delta_time: float) -> List[Packet]:
        packets: List[Packet] = []
        if current_time > self.duration_sec:
            return packets

        end_time = current_time + delta_time
        interval = (1.0 / self.rate_pps) if self.rate_pps > 0 else 0.0005
        if self.next_arrival_time < current_time:
            self.next_arrival_time = current_time + interval

        while self.next_arrival_time < end_time:
            self.seq_counter += 1
            pkt = Packet(
                flow_id=self.flow_id,
                src=self.src,
                dst=self.dst,
                size_bytes=self.packet_size_bytes,
                creation_time=self.next_arrival_time,
                seq_id=self.seq_counter
            )
            packets.append(pkt)
            self.next_arrival_time += interval

        return packets


class IoTTrafficGenerator(BaseTrafficGenerator):
    """
    Generates IoT telemetry sensor traffic.
    Characterized by small payloads (e.g., 64-128 bytes) emitted at steady frequencies.
    """
    def __init__(self, flow_config: Dict[str, Any]):
        super().__init__(flow_config)
        if "packet_size_bytes" not in flow_config:
            self.packet_size_bytes = 128
        self.interval_sec = flow_config.get("interval_ms", 100.0) / 1000.0 # 10 Hz telemetry

    def generate_packets(self, current_time: float, delta_time: float) -> List[Packet]:
        packets: List[Packet] = []
        if current_time > self.duration_sec:
            return packets

        end_time = current_time + delta_time
        if self.next_arrival_time < current_time:
            self.next_arrival_time = current_time + self.interval_sec

        while self.next_arrival_time < end_time:
            self.seq_counter += 1
            pkt = Packet(
                flow_id=self.flow_id,
                src=self.src,
                dst=self.dst,
                size_bytes=self.packet_size_bytes,
                creation_time=self.next_arrival_time,
                seq_id=self.seq_counter
            )
            packets.append(pkt)
            self.next_arrival_time += self.interval_sec

        return packets


class TrafficFlowGenerator:
    """
    Factory wrapper & backward-compatible TrafficFlowGenerator.
    Dispatches to Poisson, Bursty, Realtime, Bulk, or IoT generators based on configuration pattern.
    """
    def __init__(self, flow_config: Dict[str, Any]):
        self.flow_id = flow_config.get("id", flow_config.get("flow_id", "flow_default"))
        self.src = flow_config["src"]
        self.dst = flow_config["dst"]

        self.pattern = flow_config.get("pattern", flow_config.get("type", "poisson")).lower()
        self.config = flow_config

        if self.pattern in ["realtime", "voip"]:
            self._impl = RealtimeTrafficGenerator(flow_config)
        elif self.pattern in ["bulk", "ftp", "backup"]:
            self._impl = BulkTransferGenerator(flow_config)
        elif self.pattern in ["iot", "sensor", "telemetry"]:
            self._impl = IoTTrafficGenerator(flow_config)
        elif self.pattern in ["bursty", "video"]:
            self._impl = BurstyTrafficGenerator(flow_config)
        else:
            self._impl = PoissonTrafficGenerator(flow_config)

    @property
    def rate_pps(self) -> float:
        return self._impl.rate_pps

    @property
    def packet_size_bytes(self) -> int:
        return self._impl.packet_size_bytes

    @property
    def duration_sec(self) -> float:
        return self._impl.duration_sec

    def reset(self):
        self._impl.reset()

    def generate_packets(self, current_time: float, delta_time: float) -> List[Packet]:
        return self._impl.generate_packets(current_time, delta_time)


class MultiTrafficCoordinator:
    """
    Coordinates and multiplexes concurrent traffic generators across multiple flows each simulation step.
    """
    def __init__(self, flow_configs: List[Dict[str, Any]]):
        self.generators: Dict[str, TrafficFlowGenerator] = {
            cfg["id"]: TrafficFlowGenerator(cfg) for cfg in flow_configs
        }

    def reset(self):
        for gen in self.generators.values():
            gen.reset()

    def generate_all_packets(self, current_time: float, delta_time: float) -> List[Packet]:
        all_pkts: List[Packet] = []
        for gen in self.generators.values():
            all_pkts.extend(gen.generate_packets(current_time, delta_time))
        return all_pkts

