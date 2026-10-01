import random
import logging
from typing import Dict, Tuple, Any, List, Optional

logger = logging.getLogger(__name__)


class ChaosEngine:
    """
    Stochastic Network Chaos Engine with Link Failure Detection & Recovery Tracking.
    Simulates physical network disruptions: link flapping, latency jitter,
    random packet loss, latency spikes, and tracks Mean-Time-To-Recovery (MTTR).
    """
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config.get("chaos", {}) if config else {}
        self.enabled = self.config.get("enabled", True)

        self.link_failure_prob = self.config.get("link_failure_probability", 0.05)
        self.repair_time_sec = self.config.get("link_repair_time_sec", 2.0)
        self.jitter_std_sec = self.config.get("jitter_std_us", 20.0) / 1_000_000.0
        self.spike_prob = self.config.get("latency_spike_probability", 0.02)
        self.spike_mult = self.config.get("latency_spike_multiplier", 3.0)
        self.random_drop_rate = self.config.get("random_drop_rate", 0.01)

        # Operational state of links: (src, dst) -> {"is_up": bool, "down_until": float, "failed_at": float}
        self.link_states: Dict[Tuple[str, str], Dict[str, Any]] = {}
        
        # Recovery metrics tracking
        self.failure_history: List[Dict[str, Any]] = []
        self.recovery_times: List[float] = []

    def register_links(self, link_keys: List[Tuple[str, str]]):
        for key in link_keys:
            self.link_states[key] = {"is_up": True, "down_until": 0.0, "failed_at": 0.0}

    def reset(self):
        """Resets link states back to operational at simulation t=0."""
        for state in self.link_states.values():
            state["is_up"] = True
            state["down_until"] = 0.0
            state["failed_at"] = 0.0
        self.failure_history.clear()
        self.recovery_times.clear()

    def is_link_up(self, src: str, dst: str) -> bool:
        if not self.enabled:
            return True
        state = self.link_states.get((src, dst))
        return state["is_up"] if state else True

    def fail_link(self, src: str, dst: str, duration_sec: float, current_time: float = 0.0):
        """Explicitly injects a link outage for deterministic chaos testing."""
        key = (src, dst)
        if key in self.link_states:
            self.link_states[key]["is_up"] = False
            self.link_states[key]["down_until"] = current_time + duration_sec
            self.link_states[key]["failed_at"] = current_time
            self.failure_history.append({
                "link": key,
                "event": "failure",
                "time": current_time,
                "expected_recovery": current_time + duration_sec
            })
            logger.warning(f"💥 [MANUAL FAILURE] Link {src}->{dst} failed at t={current_time:.2f}s for {duration_sec}s")

    def recover_link(self, src: str, dst: str, current_time: float = 0.0):
        """Explicitly recovers a failed link."""
        key = (src, dst)
        if key in self.link_states and not self.link_states[key]["is_up"]:
            outage_duration = current_time - self.link_states[key].get("failed_at", current_time)
            self.link_states[key]["is_up"] = True
            self.recovery_times.append(outage_duration)
            self.failure_history.append({
                "link": key,
                "event": "recovery",
                "time": current_time,
                "outage_duration_sec": outage_duration
            })
            logger.info(f"⚡ [MANUAL RECOVERY] Link {src}->{dst} recovered at t={current_time:.2f}s (outage: {outage_duration:.2f}s)")

    def update(self, current_time: float, delta_time: float):
        """Updates link outage and recovery states."""
        if not self.enabled:
            return

        for key, state in self.link_states.items():
            # Only trigger failure outages on inter-router core links (which have candidate backup paths)
            if key[0].startswith('h') or key[1].startswith('h'):
                continue

            if not state["is_up"]:
                if current_time >= state["down_until"]:
                    state["is_up"] = True
                    outage_dur = current_time - state.get("failed_at", current_time - self.repair_time_sec)
                    self.recovery_times.append(outage_dur)
                    self.failure_history.append({
                        "link": key,
                        "event": "recovery",
                        "time": current_time,
                        "outage_duration_sec": outage_dur
                    })
                    logger.info(f"⚡ [CHAOS RECOVERY] Link {key[0]}->{key[1]} recovered at t={current_time:.2f}s (outage: {outage_dur:.2f}s)")
            else:
                # Random failure check
                if random.random() < (self.link_failure_prob * delta_time):
                    state["is_up"] = False
                    state["down_until"] = current_time + self.repair_time_sec
                    state["failed_at"] = current_time
                    self.failure_history.append({
                        "link": key,
                        "event": "failure",
                        "time": current_time,
                        "expected_recovery": current_time + self.repair_time_sec
                    })
                    logger.warning(f"💥 [CHAOS FAILURE] Link {key[0]}->{key[1]} FAILED at t={current_time:.2f}s for {self.repair_time_sec}s")

    def get_metrics(self) -> Dict[str, Any]:
        """Returns failure statistics and Mean Time To Recovery (MTTR)."""
        active_failures = [k for k, v in self.link_states.items() if not v["is_up"]]
        mttr = float(sum(self.recovery_times) / len(self.recovery_times)) if self.recovery_times else 0.0
        return {
            "total_failure_events": len([e for e in self.failure_history if e["event"] == "failure"]),
            "total_recovery_events": len(self.recovery_times),
            "currently_failed_links": active_failures,
            "mttr_sec": mttr
        }

    def sample_delay(self, base_latency_sec: float) -> float:
        """Applies delay jitter and latency spikes."""
        if not self.enabled:
            return base_latency_sec

        # Apply Gaussian delay jitter
        jitter = random.gauss(0.0, self.jitter_std_sec)
        delay = max(0.000001, base_latency_sec + jitter)

        # Apply latency spike
        if random.random() < self.spike_prob:
            delay *= self.spike_mult

        return delay

    def should_drop_packet(self) -> bool:
        """Determines if a packet is lost due to random physical noise/BER."""
        if not self.enabled:
            return False
        return random.random() < self.random_drop_rate

