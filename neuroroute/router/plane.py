from __future__ import annotations

import abc
import asyncio
import json
import struct
import time
import uuid
from dataclasses import asdict, dataclass, field
from collections import deque
from enum import IntEnum
from typing import Any, Dict, List, Optional, Tuple


#Errors 

class PacketValidationError(ValueError):
    """Raised when a Packet fails construction or validity constraints."""



class Priority(IntEnum):
    """Standard packet priority levels used across the data plane."""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


# --------------------------------------------------------------------------
# Packet
# --------------------------------------------------------------------------

@dataclass
class Packet:
    """
    Canonical unit of data transferred between router nodes.

    Fields
    ------
    packet_id:      Unique identifier for this packet. Auto-generated
                     (uuid4) if left empty.
    source:         Identifier of the originating node.
    destination:    Identifier of the target node.
    payload:        Raw payload bytes (str payloads are UTF-8 encoded
                     automatically).
    priority:       Integer priority, see `Priority` (0-3).
    ttl:            Time-to-live, decremented on every hop. Must stay > 0.
    creation_time:  Unix timestamp of creation (defaults to time.time()).
    hop_history:    Ordered list of node_ids the packet has traversed.
    """

    packet_id: str
    source: str
    destination: str
    payload: bytes
    priority: int = int(Priority.NORMAL)
    ttl: int = 64
    creation_time: float = field(default_factory=time.time)
    hop_history: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.packet_id:
            self.packet_id = str(uuid.uuid4())

        if isinstance(self.payload, str):
            self.payload = self.payload.encode("utf-8")   # so that regular text can be used as payload

        if not isinstance(self.payload, (bytes, bytearray)):  # verify the data in payload is 
            raise PacketValidationError(
                f"payload must be bytes or str, got {type(self.payload)!r}"
            )

        self.payload = bytes(self.payload) # ensure payload is immutable bytes

        if not isinstance(self.source, str) or not self.source: # verify source is a non-empty string
            raise PacketValidationError("source must be a non-empty string")


        if not isinstance(self.destination, str) or not self.destination:  # verify destination is a non-empty string
            raise PacketValidationError("destination must be a non-empty string")

        if self.source == self.destination: # verify source and destination are not the same
            raise PacketValidationError("source and destination must differ")

        if not isinstance(self.priority, int) or not (0 <= self.priority <= 3): # verify priority is an integer within the range [0, 3]
            raise PacketValidationError(
                "priority must be an int within [0, 3] (see Priority enum)"
            )
        if not isinstance(self.ttl, int) or self.ttl <= 0: # verify ttl is a positive integer
            raise PacketValidationError("ttl must be a positive integer")

        if self.creation_time < 0: # verify creation_time is non-negative
            raise PacketValidationError("creation_time must be non-negative")

        if not isinstance(self.hop_history, list) or not all(
            isinstance(h, str) for h in self.hop_history
        ): # verify hop_history is a list of strings
            raise PacketValidationError("hop_history must be a list of node-id strings")

    # ---- lifecycle helpers -------------------------------------------------

    def record_hop(self, node_id: str) -> "Packet":
        """
        Return a *new* Packet reflecting transit through `node_id`:
        ttl decremented by one and node_id appended to hop_history.

        Immutable-update style keeps packets safe to share across
        concurrently-running coroutines/threads without aliasing bugs.
        """
        if not node_id:
            raise PacketValidationError("node_id must be a non-empty string")
        if self.ttl - 1 <= 0:
            raise PacketValidationError(f"packet {self.packet_id} expired (ttl exhausted)")
        return Packet(
            packet_id=self.packet_id,
            source=self.source,
            destination=self.destination,
            payload=self.payload,
            priority=self.priority,
            ttl=self.ttl - 1,
            creation_time=self.creation_time,
            hop_history=[*self.hop_history, node_id],
        )

    def is_expired(self) -> bool:
        return self.ttl <= 0

    def age_seconds(self, now: Optional[float] = None) -> float:
        return (now if now is not None else time.time()) - self.creation_time

    # ---- serialization -------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe dict representation (payload hex-encoded)."""
        d = asdict(self)
        d["payload"] = self.payload.hex()
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Packet":
        data = dict(data)
        try:
            payload_hex = data.pop("payload")
        except KeyError as exc:
            raise PacketValidationError("serialized packet missing 'payload' field") from exc
        try:
            payload = bytes.fromhex(payload_hex)
        except (TypeError, ValueError) as exc:
            raise PacketValidationError("payload field is not valid hex") from exc
        try:
            return cls(payload=payload, **data)
        except TypeError as exc:
            raise PacketValidationError(f"malformed packet dict: {exc}") from exc

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), separators=(",", ":"), sort_keys=True)

    @classmethod
    def from_json(cls, raw: str) -> "Packet":
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise PacketValidationError(f"invalid packet JSON: {exc}") from exc
        return cls.from_dict(data)

    def to_bytes(self) -> bytes:
        """
        Wire format: 4-byte big-endian length prefix followed by UTF-8 JSON.
        Length-prefixing keeps this compatible with streaming transports
        where message boundaries aren't otherwise preserved.
        """
        body = self.to_json().encode("utf-8")
        header = struct.pack(">I", len(body))
        return header + body

    @classmethod
    def from_bytes(cls, raw: bytes) -> "Packet":
        if len(raw) < 4:
            raise PacketValidationError("byte stream too short to contain a valid packet header")
        (length,) = struct.unpack(">I", raw[:4])
        body = raw[4:4 + length]
        if len(body) != length:
            raise PacketValidationError("byte stream length does not match declared packet length")
        return cls.from_json(body.decode("utf-8"))

    def __repr__(self) -> str:
        return (
            f"Packet(id={self.packet_id[:8]}…, {self.source}->{self.destination}, "
            f"priority={self.priority}, ttl={self.ttl}, hops={len(self.hop_history)})"
        )


# --------------------------------------------------------------------------
# Routing table entries
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class RouteEntry:
    """A single routing table entry: where to send matching packets next."""
    destination_prefix: str
    next_hop: str
    metric: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------
# BaseRouterNode
# --------------------------------------------------------------------------

class BaseRouterNode(abc.ABC):
    """
    Abstract interface every router node in the data plane must implement.

    Concerns owned by this base class:
      * Bounded inbound buffer (`asyncio.Queue`), exposing queue length and
        buffer size for backpressure/telemetry.
      * A routing table mapping destination prefixes -> `RouteEntry`.
      * The `enqueue`/`dequeue` primitives used by the simulation scheduler.

    Concerns left to subclasses:
      * `lookup_route`   — how a destination is resolved (exact match,
                            longest-prefix match, hashing, etc.).
      * `forward`         — the actual forwarding decision/side effect for
                            a dequeued packet (e.g. handing it to a peer
                            node's queue, a NIC simulation, etc.).
    """

    def __init__(self, node_id: str, buffer_size: int = 1024) -> None:
        if not node_id:
            raise ValueError("node_id must be a non-empty string")
        if buffer_size <= 0:
            raise ValueError("buffer_size must be a positive integer")
        self.node_id = node_id
        self._buffer_size = buffer_size
        self._queue: "asyncio.Queue[Packet]" = asyncio.Queue(maxsize=buffer_size)
        self._routing_table: Dict[str, RouteEntry] = {}

    # ---- introspection -------------------------------------------------

    @property
    def buffer_size(self) -> int:
        """Maximum number of packets this node's inbound queue can hold."""
        return self._buffer_size

    @property
    def queue_length(self) -> int:
        """Current number of packets waiting in the inbound queue."""
        return self._queue.qsize()

    @property
    def is_full(self) -> bool:
        return self._queue.full()

    @property
    def is_empty(self) -> bool:
        return self._queue.empty()

    @property
    def routing_table(self) -> Dict[str, RouteEntry]:
        """Read-only snapshot of the routing table."""
        return dict(self._routing_table)

    def _invalidate_cache(self) -> None:
        if hasattr(self, "_route_cache"):
            getattr(self, "_route_cache").clear()

    def add_route(self, destination_prefix: str, next_hop: str, metric: int = 1) -> None:
        if not destination_prefix:
            raise ValueError("destination_prefix must be a non-empty string")
        if not next_hop:
            raise ValueError("next_hop must be a non-empty string")
        self._routing_table[destination_prefix] = RouteEntry(
            destination_prefix=destination_prefix, next_hop=next_hop, metric=metric
        )
        self._invalidate_cache()

    def remove_route(self, destination_prefix: str) -> None:
        self._routing_table.pop(destination_prefix, None)
        self._invalidate_cache()

    @abc.abstractmethod
    def lookup_route(self, destination: str) -> Optional[RouteEntry]:
        """Resolve a destination to a `RouteEntry`, or None if unreachable."""
        raise NotImplementedError

    # ---- queueing --------------------------------------------------------

    async def enqueue(self, packet: Packet) -> bool:
        """
        Attempt to place a packet on the inbound queue without blocking.
        Returns False (drop) if the buffer is full, True if accepted.
        """
        if self._queue.full():
            return False
        await self._queue.put(packet)
        return True

    async def dequeue(self) -> Packet:
        """Await and remove the next packet from the inbound queue."""
        return await self._queue.get()

    # ---- forwarding --------------------------------------------------------

    @abc.abstractmethod
    async def forward(self, packet: Packet) -> Tuple[bool, Optional[str]]:
        """
        Forward `packet` toward its destination.

        Returns
        -------
        (success, next_hop_node_id): `success` indicates whether the
        forwarding action was accepted (e.g. downstream queue had room);
        `next_hop_node_id` is the id of the node the packet was handed to,
        or None if the packet was dropped / had no route / was delivered
        locally.
        """
        raise NotImplementedError

    async def process_one(self) -> Optional[Packet]:
        """
        Convenience driver: dequeue one packet, drop it if expired,
        forward it, and return the hop-updated packet on success or
        None if it was dropped.
        """
        packet = await self.dequeue()
        if packet.is_expired():
            return None
        success, next_hop = await self.forward(packet)
        if not success or next_hop is None:
            return None
        return packet.record_hop(self.node_id)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(node_id={self.node_id!r}, "
            f"queue={self.queue_length}/{self.buffer_size}, "
            f"routes={len(self._routing_table)})"
        )


# --------------------------------------------------------------------------
# SimRouterNode
# --------------------------------------------------------------------------

class SimRouterNode(BaseRouterNode):
    """
    Concrete Router Node used in simulation runs.
    Integrates with TopologyManager, routing strategies, and peer nodes.
    """

    def __init__(
        self,
        node_id: str,
        topology_manager: Any = None,
        strategy: Any = None,
        buffer_size: int = 1024,
        processing_delay: float = 0.0,
    ) -> None:
        super().__init__(node_id, buffer_size=buffer_size)
        self.topology_manager = topology_manager
        self.strategy = strategy
        self.processing_delay = float(processing_delay)
        self.peers: Dict[str, BaseRouterNode] = {}
        self.delivered: List[Packet] = []

        # Route lookup cache for O(1) fast-path routing
        self._route_cache: Dict[str, Optional[RouteEntry]] = {}

        # Telemetry & metrics
        self._drop_count: int = 0
        self._processed_count: int = 0
        self._total_wait_time: float = 0.0

        # Runtime task management
        self._running: bool = False
        self._task: Optional[asyncio.Task] = None

        # Register with TopologyManager for centralized route pushes
        if self.topology_manager is not None and hasattr(self.topology_manager, 'register_node'):
            self.topology_manager.register_node(self)

    def set_peers(self, peers: Dict[str, BaseRouterNode]) -> None:
        """Register reference to all peer router nodes in the network."""
        self.peers = peers

    def receive_routes(self, routes: Dict[str, Optional[str]]) -> None:
        """Receive precomputed routes from TopologyManager and populate cache.

        Called by TopologyManager._push_routes_to_nodes() whenever the
        topology changes. Populates _route_cache with RouteEntry objects
        for O(1) next-hop lookups, eliminating per-packet Dijkstra calls.
        """
        self._route_cache.clear()
        for destination, next_hop in routes.items():
            if next_hop is not None and destination != self.node_id:
                self._route_cache[destination] = RouteEntry(
                    destination_prefix=destination,
                    next_hop=next_hop,
                )

    # ---- Peer Management -----------------------------------------------

    def link_peer(self, peer: "RouterNode", metric: int = 1) -> None:
        """Connect a peer node and add a direct route to it."""
        if not isinstance(peer, BaseRouterNode):
            raise TypeError("peer must be an instance of BaseRouterNode")
        self.peers[peer.node_id] = peer
        self.add_route(peer.node_id, peer.node_id, metric=metric)

    def unlink_peer(self, peer_id: str) -> None:
        """Disconnect a peer node and remove its direct route."""
        self.peers.pop(peer_id, None)
        self.remove_route(peer_id)

    # ---- Route Lookup --------------------------------------------------

    def lookup_route(self, destination: str, prev_hop: Optional[str] = None) -> Optional[RouteEntry]:
        """
        Resolve destination to a RouteEntry.
        Supports cached O(1) lookup, exact destination match, and prefix matching.

        When a dynamic strategy (e.g. Q-learning) is set, the cache is bypassed
        so the strategy is consulted on every routing decision.

        Args:
            destination: Target node to route toward.
            prev_hop: The node this packet just came from (for anti-bounce).
        """
        # Dynamic strategies must be consulted every time — never cache their decisions
        if self.strategy:
            # Pass prev_hop if the strategy supports it (RL strategies do)
            if prev_hop is not None and hasattr(self.strategy, 'get_next_hop'):
                import inspect
                sig = inspect.signature(self.strategy.get_next_hop)
                if 'prev_hop' in sig.parameters:
                    next_hop = self.strategy.get_next_hop(self.node_id, destination, prev_hop=prev_hop)
                else:
                    next_hop = self.strategy.get_next_hop(self.node_id, destination)
            else:
                next_hop = self.strategy.get_next_hop(self.node_id, destination)
            if next_hop:
                return RouteEntry(destination_prefix=destination, next_hop=next_hop)

        # Static routing table: use cache for O(1) fast-path
        if destination in self._route_cache:
            return self._route_cache[destination]

        if destination in self._routing_table:
            res = self._routing_table[destination]
            self._route_cache[destination] = res
            return res

        best_match: Optional[RouteEntry] = None
        longest_len = -1
        for prefix, entry in self._routing_table.items():
            if destination.startswith(prefix) and len(prefix) > longest_len:
                best_match = entry
                longest_len = len(prefix)
        self._route_cache[destination] = best_match
        return best_match

    # ---- Queue & Overflow Management -----------------------------------

    async def enqueue(self, packet: Packet) -> bool:
        """
        Attempt to place a packet on the inbound queue without blocking.
        If the queue is full, the packet is dropped and drop_count is incremented.
        """
        if self._queue.full():
            self._drop_count += 1
            return False
        await self._queue.put((packet, time.time()))
        return True

    async def dequeue(self) -> Packet:
        """
        Await and remove the next packet from the inbound queue.
        Calculates queue wait time and updates metrics.
        """
        item = await self._queue.get()
        if isinstance(item, tuple):
            packet, arrival_time = item
            wait_time = max(0.0, time.time() - arrival_time)
            self._total_wait_time += wait_time
            self._processed_count += 1
            return packet
        return item

    # ---- Forwarding ----------------------------------------------------

    async def forward(self, packet: Packet) -> Tuple[bool, Optional[str]]:
        """Forward packet to destination or next hop peer."""
        if packet.destination == self.node_id:
            if packet not in self.delivered:
                self.delivered.append(packet)
            return True, None

        route = self.lookup_route(packet.destination)
        if route is None:
            self._drop_count += 1
            return False, None

        next_hop = route.next_hop
        if not next_hop or next_hop not in self.peers:
            self._drop_count += 1
            return False, None

        peer_node = self.peers[next_hop]

        try:
            updated_packet = packet.record_hop(self.node_id)
        except PacketValidationError:
            self._drop_count += 1
            return False, None

        success = await peer_node.enqueue(updated_packet)
        if success:
            return True, next_hop
        self._drop_count += 1
        return False, None

    async def run(self, stop_event: asyncio.Event, stats: Dict[str, Any]) -> None:
        """
        Main execution loop for this router node.
        Dequeues incoming packets, applies latency delays, records hops, and forwards.
        Supports continuous learning: feeds routing outcomes back to the strategy.
        """
        _can_learn = self.strategy is not None and hasattr(self.strategy, "record_outcome")

        while not stop_event.is_set() or not self.is_empty:
            try:
                packet = await asyncio.wait_for(self.dequeue(), timeout=0.05)
            except asyncio.TimeoutError:
                continue

            if packet.is_expired():
                stats["packets_dropped"] += 1
                continue

            if packet.destination == self.node_id:
                delivery_time = time.time() - packet.creation_time
                stats["packets_delivered"] += 1
                stats["total_latency"] += delivery_time
                # Reward the previous hop's routing decision that sent the packet here
                if _can_learn and packet.hop_history:
                    prev_node = packet.hop_history[-1]
                    self.strategy.record_outcome(
                        prev_node, packet.destination, self.node_id,
                        delivered=True, dropped=False,
                    )
                continue

            # Use lookup_route() for a single consistent routing decision
            # Extract prev_hop from packet history for anti-bounce logic
            prev_hop = packet.hop_history[-1] if packet.hop_history else None
            route = self.lookup_route(packet.destination, prev_hop=prev_hop)
            if route is None or not route.next_hop:
                stats["packets_dropped"] += 1
                if _can_learn:
                    self.strategy.record_outcome(
                        self.node_id, packet.destination, self.node_id,
                        delivered=False, dropped=True,
                    )
                continue

            next_hop = route.next_hop
            if next_hop not in self.peers:
                stats["packets_dropped"] += 1
                if _can_learn:
                    self.strategy.record_outcome(
                        self.node_id, packet.destination, next_hop,
                        delivered=False, dropped=True,
                    )
                continue

            # Simulate link latency if applicable
            if self.topology_manager and self.topology_manager.is_connected(self.node_id, next_hop):
                try:
                    metrics = self.topology_manager.get_link_metrics(self.node_id, next_hop)
                    latency = float(metrics.get("latency", 0.0))
                    if latency > 0:
                        await asyncio.sleep(latency / 1000.0)
                except KeyError:
                    pass

            # Forward directly to the chosen peer (avoid re-routing via forward/lookup_route)
            peer_node = self.peers[next_hop]
            try:
                updated_packet = packet.record_hop(self.node_id)
            except PacketValidationError:
                stats["packets_dropped"] += 1
                if _can_learn:
                    self.strategy.record_outcome(
                        self.node_id, packet.destination, next_hop,
                        delivered=False, dropped=True,
                    )
                continue

            success = await peer_node.enqueue(updated_packet)
            if not success:
                stats["packets_dropped"] += 1
                if _can_learn:
                    self.strategy.record_outcome(
                        self.node_id, packet.destination, next_hop,
                        delivered=False, dropped=True,
                    )
            else:
                # Successful forward — intermediate hop feedback
                if _can_learn:
                    self.strategy.record_outcome(
                        self.node_id, packet.destination, next_hop,
                        delivered=False, dropped=False,
                    )
    async def _forward_loop(self) -> None:
        """Worker loop running as an asyncio task to process inbound queue."""
        while self._running:
            try:
                packet = await self.dequeue()
            except asyncio.CancelledError:
                break
            except Exception:
                continue

            if self.processing_delay > 0:
                await asyncio.sleep(self.processing_delay)

            if packet.is_expired():
                self._drop_count += 1
                if hasattr(self, "_queue") and hasattr(self._queue, "task_done"):
                    try:
                        self._queue.task_done()
                    except Exception:
                        pass
                continue

            await self.forward(packet)
            if hasattr(self, "_queue") and hasattr(self._queue, "task_done"):
                try:
                    self._queue.task_done()
                except Exception:
                    pass

    def start(self) -> None:
        """Start the background forwarding loop."""
        if not self._running:
            self._running = True
            self._task = asyncio.create_task(self._forward_loop())

    async def stop(self) -> None:
        """Stop the background forwarding loop gracefully."""
        if self._running:
            self._running = False
            if self._task and not self._task.done():
                self._task.cancel()
                try:
                    await self._task
                except asyncio.CancelledError:
                    pass
            self._task = None

    @property
    def drop_count(self) -> int:
        return self._drop_count

    @property
    def processed_count(self) -> int:
        return self._processed_count

    @property
    def average_wait_time(self) -> float:
        if self._processed_count == 0:
            return 0.0
        return self._total_wait_time / self._processed_count

    def get_metrics(self) -> Dict[str, Any]:
        """Return dynamic telemetry metrics dictionary."""
        return {
            "node_id": self.node_id,
            "queue_depth": self.queue_length,
            "buffer_size": self.buffer_size,
            "drop_count": self.drop_count,
            "processed_count": self.processed_count,
            "average_wait_time": self.average_wait_time,
            "delivered_count": len(self.delivered),
        }


RouterNode = SimRouterNode


# --------------------------------------------------------------------------
# FastQueue (Low-Overhead Queue)
# --------------------------------------------------------------------------

class FastQueue:
    """
    Low-overhead bounded queue utilizing collections.deque and an asyncio.Event
    for zero-allocation fast-path put/get operations.
    """

    __slots__ = ("_deque", "_maxsize", "_event")

    def __init__(self, maxsize: int = 1024) -> None:
        if maxsize <= 0:
            raise ValueError("maxsize must be a positive integer")
        self._maxsize: int = maxsize
        self._deque: deque = deque()
        self._event: asyncio.Event = asyncio.Event()

    def qsize(self) -> int:
        return len(self._deque)

    def empty(self) -> bool:
        return len(self._deque) == 0

    def full(self) -> bool:
        return len(self._deque) >= self._maxsize

    def put_nowait(self, item: Any) -> bool:
        """Non-blocking put. Returns False if queue is full."""
        if len(self._deque) >= self._maxsize:
            return False
        self._deque.append(item)
        if not self._event.is_set():
            self._event.set()
        return True

    def get_nowait(self) -> Optional[Any]:
        """Non-blocking get. Returns None if queue is empty."""
        if not self._deque:
            return None
        item = self._deque.popleft()
        if not self._deque:
            self._event.clear()
        return item

    async def get(self) -> Any:
        """Async get. Waits if queue is empty."""
        while not self._deque:
            self._event.clear()
            await self._event.wait()
        item = self._deque.popleft()
        if not self._deque:
            self._event.clear()
        return item


# --------------------------------------------------------------------------
# FastRouterNode (High-Throughput Low-Overhead Router Node)
# --------------------------------------------------------------------------

class FastRouterNode(BaseRouterNode):
    """
    High-throughput router node optimized for per-hop processing latencies < 5 µs.

    Uses `FastQueue` and O(1) cached route lookups to avoid standard
    `asyncio.Queue` Future allocation overheads.
    """

    def __init__(self, node_id: str, buffer_size: int = 1024) -> None:
        super().__init__(node_id, buffer_size=buffer_size)
        self.fast_queue: FastQueue = FastQueue(maxsize=buffer_size)
        self.peers: Dict[str, FastRouterNode] = {}
        self.delivered: List[Packet] = []
        self._route_cache: Dict[str, Optional[RouteEntry]] = {}

        # Telemetry counters
        self._drop_count: int = 0
        self._processed_count: int = 0
        self._running: bool = False
        self._task: Optional[asyncio.Task] = None

    @property
    def queue_length(self) -> int:
        return self.fast_queue.qsize()

    @property
    def is_full(self) -> bool:
        return self.fast_queue.full()

    @property
    def is_empty(self) -> bool:
        return self.fast_queue.empty()

    def link_peer(self, peer: "FastRouterNode", metric: int = 1) -> None:
        if not isinstance(peer, BaseRouterNode):
            raise TypeError("peer must be an instance of BaseRouterNode")
        self.peers[peer.node_id] = peer
        self.add_route(peer.node_id, peer.node_id, metric=metric)

    def unlink_peer(self, peer_id: str) -> None:
        self.peers.pop(peer_id, None)
        self.remove_route(peer_id)

    def lookup_route(self, destination: str) -> Optional[RouteEntry]:
        if destination in self._route_cache:
            return self._route_cache[destination]

        if destination in self._routing_table:
            res = self._routing_table[destination]
            self._route_cache[destination] = res
            return res

        best_match: Optional[RouteEntry] = None
        longest_len = -1
        for prefix, entry in self._routing_table.items():
            if destination.startswith(prefix) and len(prefix) > longest_len:
                best_match = entry
                longest_len = len(prefix)
        self._route_cache[destination] = best_match
        return best_match

    async def enqueue(self, packet: Packet) -> bool:
        accepted = self.fast_queue.put_nowait(packet)
        if not accepted:
            self._drop_count += 1
        return accepted

    async def dequeue(self) -> Packet:
        return await self.fast_queue.get()

    def enqueue_fast(self, packet: Packet) -> bool:
        """Synchronous fast-path non-blocking enqueue."""
        accepted = self.fast_queue.put_nowait(packet)
        if not accepted:
            self._drop_count += 1
        return accepted

    def process_one_fast(self) -> Optional[Packet]:
        """
        Synchronous fast-path processing of one packet (< 5 microseconds per hop).
        Dequeues non-blockingly, checks expiration, and forwards directly.
        """
        packet = self.fast_queue.get_nowait()
        if packet is None:
            return None

        self._processed_count += 1

        if packet.is_expired():
            self._drop_count += 1
            return None

        if packet.destination == self.node_id:
            self.delivered.append(packet)
            return packet

        route = self.lookup_route(packet.destination)
        if route is None:
            self._drop_count += 1
            return None

        peer = self.peers.get(route.next_hop)
        if peer is None:
            self._drop_count += 1
            return None

        try:
            updated_packet = packet.record_hop(self.node_id)
        except PacketValidationError:
            self._drop_count += 1
            return None

        if not peer.enqueue_fast(updated_packet):
            return None

        return updated_packet

    async def forward(self, packet: Packet) -> Tuple[bool, Optional[str]]:
        if packet.destination == self.node_id:
            self.delivered.append(packet)
            return True, None

        route = self.lookup_route(packet.destination)
        if route is None:
            self._drop_count += 1
            return False, None

        peer = self.peers.get(route.next_hop)
        if peer is None:
            self._drop_count += 1
            return False, None

        try:
            updated_packet = packet.record_hop(self.node_id)
        except PacketValidationError:
            self._drop_count += 1
            return False, None

        accepted = peer.enqueue_fast(updated_packet)
        if not accepted:
            return False, route.next_hop

        return True, route.next_hop

    async def _forward_loop(self) -> None:
        while self._running:
            try:
                packet = await self.fast_queue.get()
            except asyncio.CancelledError:
                break
            except Exception:
                continue

            self._processed_count += 1

            if packet.is_expired():
                self._drop_count += 1
                continue

            await self.forward(packet)

    def start(self) -> None:
        if not self._running:
            self._running = True
            self._task = asyncio.create_task(self._forward_loop())

    async def stop(self) -> None:
        if self._running:
            self._running = False
            if self._task and not self._task.done():
                self._task.cancel()
                try:
                    await self._task
                except asyncio.CancelledError:
                    pass
            self._task = None

    @property
    def drop_count(self) -> int:
        return self._drop_count

    @property
    def processed_count(self) -> int:
        return self._processed_count
