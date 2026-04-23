"""
Environment generation: sensor subnets, independent nodes, UAVs, multiple base stations,
and IoT communication graph H.
"""
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple


# ═══════════════════════════════════════════════════════════════════════════
# Data classes
# ═══════════════════════════════════════════════════════════════════════════
@dataclass
class SensorNode:
    id: int
    x: float
    y: float
    subnet_id: int                      # -1 for independent nodes
    e_max: float
    e_current: float
    is_independent: bool = False        # True for S^ind nodes
    buffer_capacity: int = 20           # C_i
    # ── packet tracking (TTL-aware) ──────────────────────────────────
    packet_queue: list = field(default_factory=list)
    packets_collected: int = 0
    packets_generated: int = 0
    packets_expired: int = 0
    last_collected_time: float = 0.0
    last_collected_round: int = -1
    is_convergence: bool = False
    # ── buffer for multi-hop forwarding ──────────────────────────────
    forward_buffer: list = field(default_factory=list)


@dataclass
class Subnet:
    id: int
    nodes: List[SensorNode] = field(default_factory=list)
    convergence_node_id: int = -1
    initial_convergence_node_id: int = -1
    centroid: Optional[np.ndarray] = None

    def compute_centroid(self):
        if not self.nodes:
            self.centroid = np.array([0.0, 0.0])
        else:
            self.centroid = np.array(
                [[n.x, n.y] for n in self.nodes]
            ).mean(axis=0)
        return self.centroid


@dataclass
class UAV:
    id: int
    x: float
    y: float
    z: float
    e_max: float
    e_current: float
    cluster_id: int = -1
    role: str = "solo"          # "sender" | "receiver" | "solo" | "terminal" | "relay"
    partner_id: int = -1
    assigned_bs_id: int = 0     # which BS this UAV is assigned to
    path: list = field(default_factory=list)
    speed: float = 10.0
    total_packets_delivered: int = 0
    total_distance: float = 0.0
    relay_buffer: int = 0
    data_buffer: int = 0
    collected_sensor_ids: list = field(default_factory=list)


@dataclass
class BaseStation:
    id: int = 0
    x: float = 100.0
    y: float = 100.0
    z: float = 50.0
    num_chargers: int = 2
    num_assigned_uavs: int = 0
    strength: float = 0.0              # Ψ_l (BS strength)
    charging_queue: list = field(default_factory=list)
    packets_received: int = 0
    zone_nodes: list = field(default_factory=list)  # gateway IDs in this BS zone


# ═══════════════════════════════════════════════════════════════════════════
# IoT Communication Graph Edge
# ═══════════════════════════════════════════════════════════════════════════
@dataclass
class IoTEdge:
    node_a_id: int
    node_b_id: int
    distance: float
    weight: float           # w^II — 0 for same-subnet, radio model otherwise
    same_subnet: bool


# ═══════════════════════════════════════════════════════════════════════════
# Environment
# ═══════════════════════════════════════════════════════════════════════════
class Environment:
    def __init__(self, config):
        self.config = config
        self.rng = np.random.RandomState(config.seed)
        self.sensor_nodes: List[SensorNode] = []
        self.subnets: List[Subnet] = []
        self.uavs: List[UAV] = []
        self.base_stations: List[BaseStation] = []
        self.iot_graph: Dict[int, List[IoTEdge]] = {}  # adjacency list for H
        self.independent_nodes: List[SensorNode] = []
        self.current_time = 0.0
        self.current_round = 0
        self.total_packets_generated = 0
        self.total_packets_delivered = 0
        self.total_packets_expired = 0

        # Node classification results (set by Phase 1)
        self.node_classes: Dict[int, str] = {}  # node_id -> 'P'/'C'/'I'

    @property
    def base_station(self):
        """Backward-compatible: return first BS."""
        return self.base_stations[0] if self.base_stations else None

    # ── public API ────────────────────────────────────────────────────────
    def generate(self):
        """Create the full environment."""
        self._generate_base_stations()
        self._generate_subnets()
        self._generate_independent_nodes()
        self._generate_uavs()
        self._build_iot_graph()
        return self

    def generate_data(self, round_num):
        """Each sensor produces one data-unit this round."""
        for node in self.sensor_nodes:
            if len(node.packet_queue) >= node.buffer_capacity:
                continue  # buffer full
            n_new = int(self.config.data_gen_rate)
            for _ in range(n_new):
                node.packet_queue.append(round_num)
            node.packets_generated += n_new
            self.total_packets_generated += n_new

    def expire_stale_packets(self, current_round):
        """Remove packets older than delta_T rounds."""
        expired = 0
        delta = self.config.delta_T
        for node in self.sensor_nodes:
            before = len(node.packet_queue)
            node.packet_queue = [
                r for r in node.packet_queue
                if current_round - r <= delta
            ]
            n_exp = before - len(node.packet_queue)
            node.packets_expired += n_exp
            expired += n_exp
        self.total_packets_expired += expired
        return expired

    def get_all_convergence_nodes(self):
        conv = []
        for sub in self.subnets:
            if sub.convergence_node_id >= 0:
                node = next(
                    (n for n in sub.nodes if n.id == sub.convergence_node_id),
                    None,
                )
                if node is not None:
                    conv.append(node)
        return conv

    def nearest_bs(self, x, y):
        """Return the BaseStation closest to (x, y)."""
        best, bd = self.base_stations[0], float("inf")
        for bs in self.base_stations:
            d = np.hypot(x - bs.x, y - bs.y)
            if d < bd:
                bd, best = d, bs
        return best

    # ── internals ─────────────────────────────────────────────────────────
    def _generate_base_stations(self):
        cfg = self.config
        for i, (bx, by, bz) in enumerate(cfg.base_station_positions):
            n_chargers = cfg.n_chargers_per_bs[i] if i < len(cfg.n_chargers_per_bs) else 2
            self.base_stations.append(
                BaseStation(id=i, x=bx, y=by, z=bz, num_chargers=n_chargers)
            )

    def _generate_subnets(self):
        cfg = self.config
        node_id = 0
        margin = 100
        centres = []

        for _ in range(cfg.num_subnets):
            for _attempt in range(200):
                cx = self.rng.uniform(margin, cfg.area_width - margin)
                cy = self.rng.uniform(margin, cfg.area_height - margin)
                if all(
                    np.hypot(cx - sc[0], cy - sc[1]) > 150 for sc in centres
                ):
                    break
            centres.append((cx, cy))

        for sid, (cx, cy) in enumerate(centres):
            n_nodes = self.rng.randint(
                cfg.nodes_per_subnet_min, cfg.nodes_per_subnet_max + 1
            )
            sub = Subnet(id=sid)
            for _ in range(n_nodes):
                nx = np.clip(cx + self.rng.normal(0, 30), 0, cfg.area_width)
                ny = np.clip(cy + self.rng.normal(0, 30), 0, cfg.area_height)
                node = SensorNode(
                    id=node_id, x=nx, y=ny, subnet_id=sid,
                    e_max=cfg.sensor_e_max, e_current=cfg.sensor_e_max,
                    is_independent=False,
                    buffer_capacity=cfg.sensor_buffer_capacity,
                )
                sub.nodes.append(node)
                self.sensor_nodes.append(node)
                node_id += 1
            sub.compute_centroid()
            self.subnets.append(sub)

    def _generate_independent_nodes(self):
        """Scatter S^ind independent nodes across the area."""
        cfg = self.config
        node_id = len(self.sensor_nodes)
        for _ in range(cfg.num_independent_nodes):
            nx = self.rng.uniform(50, cfg.area_width - 50)
            ny = self.rng.uniform(50, cfg.area_height - 50)
            node = SensorNode(
                id=node_id, x=nx, y=ny, subnet_id=-1,
                e_max=cfg.sensor_e_max, e_current=cfg.sensor_e_max,
                is_independent=True,
                buffer_capacity=cfg.sensor_buffer_capacity,
            )
            self.sensor_nodes.append(node)
            self.independent_nodes.append(node)
            node_id += 1

    def _generate_uavs(self):
        cfg = self.config
        # Start UAVs at nearest BS (round-robin initially)
        for i in range(cfg.num_uavs):
            bs = self.base_stations[i % len(self.base_stations)]
            self.uavs.append(
                UAV(
                    id=i,
                    x=bs.x, y=bs.y, z=cfg.uav_altitude,
                    e_max=cfg.E_n_max, e_current=cfg.E_n_max,
                    speed=cfg.uav_speed,
                    assigned_bs_id=bs.id,
                )
            )

    def _build_iot_graph(self):
        """Build IoT communication graph H with edge weights w^II.

        Eq 3-5: weight is 0 for same-subnet edges, first-order radio model otherwise.
        """
        cfg = self.config
        self.iot_graph = {n.id: [] for n in self.sensor_nodes}

        nodes = self.sensor_nodes
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                na, nb = nodes[i], nodes[j]
                d = np.hypot(na.x - nb.x, na.y - nb.y)
                if d > cfg.d_II_max:
                    continue

                same_sub = (na.subnet_id == nb.subnet_id
                            and na.subnet_id >= 0
                            and not na.is_independent
                            and not nb.is_independent)

                if same_sub:
                    w = 0.0  # free intra-subnet communication
                else:
                    # First-order radio energy model (Eq 3-5)
                    bits = cfg.packet_size_bits
                    d_cross = np.sqrt(cfg.eps_fs / cfg.eps_mp) if cfg.eps_mp > 0 else 1e6
                    e_tx = cfg.E_ele * bits
                    if d < d_cross:
                        e_tx += cfg.eps_fs * bits * d ** 2
                    else:
                        e_tx += cfg.eps_mp * bits * d ** 4
                    e_rx = cfg.E_ele * bits
                    w = e_tx + e_rx

                edge_ab = IoTEdge(na.id, nb.id, d, w, same_sub)
                edge_ba = IoTEdge(nb.id, na.id, d, w, same_sub)
                self.iot_graph[na.id].append(edge_ab)
                self.iot_graph[nb.id].append(edge_ba)
