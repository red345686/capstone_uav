"""
Environment generation: sensor subnets, UAVs, and base station.
"""
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional


# ═══════════════════════════════════════════════════════════════════════════
# Data classes
# ═══════════════════════════════════════════════════════════════════════════
@dataclass
class SensorNode:
    id: int
    x: float
    y: float
    subnet_id: int
    e_max: float
    e_current: float
    # ── packet tracking (TTL‑aware) ──────────────────────────────────
    packet_queue: list = field(default_factory=list)   # generation rounds
    packets_collected: int = 0
    packets_generated: int = 0
    packets_expired: int = 0
    last_collected_time: float = 0.0
    last_collected_round: int = -1
    is_convergence: bool = False


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
    role: str = "solo"          # "sender" | "receiver" | "solo"
    partner_id: int = -1
    path: list = field(default_factory=list)
    speed: float = 10.0
    total_packets_delivered: int = 0
    total_distance: float = 0.0
    relay_buffer: int = 0       # data‑units received via A2A relay
    data_buffer: int = 0        # packets collected, not yet delivered to BS
    collected_sensor_ids: list = field(default_factory=list)  # sensor IDs in buffer


@dataclass
class BaseStation:
    x: float
    y: float
    z: float
    charging_queue: list = field(default_factory=list)
    packets_received: int = 0


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
        self.base_station = BaseStation(config.bs_x, config.bs_y, config.bs_z)
        self.current_time = 0.0
        self.current_round = 0
        self.total_packets_generated = 0
        self.total_packets_delivered = 0
        self.total_packets_expired = 0

    # ── public API ────────────────────────────────────────────────────────
    def generate(self):
        """Create the full environment (subnets + UAVs)."""
        self._generate_subnets()
        self._generate_uavs()
        return self

    def generate_data(self, round_num):
        """Each sensor produces one data‑unit this round (512 KB)."""
        for node in self.sensor_nodes:
            n_new = int(self.config.data_gen_rate)
            for _ in range(n_new):
                node.packet_queue.append(round_num)
            node.packets_generated += n_new
            self.total_packets_generated += n_new

    def expire_stale_packets(self, current_round):
        """Remove packets older than δ_T rounds (TTL expiry).

        Returns the number of expired packets this tick.
        """
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

    # ── internals ─────────────────────────────────────────────────────────
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
                )
                sub.nodes.append(node)
                self.sensor_nodes.append(node)
                node_id += 1
            sub.compute_centroid()
            self.subnets.append(sub)

    def _generate_uavs(self):
        cfg = self.config
        for i in range(cfg.num_uavs):
            self.uavs.append(
                UAV(
                    id=i,
                    x=cfg.bs_x, y=cfg.bs_y, z=cfg.uav_altitude,
                    e_max=cfg.E_n_max, e_current=cfg.E_n_max,
                    speed=cfg.uav_speed,
                )
            )
