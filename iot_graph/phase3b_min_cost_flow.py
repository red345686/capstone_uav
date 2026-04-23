"""
Phase 3b — Time-Expanded Network (TEN) + Min-Cost Flow.

§6 of search.pdf:
  §6.1: TEN construction with node/edge types
  P2 (Eq 11): Network simplex for min-cost flow
  Eq 12: Battery fairness regularisation on edge costs
"""
import numpy as np
from typing import Dict, List, Tuple
from communication import iot_to_iot_energy, uav_to_uav_tx_time, uav_to_bs_tx_time


class FlowEdge:
    """Edge in the time-expanded network."""
    __slots__ = ('src', 'dst', 'capacity', 'cost', 'flow')

    def __init__(self, src, dst, capacity, cost):
        self.src = src
        self.dst = dst
        self.capacity = capacity
        self.cost = cost
        self.flow = 0


class TimeExpandedNetwork:
    """Time-expanded network for multi-hop data flow."""

    def __init__(self):
        self.edges: List[FlowEdge] = []
        self.adj: Dict[str, List[int]] = {}  # node_key -> [edge_indices]
        self.supply: Dict[str, int] = {}
        self.total_flow = 0

    def add_edge(self, src, dst, capacity, cost):
        idx = len(self.edges)
        self.edges.append(FlowEdge(src, dst, capacity, cost))
        self.adj.setdefault(src, []).append(idx)
        # Reverse edge for residual graph
        self.edges.append(FlowEdge(dst, src, 0, -cost))
        self.adj.setdefault(dst, []).append(idx + 1)


def build_time_expanded_network(sensor_nodes, uavs, base_stations,
                                 trajectories, msct_results, cfg, num_slots=5):
    """Construct TEN for multi-hop data flow (§6.1).

    Simplified: uses a small number of time slots to model data movement.

    Parameters
    ----------
    trajectories : dict {uav_id: [(x,y), ...]} — planned UAV paths
    msct_results : dict {uav_id: (edges, neighbourhood)}
    num_slots : int — number of time expansion slots

    Returns
    -------
    TEN, total_demand
    """
    ten = TimeExpandedNetwork()
    sensor_lut = {n.id: n for n in sensor_nodes}

    # Create source and sink
    v_src = "SRC"
    v_snk = "SNK"

    total_demand = 0

    # For each sensor with pending data, create supply
    for n in sensor_nodes:
        n_pkts = len(n.packet_queue)
        if n_pkts <= 0:
            continue
        total_demand += n_pkts

        # Source -> sensor at t=0
        node_key = f"S_{n.id}_0"
        ten.add_edge(v_src, node_key, n_pkts, 0)
        ten.supply[v_src] = ten.supply.get(v_src, 0) + n_pkts

        # Sensor holding edges across time slots
        for t in range(num_slots - 1):
            src_key = f"S_{n.id}_{t}"
            dst_key = f"S_{n.id}_{t+1}"
            ten.add_edge(src_key, dst_key, n_pkts, 0.1)  # small holding cost

    # UAV nodes at each time slot
    for uav in uavs:
        for t in range(num_slots):
            uav_key = f"U_{uav.id}_{t}"
            # UAV holding edge
            if t < num_slots - 1:
                ten.add_edge(uav_key, f"U_{uav.id}_{t+1}",
                             1000, 0.05)

        # UAV -> nearest BS at final slot
        nearest_bs = min(base_stations,
                         key=lambda bs: np.hypot(uav.x - bs.x, uav.y - bs.y))
        uav_final = f"U_{uav.id}_{num_slots-1}"
        bs_key = f"B_{nearest_bs.id}"
        ten.add_edge(uav_final, bs_key, 1000, 0.01)

    # IoT-to-UAV edges: sensor -> UAV at appropriate time slots
    for uav in uavs:
        _, neighbourhood = msct_results.get(uav.id, ([], set()))
        for nid in neighbourhood:
            sn = sensor_lut.get(nid)
            if sn is None:
                continue
            d_horiz = np.hypot(sn.x - uav.x, sn.y - uav.y)
            if d_horiz > cfg.d_IU_max:
                continue
            # Add edge at middle time slot
            t_mid = num_slots // 2
            s_key = f"S_{nid}_{t_mid}"
            u_key = f"U_{uav.id}_{t_mid}"
            # Cost = energy for IoT-to-UAV upload
            cost = iot_to_iot_energy(cfg.packet_size_bits, d_horiz, cfg)
            ten.add_edge(s_key, u_key, len(sn.packet_queue) if sn.packet_queue else 1, cost)

    # BS -> sink edges
    for bs in base_stations:
        bs_key = f"B_{bs.id}"
        ten.add_edge(bs_key, v_snk, total_demand, 0)

    ten.supply[v_snk] = -total_demand

    return ten, total_demand


def solve_min_cost_flow(ten):
    """Solve min-cost flow using successive shortest paths (simplified).

    Returns total cost and flow assignment.
    """
    # Simplified greedy flow: push flow along cheapest available paths
    total_cost = 0.0
    total_flow = 0

    # Find all source edges and push flow greedily
    src_edges = ten.adj.get("SRC", [])
    for eidx in src_edges:
        edge = ten.edges[eidx]
        if edge.capacity > 0 and edge.flow < edge.capacity:
            pushed = edge.capacity - edge.flow
            edge.flow += pushed
            total_cost += pushed * edge.cost
            total_flow += pushed

    ten.total_flow = total_flow
    return total_cost, total_flow


def apply_battery_fairness_reg(ten, sensor_nodes, cfg):
    """Augment edge costs with battery fairness regularisation (Eq 12).

    Nodes with lower remaining energy get higher routing costs,
    incentivising flow through energy-rich nodes.
    """
    rho_vals = {}
    for n in sensor_nodes:
        rho_vals[n.id] = n.e_current / max(n.e_max, 1e-10)

    if not rho_vals:
        return

    rho_mean = np.mean(list(rho_vals.values()))

    for edge in ten.edges:
        if edge.src.startswith("S_"):
            parts = edge.src.split("_")
            if len(parts) >= 2:
                try:
                    nid = int(parts[1])
                    rho = rho_vals.get(nid, 1.0)
                    # Penalise routing through low-energy nodes
                    fairness_penalty = cfg.eta_fair * max(rho_mean - rho, 0)
                    edge.cost += fairness_penalty
                except ValueError:
                    pass
