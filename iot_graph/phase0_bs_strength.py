"""
Phase 0 — BS Strength Ranking + Power-Voronoi Pre-Division.

§2 of search.pdf:
  Eq 1: Ψ_l = w_c·(n_c^l / n_c^max) + w_n·(N_l^UAV / N)
  Eq 2-3: Power-Voronoi partition with radii r_l ∝ Ψ_l
  Eq 4: Assign each gateway to nearest BS zone
"""
import numpy as np
from typing import List, Dict, Tuple


def compute_bs_strength(base_stations, num_uavs, cfg):
    """Compute BS strength Ψ_l for each base station (Eq 1).

    Ψ_l = w_c·(n_c^l / n_c^max) + w_n·(N_l^UAV / N)
    """
    n_c_max = max(bs.num_chargers for bs in base_stations) if base_stations else 1
    N = max(num_uavs, 1)

    for bs in base_stations:
        # Initial UAV assignment: round-robin
        bs.num_assigned_uavs = sum(1 for i in range(num_uavs)
                                   if i % len(base_stations) == bs.id)
        bs.strength = (cfg.w_c * (bs.num_chargers / max(n_c_max, 1))
                       + cfg.w_n * (bs.num_assigned_uavs / N))

    return {bs.id: bs.strength for bs in base_stations}


def build_power_voronoi(base_stations, cfg):
    """Build power-Voronoi zones Z_l (Eq 2-3).

    Each point (x,y) is assigned to BS_l that minimises:
        d(x,y, b_l)² - r_l²
    where r_l ∝ Ψ_l.

    Returns a function: assign(x, y) -> bs_id
    and a list of zone boundaries (for plotting).
    """
    if len(base_stations) == 1:
        def assign_single(x, y):
            return base_stations[0].id
        return assign_single, {}

    # Radius proportional to strength
    max_strength = max(bs.strength for bs in base_stations) if base_stations else 1.0
    radii = {}
    for bs in base_stations:
        # Scale radius: stronger BS gets larger zone
        radii[bs.id] = (bs.strength / max(max_strength, 1e-10)) * 200.0  # scale factor

    def assign(x, y):
        best_id, best_cost = base_stations[0].id, float("inf")
        for bs in base_stations:
            d_sq = (x - bs.x) ** 2 + (y - bs.y) ** 2
            cost = d_sq - radii[bs.id] ** 2
            if cost < best_cost:
                best_cost = cost
                best_id = bs.id
        return best_id

    return assign, radii


def assign_gateways_to_zones(gateways, zone_assign_fn, base_stations):
    """Assign each gateway node to its nearest BS zone (Eq 4).

    Parameters
    ----------
    gateways : list of SensorNode (convergence nodes)
    zone_assign_fn : callable(x, y) -> bs_id

    Returns
    -------
    dict: {bs_id: [gateway_node, ...]}
    """
    zones = {bs.id: [] for bs in base_stations}
    for gw in gateways:
        bs_id = zone_assign_fn(gw.x, gw.y)
        zones[bs_id].append(gw)
    return zones


def assign_all_nodes_to_zones(sensor_nodes, zone_assign_fn, base_stations):
    """Assign every sensor node to a BS zone.

    Returns dict: {bs_id: [node_id, ...]}
    """
    zones = {bs.id: [] for bs in base_stations}
    for n in sensor_nodes:
        bs_id = zone_assign_fn(n.x, n.y)
        zones[bs_id].append(n.id)
    return zones
