"""
Phase 2 — MSCT (Minimum Spanning Cost Trees) per UAV.

§4 of search.pdf, Algorithm 1:
  Modified Prim's on H rooted at UAV's assigned gateway.
  Prune when tree path cost > κ · c^LoS (Eq 6-7).
  Return tree T_n and effective neighbourhood N_n^eff (Eq 8).
"""
import numpy as np
from typing import Dict, List, Set, Tuple
import heapq


def compute_los_reference_cost(node_x, node_y, uav_x, uav_y, uav_z, cfg):
    """Compute LoS reference cost c^LoS (Eq 6).

    This is the direct IoT-to-UAV communication cost used as baseline.
    """
    d_horiz = np.hypot(node_x - uav_x, node_y - uav_y)
    d_3d = np.sqrt(d_horiz ** 2 + uav_z ** 2)
    # Energy cost for direct upload to UAV
    bits = cfg.packet_size_bits
    d_cross = np.sqrt(cfg.eps_fs / cfg.eps_mp) if cfg.eps_mp > 0 else 1e6
    e_tx = cfg.E_ele * bits
    if d_3d < d_cross:
        e_tx += cfg.eps_fs * bits * d_3d ** 2
    else:
        e_tx += cfg.eps_mp * bits * d_3d ** 4
    return e_tx


def build_msct(iot_graph, root_node_id, uav_pos, cfg, sensor_lut):
    """Build MSCT rooted at root_node_id using modified Prim's (Algorithm 1).

    Parameters
    ----------
    iot_graph : dict {node_id: [IoTEdge, ...]}
    root_node_id : int
    uav_pos : (x, y, z) — UAV position
    cfg : SimConfig
    sensor_lut : dict {node_id: SensorNode}

    Returns
    -------
    tree_edges : list of (parent_id, child_id, weight)
    neighbourhood : set of node_ids reachable within budget
    """
    if root_node_id not in iot_graph:
        return [], {root_node_id}

    uav_x, uav_y, uav_z = uav_pos

    # Reference cost for the root
    root_node = sensor_lut.get(root_node_id)
    if root_node is None:
        return [], {root_node_id}

    c_los = compute_los_reference_cost(
        root_node.x, root_node.y, uav_x, uav_y, uav_z, cfg
    )
    budget = cfg.kappa_msct * max(c_los, 1e-10)

    # Modified Prim's algorithm
    visited = {root_node_id}
    tree_edges = []
    path_costs = {root_node_id: 0.0}  # cumulative cost from root

    # Priority queue: (cost_to_reach, node_id, parent_id, edge_weight)
    pq = []
    for edge in iot_graph.get(root_node_id, []):
        heapq.heappush(pq, (edge.weight, edge.node_b_id, root_node_id, edge.weight))

    while pq:
        cum_cost, node_id, parent_id, edge_w = heapq.heappop(pq)

        if node_id in visited:
            continue

        # Prune if cumulative cost exceeds budget (Eq 7)
        if cum_cost > budget:
            continue

        visited.add(node_id)
        path_costs[node_id] = cum_cost
        tree_edges.append((parent_id, node_id, edge_w))

        # Expand neighbours
        for edge in iot_graph.get(node_id, []):
            if edge.node_b_id not in visited:
                new_cost = cum_cost + edge.weight
                heapq.heappush(pq, (new_cost, edge.node_b_id, node_id, edge.weight))

    return tree_edges, visited


def build_all_mscts(env, conv_nodes, uav_assignments):
    """Build MSCT for each UAV based on assigned convergence nodes.

    Parameters
    ----------
    env : Environment
    conv_nodes : list of SensorNode (convergence nodes)
    uav_assignments : dict {uav_id: [convergence_node, ...]}

    Returns
    -------
    msct_results : dict {uav_id: (tree_edges, neighbourhood)}
    """
    cfg = env.config
    sensor_lut = {n.id: n for n in env.sensor_nodes}
    results = {}

    for uav in env.uavs:
        assigned_cns = uav_assignments.get(uav.id, [])
        all_edges = []
        all_neighbourhood = set()

        for cn in assigned_cns:
            uav_pos = (uav.x, uav.y, uav.z)
            edges, nbhood = build_msct(
                env.iot_graph, cn.id, uav_pos, cfg, sensor_lut
            )
            all_edges.extend(edges)
            all_neighbourhood.update(nbhood)

        results[uav.id] = (all_edges, all_neighbourhood)

    return results
