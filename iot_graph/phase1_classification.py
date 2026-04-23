"""
Phase 1 — Node Classification (P/C/I labelling).

§3 of search.pdf:
  Eq 5: Classify gateways as Proximal / Contested / Isolated
    P: within d_prox of assigned BS
    I: farther than d_iso from any BS zone boundary
    C: everything else
"""
import numpy as np
from typing import Dict, List, Set, Tuple


def classify_nodes(gateways, zone_assignments, base_stations, cfg,
                   all_sensor_nodes=None, zone_assign_fn=None):
    """Classify nodes as P (proximal), C (contested), I (isolated).

    Classifies gateways first, then propagates labels to all sensor nodes
    in the same subnet.  Independent nodes get their own label based on
    distance to nearest BS.

    Returns
    -------
    (G_P, G_C, G_I) — three lists of gateway nodes
    node_classes — dict {node_id: 'P'/'C'/'I'} for ALL sensor nodes
    """
    bs_lut = {bs.id: bs for bs in base_stations}
    G_P, G_C, G_I = [], [], []
    node_classes = {}

    # Build flat list: (gateway, assigned_bs_id)
    gw_assignments = []
    for bs_id, gw_list in zone_assignments.items():
        for gw in gw_list:
            gw_assignments.append((gw, bs_id))

    # Classify gateways and record per-subnet label
    subnet_class = {}  # subnet_id -> 'P'/'C'/'I'

    for gw, assigned_bs_id in gw_assignments:
        bs = bs_lut[assigned_bs_id]
        d_to_bs = np.hypot(gw.x - bs.x, gw.y - bs.y)

        if d_to_bs <= cfg.d_prox:
            G_P.append(gw)
            node_classes[gw.id] = 'P'
            if gw.subnet_id >= 0:
                subnet_class[gw.subnet_id] = 'P'
            continue

        min_d_other = float("inf")
        for other_bs in base_stations:
            if other_bs.id == assigned_bs_id:
                continue
            d_other = np.hypot(gw.x - other_bs.x, gw.y - other_bs.y)
            min_d_other = min(min_d_other, d_other)

        if min_d_other >= cfg.d_iso:
            G_I.append(gw)
            node_classes[gw.id] = 'I'
            if gw.subnet_id >= 0:
                subnet_class[gw.subnet_id] = 'I'
        else:
            G_C.append(gw)
            node_classes[gw.id] = 'C'
            if gw.subnet_id >= 0:
                subnet_class[gw.subnet_id] = 'C'

    # Propagate gateway label to all sensors in same subnet
    if all_sensor_nodes is not None:
        for n in all_sensor_nodes:
            if n.id in node_classes:
                continue  # already classified (is a gateway)
            if n.subnet_id >= 0 and n.subnet_id in subnet_class:
                node_classes[n.id] = subnet_class[n.subnet_id]
            elif n.is_independent:
                # Classify independent nodes by distance to nearest BS
                min_d = min(np.hypot(n.x - bs.x, n.y - bs.y)
                            for bs in base_stations)
                if min_d <= cfg.d_prox:
                    node_classes[n.id] = 'P'
                elif min_d >= cfg.d_iso:
                    node_classes[n.id] = 'I'
                else:
                    node_classes[n.id] = 'C'

    return G_P, G_C, G_I, node_classes


def select_convergence_nodes(env):
    """Select / update convergence nodes for every subnet.

    Kept from original Phase 1 for backward compatibility.
    """
    cfg = env.config
    safety = cfg.sensor_e_max * cfg.safety_threshold_ratio

    for sub in env.subnets:
        if not sub.nodes:
            continue

        centroid = sub.compute_centroid()
        positions = np.array([[n.x, n.y] for n in sub.nodes])

        if sub.initial_convergence_node_id < 0:
            dists = np.linalg.norm(positions - centroid, axis=1)
            idx = int(np.argmin(dists))
            sub.initial_convergence_node_id = sub.nodes[idx].id
            sub.convergence_node_id = sub.nodes[idx].id
            for n in sub.nodes:
                n.is_convergence = (n.id == sub.nodes[idx].id)
        else:
            cur_cn = next(
                (n for n in sub.nodes if n.id == sub.convergence_node_id),
                None,
            )
            if cur_cn is not None and cur_cn.e_current >= safety:
                for n in sub.nodes:
                    n.is_convergence = (n.id == sub.convergence_node_id)
                continue

            # CN depleted → replace with highest-energy node near centroid
            for n in sub.nodes:
                n.is_convergence = False

            cands = [n for n in sub.nodes if n.e_current >= safety]
            if cands:
                best = min(cands, key=lambda n: np.hypot(n.x - centroid[0],
                                                          n.y - centroid[1]))
            else:
                best = max(sub.nodes, key=lambda n: n.e_current)

            sub.convergence_node_id = best.id
            best.is_convergence = True

    return env.get_all_convergence_nodes()
