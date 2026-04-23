"""
Phase 3a — Gateway Selection + Strength-Weighted Division.

§5 of search.pdf:
  Eq 9: Pick gateway per component minimising max eccentricity
  Algorithm 2: BS-strength-seeded K-means with load balancing (Eq 10)
"""
import numpy as np
from typing import Dict, List


def select_gateways(subnets, sensor_lut):
    """Select gateway (convergence node) per subnet minimising eccentricity.

    For each subnet, the gateway is the node closest to the centroid.
    """
    gateways = []
    for sub in subnets:
        if not sub.nodes:
            continue
        centroid = sub.compute_centroid()
        best = min(sub.nodes, key=lambda n: np.hypot(n.x - centroid[0],
                                                      n.y - centroid[1]))
        gateways.append(best)
    return gateways


def strength_weighted_division(conv_nodes, base_stations, num_uavs, cfg, rng):
    """BS-strength-seeded K-means with load balancing (Algorithm 2, Eq 10).

    Partitions convergence nodes among UAVs using K-means initialised
    with BS positions weighted by strength.

    Returns
    -------
    clusters : dict {uav_id: [subnet_id, ...]}
    """
    k = num_uavs
    if not conv_nodes:
        return {i: [] for i in range(k)}
    if len(conv_nodes) <= k:
        clusters = {i: [] for i in range(k)}
        for i, cn in enumerate(conv_nodes):
            clusters[i % k].append(cn.subnet_id)
        return clusters

    positions = np.array([[n.x, n.y] for n in conv_nodes])

    # Initialise centroids: use BS positions + random perturbation
    centroids = []
    for i in range(k):
        bs = base_stations[i % len(base_stations)]
        # Perturb slightly to break ties
        cx = bs.x + rng.normal(0, 50)
        cy = bs.y + rng.normal(0, 50)
        centroids.append([cx, cy])
    centroids = np.array(centroids)

    # K-means iterations
    labels = np.zeros(len(conv_nodes), dtype=int)
    for _ in range(100):
        for i in range(len(conv_nodes)):
            labels[i] = int(np.argmin(
                np.linalg.norm(centroids - positions[i], axis=1)
            ))
        new_c = centroids.copy()
        for j in range(k):
            members = positions[labels == j]
            if len(members):
                new_c[j] = members.mean(axis=0)
        if np.allclose(centroids, new_c):
            break
        centroids = new_c

    clusters = {i: [] for i in range(k)}
    for idx, lbl in enumerate(labels):
        clusters[lbl].append(conv_nodes[idx].subnet_id)

    # Load balancing
    clusters = _balance(clusters, conv_nodes, cfg)
    return clusters


def _balance(clusters, conv_nodes, cfg):
    """Balance cluster sizes within gamma_B threshold."""
    cn_lut = {n.subnet_id: n for n in conv_nodes}

    for _ in range(30):
        counts = {cid: len(sids) for cid, sids in clusters.items()}
        if not counts:
            break
        hi = max(counts, key=counts.get)
        lo = min(counts, key=counts.get)
        if counts[hi] - counts[lo] <= max(cfg.gamma_B // 5, 2):
            break

        # Move nearest node from heavy to light cluster
        lo_pos = [np.array([cn_lut[sid].x, cn_lut[sid].y])
                  for sid in clusters[lo] if sid in cn_lut]
        if not lo_pos:
            break
        lo_centroid = np.mean(lo_pos, axis=0)

        best_sid, best_d = None, float("inf")
        for sid in clusters[hi]:
            if sid in cn_lut:
                d = np.linalg.norm(
                    np.array([cn_lut[sid].x, cn_lut[sid].y]) - lo_centroid
                )
                if d < best_d:
                    best_d, best_sid = d, sid

        if best_sid is not None:
            clusters[hi].remove(best_sid)
            clusters[lo].append(best_sid)
        else:
            break

    return clusters
