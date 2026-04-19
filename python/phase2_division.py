"""
Phase 2 — Mission Area Division (K-Means + Load Balancing).

Cluster convergence nodes into N_A groups, then balance so that
the difference in sensor counts stays within gamma_B.
"""
import numpy as np


# ═══════════════════════════════════════════════════════════════════════════
# K-Means (with k‑means++ init)
# ═══════════════════════════════════════════════════════════════════════════
def _kmeans(positions, k, rng, max_iter=100):
    n = len(positions)
    if n <= k:
        return list(range(n))

    # k‑means++ initialisation
    centroids = [positions[rng.randint(n)].copy()]
    for _ in range(1, k):
        d2 = np.array([
            min(np.sum((p - c) ** 2) for c in centroids) for p in positions
        ])
        probs = d2 / d2.sum()
        centroids.append(positions[rng.choice(n, p=probs)].copy())
    centroids = np.array(centroids)

    labels = np.zeros(n, dtype=int)
    for _ in range(max_iter):
        for i in range(n):
            labels[i] = int(np.argmin(np.linalg.norm(centroids - positions[i], axis=1)))
        new_c = centroids.copy()
        for j in range(k):
            members = positions[labels == j]
            if len(members):
                new_c[j] = members.mean(axis=0)
        if np.allclose(centroids, new_c):
            break
        centroids = new_c

    return labels.tolist()


# ═══════════════════════════════════════════════════════════════════════════
# Load balancing
# ═══════════════════════════════════════════════════════════════════════════
def _balance(clusters, subnets, conv_nodes, cfg):
    sub_lut = {s.id: s for s in subnets}
    cn_lut  = {n.subnet_id: n for n in conv_nodes}

    for _ in range(30):
        counts = {
            cid: sum(len(sub_lut[sid].nodes) for sid in sids if sid in sub_lut)
            for cid, sids in clusters.items()
        }
        if not counts:
            break
        hi = max(counts, key=counts.get)
        lo = min(counts, key=counts.get)
        if counts[hi] - counts[lo] <= cfg.gamma_B:
            break

        # centroid of the "light" cluster
        lo_pos = [
            np.array([cn_lut[sid].x, cn_lut[sid].y])
            for sid in clusters[lo] if sid in cn_lut
        ]
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


# ═══════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════
def divide_area(conv_nodes, subnets, config, rng):
    """Return {cluster_id: [subnet_id, …]} after K‑Means + balancing."""
    k = config.num_uavs
    if not conv_nodes:
        return {i: [] for i in range(k)}
    if len(conv_nodes) <= k:
        clusters = {i: [] for i in range(k)}
        for i, cn in enumerate(conv_nodes):
            clusters[i % k].append(cn.subnet_id)
        return clusters

    positions = np.array([[n.x, n.y] for n in conv_nodes])
    labels = _kmeans(positions, k, rng)

    clusters = {i: [] for i in range(k)}
    for idx, lbl in enumerate(labels):
        clusters[lbl].append(conv_nodes[idx].subnet_id)

    return _balance(clusters, subnets, conv_nodes, config)
