"""
Phase 1 — Convergence Node Selection  (Algorithm 1).

• Initial:  sensor node *closest* to the subnet centroid  (Algorithm 1).
• Update:   when current CN energy < safety threshold, use
            angular-zone replacement  (Eq 33):

            θ_i = arctan((y_i - y_{i+1}) / (x_i - x_{i+1}))
                + arctan((y_i - y_{i-1}) / (x_i - x_{i-1}))

            Search within this angular zone for a replacement node that
            has sufficient remaining energy, selecting the one nearest
            to the depleted CN.
"""
import numpy as np


# ═══════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════
def select_convergence_nodes(env):
    """Select / update convergence nodes for every subnet.  Returns list."""
    cfg = env.config
    safety = cfg.sensor_e_max * cfg.safety_threshold_ratio  # F_R threshold

    for sub in env.subnets:
        if not sub.nodes:
            continue

        centroid = sub.compute_centroid()
        positions = np.array([[n.x, n.y] for n in sub.nodes])

        if sub.initial_convergence_node_id < 0:
            # ── first‑time selection: *closest* to centroid ───────────
            dists = np.linalg.norm(positions - centroid, axis=1)
            idx = int(np.argmin(dists))
            sub.initial_convergence_node_id = sub.nodes[idx].id
            sub.convergence_node_id = sub.nodes[idx].id
            for n in sub.nodes:
                n.is_convergence = (n.id == sub.nodes[idx].id)
        else:
            # ── check whether current CN needs replacement ────────────
            cur_cn = next(
                (n for n in sub.nodes if n.id == sub.convergence_node_id),
                None,
            )
            if cur_cn is not None and cur_cn.e_current >= safety:
                # Current CN is healthy — keep it
                for n in sub.nodes:
                    n.is_convergence = (n.id == sub.convergence_node_id)
                continue

            # CN depleted → use angular‑zone replacement (Eq 33)
            for n in sub.nodes:
                n.is_convergence = False

            replacement = _angular_zone_replacement(
                sub, cur_cn, safety, centroid
            )
            if replacement is not None:
                sub.convergence_node_id = replacement.id
                replacement.is_convergence = True
            else:
                # Absolute fallback: node with most remaining energy
                best = max(sub.nodes, key=lambda n: n.e_current)
                sub.convergence_node_id = best.id
                best.is_convergence = True

    return env.get_all_convergence_nodes()


# ═══════════════════════════════════════════════════════════════════════════
# Angular‑zone replacement  (Eq 33)
# ═══════════════════════════════════════════════════════════════════════════
def _angular_zone_replacement(sub, depleted_cn, safety, centroid):
    """Find a replacement CN within the angular zone of *depleted_cn*.

    1.  Sort all subnet nodes by polar angle w.r.t. centroid.
    2.  Identify the depleted CN and its angular neighbours (i−1, i+1).
    3.  Compute the zone width  θ_i  (Eq 33).
    4.  Search for eligible replacements inside that zone.
    5.  Pick the nearest one to the depleted CN.
    """
    if depleted_cn is None or len(sub.nodes) < 3:
        # Not enough nodes for angular‑zone logic — simple fallback
        cands = [n for n in sub.nodes if n.e_current >= safety]
        if cands:
            return min(
                cands,
                key=lambda n: np.hypot(n.x - centroid[0], n.y - centroid[1]),
            )
        return None

    # Sort by polar angle from centroid
    nodes_sorted = sorted(
        sub.nodes,
        key=lambda n: np.arctan2(n.y - centroid[1], n.x - centroid[0]),
    )

    dep_idx = next(
        (i for i, n in enumerate(nodes_sorted) if n.id == depleted_cn.id),
        None,
    )
    if dep_idx is None:
        return None

    N = len(nodes_sorted)
    ni      = nodes_sorted[dep_idx]
    ni_prev = nodes_sorted[(dep_idx - 1) % N]
    ni_next = nodes_sorted[(dep_idx + 1) % N]

    # ── Eq 33 ─────────────────────────────────────────────────────────
    theta = (
        np.arctan2(ni.y - ni_next.y, ni.x - ni_next.x)
        + np.arctan2(ni.y - ni_prev.y, ni.x - ni_prev.x)
    )
    half_zone = max(abs(theta) / 2.0, 0.1)  # floor at ~5.7° to avoid 0

    # Angular position of depleted CN w.r.t. centroid
    cn_angle = np.arctan2(ni.y - centroid[1], ni.x - centroid[0])

    # Search within zone
    candidates = []
    for n in nodes_sorted:
        if n.id == depleted_cn.id or n.e_current < safety:
            continue
        n_angle = np.arctan2(n.y - centroid[1], n.x - centroid[0])
        diff = (n_angle - cn_angle + np.pi) % (2 * np.pi) - np.pi
        if abs(diff) <= half_zone:
            dist = np.hypot(n.x - ni.x, n.y - ni.y)
            candidates.append((n, dist))

    if candidates:
        candidates.sort(key=lambda x: x[1])
        return candidates[0][0]

    # Zone was too narrow — fall back to any energetic node near centroid
    fallback = [
        n for n in sub.nodes
        if n.id != depleted_cn.id and n.e_current >= safety
    ]
    if fallback:
        return min(
            fallback,
            key=lambda n: np.hypot(n.x - centroid[0], n.y - centroid[1]),
        )
    return None
