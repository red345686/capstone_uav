"""
Phase 4 — IALNS Path Planning  (Algorithm 4).

Improved Adaptive Large Neighbourhood Search with:
  • Communicationp-oint insertion between paired subareas
  • Destroy operators: random removal, worst-case removal
  • Repair operator:  greedy insertion
  • Metropolis acceptance criterion  P_S = exp(-ΔF / (λ_A·S_A))
  • Speed adjustment (Eq 7) so paired UAVs arrive at P_n simultaneously
  • Clamping & hover delay (Eq 8, 9) when speed exceeds V_max
"""
import numpy as np


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════
def _dist(a, b):
    return np.hypot(a[0] - b[0], a[1] - b[1])


def total_path_distance(path):
    """Cost function F_c = Σ d_{i,i+1}."""
    return sum(_dist(path[i], path[i + 1]) for i in range(len(path) - 1))


def compute_communication_point(conv1, conv2):
    """Midpoint between the two closest convergence nodes of paired areas."""
    if not conv1 or not conv2:
        return None
    best, bd = None, float("inf")
    for n1 in conv1:
        for n2 in conv2:
            d = _dist((n1.x, n1.y), (n2.x, n2.y))
            if d < bd:
                bd = d
                best = ((n1.x + n2.x) / 2, (n1.y + n2.y) / 2)
    return np.array(best) if best else None


# ═══════════════════════════════════════════════════════════════════════════
# Initial solution — nearest‑neighbour
# ═══════════════════════════════════════════════════════════════════════════
def _nn_path(start, waypoints, end):
    if not waypoints:
        return [start, end]
    path, remaining, cur = [start], list(waypoints), start
    while remaining:
        idx = int(np.argmin([_dist(cur, p) for p in remaining]))
        cur = remaining.pop(idx)
        path.append(cur)
    path.append(end)
    return path


# ═══════════════════════════════════════════════════════════════════════════
# Destroy operators
# ═══════════════════════════════════════════════════════════════════════════
def _destroy_random(path, n_rem, rng):
    if len(path) <= 3:
        return list(path), []
    inner = list(range(1, len(path) - 1))
    n_rem = min(n_rem, len(inner))
    idxs = sorted(rng.choice(inner, n_rem, replace=False), reverse=True)
    p, removed = list(path), []
    for i in idxs:
        removed.append(p.pop(i))
    return p, removed


def _destroy_worst(path, n_rem):
    if len(path) <= 3:
        return list(path), []
    p, removed = list(path), []
    for _ in range(min(n_rem, len(p) - 2)):
        if len(p) <= 2:
            break
        savings = []
        for i in range(1, len(p) - 1):
            detour = _dist(p[i - 1], p[i]) + _dist(p[i], p[i + 1])
            direct = _dist(p[i - 1], p[i + 1])
            savings.append(detour - direct)
        wi = int(np.argmax(savings)) + 1
        removed.append(p.pop(wi))
    return p, removed


# ═══════════════════════════════════════════════════════════════════════════
# Repair operator — greedy insertion
# ═══════════════════════════════════════════════════════════════════════════
def _repair_greedy(path, removed):
    for node in removed:
        best_i, best_c = 1, float("inf")
        for i in range(1, len(path)):
            c = (_dist(path[i - 1], node) + _dist(node, path[i])
                 - _dist(path[i - 1], path[i]))
            if c < best_c:
                best_c, best_i = c, i
        path.insert(best_i, node)
    return path


# ═══════════════════════════════════════════════════════════════════════════
# IALNS main loop
# ═══════════════════════════════════════════════════════════════════════════
def ialns_path_planning(uav, conv_nodes, comm_point, base_station, cfg, rng,
                         start_pos=None, end_pos=None):
    """Return optimised path [(x,y), …] for one UAV.

    Parameters
    ----------
    start_pos : tuple or None – starting (x,y); defaults to base station.
    end_pos   : tuple or None – ending (x,y); defaults to base station.
                For senders, set to the communication point so the UAV
                does not waste energy returning to BS (core MGDC benefit).
    """
    bs = (base_station.x, base_station.y)
    start = tuple(start_pos) if start_pos is not None else bs
    end   = tuple(end_pos)   if end_pos   is not None else bs

    wps = [(n.x, n.y) for n in conv_nodes]
    if comm_point is not None:
        cp_tuple = tuple(comm_point)
        # Only add comm_point as an intermediate waypoint if it is NOT
        # already the start or end of the path.
        if (_dist(cp_tuple, start) > 5.0 and _dist(cp_tuple, end) > 5.0):
            wps.append(cp_tuple)

    if not wps:
        return [start, end]

    path = _nn_path(start, wps, end)
    best_path = list(path)
    best_cost = total_path_distance(path)
    cur_cost = best_cost
    temp = cfg.ialns_temp_init
    d_weights = [1.0, 1.0]      # random, worst

    for _ in range(cfg.ialns_iterations):
        n_rem = max(1, int(len(path) * cfg.ialns_destroy_fraction))
        if len(path) <= 3:
            n_rem = min(1, len(path) - 2)

        # select destroy operator (adaptive weights)
        probs = np.array(d_weights) / sum(d_weights)
        choice = rng.choice(len(d_weights), p=probs)
        if choice == 0:
            new_p, rem = _destroy_random(list(path), n_rem, rng)
        else:
            new_p, rem = _destroy_worst(list(path), n_rem)

        new_p = _repair_greedy(new_p, rem)
        new_cost = total_path_distance(new_p)
        delta = new_cost - cur_cost

        # ── Metropolis acceptance  P_S = exp(−Δ / T)  ──────────────
        if delta < 0:
            path, cur_cost = new_p, new_cost
            if new_cost < best_cost:
                best_path, best_cost = list(new_p), new_cost
                d_weights[choice] += 0.1   # reward successful operator
        elif temp > 1e-8 and rng.random() < np.exp(-delta / temp):
            path, cur_cost = new_p, new_cost

        temp *= cfg.ialns_temp_decay          # cool down (ξ_A approach)

    return best_path


# ═══════════════════════════════════════════════════════════════════════════
# Speed adjustment for rendezvous  (Eq 7, 8, 9)
# ═══════════════════════════════════════════════════════════════════════════
def adjust_speeds_for_rendezvous(path_s, path_r, comm_point, cfg):
    """Return (v_sender, v_receiver) so both arrive at P_n simultaneously.

    Eq 7:  V_n = dist_n_to_P / t_rendezvous
    Eq 8,9: if V_n > V_max  →  clamp to V_max and UAV hovers/waits.
    """
    if comm_point is None:
        return cfg.uav_speed, cfg.uav_speed

    cp = tuple(comm_point) if not isinstance(comm_point, tuple) else comm_point

    def dist_to_cp(path):
        """Distance along the path from start to the communication point."""
        for i, p in enumerate(path):
            if _dist(p, cp) < 5.0:
                return sum(_dist(path[j], path[j + 1]) for j in range(i))
        return total_path_distance(path) / 2

    ds, dr = dist_to_cp(path_s), dist_to_cp(path_r)

    if ds <= 0 and dr <= 0:
        return cfg.uav_speed, cfg.uav_speed

    # Proportional speed adjustment  (Eq 7)
    # Both must arrive at same t_rendezvous  →  V_n ∝ dist_n
    d_max = max(ds, dr)
    if d_max <= 0:
        return cfg.uav_speed, cfg.uav_speed

    t_rendezvous = d_max / cfg.uav_speed      # slower UAV at base speed
    v_s = ds / t_rendezvous if t_rendezvous > 0 else cfg.uav_speed
    v_r = dr / t_rendezvous if t_rendezvous > 0 else cfg.uav_speed

    # ── Clamping  (Eq 8, 9): force hover / wait if out of range ──────
    v_s = float(np.clip(v_s, cfg.uav_speed_min, cfg.uav_speed_max))
    v_r = float(np.clip(v_r, cfg.uav_speed_min, cfg.uav_speed_max))

    return v_s, v_r
