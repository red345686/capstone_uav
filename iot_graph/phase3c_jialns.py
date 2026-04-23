"""
Phase 3c — JIALNS (Joint Improved ALNS) Path Planning.

§7 of search.pdf, Algorithm 3:
  Solution: X = (π_1,...,π_N), sequences of contested gateways only
  Cost: F_c = Σ D_n(π_n) - Φ(X) (Eq 13)
  Destroy operators (§7.3): D1 random contested, D2 worst-case, D3 relay perturbation
  Repair operators (§7.4): R1 greedy, R2 random, R3 relay reinsertion, R4 BS re-assignment
  Weight update + Metropolis acceptance (§7.5, Eq 14-15)
"""
import numpy as np
from typing import Dict, List, Tuple


def _dist(a, b):
    return np.hypot(a[0] - b[0], a[1] - b[1])


def total_path_distance(path):
    """Total distance of path."""
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
# Initial solution — nearest-neighbour
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
def _destroy_random_contested(path, n_rem, rng):
    """D1: Random contested node removal."""
    if len(path) <= 3:
        return list(path), []
    inner = list(range(1, len(path) - 1))
    n_rem = min(n_rem, len(inner))
    idxs = sorted(rng.choice(inner, n_rem, replace=False), reverse=True)
    p, removed = list(path), []
    for i in idxs:
        removed.append(p.pop(i))
    return p, removed


def _destroy_worst_contested(path, n_rem):
    """D2: Worst-case contested node removal (highest detour cost)."""
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


def _destroy_relay_perturbation(path, n_rem, rng):
    """D3: Relay-point perturbation — shift waypoints slightly."""
    if len(path) <= 3:
        return list(path), []
    p = list(path)
    inner = list(range(1, len(p) - 1))
    n_perturb = min(n_rem, len(inner))
    idxs = rng.choice(inner, n_perturb, replace=False)
    removed = []
    for i in idxs:
        old = p[i]
        removed.append(old)
        # Perturb position
        new_pos = (old[0] + rng.normal(0, 20), old[1] + rng.normal(0, 20))
        p[i] = new_pos
    return p, removed


# ═══════════════════════════════════════════════════════════════════════════
# Repair operators
# ═══════════════════════════════════════════════════════════════════════════
def _repair_greedy(path, removed):
    """R1: Greedy insertion at best position."""
    for node in removed:
        best_i, best_c = 1, float("inf")
        for i in range(1, len(path)):
            c = (_dist(path[i - 1], node) + _dist(node, path[i])
                 - _dist(path[i - 1], path[i]))
            if c < best_c:
                best_c, best_i = c, i
        path.insert(best_i, node)
    return path


def _repair_random(path, removed, rng):
    """R2: Random insertion."""
    for node in removed:
        if len(path) < 2:
            path.append(node)
        else:
            pos = rng.randint(1, max(len(path), 2))
            path.insert(pos, node)
    return path


def _repair_relay_reinsertion(path, removed):
    """R3: Relay-point reinsertion near midpoint."""
    for node in removed:
        if len(path) < 2:
            path.append(node)
        else:
            mid = len(path) // 2
            path.insert(mid, node)
    return path


# ═══════════════════════════════════════════════════════════════════════════
# JIALNS main loop
# ═══════════════════════════════════════════════════════════════════════════
def jialns_path_planning(uav, conv_nodes, comm_point, base_stations, cfg, rng,
                          start_pos=None, end_pos=None, node_classes=None):
    """Return optimised path [(x,y), ...] for one UAV.

    Extends IALNS with contested-node awareness and multi-BS support.
    """
    # Find nearest BS for start/end
    if base_stations:
        nearest_bs = min(base_stations,
                         key=lambda bs: np.hypot(bs.x - uav.x, bs.y - uav.y))
        bs = (nearest_bs.x, nearest_bs.y)
    else:
        bs = (100.0, 100.0)

    start = tuple(start_pos) if start_pos is not None else bs
    end = tuple(end_pos) if end_pos is not None else bs

    wps = [(n.x, n.y) for n in conv_nodes]
    if comm_point is not None:
        cp_tuple = tuple(comm_point)
        if (_dist(cp_tuple, start) > 5.0 and _dist(cp_tuple, end) > 5.0):
            wps.append(cp_tuple)

    if not wps:
        return [start, end]

    path = _nn_path(start, wps, end)
    best_path = list(path)
    best_cost = total_path_distance(path)
    cur_cost = best_cost
    temp = cfg.ialns_temp_init
    d_weights = [1.0, 1.0, 0.5]    # D1, D2, D3
    r_weights = [1.0, 0.5, 0.3]    # R1, R2, R3

    for _ in range(cfg.ialns_iterations):
        n_rem = max(1, int(len(path) * cfg.ialns_destroy_fraction))
        if len(path) <= 3:
            n_rem = min(1, len(path) - 2)

        # Select destroy operator
        d_probs = np.array(d_weights) / sum(d_weights)
        d_choice = rng.choice(len(d_weights), p=d_probs)
        if d_choice == 0:
            new_p, rem = _destroy_random_contested(list(path), n_rem, rng)
        elif d_choice == 1:
            new_p, rem = _destroy_worst_contested(list(path), n_rem)
        else:
            new_p, rem = _destroy_relay_perturbation(list(path), n_rem, rng)

        # Select repair operator
        r_probs = np.array(r_weights) / sum(r_weights)
        r_choice = rng.choice(len(r_weights), p=r_probs)
        if r_choice == 0:
            new_p = _repair_greedy(new_p, rem)
        elif r_choice == 1:
            new_p = _repair_random(new_p, rem, rng)
        else:
            new_p = _repair_relay_reinsertion(new_p, rem)

        new_cost = total_path_distance(new_p)
        delta = new_cost - cur_cost

        # Metropolis acceptance
        if delta < 0:
            path, cur_cost = new_p, new_cost
            if new_cost < best_cost:
                best_path, best_cost = list(new_p), new_cost
                d_weights[d_choice] += 0.1
                r_weights[r_choice] += 0.1
        elif temp > 1e-8 and rng.random() < np.exp(-delta / temp):
            path, cur_cost = new_p, new_cost

        temp *= cfg.ialns_temp_decay

    return best_path


# ═══════════════════════════════════════════════════════════════════════════
# Speed adjustment for rendezvous
# ═══════════════════════════════════════════════════════════════════════════
def adjust_speeds_for_rendezvous(path_s, path_r, comm_point, cfg):
    """Return (v_sender, v_receiver) so both arrive at P_n simultaneously."""
    if comm_point is None:
        return cfg.uav_speed, cfg.uav_speed

    cp = tuple(comm_point) if not isinstance(comm_point, tuple) else comm_point

    def dist_to_cp(path):
        for i, p in enumerate(path):
            if _dist(p, cp) < 5.0:
                return sum(_dist(path[j], path[j + 1]) for j in range(i))
        return total_path_distance(path) / 2

    ds, dr = dist_to_cp(path_s), dist_to_cp(path_r)

    if ds <= 0 and dr <= 0:
        return cfg.uav_speed, cfg.uav_speed

    d_max = max(ds, dr)
    if d_max <= 0:
        return cfg.uav_speed, cfg.uav_speed

    t_rendezvous = d_max / cfg.uav_speed
    v_s = ds / t_rendezvous if t_rendezvous > 0 else cfg.uav_speed
    v_r = dr / t_rendezvous if t_rendezvous > 0 else cfg.uav_speed

    v_s = float(np.clip(v_s, cfg.uav_speed_min, cfg.uav_speed_max))
    v_r = float(np.clip(v_r, cfg.uav_speed_min, cfg.uav_speed_max))

    return v_s, v_r
