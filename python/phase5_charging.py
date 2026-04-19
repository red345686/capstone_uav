"""
Phase 5 — Alternating Charging Mode  (Algorithm 5).

After T_C rounds, paired UAVs swap subareas:
    A_R^j  ←  A_N^{j+1}
The new proximal UAV returns to the base station for charging.

Base Station Charging  (Eq 12):
    Full recharge to E_max.

M/M/1 Queue Wait Time  (Eq 14, 15):
    If multiple UAVs return simultaneously, model wait times
    with exponential queuing distribution.
"""
import numpy as np
from energy_model import hover_energy


def swap_roles(uavs, pairs, clusters):
    """Swap sender ↔ receiver and their cluster assignments (Algorithm 5).

    Returns the updated pairs list (receiver, sender swapped).
    """
    lut = {u.id: u for u in uavs}
    new_pairs = []
    for sid, rid in pairs:
        s, r = lut[sid], lut[rid]
        s.role, r.role = r.role, s.role
        s.cluster_id, r.cluster_id = r.cluster_id, s.cluster_id
        new_pairs.append((rid, sid))
    return new_pairs


def mm1m_queue_time(lam, mu, m, n_wait):
    """Expected sojourn time under M/M/1/m  (Eq 14, 15).

    Parameters
    ----------
    lam    : float - arrival rate
    mu     : float - service rate
    m      : int   - system capacity
    n_wait : int   - current queue length (position of arriving UAV)

    Returns
    -------
    float - expected time in system (wait + service)
    """
    if mu <= 0:
        return float("inf")
    rho = lam / mu
    svc = 1.0 / mu

    if abs(rho - 1.0) < 1e-10:
        # Special case ρ = 1
        Lq = n_wait / 2.0
    elif rho ** m > 1e30:
        Lq = m - 1.0
    else:
        num = rho * (1.0 - (m + 1) * rho ** m + m * rho ** (m + 1))
        den = (1.0 - rho) * (1.0 - rho ** (m + 1))
        Lq = num / den if abs(den) > 1e-10 else m / 2.0
    return max(Lq * svc + svc, svc)         # at least one service time


def handle_charging(uavs, pairs, base_station, config, env):
    """Recharge the new proximal UAVs (post‑swap).

    The returning UAV:
      1. flies back to BS  (energy cost subtracted elsewhere)
      2. waits in the M/M/1/m queue  (Eq 14, 15)
      3. receives full recharge       (Eq 12)
    """
    lut = {u.id: u for u in uavs}
    m = config.num_uavs                             # system capacity
    mu = 0.1                                        # service rate
    lam = len(pairs) / max(config.round_duration, 1.0)

    for _sid, rid in pairs:
        rcv = lut[rid]

        # Position in queue
        n_wait = len(base_station.charging_queue)
        wait_time = mm1m_queue_time(lam, mu, m, n_wait)

        # Energy consumed while hovering / waiting at BS
        rcv.e_current -= hover_energy(wait_time, config)
        rcv.e_current = max(rcv.e_current, 0.0)

        base_station.charging_queue.append(rcv.id)

        # Full recharge  (Eq 12)
        rcv.e_current = rcv.e_max
        rcv.x, rcv.y = base_station.x, base_station.y

    base_station.charging_queue.clear()


def should_swap(uavs, pairs, round_num, config):
    """Check continuous E_rem monitoring trigger (Algorithm 5).

    Swap is forced when:
      • The round hits the charging swap period T_C, OR
      • A remote UAV's remaining energy drops below 20 % of E_max.
    """
    if round_num > 0 and round_num % config.T_C == 0:
        return True
    lut = {u.id: u for u in uavs}
    for sid, _rid in pairs:
        remote = lut[sid]
        if remote.e_current < 0.2 * remote.e_max:
            return True
    return False
