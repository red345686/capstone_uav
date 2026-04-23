"""
Phase 3d — Extended Relay Matching + Multi-BS Alternating Charging.

§8 of search.pdf:
  Algorithm 4: Extended relay matching with BS zone awareness
    - Classify UAVs as terminal or relay based on zone boundary
    - Sender preference: min_l ||q_m - b_l|| (Eq 16)
    - Receiver preference: energy efficiency (Eq 16 context)
    - Gale-Shapley deferred acceptance
  Eq 17: Alternating charging at nearest non-saturated BS
"""
import numpy as np
from communication import energy_efficiency_metric
from energy_model import hover_energy


def classify_uavs_extended(uavs, clusters, zone_assign_fn, base_stations, cfg):
    """Classify UAVs as terminal or relay based on zone boundary.

    Terminal UAVs: assigned to a single BS zone (their cluster is entirely
    within one zone).
    Relay UAVs: assigned to contested areas spanning multiple zones.
    """
    bs_lut = {bs.id: bs for bs in base_stations}
    terminal_ids = set()
    relay_ids = set()

    for u in uavs:
        # Determine if UAV's cluster spans multiple zones
        # Simple heuristic: terminal if close to its assigned BS
        assigned_bs = bs_lut.get(u.assigned_bs_id, base_stations[0])
        d_to_bs = np.hypot(u.x - assigned_bs.x, u.y - assigned_bs.y)

        if d_to_bs < cfg.d_prox * 2:
            terminal_ids.add(u.id)
            u.role = "terminal"
        else:
            relay_ids.add(u.id)
            u.role = "relay"

    return list(terminal_ids), list(relay_ids)


def extended_relay_matching(uavs, clusters, subnets, base_stations, cfg,
                             zone_assign_fn=None):
    """Extended Gale-Shapley relay matching (Algorithm 4).

    Returns list[(sender_id, receiver_id)].
    """
    uav_lut = {u.id: u for u in uavs}
    sub_lut = {s.id: s for s in subnets}
    bs_lut = {bs.id: bs for bs in base_stations}

    if len(uavs) < 2:
        for u in uavs:
            u.role, u.partner_id = "solo", -1
        return []

    # Classify into senders (remote) and receivers (proximal) based on
    # distance to nearest BS
    dists = []
    for u in uavs:
        # Centroid of assigned cluster
        sids = clusters.get(u.cluster_id, clusters.get(u.id, []))
        pts = []
        for sid in sids:
            if sid in sub_lut and sub_lut[sid].centroid is not None:
                pts.append(sub_lut[sid].centroid)
        if pts:
            centroid = np.mean(pts, axis=0)
        else:
            centroid = np.array([u.x, u.y])

        # Distance from cluster centroid to nearest BS
        min_d = min(np.linalg.norm(centroid - np.array([bs.x, bs.y]))
                    for bs in base_stations)
        dists.append((u.id, min_d))

    dists.sort(key=lambda x: x[1])
    mid = len(dists) // 2
    receiver_ids = [uid for uid, _ in dists[:mid]]
    sender_ids = [uid for uid, _ in dists[mid:]]

    for u in uavs:
        if u.id in sender_ids:
            u.role = "sender"
        elif u.id in receiver_ids:
            u.role = "receiver"
        else:
            u.role = "solo"

    if not sender_ids or not receiver_ids:
        for u in uavs:
            u.role, u.partner_id = "solo", -1
        return []

    # Sender preference: nearest BS (Eq 16)
    s_prefs = {}
    for sid in sender_ids:
        ranked = sorted(
            receiver_ids,
            key=lambda rid: min(
                np.hypot(uav_lut[rid].x - bs.x, uav_lut[rid].y - bs.y)
                for bs in base_stations
            ),
        )
        s_prefs[sid] = ranked

    # Receiver preference: energy efficiency
    r_prefs = {}
    for rid in receiver_ids:
        ru = uav_lut[rid]
        effs = []
        for sid in sender_ids:
            su = uav_lut[sid]
            d_horiz = np.hypot(su.x - ru.x, su.y - ru.y)
            e_eta = energy_efficiency_metric(d_horiz, cfg)
            effs.append((sid, e_eta))
        effs.sort(key=lambda x: x[1], reverse=True)
        r_prefs[rid] = [s for s, _ in effs]

    # Gale-Shapley
    next_prop = {s: 0 for s in sender_ids}
    r_match = {r: None for r in receiver_ids}
    s_match = {s: None for s in sender_ids}
    free = list(sender_ids)

    while free:
        sid = free.pop(0)
        if next_prop[sid] >= len(s_prefs.get(sid, [])):
            continue
        rid = s_prefs[sid][next_prop[sid]]
        next_prop[sid] += 1
        if r_match[rid] is None:
            r_match[rid], s_match[sid] = sid, rid
        else:
            cur = r_match[rid]
            pref = r_prefs[rid]
            if sid in pref and cur in pref and pref.index(sid) < pref.index(cur):
                r_match[rid], s_match[sid] = sid, rid
                s_match[cur] = None
                free.append(cur)
            else:
                free.append(sid)

    pairs = []
    for sid in sender_ids:
        rid = s_match[sid]
        if rid is not None:
            uav_lut[sid].partner_id = rid
            uav_lut[rid].partner_id = sid
            pairs.append((sid, rid))
        else:
            uav_lut[sid].role = "solo"
    for rid in receiver_ids:
        if r_match[rid] is None:
            uav_lut[rid].role = "solo"

    return pairs


def alternating_charging_multi_bs(uavs, pairs, base_stations, cfg, env):
    """Alternating charging with multi-BS support (Eq 17).

    Each UAV charges at nearest non-saturated BS.
    """
    lut = {u.id: u for u in uavs}
    bs_load = {bs.id: 0 for bs in base_stations}

    for _sid, rid in pairs:
        rcv = lut[rid]

        # Find nearest non-saturated BS
        sorted_bs = sorted(
            base_stations,
            key=lambda bs: np.hypot(rcv.x - bs.x, rcv.y - bs.y)
        )

        target_bs = sorted_bs[0]  # default: nearest
        for bs in sorted_bs:
            if bs_load[bs.id] < bs.num_chargers:
                target_bs = bs
                break

        # Queue wait time (simplified M/M/1)
        n_wait = bs_load[target_bs.id]
        mu = 0.1
        wait_time = (n_wait + 1) / mu

        rcv.e_current -= hover_energy(wait_time, cfg)
        rcv.e_current = max(rcv.e_current, 0.0)

        # Full recharge
        rcv.e_current = rcv.e_max
        rcv.x, rcv.y = target_bs.x, target_bs.y
        rcv.assigned_bs_id = target_bs.id

        bs_load[target_bs.id] += 1


def swap_roles(uavs, pairs, clusters):
    """Swap sender <-> receiver and their cluster assignments."""
    lut = {u.id: u for u in uavs}
    new_pairs = []
    for sid, rid in pairs:
        s, r = lut[sid], lut[rid]
        s.role, r.role = r.role, s.role
        s.cluster_id, r.cluster_id = r.cluster_id, s.cluster_id
        new_pairs.append((rid, sid))
    return new_pairs


def should_swap(uavs, pairs, round_num, config):
    """Check if role swap should happen."""
    if round_num > 0 and round_num % config.T_C == 0:
        return True
    lut = {u.id: u for u in uavs}
    for sid, _rid in pairs:
        remote = lut[sid]
        if remote.e_current < 0.2 * remote.e_max:
            return True
    return False
