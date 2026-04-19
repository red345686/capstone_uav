"""
Phase 3 — Relay Relationship Pairing  (Algorithm 3 / Gale-Shapley).

Sender preference  : ranks receivers by ascending distance from BS
                     (closer to BS ⇒ more preferred, for efficient relay).
Receiver preference : ranks senders by descending energye-fficiency E_η
                     (Eq 41 context):
                     E_{i,n} = (B / P_R) · log₂(1 + h²·P_S / σ²)
"""
import numpy as np
from communication import energy_efficiency_metric


# ─────────────────────────────────────────────────────────────────────────
def _cluster_centroid(uav_id, uavs, clusters, subnets):
    uav = next(u for u in uavs if u.id == uav_id)
    sub_lut = {s.id: s for s in subnets}
    sids = clusters.get(uav.cluster_id, [])
    pts = [sub_lut[sid].centroid for sid in sids
           if sid in sub_lut and sub_lut[sid].centroid is not None]
    if pts:
        return np.mean(pts, axis=0)
    return np.array([uav.x, uav.y])


def classify_uavs(uavs, clusters, subnets, bs):
    """Split UAVs into senders (remote) and receivers (proximal)."""
    bs_pos = np.array([bs.x, bs.y])
    dists = []
    for u in uavs:
        c = _cluster_centroid(u.id, uavs, clusters, subnets)
        dists.append((u.id, np.linalg.norm(c - bs_pos)))
    dists.sort(key=lambda x: x[1])
    mid = len(dists) // 2
    receivers = {uid for uid, _ in dists[:mid]}
    senders   = {uid for uid, _ in dists[mid:]}
    for u in uavs:
        u.role = "sender" if u.id in senders else (
            "receiver" if u.id in receivers else "solo"
        )
    return list(senders), list(receivers)


def stable_matching(uavs, clusters, subnets, base_station, config):
    """Gale‑Shapley stable matching.  Returns list[(sender_id, receiver_id)]."""
    uav_lut = {u.id: u for u in uavs}
    sub_lut = {s.id: s for s in subnets}
    sender_ids, receiver_ids = classify_uavs(
        uavs, clusters, subnets, base_station
    )
    if not sender_ids or not receiver_ids:
        for u in uavs:
            u.role, u.partner_id = "solo", -1
        return []

    bs_pos = np.array([base_station.x, base_station.y])

    def centroid(uid):
        return _cluster_centroid(uid, uavs, clusters, subnets)

    # ── Sender preference: receiver closest to BS first ───────────────
    s_prefs = {}
    for sid in sender_ids:
        ranked = sorted(
            receiver_ids,
            key=lambda rid: np.linalg.norm(centroid(rid) - bs_pos),
        )
        s_prefs[sid] = ranked

    # ── Receiver preference: highest energy‑efficiency sender first ───
    r_prefs = {}
    for rid in receiver_ids:
        rc = centroid(rid)
        effs = []
        for sid in sender_ids:
            sc = centroid(sid)
            d_horiz = np.linalg.norm(sc - rc)
            e_eta = energy_efficiency_metric(d_horiz, config)
            effs.append((sid, e_eta))
        effs.sort(key=lambda x: x[1], reverse=True)
        r_prefs[rid] = [s for s, _ in effs]

    # ── Gale‑Shapley ─────────────────────────────────────────────────
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
            if pref.index(sid) < pref.index(cur):
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
