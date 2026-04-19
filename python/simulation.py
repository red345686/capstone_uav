"""
Main simulation loop — orchestrates all five phases.

MGDC data flow (corrected architecture):
    Sensor → CN → UAV buffer → (relay at Pn) → BS delivery

Key mechanics:
  • TTL-based packet expiry  (δ_T = C_S / s ≈ 10 rounds)
  • Shannon-capacity-limited data collection  (Eq 1)
  • Air-to-Air relay at communication point P  (Eq 20-22)
  • Propulsion energy via Eq 25
  • Time-budgeted flight: UAV can only fly for round_duration seconds per round
  • Buffer-based collection: packets only count as 'delivered' at BS
  • Sender UAV: BS → CNs → comm_point Pn  (does NOT return to BS)
  • Receiver UAV: BS → CNs → comm_point Pn → BS  (delivers at BS)
  • Solo UAV: BS → CNs → BS  (delivers at BS)
"""
import numpy as np

from config import SimConfig
from environment import Environment
from energy_model import flight_energy, hover_energy, sensor_tx_energy
from communication import (
    shannon_capacity_g2a,
    shannon_capacity_a2a,
    verify_a2a_los,
)
from phase1_convergence import select_convergence_nodes
from phase2_division import divide_area
from phase3_matching import stable_matching
from phase4_pathplanning import (
    ialns_path_planning,
    compute_communication_point,
    adjust_speeds_for_rendezvous,
    total_path_distance,
)
from phase5_charging import swap_roles, handle_charging, should_swap
from metrics import MetricsTracker


# ═══════════════════════════════════════════════════════════════════════════
# Data Collection (Shannon‑limited, buffer‑based)
# ═══════════════════════════════════════════════════════════════════════════
def _collect_at_convergence_node(uav, sub, cn, env, cfg, time_remaining):
    """Collect data from *sub* via its convergence node *cn*.

    Packets go into uav.data_buffer — NOT into env.total_packets_delivered.
    Delivery to BS happens later via _deliver_to_bs.

    Parameters
    ----------
    time_remaining : float – seconds left in the round's time budget

    Returns
    -------
    (collected_count, hover_time, collected_sensor_ids)
    """
    if time_remaining <= 0:
        return 0, 0.0, []

    # G2A capacity (UAV hovers directly above CN → d_horizontal ≈ 0)
    capacity_bps = shannon_capacity_g2a(0.0, cfg)
    if capacity_bps <= 0:
        return 0, 0.0, []

    # Count pending data (only from sensors with energy)
    pending = []
    for sn in sub.nodes:
        if sn.packet_queue and sn.e_current > 0:
            pending.append(sn)
    total_units = sum(len(sn.packet_queue) for sn in pending)
    if total_units <= 0:
        return 0, 0.0, []

    total_bits = total_units * cfg.packet_size_bits

    # Hover time (capped by max_hover_time AND remaining time budget)
    hover_time = min(total_bits / capacity_bps,
                     cfg.max_hover_time,
                     time_remaining)
    collectible_bits = capacity_bps * hover_time
    collectible_units = int(collectible_bits / cfg.packet_size_bits)

    # ── Energy costs ──────────────────────────────────────────────────
    # UAV hover energy  (Eq 25 at V=0)
    uav.e_current -= hover_energy(hover_time, cfg)
    if uav.e_current <= 0:
        uav.e_current = 0
        return 0, hover_time, []

    # CN transmission energy  (P_S · hover_time)
    cn.e_current -= cfg.P_S_i * hover_time
    cn.e_current = max(cn.e_current, 0.0)

    # ── Collect from each sensor (oldest packets first) ───────────────
    collected = 0
    collected_sids = []
    for sn in pending:
        if collected >= collectible_units:
            break
        n_avail = len(sn.packet_queue)
        n_take = min(n_avail, collectible_units - collected)
        if n_take <= 0:
            continue

        # Intra‑subnet TX energy: sensor → CN  (Eq 4)
        tx_bits = n_take * cfg.packet_size_bits
        tx_d = np.hypot(sn.x - cn.x, sn.y - cn.y)
        e_tx = sensor_tx_energy(tx_bits, tx_d, cfg)
        sn.e_current -= e_tx
        if sn.e_current < 0:
            sn.e_current = 0

        # Remove oldest packets (FIFO)
        sn.packet_queue = sn.packet_queue[n_take:]
        sn.packets_collected += n_take
        collected += n_take
        collected_sids.append(sn.id)

    return collected, hover_time, collected_sids


# ═══════════════════════════════════════════════════════════════════════════
# Single‑UAV path execution  (TIME‑BUDGETED)
# ═══════════════════════════════════════════════════════════════════════════
def _execute_path(uav, path, env, cfg):
    """Fly *uav* along *path*, collecting data at CN waypoints.

    Time‑budgeted: the UAV can only fly for cfg.round_duration seconds.
    If time runs out mid‑path, UAV stops — can't visit remaining CNs.
    This is what makes UAV count matter: fewer UAVs → longer paths →
    fewer CNs reachable per round → less data collected.

    Collected packets go into uav.data_buffer (NOT env counters).

    Returns
    -------
    (total_collected, time_used)
    """
    # Build quick lookup: rounded (x, y) → (subnet, convergence‑node)
    cn_lut = {}
    for sub in env.subnets:
        if sub.convergence_node_id < 0:
            continue
        node = next(
            (n for n in sub.nodes if n.id == sub.convergence_node_id), None
        )
        if node:
            cn_lut[(round(node.x, 4), round(node.y, 4))] = (sub, node)

    collected = 0
    time_used = 0.0
    time_budget = cfg.round_duration

    for i in range(len(path) - 1):
        d = np.hypot(path[i + 1][0] - path[i][0],
                     path[i + 1][1] - path[i][1])

        # ── Flight time for this segment ──────────────────────────────
        speed = max(uav.speed, cfg.uav_speed_min)
        flight_time = d / speed if speed > 0 else 0.0

        # Check time budget
        if time_used + flight_time > time_budget:
            # Can't complete this segment — stop at partial position
            remaining_time = time_budget - time_used
            if remaining_time > 0 and d > 0:
                frac = (remaining_time * speed) / d
                frac = min(frac, 1.0)
                uav.x = path[i][0] + frac * (path[i + 1][0] - path[i][0])
                uav.y = path[i][1] + frac * (path[i + 1][1] - path[i][1])
                partial_d = remaining_time * speed
                uav.e_current -= flight_energy(partial_d, speed, cfg)
                uav.total_distance += partial_d
            time_used = time_budget
            break

        # ── Propulsion energy  (Eq 25) ────────────────────────────────
        uav.e_current -= flight_energy(d, speed, cfg)
        uav.total_distance += d
        time_used += flight_time
        if uav.e_current <= 0:
            uav.e_current = 0
            break

        # ── Check if waypoint is near a convergence node ──────────────
        wp = path[i + 1]
        for (cx, cy), (sub, cn) in cn_lut.items():
            if np.hypot(wp[0] - cx, wp[1] - cy) < 5.0:
                time_remaining = time_budget - time_used
                c, hover_t, sids = _collect_at_convergence_node(
                    uav, sub, cn, env, cfg, time_remaining
                )
                time_used += hover_t
                collected += c
                uav.collected_sensor_ids.extend(sids)
                if uav.e_current <= 0:
                    break
                break  # only one CN match per waypoint

        if uav.e_current <= 0 or time_used >= time_budget:
            break

    # Accumulate into UAV data buffer (NOT env counters)
    uav.data_buffer += collected

    # Update UAV position to last reached waypoint
    if path and time_used < time_budget:
        uav.x, uav.y = path[-1]

    return collected, time_used


# ═══════════════════════════════════════════════════════════════════════════
# Delivery to BS  (only here do packets count as "delivered")
# ═══════════════════════════════════════════════════════════════════════════
def _deliver_to_bs(uav, env):
    """Transfer uav.data_buffer → env.total_packets_delivered.

    Called only when a receiver/solo UAV is at BS after completing its path.
    Updates AoI timestamps for all sensors whose data is in the buffer.
    """
    if uav.data_buffer <= 0:
        return

    # Count delivered
    env.total_packets_delivered += uav.data_buffer
    uav.total_packets_delivered += uav.data_buffer

    # Update AoI: mark last_collected_time for sensors whose packets
    # were just delivered to BS
    for sid in uav.collected_sensor_ids:
        sn = next((n for n in env.sensor_nodes if n.id == sid), None)
        if sn is not None:
            sn.last_collected_time = env.current_time
            sn.last_collected_round = env.current_round

    # Clear buffer
    uav.data_buffer = 0
    uav.collected_sensor_ids = []


# ═══════════════════════════════════════════════════════════════════════════
# Air‑to‑Air Relay  (Eq 20, 21, 22) — buffer transfer
# ═══════════════════════════════════════════════════════════════════════════
def _a2a_relay(sender_uav, receiver_uav, comm_point, cfg):
    """Transfer sender's data_buffer + collected_sensor_ids to receiver.

    1. Verify LoS  (Eq 20, 21).
    2. Compute A2A capacity  (Eq 22 context).
    3. Transfer data buffer.
    """
    if comm_point is None or sender_uav.data_buffer <= 0:
        return

    # LoS check
    s_pos = (sender_uav.x, sender_uav.y)
    r_pos = (receiver_uav.x, receiver_uav.y)
    if not verify_a2a_los(s_pos, r_pos, cfg):
        return

    d_a2a = max(np.hypot(s_pos[0] - r_pos[0], s_pos[1] - r_pos[1]), 1.0)
    capacity = shannon_capacity_a2a(d_a2a, cfg)
    if capacity <= 0:
        return

    data_bits = sender_uav.data_buffer * cfg.packet_size_bits
    transfer_time = data_bits / capacity

    # Both hover during transfer
    sender_uav.e_current -= hover_energy(transfer_time, cfg)
    receiver_uav.e_current -= hover_energy(transfer_time, cfg)
    sender_uav.e_current = max(sender_uav.e_current, 0.0)
    receiver_uav.e_current = max(receiver_uav.e_current, 0.0)

    # Move data: sender's buffer → receiver's buffer
    receiver_uav.data_buffer += sender_uav.data_buffer
    receiver_uav.collected_sensor_ids.extend(sender_uav.collected_sensor_ids)
    sender_uav.data_buffer = 0
    sender_uav.collected_sensor_ids = []


# ═══════════════════════════════════════════════════════════════════════════
# Main entry point
# ═══════════════════════════════════════════════════════════════════════════
def run_simulation(config: SimConfig,
                   with_alternating_charging: bool = True,
                   verbose: bool = True):
    """Run the full 5‑phase simulation.

    Returns
    -------
    env, metrics, all_paths, all_clusters
    """
    rng = np.random.RandomState(config.seed)
    env = Environment(config)
    env.generate()
    metrics = MetricsTracker()
    all_paths: dict = {}
    all_clusters: dict = {}
    pairs = []

    if verbose:
        total_sensors = len(env.sensor_nodes)
        print(f"=== Simulation Start ===")
        print(f"  Area      : {config.area_width}×{config.area_height} m")
        print(f"  UAVs      : {config.num_uavs}")
        print(f"  Subnets   : {len(env.subnets)}")
        print(f"  Sensors   : {total_sensors}")
        print(f"  V_max     : {config.uav_speed_max} m/s")
        print(f"  v_0       : {config.v_0} m/s")
        print(f"  TTL (delta_T) : {config.delta_T} rounds")
        print(f"  Round dur : {config.round_duration}s")
        print(f"  Charging  : {'ON' if with_alternating_charging else 'OFF'}")
        print()

    for rnd in range(config.max_rounds):
        env.current_round = rnd
        env.current_time = (rnd + 1) * config.round_duration

        # ── Generate data (512 KB per sensor per round) ───────────────
        env.generate_data(rnd)

        # ── TTL expiry: delete packets older than δ_T rounds ──────────
        env.expire_stale_packets(rnd)

        # ── Phase 1: Convergence Node Selection (Algorithm 1) ─────────
        conv_nodes = select_convergence_nodes(env)
        if not conv_nodes:
            break

        # ── Phase 2: Mission Area Division (Algorithm 2) ──────────────
        clusters = divide_area(conv_nodes, env.subnets, config, rng)
        all_clusters[rnd] = dict(clusters)
        for u in env.uavs:
            u.cluster_id = u.id if u.id in clusters else -1

        # ── Phase 3: Relay Matching (Algorithm 3 / Gale‑Shapley) ──────
        if config.num_uavs > 1:
            pairs = stable_matching(
                env.uavs, clusters, env.subnets,
                env.base_station, config,
            )
        else:
            env.uavs[0].role = "solo"
            pairs = []

        # ── Phase 4: Path Planning (Algorithm 4 / IALNS) ─────────────
        comm_points: dict = {}
        sub_lut = {s.id: s for s in env.subnets}
        for sid, rid in pairs:
            s_sids = clusters.get(
                next(u for u in env.uavs if u.id == sid).cluster_id, []
            )
            r_sids = clusters.get(
                next(u for u in env.uavs if u.id == rid).cluster_id, []
            )
            s_cn = [n for n in conv_nodes if n.subnet_id in s_sids]
            r_cn = [n for n in conv_nodes if n.subnet_id in r_sids]
            cp = compute_communication_point(s_cn, r_cn)
            if cp is not None:
                comm_points[sid] = cp
                comm_points[rid] = cp

        # Build paths with correct start/end positions:
        #   Sender:   current_pos → CNs → comm_point  (NOT back to BS)
        #   Receiver: current_pos → CNs → comm_point → BS
        #   Solo:     BS → CNs → BS
        bs_pos = (env.base_station.x, env.base_station.y)
        uav_lut = {u.id: u for u in env.uavs}

        round_paths: dict = {}
        for u in env.uavs:
            if u.e_current <= 0:
                continue
            u_sids = clusters.get(u.cluster_id, [])
            u_cn = [n for n in conv_nodes if n.subnet_id in u_sids]
            cp = comm_points.get(u.id)

            # Determine start/end based on role
            start = (u.x, u.y)
            if u.role == "sender":
                # Sender: ends at comm_point (core MGDC benefit — no wasted
                # return flight to BS)
                end = tuple(cp) if cp is not None else bs_pos
            else:
                # Receiver and Solo: end at BS
                end = bs_pos

            round_paths[u.id] = ialns_path_planning(
                u, u_cn, cp, env.base_station, config, rng,
                start_pos=start, end_pos=end,
            )

        # Speed synchronisation  (Eq 7, 8, 9)
        for sid, rid in pairs:
            if sid in round_paths and rid in round_paths:
                cp = comm_points.get(sid)
                if cp is not None:
                    vs, vr = adjust_speeds_for_rendezvous(
                        round_paths[sid], round_paths[rid], cp, config,
                    )
                    uav_lut[sid].speed = vs
                    uav_lut[rid].speed = vr

        # ── Snapshot energy before execution ──────────────────────────
        energy_before = sum(u.e_current for u in env.uavs)

        # ── Reset UAV buffers for this round ──────────────────────────
        for u in env.uavs:
            u.data_buffer = 0
            u.collected_sensor_ids = []

        # ── Execute flight paths (data collection + energy drain) ─────
        for u in env.uavs:
            if u.id in round_paths and u.e_current > 0:
                _execute_path(u, round_paths[u.id], env, config)

        # ── A2A Relay at communication point (sender → receiver) ──────
        for sid, rid in pairs:
            if sid in uav_lut and rid in uav_lut:
                cp = comm_points.get(sid)
                _a2a_relay(uav_lut[sid], uav_lut[rid], cp, config)

        # ── Deliver to BS (receiver / solo only) ──────────────────────
        for u in env.uavs:
            if u.role in ("receiver", "solo"):
                _deliver_to_bs(u, env)

        # ── Track energy consumed this round ──────────────────────────
        energy_after = sum(u.e_current for u in env.uavs)
        metrics.record_energy_spent(max(0, energy_before - energy_after))

        all_paths[rnd] = round_paths

        # ── Metrics snapshot ──────────────────────────────────────────
        metrics.update(env, rnd, env.current_time)

        # ── Phase 5: Alternating Charging (Algorithm 5) ───────────────
        if (with_alternating_charging and pairs
                and should_swap(env.uavs, pairs, rnd, config)):
            pairs = swap_roles(env.uavs, pairs, clusters)
            handle_charging(env.uavs, pairs, env.base_station, config, env)

        # ── Termination check ─────────────────────────────────────────
        active = [u for u in env.uavs if u.e_current > 0]
        if not active:
            if verbose:
                print(f"  Round {rnd}: all UAVs depleted -- stopping.")
            break

        # Stop if no sensor has energy to transmit (no more useful work)
        live_sensors = sum(1 for sn in env.sensor_nodes if sn.e_current > 0)
        if live_sensors == 0:
            if verbose:
                print(f"  Round {rnd}: all sensors depleted -- stopping.")
            break

        if verbose and rnd % 10 == 0:
            s = metrics.get_summary()
            print(f"  Round {rnd:3d} | AoI {s['final_aoi']:8.1f} | "
                  f"ePDR {s['effective_pdr']:.3f} | active {len(active)} | "
                  f"sensors {live_sensors}")

    if verbose:
        s = metrics.get_summary()
        print(f"\n=== Simulation Complete ===")
        print(f"  Rounds : {s['total_rounds']}")
        print(f"  Runtime: {s['runtime']:.0f} s")
        print(f"  Avg AoI: {s['avg_aoi']:.2f} rounds")
        print(f"  Eff PDR: {s['effective_pdr']:.3f}")
        print(f"  Packets: {s['total_packets_delivered']}"
              f"/{s['total_packets_generated']}"
              f" (expired: {s['total_packets_expired']})")
        print(f"  Energy : {s['total_energy_consumed']/1e3:.1f} kJ consumed")
        print(f"  E/pkt  : {s['energy_per_packet']:.1f} J/packet")

    return env, metrics, all_paths, all_clusters
