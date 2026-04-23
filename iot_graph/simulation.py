"""
Main simulation loop — orchestrates the 4-phase joint iterative search.

Multi-Hop IoT-UAV data flow:
    Sensor → (IoT multi-hop via H) → CN → UAV buffer → (UAV relay) → BS delivery

Key mechanics:
  - Multiple base stations with power-Voronoi zones
  - IoT-to-IoT multi-hop routing via graph H
  - UAV-to-UAV relay via V[t] graph
  - Buffer dynamics with per-node TTL and flow conservation
  - Battery fairness tracking (Δ_batt)
  - Composite objective: α·A_P + β·E_total + γ·Δ_batt
"""
import numpy as np

from config import SimConfig
from environment import Environment
from energy_model import flight_energy, hover_energy, sensor_tx_energy
from communication import (
    shannon_capacity_g2a,
    shannon_capacity_a2a,
    verify_a2a_los,
    iot_to_uav_valid,
)
from search_algorithm import joint_iterative_search
from phase3d_relay_charging import (
    swap_roles,
    should_swap,
    alternating_charging_multi_bs,
)
from metrics import MetricsTracker


# ═══════════════════════════════════════════════════════════════════════════
# Data Collection (Shannon-limited, buffer-based)
# ═══════════════════════════════════════════════════════════════════════════
def _collect_at_convergence_node(uav, sub, cn, env, cfg, time_remaining):
    """Collect data from *sub* via its convergence node *cn*.

    Packets go into uav.data_buffer — delivery to BS happens later.
    """
    if time_remaining <= 0:
        return 0, 0.0, []

    capacity_bps = shannon_capacity_g2a(0.0, cfg)
    if capacity_bps <= 0:
        return 0, 0.0, []

    pending = [sn for sn in sub.nodes if sn.packet_queue and sn.e_current > 0]
    total_units = sum(len(sn.packet_queue) for sn in pending)
    if total_units <= 0:
        return 0, 0.0, []

    total_bits = total_units * cfg.packet_size_bits
    hover_time = min(total_bits / capacity_bps, cfg.max_hover_time, time_remaining)
    collectible_bits = capacity_bps * hover_time
    collectible_units = int(collectible_bits / cfg.packet_size_bits)

    uav.e_current -= hover_energy(hover_time, cfg)
    if uav.e_current <= 0:
        uav.e_current = 0
        return 0, hover_time, []

    cn.e_current -= cfg.P_S_i * hover_time
    cn.e_current = max(cn.e_current, 0.0)

    collected = 0
    collected_sids = []
    for sn in pending:
        if collected >= collectible_units:
            break
        n_avail = len(sn.packet_queue)
        n_take = min(n_avail, collectible_units - collected)
        if n_take <= 0:
            continue

        tx_bits = n_take * cfg.packet_size_bits
        tx_d = np.hypot(sn.x - cn.x, sn.y - cn.y)
        e_tx = sensor_tx_energy(tx_bits, tx_d, cfg)
        sn.e_current -= e_tx
        if sn.e_current < 0:
            sn.e_current = 0

        sn.packet_queue = sn.packet_queue[n_take:]
        sn.packets_collected += n_take
        collected += n_take
        collected_sids.append(sn.id)

    return collected, hover_time, collected_sids


def _collect_from_independent(uav, node, env, cfg, time_remaining):
    """Collect data from an independent node directly."""
    if time_remaining <= 0 or not node.packet_queue or node.e_current <= 0:
        return 0, 0.0

    d_horiz = np.hypot(node.x - uav.x, node.y - uav.y)
    if not iot_to_uav_valid(d_horiz, cfg.uav_altitude, cfg):
        return 0, 0.0

    capacity_bps = shannon_capacity_g2a(d_horiz, cfg)
    if capacity_bps <= 0:
        return 0, 0.0

    n_pkts = len(node.packet_queue)
    total_bits = n_pkts * cfg.packet_size_bits
    hover_time = min(total_bits / capacity_bps, cfg.max_hover_time, time_remaining)
    collectible = int(capacity_bps * hover_time / cfg.packet_size_bits)
    n_take = min(collectible, n_pkts)

    if n_take <= 0:
        return 0, 0.0

    uav.e_current -= hover_energy(hover_time, cfg)
    if uav.e_current <= 0:
        uav.e_current = 0
        return 0, hover_time

    tx_bits = n_take * cfg.packet_size_bits
    e_tx = sensor_tx_energy(tx_bits, d_horiz, cfg)
    node.e_current -= e_tx
    node.e_current = max(node.e_current, 0.0)

    node.packet_queue = node.packet_queue[n_take:]
    node.packets_collected += n_take

    uav.data_buffer += n_take
    uav.collected_sensor_ids.append(node.id)

    return n_take, hover_time


# ═══════════════════════════════════════════════════════════════════════════
# Single-UAV path execution (TIME-BUDGETED)
# ═══════════════════════════════════════════════════════════════════════════
def _execute_path(uav, path, env, cfg):
    """Fly *uav* along *path*, collecting data at CN waypoints."""
    cn_lut = {}
    for sub in env.subnets:
        if sub.convergence_node_id < 0:
            continue
        node = next(
            (n for n in sub.nodes if n.id == sub.convergence_node_id), None
        )
        if node:
            cn_lut[(round(node.x, 4), round(node.y, 4))] = (sub, node)

    # Also build lookup for independent nodes
    ind_lut = {(round(n.x, 4), round(n.y, 4)): n for n in env.independent_nodes}

    collected = 0
    time_used = 0.0
    time_budget = cfg.round_duration

    for i in range(len(path) - 1):
        d = np.hypot(path[i + 1][0] - path[i][0],
                     path[i + 1][1] - path[i][1])

        speed = max(uav.speed, cfg.uav_speed_min)
        flight_time = d / speed if speed > 0 else 0.0

        if time_used + flight_time > time_budget:
            remaining_time = time_budget - time_used
            if remaining_time > 0 and d > 0:
                frac = min((remaining_time * speed) / d, 1.0)
                uav.x = path[i][0] + frac * (path[i + 1][0] - path[i][0])
                uav.y = path[i][1] + frac * (path[i + 1][1] - path[i][1])
                partial_d = remaining_time * speed
                uav.e_current -= flight_energy(partial_d, speed, cfg)
                uav.total_distance += partial_d
            time_used = time_budget
            break

        uav.e_current -= flight_energy(d, speed, cfg)
        uav.total_distance += d
        time_used += flight_time
        if uav.e_current <= 0:
            uav.e_current = 0
            break

        wp = path[i + 1]

        # Check CN waypoints
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
                break

        # Check independent nodes nearby
        for (ix, iy), ind_node in ind_lut.items():
            if np.hypot(wp[0] - ix, wp[1] - iy) < 10.0:
                time_remaining = time_budget - time_used
                c, hover_t = _collect_from_independent(
                    uav, ind_node, env, cfg, time_remaining
                )
                time_used += hover_t
                collected += c
                break

        if uav.e_current <= 0 or time_used >= time_budget:
            break

    uav.data_buffer += collected

    if path and time_used < time_budget:
        uav.x, uav.y = path[-1]

    return collected, time_used


# ═══════════════════════════════════════════════════════════════════════════
# Delivery to BS (supports any base station)
# ═══════════════════════════════════════════════════════════════════════════
def _deliver_to_bs(uav, env, metrics=None):
    """Transfer uav.data_buffer → env.total_packets_delivered.

    Delivers to the nearest BS.
    """
    if uav.data_buffer <= 0:
        return

    target_bs = env.nearest_bs(uav.x, uav.y)

    env.total_packets_delivered += uav.data_buffer
    uav.total_packets_delivered += uav.data_buffer
    target_bs.packets_received += uav.data_buffer

    if metrics is not None:
        metrics.record_bs_delivery(target_bs.id, uav.data_buffer)

    for sid in uav.collected_sensor_ids:
        sn = next((n for n in env.sensor_nodes if n.id == sid), None)
        if sn is not None:
            sn.last_collected_time = env.current_time
            sn.last_collected_round = env.current_round

    uav.data_buffer = 0
    uav.collected_sensor_ids = []


# ═══════════════════════════════════════════════════════════════════════════
# Air-to-Air Relay (multi-hop UAV relay via V[t])
# ═══════════════════════════════════════════════════════════════════════════
def _a2a_relay(sender_uav, receiver_uav, comm_point, cfg):
    """Transfer sender's data_buffer to receiver via A2A relay."""
    if comm_point is None or sender_uav.data_buffer <= 0:
        return

    s_pos = (sender_uav.x, sender_uav.y)
    r_pos = (receiver_uav.x, receiver_uav.y)
    if not verify_a2a_los(s_pos, r_pos, cfg):
        return

    d_a2a = max(np.hypot(s_pos[0] - r_pos[0], s_pos[1] - r_pos[1]), 1.0)

    # Check UAV-to-UAV distance threshold
    if d_a2a > cfg.d_UU_th:
        return

    capacity = shannon_capacity_a2a(d_a2a, cfg)
    if capacity <= 0:
        return

    data_bits = sender_uav.data_buffer * cfg.packet_size_bits
    transfer_time = data_bits / capacity

    sender_uav.e_current -= hover_energy(transfer_time, cfg)
    receiver_uav.e_current -= hover_energy(transfer_time, cfg)
    sender_uav.e_current = max(sender_uav.e_current, 0.0)
    receiver_uav.e_current = max(receiver_uav.e_current, 0.0)

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
    """Run the 4-phase joint iterative search simulation.

    Returns
    -------
    env, metrics, all_paths, all_clusters, zone_assign_fn, node_classes
    """
    rng = np.random.RandomState(config.seed)
    env = Environment(config)
    env.generate()
    metrics = MetricsTracker()
    all_paths: dict = {}
    all_clusters: dict = {}
    pairs = []
    zone_assign_fn = None
    node_classes = {}

    if verbose:
        total_sensors = len(env.sensor_nodes)
        n_ind = len(env.independent_nodes)
        print(f"=== Simulation Start ===")
        print(f"  Area       : {config.area_width}x{config.area_height} m")
        print(f"  UAVs       : {config.num_uavs}")
        print(f"  Base Stns  : {config.num_base_stations}")
        print(f"  Subnets    : {len(env.subnets)}")
        print(f"  Sensors    : {total_sensors} ({n_ind} independent)")
        print(f"  Charging   : {'ON' if with_alternating_charging else 'OFF'}")
        print(f"  Obj weights: alpha={config.alpha_obj}, "
              f"beta={config.beta_obj}, gamma={config.gamma_obj}")
        print()

    for rnd in range(config.max_rounds):
        env.current_round = rnd
        env.current_time = (rnd + 1) * config.round_duration

        # ── Generate data ───────────────────────────────────────────────
        env.generate_data(rnd)
        env.expire_stale_packets(rnd)

        # ── 4-Phase Joint Iterative Search ──────────────────────────────
        result = joint_iterative_search(env, config, rng, verbose=(verbose and rnd == 0))
        (round_paths, clusters, pairs, comm_points,
         zone_assign_fn, node_classes, search_costs) = result

        all_clusters[rnd] = dict(clusters)
        all_paths[rnd] = round_paths

        if rnd == 0:
            metrics.search_cost_history = search_costs

        for u in env.uavs:
            u.cluster_id = u.id if u.id in clusters else (
                min(clusters.keys()) if clusters else -1
            )

        # ── Snapshot energy before execution ────────────────────────────
        energy_before = sum(u.e_current for u in env.uavs)

        # ── Reset UAV buffers ───────────────────────────────────────────
        for u in env.uavs:
            u.data_buffer = 0
            u.collected_sensor_ids = []

        # ── Execute flight paths ────────────────────────────────────────
        for u in env.uavs:
            if u.id in round_paths and u.e_current > 0:
                _execute_path(u, round_paths[u.id], env, config)

        # ── A2A Relay ───────────────────────────────────────────────────
        uav_lut = {u.id: u for u in env.uavs}
        for sid, rid in pairs:
            if sid in uav_lut and rid in uav_lut:
                cp = comm_points.get(sid)
                _a2a_relay(uav_lut[sid], uav_lut[rid], cp, config)

        # ── Deliver to BS ───────────────────────────────────────────────
        for u in env.uavs:
            if u.role in ("receiver", "solo", "terminal"):
                _deliver_to_bs(u, env, metrics)

        # ── Track energy consumed ───────────────────────────────────────
        energy_after = sum(u.e_current for u in env.uavs)
        metrics.record_energy_spent(max(0, energy_before - energy_after))

        # ── Metrics snapshot ────────────────────────────────────────────
        metrics.update(env, rnd, env.current_time)

        # ── Alternating Charging (multi-BS) ─────────────────────────────
        if (with_alternating_charging and pairs
                and should_swap(env.uavs, pairs, rnd, config)):
            pairs = swap_roles(env.uavs, pairs, clusters)
            alternating_charging_multi_bs(
                env.uavs, pairs, env.base_stations, config, env
            )

        # ── Termination check ───────────────────────────────────────────
        active = [u for u in env.uavs if u.e_current > 0]
        if not active:
            if verbose:
                print(f"  Round {rnd}: all UAVs depleted -- stopping.")
            break

        live_sensors = sum(1 for sn in env.sensor_nodes if sn.e_current > 0)
        if live_sensors == 0:
            if verbose:
                print(f"  Round {rnd}: all sensors depleted -- stopping.")
            break

        if verbose and rnd % 10 == 0:
            s = metrics.get_summary()
            print(f"  Round {rnd:3d} | AoI {s['final_aoi']:8.1f} | "
                  f"ePDR {s['effective_pdr']:.3f} | "
                  f"fair {s['final_fairness']:.3f} | "
                  f"comp {s['final_composite']:.4f} | "
                  f"active {len(active)}")

    if verbose:
        s = metrics.get_summary()
        print(f"\n=== Simulation Complete ===")
        print(f"  Rounds    : {s['total_rounds']}")
        print(f"  Avg AoI   : {s['avg_aoi']:.2f} rounds")
        print(f"  Eff PDR   : {s['effective_pdr']:.3f}")
        print(f"  Fairness  : {s['final_fairness']:.4f} (Δ_batt)")
        print(f"  Composite : {s['final_composite']:.4f}")
        print(f"  Packets   : {s['total_packets_delivered']}"
              f"/{s['total_packets_generated']}"
              f" (expired: {s['total_packets_expired']})")
        print(f"  Energy    : {s['total_energy_consumed']/1e3:.1f} kJ consumed")
        print(f"  Per-BS    : {s['per_bs_delivered']}")

    return env, metrics, all_paths, all_clusters, zone_assign_fn, node_classes
