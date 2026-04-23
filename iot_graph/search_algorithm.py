"""
Joint Iterative Search — Algorithm 5 from search.pdf §9.

4-phase pipeline:
  Phase 0: BS strength + Voronoi (once)
  Phase 1: Node classification (once)
  Phase 2: MSCT per UAV (once)
  Phase 3: Iterative loop (R iterations)
    - JIALNS → TEN + min-cost flow → battery fairness → convergence check
    - Re-run division + matching if fairness threshold exceeded
"""
import numpy as np

from phase0_bs_strength import (
    compute_bs_strength,
    build_power_voronoi,
    assign_gateways_to_zones,
    assign_all_nodes_to_zones,
)
from phase1_classification import classify_nodes, select_convergence_nodes
from phase2_msct import build_all_mscts
from phase3a_gateway_division import strength_weighted_division
from phase3b_min_cost_flow import (
    build_time_expanded_network,
    solve_min_cost_flow,
    apply_battery_fairness_reg,
)
from phase3c_jialns import (
    jialns_path_planning,
    compute_communication_point,
    adjust_speeds_for_rendezvous,
    total_path_distance,
)
from phase3d_relay_charging import (
    extended_relay_matching,
    swap_roles,
    should_swap,
    alternating_charging_multi_bs,
)


def joint_iterative_search(env, cfg, rng, verbose=False):
    """Run the joint 4-phase iterative search (Algorithm 5).

    Returns
    -------
    round_paths : dict {uav_id: [(x,y), ...]}
    clusters : dict {cluster_id: [subnet_id, ...]}
    pairs : list[(sender_id, receiver_id)]
    comm_points : dict {uav_id: (x,y)}
    zone_assign_fn : callable
    node_classes : dict {node_id: 'P'/'C'/'I'}
    search_costs : list of F* per iteration
    """
    # ── Phase 0: BS Strength + Power-Voronoi (once) ─────────────────
    strengths = compute_bs_strength(env.base_stations, cfg.num_uavs, cfg)
    zone_assign_fn, zone_radii = build_power_voronoi(env.base_stations, cfg)

    # Assign all nodes to zones
    node_zones = assign_all_nodes_to_zones(
        env.sensor_nodes, zone_assign_fn, env.base_stations
    )
    for bs in env.base_stations:
        bs.zone_nodes = node_zones.get(bs.id, [])

    # ── Phase 1: Convergence node selection + classification ────────
    conv_nodes = select_convergence_nodes(env)
    if not conv_nodes:
        return {}, {i: [] for i in range(cfg.num_uavs)}, [], {}, zone_assign_fn, {}, []

    # Assign gateways to zones
    gw_zones = assign_gateways_to_zones(conv_nodes, zone_assign_fn, env.base_stations)

    # Classify nodes (all sensors, not just gateways)
    G_P, G_C, G_I, node_classes = classify_nodes(
        conv_nodes, gw_zones, env.base_stations, cfg,
        all_sensor_nodes=env.sensor_nodes, zone_assign_fn=zone_assign_fn,
    )
    env.node_classes = node_classes

    if verbose:
        print(f"    Phase 1: P={len(G_P)}, C={len(G_C)}, I={len(G_I)} gateways")

    # ── Phase 2: MSCT per UAV ───────────────────────────────────────
    # First, divide area to get initial UAV assignments
    clusters = strength_weighted_division(
        conv_nodes, env.base_stations, cfg.num_uavs, cfg, rng
    )
    for u in env.uavs:
        u.cluster_id = u.id if u.id in clusters else (
            min(clusters.keys()) if clusters else -1
        )

    # Build UAV assignments for MSCT
    uav_assignments = {}
    for u in env.uavs:
        cid = u.cluster_id if u.cluster_id in clusters else u.id
        sids = clusters.get(cid, [])
        uav_assignments[u.id] = [cn for cn in conv_nodes if cn.subnet_id in sids]

    msct_results = build_all_mscts(env, conv_nodes, uav_assignments)

    if verbose:
        for uid, (edges, nbhood) in msct_results.items():
            print(f"    Phase 2: UAV {uid} MSCT has {len(edges)} edges, "
                  f"{len(nbhood)} nodes in neighbourhood")

    # ── Phase 3: Iterative search loop ──────────────────────────────
    search_costs = []
    best_cost = float("inf")
    best_paths = {}
    best_pairs = []
    best_comm_points = {}

    for iteration in range(cfg.jialns_R):
        # Phase 3a: Relay matching
        pairs = extended_relay_matching(
            env.uavs, clusters, env.subnets, env.base_stations, cfg,
            zone_assign_fn=zone_assign_fn,
        )

        # Phase 3c: JIALNS path planning
        comm_points = {}
        sub_lut = {s.id: s for s in env.subnets}
        uav_lut = {u.id: u for u in env.uavs}

        for sid, rid in pairs:
            s_sids = clusters.get(
                next(u for u in env.uavs if u.id == sid).cluster_id,
                clusters.get(sid, [])
            )
            r_sids = clusters.get(
                next(u for u in env.uavs if u.id == rid).cluster_id,
                clusters.get(rid, [])
            )
            s_cn = [n for n in conv_nodes if n.subnet_id in s_sids]
            r_cn = [n for n in conv_nodes if n.subnet_id in r_sids]
            cp = compute_communication_point(s_cn, r_cn)
            if cp is not None:
                comm_points[sid] = cp
                comm_points[rid] = cp

        # Build paths
        round_paths = {}
        for u in env.uavs:
            if u.e_current <= 0:
                continue
            cid = u.cluster_id if u.cluster_id in clusters else u.id
            u_sids = clusters.get(cid, [])
            u_cn = [n for n in conv_nodes if n.subnet_id in u_sids]
            cp = comm_points.get(u.id)

            # Determine start/end
            start = (u.x, u.y)
            nearest_bs = env.nearest_bs(u.x, u.y)
            bs_pos = (nearest_bs.x, nearest_bs.y)

            if u.role == "sender":
                end = tuple(cp) if cp is not None else bs_pos
            else:
                end = bs_pos

            round_paths[u.id] = jialns_path_planning(
                u, u_cn, cp, env.base_stations, cfg, rng,
                start_pos=start, end_pos=end,
                node_classes=node_classes,
            )

        # Speed synchronisation
        for sid, rid in pairs:
            if sid in round_paths and rid in round_paths:
                cp = comm_points.get(sid)
                if cp is not None:
                    vs, vr = adjust_speeds_for_rendezvous(
                        round_paths[sid], round_paths[rid], cp, cfg,
                    )
                    uav_lut[sid].speed = vs
                    uav_lut[rid].speed = vr

        # Phase 3b: Build TEN and solve min-cost flow
        ten, total_demand = build_time_expanded_network(
            env.sensor_nodes, env.uavs, env.base_stations,
            round_paths, msct_results, cfg,
        )

        # Apply battery fairness regularisation
        apply_battery_fairness_reg(ten, env.sensor_nodes, cfg)

        # Solve flow
        flow_cost, flow_amount = solve_min_cost_flow(ten)

        # Compute total cost: sum of path distances + flow cost
        total_cost = sum(total_path_distance(p) for p in round_paths.values())
        total_cost += flow_cost
        search_costs.append(total_cost)

        if total_cost < best_cost:
            best_cost = total_cost
            best_paths = dict(round_paths)
            best_pairs = list(pairs)
            best_comm_points = dict(comm_points)

        if verbose:
            print(f"    Phase 3 iter {iteration}: cost={total_cost:.1f}, "
                  f"flow={flow_amount}")

    return (best_paths, clusters, best_pairs, best_comm_points,
            zone_assign_fn, node_classes, search_costs)
