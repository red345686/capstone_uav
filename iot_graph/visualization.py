"""
Visualization utilities — environment plots, metrics curves, UAV-count comparison,
AoI heatmap, PDR bar chart, battery life curves, battery fairness, BS zones,
search convergence.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                     # noqa: E402
import matplotlib.patches as mpatches               # noqa: E402
import numpy as np                                   # noqa: E402
from scipy.interpolate import griddata               # noqa: E402

COLORS = [
    "#e74c3c", "#3498db", "#2ecc71", "#f39c12",
    "#9b59b6", "#1abc9c", "#e67e22", "#34495e",
]

BS_COLORS = ["#0000FF", "#FF00FF", "#00CED1", "#FF4500"]
CLASS_COLORS = {"P": "#2ecc71", "C": "#f39c12", "I": "#e74c3c"}


# ═══════════════════════════════════════════════════════════════════════════
# Environment snapshot
# ═══════════════════════════════════════════════════════════════════════════
def plot_environment(env, clusters=None, paths=None, comm_points=None,
                     round_num=0, save_path=None, title=None,
                     zone_assign_fn=None, node_classes=None):
    fig, ax = plt.subplots(figsize=(12, 12))
    cfg = env.config
    ax.set_xlim(-20, cfg.area_width + 20)
    ax.set_ylim(-20, cfg.area_height + 20)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_facecolor("#f8f9fa")
    ax.add_patch(plt.Rectangle((0, 0), cfg.area_width, cfg.area_height,
                                fill=False, ec="black", lw=2))

    # BS zone background (power-Voronoi)
    if zone_assign_fn and len(env.base_stations) > 1:
        grid_x = np.linspace(0, cfg.area_width, 100)
        grid_y = np.linspace(0, cfg.area_height, 100)
        zone_grid = np.zeros((100, 100))
        for ix, gx in enumerate(grid_x):
            for iy, gy in enumerate(grid_y):
                zone_grid[iy, ix] = zone_assign_fn(gx, gy)
        ax.contourf(grid_x, grid_y, zone_grid, levels=len(env.base_stations),
                    alpha=0.08, cmap="tab10")

    # Sensor nodes (color by P/C/I classification or cluster)
    for sub in env.subnets:
        for n in sub.nodes:
            if node_classes and n.id in node_classes:
                color = CLASS_COLORS.get(node_classes[n.id], "#cccccc")
            elif clusters:
                color = "#cccccc"
                for cid, sids in clusters.items():
                    if sub.id in sids:
                        color = COLORS[cid % len(COLORS)]
                        break
            else:
                color = "#cccccc"
            mk = "*" if n.is_convergence else "o"
            sz = 150 if n.is_convergence else 30
            ax.scatter(n.x, n.y, c=color, marker=mk, s=sz,
                       alpha=0.85, edgecolors="k", linewidth=0.5, zorder=3)

    # Independent nodes
    for n in env.independent_nodes:
        color = CLASS_COLORS.get(node_classes.get(n.id, "I"), "#e74c3c") if node_classes else "#e74c3c"
        ax.scatter(n.x, n.y, c=color, marker="d", s=50,
                   alpha=0.85, edgecolors="k", linewidth=0.5, zorder=3)

    # Base stations (multiple, different colors)
    for bs in env.base_stations:
        bc = BS_COLORS[bs.id % len(BS_COLORS)]
        ax.scatter(bs.x, bs.y, c=bc, marker="^", s=400,
                   edgecolors="k", lw=2, zorder=5)
        ax.annotate(f"BS{bs.id}", (bs.x, bs.y),
                    fontsize=10, fontweight="bold", ha="center",
                    xytext=(0, 18), textcoords="offset points", color=bc)

    # UAV paths
    if paths:
        for uid, path in paths.items():
            if not path:
                continue
            c = COLORS[uid % len(COLORS)]
            xs, ys = zip(*path)
            ax.plot(xs, ys, "-", color=c, lw=2, alpha=0.7, zorder=4)
            for i in range(0, len(path) - 1, max(1, len(path) // 5)):
                ax.annotate(
                    "", xy=path[i + 1], xytext=path[i],
                    arrowprops=dict(arrowstyle="->", color=c, lw=1.5),
                    zorder=4,
                )

    # Communication points
    if comm_points:
        seen = set()
        for _uid, cp in comm_points.items():
            key = (round(cp[0], 1), round(cp[1], 1))
            if key in seen:
                continue
            seen.add(key)
            ax.scatter(cp[0], cp[1], c="yellow", marker="D", s=200,
                       edgecolors="k", lw=2, zorder=5)
            ax.annotate("Pn", (cp[0], cp[1]), fontsize=8, fontweight="bold",
                        ha="center", xytext=(0, 10), textcoords="offset points")

    # UAV markers
    for u in env.uavs:
        c = COLORS[u.id % len(COLORS)]
        ax.scatter(u.x, u.y, c=c, marker="s", s=200,
                   edgecolors="k", lw=2, zorder=6)
        ax.annotate(f"UAV{u.id} ({u.role})", (u.x, u.y),
                    fontsize=8, ha="center",
                    xytext=(0, 12), textcoords="offset points")

    # Legend
    handles = [
        plt.Line2D([], [], marker="o", color=CLASS_COLORS["P"], ls="", ms=6, label="Proximal (P)"),
        plt.Line2D([], [], marker="o", color=CLASS_COLORS["C"], ls="", ms=6, label="Contested (C)"),
        plt.Line2D([], [], marker="o", color=CLASS_COLORS["I"], ls="", ms=6, label="Isolated (I)"),
        plt.Line2D([], [], marker="d", color="#e74c3c", ls="", ms=6, label="Independent"),
        plt.Line2D([], [], marker="*", color="gray", ls="", ms=10, label="Convergence Node"),
        plt.Line2D([], [], marker="D", color="yellow", ls="", ms=8, label="Comm. Point"),
    ]
    for i, bs in enumerate(env.base_stations):
        handles.append(plt.Line2D([], [], marker="^",
                                   color=BS_COLORS[i % len(BS_COLORS)],
                                   ls="", ms=10, label=f"BS {bs.id}"))
    for i in range(cfg.num_uavs):
        handles.append(mpatches.Patch(color=COLORS[i % len(COLORS)],
                                       label=f"UAV {i} area"))
    ax.legend(handles=handles, loc="upper right", fontsize=8)
    ax.set_title(title or f"Multi-Hop IoT-UAV – Round {round_num}",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# Metrics over time
# ═══════════════════════════════════════════════════════════════════════════
def plot_metrics(metrics_list, labels, save_path=None):
    fig, axes = plt.subplots(2, 4, figsize=(24, 10))
    fig.suptitle("Simulation Metrics Comparison", fontsize=16, fontweight="bold")
    cs = COLORS[: len(metrics_list)]

    for m, lbl, c in zip(metrics_list, labels, cs):
        axes[0, 0].plot(m.round_times, m.aoi_history,
                        "-o", color=c, label=lbl, lw=2, ms=3)
        axes[0, 1].plot(m.round_times, m.pdr_history,
                        "-o", color=c, label=lbl, lw=2, ms=3)
        axes[0, 2].plot(m.round_times, m.energy_history,
                        "-o", color=c, label=lbl, lw=2, ms=3)
        axes[0, 3].plot(m.round_times, m.fairness_history,
                        "-o", color=c, label=lbl, lw=2, ms=3)
        axes[1, 0].plot(m.round_times,
                        [e / 1e3 for e in m.energy_consumed_history],
                        "-o", color=c, label=lbl, lw=2, ms=3)
        axes[1, 1].plot(m.round_times, m.composite_obj_history,
                        "-o", color=c, label=lbl, lw=2, ms=3)

    for ax, yl, t in zip(
        [axes[0, 0], axes[0, 1], axes[0, 2], axes[0, 3],
         axes[1, 0], axes[1, 1]],
        ["System AoI", "PDR", "Remaining Energy (J)", "Battery Fairness (Δ_batt)",
         "Cumulative Energy (kJ)", "Composite Objective"],
        ["Age of Information", "Packet Delivery Ratio",
         "Remaining Energy", "Battery Fairness",
         "Cumulative Energy Consumed", "Composite Objective (Eq 23)"],
    ):
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(yl)
        ax.set_title(t)
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Runtime bar chart
    ax = axes[1, 2]
    x = np.arange(len(labels))
    rt = [m.runtime for m in metrics_list]
    bars = ax.bar(x, rt, color=cs, alpha=0.8, edgecolor="k")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylabel("Runtime (s)")
    ax.set_title("Total Runtime")
    for b, v in zip(bars, rt):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.5,
                f"{v:.0f}s", ha="center", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    # Energy per packet
    ax = axes[1, 3]
    epp = [m.get_summary()["energy_per_packet"] for m in metrics_list]
    bars = ax.bar(x, epp, color=cs, alpha=0.8, edgecolor="k")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylabel("Energy / Packet (J)")
    ax.set_title("Energy Efficiency")
    for b, v in zip(bars, epp):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.5,
                f"{v:.0f}", ha="center", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# UAV count sweep
# ═══════════════════════════════════════════════════════════════════════════
def plot_uav_count_comparison(results, save_path=None):
    fig, axes = plt.subplots(2, 4, figsize=(24, 10))
    fig.suptitle("Performance vs Number of UAVs", fontsize=16, fontweight="bold")
    ns = sorted(results.keys())
    x = np.arange(len(ns))
    w = 0.35

    # Row 1 — use avg_aoi (more stable than final_aoi which can spike at end)
    axes[0, 0].bar(x - w/2, [results[n].get("avg_aoi_w", results[n]["avg_aoi"]) for n in ns],
                   w, label="With Charging", color="#3498db", alpha=0.8, ec="k")
    axes[0, 0].bar(x + w/2, [results[n].get("avg_aoi_wo", results[n]["avg_aoi"]) for n in ns],
                   w, label="Without Charging", color="#e74c3c", alpha=0.8, ec="k")
    axes[0, 0].set(xlabel="# UAVs", ylabel="AoI (rounds)", title="Avg AoI vs UAV Count")
    axes[0, 0].set_xticks(x); axes[0, 0].set_xticklabels(ns)
    axes[0, 0].legend(); axes[0, 0].grid(True, alpha=0.3, axis="y")

    axes[0, 1].bar(x - w/2, [results[n].get("pdr_with", results[n]["final_pdr"]) for n in ns],
                   w, label="With Charging", color="#2ecc71", alpha=0.8, ec="k")
    axes[0, 1].bar(x + w/2, [results[n].get("pdr_without", results[n]["final_pdr"]) for n in ns],
                   w, label="Without Charging", color="#e74c3c", alpha=0.8, ec="k")
    axes[0, 1].set(xlabel="# UAVs", ylabel="Effective PDR",
                   title="Eff. PDR (del / (del+exp))")
    axes[0, 1].set_xticks(x); axes[0, 1].set_xticklabels(ns)
    axes[0, 1].legend(); axes[0, 1].grid(True, alpha=0.3, axis="y")

    axes[0, 2].bar(x - w/2, [results[n].get("runtime_with", 0) for n in ns],
                   w, label="With Charging", color="#2ecc71", alpha=0.8, ec="k")
    axes[0, 2].bar(x + w/2, [results[n].get("runtime_without", 0) for n in ns],
                   w, label="Without Charging", color="#e74c3c", alpha=0.8, ec="k")
    axes[0, 2].set(xlabel="# UAVs", ylabel="Runtime (s)", title="Runtime Comparison")
    axes[0, 2].set_xticks(x); axes[0, 2].set_xticklabels(ns)
    axes[0, 2].legend(); axes[0, 2].grid(True, alpha=0.3, axis="y")

    # Fairness
    axes[0, 3].bar(x - w/2, [results[n].get("fairness_with", 0) for n in ns],
                   w, label="With Charging", color="#3498db", alpha=0.8, ec="k")
    axes[0, 3].bar(x + w/2, [results[n].get("fairness_without", 0) for n in ns],
                   w, label="Without Charging", color="#e74c3c", alpha=0.8, ec="k")
    axes[0, 3].set(xlabel="# UAVs", ylabel="Δ_batt", title="Battery Fairness")
    axes[0, 3].set_xticks(x); axes[0, 3].set_xticklabels(ns)
    axes[0, 3].legend(); axes[0, 3].grid(True, alpha=0.3, axis="y")

    # Row 2
    axes[1, 0].bar(
        x - w/2, [results[n].get("energy_consumed_with", 0) / 1e3 for n in ns],
        w, label="With Charging", color="#2ecc71", alpha=0.8, ec="k")
    axes[1, 0].bar(
        x + w/2, [results[n].get("energy_consumed_without", 0) / 1e3 for n in ns],
        w, label="Without Charging", color="#e74c3c", alpha=0.8, ec="k")
    axes[1, 0].set(xlabel="# UAVs", ylabel="Energy (kJ)", title="Total Energy Consumed")
    axes[1, 0].set_xticks(x); axes[1, 0].set_xticklabels(ns)
    axes[1, 0].legend(); axes[1, 0].grid(True, alpha=0.3, axis="y")

    axes[1, 1].bar(
        x - w/2, [results[n].get("epp_with", 0) for n in ns],
        w, label="With Charging", color="#2ecc71", alpha=0.8, ec="k")
    axes[1, 1].bar(
        x + w/2, [results[n].get("epp_without", 0) for n in ns],
        w, label="Without Charging", color="#e74c3c", alpha=0.8, ec="k")
    axes[1, 1].set(xlabel="# UAVs", ylabel="Energy / Packet (J)",
                   title="Energy Efficiency")
    axes[1, 1].set_xticks(x); axes[1, 1].set_xticklabels(ns)
    axes[1, 1].legend(); axes[1, 1].grid(True, alpha=0.3, axis="y")

    axes[1, 2].bar(
        x - w/2, [results[n].get("avg_del_w", 0) for n in ns],
        w, label="With Charging", color="#2ecc71", alpha=0.8, ec="k")
    axes[1, 2].bar(
        x + w/2, [results[n].get("avg_del_wo", 0) for n in ns],
        w, label="Without Charging", color="#e74c3c", alpha=0.8, ec="k")
    axes[1, 2].set(xlabel="# UAVs", ylabel="Packets / Round",
                   title="Avg Delivery per Round")
    axes[1, 2].set_xticks(x); axes[1, 2].set_xticklabels(ns)
    axes[1, 2].legend(); axes[1, 2].grid(True, alpha=0.3, axis="y")

    # Composite objective
    axes[1, 3].bar(
        x - w/2, [results[n].get("composite_with", 0) for n in ns],
        w, label="With Charging", color="#3498db", alpha=0.8, ec="k")
    axes[1, 3].bar(
        x + w/2, [results[n].get("composite_without", 0) for n in ns],
        w, label="Without Charging", color="#e74c3c", alpha=0.8, ec="k")
    axes[1, 3].set(xlabel="# UAVs", ylabel="Composite Obj.",
                   title="Composite Objective (Eq 23)")
    axes[1, 3].set_xticks(x); axes[1, 3].set_xticklabels(ns)
    axes[1, 3].legend(); axes[1, 3].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# AoI Heatmap (multi-BS)
# ═══════════════════════════════════════════════════════════════════════════
def plot_aoi_heatmap(metrics, env, save_path=None, title=None):
    fig, ax = plt.subplots(figsize=(12, 10))

    node_ids = list(metrics.per_node_aoi.keys())
    if not node_ids:
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes, fontsize=14)
        plt.close()
        return fig

    xs = np.array([metrics.per_node_positions[nid][0] for nid in node_ids])
    ys = np.array([metrics.per_node_positions[nid][1] for nid in node_ids])
    aoi_vals = np.array([metrics.per_node_aoi[nid] for nid in node_ids])

    cfg = env.config
    grid_x = np.linspace(0, cfg.area_width, 200)
    grid_y = np.linspace(0, cfg.area_height, 200)
    gx, gy = np.meshgrid(grid_x, grid_y)

    try:
        grid_aoi = griddata((xs, ys), aoi_vals, (gx, gy), method="cubic")
        grid_aoi = np.nan_to_num(grid_aoi, nan=np.nanmean(aoi_vals))
        # Clip to valid range — cubic interpolation can produce negatives
        grid_aoi = np.clip(grid_aoi, 0, np.max(aoi_vals) * 1.1)
    except Exception:
        grid_aoi = griddata((xs, ys), aoi_vals, (gx, gy), method="nearest")

    im = ax.pcolormesh(grid_x, grid_y, grid_aoi, cmap="YlOrRd", shading="auto",
                       vmin=0, vmax=np.max(aoi_vals))
    plt.colorbar(im, ax=ax, label="Age of Information (rounds)")

    ax.scatter(xs, ys, c=aoi_vals, cmap="YlOrRd", s=40, edgecolors="k",
               linewidth=0.5, zorder=3, vmin=aoi_vals.min(), vmax=aoi_vals.max())

    # Multiple base stations
    for bs in env.base_stations:
        bc = BS_COLORS[bs.id % len(BS_COLORS)]
        ax.scatter(bs.x, bs.y, c=bc, marker="^", s=300,
                   edgecolors="k", lw=2, zorder=5)
        ax.annotate(f"BS{bs.id}", (bs.x, bs.y), fontsize=10, fontweight="bold",
                    ha="center", xytext=(0, 15), textcoords="offset points",
                    color=bc)

    ax.set_xlim(0, cfg.area_width)
    ax.set_ylim(0, cfg.area_height)
    ax.set_aspect("equal")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(title or "System AoI Heatmap", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# PDR Bar Chart
# ═══════════════════════════════════════════════════════════════════════════
def plot_pdr_barchart(results, save_path=None):
    fig, ax = plt.subplots(figsize=(10, 6))
    ns = sorted(results.keys())
    x = np.arange(len(ns))
    w = 0.35

    pdr_with = [results[n].get("pdr_with", results[n].get("final_pdr", 0)) for n in ns]
    pdr_without = [results[n].get("pdr_without", results[n].get("final_pdr", 0)) for n in ns]

    bars1 = ax.bar(x - w/2, pdr_with, w, label="With Charging",
                   color="#2ecc71", alpha=0.85, edgecolor="k")
    bars2 = ax.bar(x + w/2, pdr_without, w, label="Without Charging",
                   color="#e74c3c", alpha=0.85, edgecolor="k")

    for b, v in zip(bars1, pdr_with):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.005,
                f"{v:.3f}", ha="center", fontsize=9, fontweight="bold")
    for b, v in zip(bars2, pdr_without):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.005,
                f"{v:.3f}", ha="center", fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{n} UAV{'s' if n > 1 else ''}" for n in ns])
    ax.set_ylabel("Packet Delivery Ratio")
    ax.set_title("Packet Delivery Ratio vs UAV Count", fontsize=14, fontweight="bold")
    ax.set_ylim(0, min(max(max(pdr_with), max(pdr_without)) * 1.15, 1.05))
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# Battery Life Curves (multi-BS)
# ═══════════════════════════════════════════════════════════════════════════
def plot_battery_life(metrics_with, metrics_without, num_uavs, save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f"Battery Life Comparison — {num_uavs} UAV(s)",
                 fontsize=14, fontweight="bold")

    for ax, met, label_pref in zip(
        axes, [metrics_with, metrics_without],
        ["With Charging", "Without Charging"]
    ):
        for uid, e_vals in met.per_uav_energy.items():
            t = met.round_times[:len(e_vals)]
            ax.plot(t, [e / 1e3 for e in e_vals], "-o",
                    color=COLORS[uid % len(COLORS)], lw=2, ms=3,
                    label=f"UAV {uid}")
        t = met.round_times[:len(met.energy_history)]
        ax.plot(t, [e / 1e3 for e in met.energy_history], "--k", lw=2,
                alpha=0.5, label="Total")

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Remaining Energy (kJ)")
        ax.set_title(label_pref)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# Battery Fairness Plot
# ═══════════════════════════════════════════════════════════════════════════
def plot_battery_fairness(metrics_with, metrics_without, env, num_uavs,
                          save_path=None):
    """Sensor battery depletion distribution (rho_i histogram + delta_batt)."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Battery Fairness — {num_uavs} UAV(s)",
                 fontsize=14, fontweight="bold")

    # Panel 1: rho_i distribution (final round, with charging)
    if metrics_with.rho_history:
        rho_final = metrics_with.rho_history[-1]
        axes[0].hist(rho_final, bins=20, color="#3498db", alpha=0.8,
                     edgecolor="k", label="With Charging")
    if metrics_without.rho_history:
        rho_final_wo = metrics_without.rho_history[-1]
        axes[0].hist(rho_final_wo, bins=20, color="#e74c3c", alpha=0.5,
                     edgecolor="k", label="Without Charging")
    axes[0].set_xlabel("Battery Ratio (rho_i = E_i / E_max)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Sensor Battery Distribution (Final Round)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Panel 2: delta_batt over time
    t_w = metrics_with.round_times[:len(metrics_with.fairness_history)]
    t_wo = metrics_without.round_times[:len(metrics_without.fairness_history)]
    axes[1].plot(t_w, metrics_with.fairness_history, "-o", color="#3498db",
                 lw=2, ms=3, label="With Charging")
    axes[1].plot(t_wo, metrics_without.fairness_history, "-o", color="#e74c3c",
                 lw=2, ms=3, label="Without Charging")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Δ_batt")
    axes[1].set_title("Battery Fairness Over Time (Eq 21-22)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Panel 3: composite objective over time
    t_w2 = metrics_with.round_times[:len(metrics_with.composite_obj_history)]
    t_wo2 = metrics_without.round_times[:len(metrics_without.composite_obj_history)]
    axes[2].plot(t_w2, metrics_with.composite_obj_history, "-o", color="#3498db",
                 lw=2, ms=3, label="With Charging")
    axes[2].plot(t_wo2, metrics_without.composite_obj_history, "-o", color="#e74c3c",
                 lw=2, ms=3, label="Without Charging")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Composite Objective")
    axes[2].set_title("Composite Objective (Eq 23)")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# BS Zone Map
# ═══════════════════════════════════════════════════════════════════════════
def plot_bs_zone_map(env, zone_assign_fn, node_classes=None, save_path=None):
    """Power-Voronoi zone visualization with node classifications."""
    fig, ax = plt.subplots(figsize=(12, 10))
    cfg = env.config

    # Zone background
    if zone_assign_fn:
        grid_x = np.linspace(0, cfg.area_width, 200)
        grid_y = np.linspace(0, cfg.area_height, 200)
        zone_grid = np.zeros((200, 200))
        for ix, gx in enumerate(grid_x):
            for iy, gy in enumerate(grid_y):
                zone_grid[iy, ix] = zone_assign_fn(gx, gy)
        ax.pcolormesh(grid_x, grid_y, zone_grid, cmap="tab10",
                      alpha=0.15, shading="auto")

    # Sensor nodes
    for n in env.sensor_nodes:
        if node_classes and n.id in node_classes:
            color = CLASS_COLORS.get(node_classes[n.id], "#cccccc")
        else:
            color = "#cccccc"
        mk = "d" if n.is_independent else ("*" if n.is_convergence else "o")
        sz = 60 if n.is_convergence else 20
        ax.scatter(n.x, n.y, c=color, marker=mk, s=sz,
                   alpha=0.8, edgecolors="k", linewidth=0.3, zorder=3)

    # Base stations
    for bs in env.base_stations:
        bc = BS_COLORS[bs.id % len(BS_COLORS)]
        ax.scatter(bs.x, bs.y, c=bc, marker="^", s=400,
                   edgecolors="k", lw=2, zorder=5)
        ax.annotate(f"BS{bs.id} (Ψ={bs.strength:.2f})", (bs.x, bs.y),
                    fontsize=9, fontweight="bold", ha="center",
                    xytext=(0, 18), textcoords="offset points", color=bc)

    handles = [
        plt.Line2D([], [], marker="o", color=CLASS_COLORS["P"], ls="", ms=6, label="Proximal"),
        plt.Line2D([], [], marker="o", color=CLASS_COLORS["C"], ls="", ms=6, label="Contested"),
        plt.Line2D([], [], marker="o", color=CLASS_COLORS["I"], ls="", ms=6, label="Isolated"),
        plt.Line2D([], [], marker="d", color="#e74c3c", ls="", ms=6, label="Independent"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=10)
    ax.set_xlim(0, cfg.area_width)
    ax.set_ylim(0, cfg.area_height)
    ax.set_aspect("equal")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("Power-Voronoi BS Zones + Node Classification",
                 fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# Search Convergence
# ═══════════════════════════════════════════════════════════════════════════
def plot_search_convergence(metrics, save_path=None):
    """F* vs iteration for the joint iterative search."""
    fig, ax = plt.subplots(figsize=(8, 5))
    costs = metrics.search_cost_history
    if costs:
        # Plot raw cost and best-so-far
        best_so_far = []
        best = float("inf")
        for c in costs:
            best = min(best, c)
            best_so_far.append(best)
        ax.plot(range(len(costs)), costs, "-o", color="#3498db", lw=1.5,
                ms=5, alpha=0.5, label="Iteration cost")
        ax.plot(range(len(best_so_far)), best_so_far, "-s", color="#e74c3c",
                lw=2.5, ms=6, label="Best so far (F*)")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Total Cost (F*)")
        ax.set_title("Joint Search Convergence", fontsize=14, fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, "No search data", ha="center", va="center",
                transform=ax.transAxes, fontsize=14)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return fig
