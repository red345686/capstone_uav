"""
Visualization utilities — environment plots, metrics curves, UAV-count comparison,
AoI heatmap (Figure 20), PDR bar chart (Figure 23), Battery life curves (Figure 24).
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


# ═══════════════════════════════════════════════════════════════════════════
# Environment snapshot
# ═══════════════════════════════════════════════════════════════════════════
def plot_environment(env, clusters=None, paths=None, comm_points=None,
                     round_num=0, save_path=None, title=None):
    fig, ax = plt.subplots(figsize=(12, 12))
    cfg = env.config
    ax.set_xlim(-20, cfg.area_width + 20)
    ax.set_ylim(-20, cfg.area_height + 20)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_facecolor("#f8f9fa")
    ax.add_patch(plt.Rectangle((0, 0), cfg.area_width, cfg.area_height,
                                fill=False, ec="black", lw=2))

    # sensor nodes
    for sub in env.subnets:
        color = "#cccccc"
        if clusters:
            for cid, sids in clusters.items():
                if sub.id in sids:
                    color = COLORS[cid % len(COLORS)]
                    break
        for n in sub.nodes:
            mk = "*" if n.is_convergence else "o"
            sz = 150 if n.is_convergence else 30
            ax.scatter(n.x, n.y, c=color, marker=mk, s=sz,
                       alpha=0.85, edgecolors="k", linewidth=0.5, zorder=3)

    # base station
    ax.scatter(env.base_station.x, env.base_station.y,
               c="red", marker="^", s=300, edgecolors="k", lw=2, zorder=5)
    ax.annotate("BS", (env.base_station.x, env.base_station.y),
                fontsize=10, fontweight="bold", ha="center",
                xytext=(0, 15), textcoords="offset points")

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

    # communication points
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

    # legend
    handles = [
        plt.Line2D([], [], marker="^", color="red", ls="", ms=10, label="Base Station"),
        plt.Line2D([], [], marker="o", color="gray", ls="", ms=6, label="Sensor"),
        plt.Line2D([], [], marker="*", color="gray", ls="", ms=10, label="Convergence Node"),
        plt.Line2D([], [], marker="D", color="yellow", ls="", ms=8, label="Comm. Point"),
    ]
    for i in range(cfg.num_uavs):
        handles.append(mpatches.Patch(color=COLORS[i % len(COLORS)],
                                       label=f"UAV {i} area"))
    ax.legend(handles=handles, loc="upper right", fontsize=9)
    ax.set_title(title or f"UAV Data Collection – Round {round_num}",
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
    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    fig.suptitle("Simulation Metrics Comparison", fontsize=16, fontweight="bold")
    cs = COLORS[: len(metrics_list)]

    for m, lbl, c in zip(metrics_list, labels, cs):
        axes[0, 0].plot(m.round_times, m.aoi_history,
                        "-o", color=c, label=lbl, lw=2, ms=3)
        axes[0, 1].plot(m.round_times, m.pdr_history,
                        "-o", color=c, label=lbl, lw=2, ms=3)
        axes[0, 2].plot(m.round_times, m.energy_history,
                        "-o", color=c, label=lbl, lw=2, ms=3)
        axes[1, 0].plot(m.round_times,
                        [e / 1e3 for e in m.energy_consumed_history],
                        "-o", color=c, label=lbl, lw=2, ms=3)

    for ax, yl, t in zip(
        [axes[0, 0], axes[0, 1], axes[0, 2], axes[1, 0]],
        ["System AoI", "PDR", "Remaining Energy (J)", "Cumulative Energy (kJ)"],
        ["Age of Information", "Packet Delivery Ratio",
         "Remaining Energy", "Cumulative Energy Consumed"],
    ):
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(yl)
        ax.set_title(t)
        ax.legend()
        ax.grid(True, alpha=0.3)

    # runtime bar chart
    ax = axes[1, 1]
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

    # energy per packet
    ax = axes[1, 2]
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
    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    fig.suptitle("Performance vs Number of UAVs", fontsize=16, fontweight="bold")
    ns = sorted(results.keys())
    x = np.arange(len(ns))
    w = 0.35

    # Row 1: AoI, Effective PDR, Runtime
    axes[0, 0].bar(x - w/2, [results[n].get("final_aoi_w", results[n]["avg_aoi"]) for n in ns],
                   w, label="With Charging", color="#3498db", alpha=0.8, ec="k")
    axes[0, 0].bar(x + w/2, [results[n].get("final_aoi_wo", results[n]["avg_aoi"]) for n in ns],
                   w, label="Without Charging", color="#e74c3c", alpha=0.8, ec="k")
    axes[0, 0].set(xlabel="# UAVs", ylabel="AoI (rounds)", title="Final AoI vs UAV Count")
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

    # Row 2: Total Energy Consumed, Energy per Packet, Total Distance
    axes[1, 0].bar(
        x - w/2, [results[n].get("energy_consumed_with", 0) / 1e3 for n in ns],
        w, label="With Charging", color="#2ecc71", alpha=0.8, ec="k")
    axes[1, 0].bar(
        x + w/2, [results[n].get("energy_consumed_without", 0) / 1e3 for n in ns],
        w, label="Without Charging", color="#e74c3c", alpha=0.8, ec="k")
    axes[1, 0].set(xlabel="# UAVs", ylabel="Energy (kJ)",
                   title="Total Energy Consumed")
    axes[1, 0].set_xticks(x); axes[1, 0].set_xticklabels(ns)
    axes[1, 0].legend(); axes[1, 0].grid(True, alpha=0.3, axis="y")

    axes[1, 1].bar(
        x - w/2, [results[n].get("epp_with", 0) for n in ns],
        w, label="With Charging", color="#2ecc71", alpha=0.8, ec="k")
    axes[1, 1].bar(
        x + w/2, [results[n].get("epp_without", 0) for n in ns],
        w, label="Without Charging", color="#e74c3c", alpha=0.8, ec="k")
    axes[1, 1].set(xlabel="# UAVs", ylabel="Energy / Packet (J)",
                   title="Energy Efficiency (lower = better)")
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

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# Figure 20 — AoI Heatmap
# ═══════════════════════════════════════════════════════════════════════════
def plot_aoi_heatmap(metrics, env, save_path=None, title=None):
    """Spatial heatmap of per‑sensor AoI across the mission area."""
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

    # Interpolate onto a grid
    grid_x = np.linspace(0, cfg.area_width, 200)
    grid_y = np.linspace(0, cfg.area_height, 200)
    gx, gy = np.meshgrid(grid_x, grid_y)

    try:
        grid_aoi = griddata((xs, ys), aoi_vals, (gx, gy), method="cubic")
        grid_aoi = np.nan_to_num(grid_aoi, nan=np.nanmean(aoi_vals))
    except Exception:
        grid_aoi = griddata((xs, ys), aoi_vals, (gx, gy), method="nearest")

    im = ax.pcolormesh(grid_x, grid_y, grid_aoi, cmap="YlOrRd", shading="auto")
    cbar = plt.colorbar(im, ax=ax, label="Age of Information (s)")

    # Overlay sensor positions
    ax.scatter(xs, ys, c=aoi_vals, cmap="YlOrRd", s=40, edgecolors="k",
               linewidth=0.5, zorder=3, vmin=aoi_vals.min(), vmax=aoi_vals.max())

    # Base station
    ax.scatter(cfg.bs_x, cfg.bs_y, c="blue", marker="^", s=300,
               edgecolors="k", lw=2, zorder=5)
    ax.annotate("BS", (cfg.bs_x, cfg.bs_y), fontsize=10, fontweight="bold",
                ha="center", xytext=(0, 15), textcoords="offset points",
                color="blue")

    ax.set_xlim(0, cfg.area_width)
    ax.set_ylim(0, cfg.area_height)
    ax.set_aspect("equal")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(title or "System AoI Heatmap (Eq 17)", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# Figure 23 — PDR Bar Chart
# ═══════════════════════════════════════════════════════════════════════════
def plot_pdr_barchart(results, save_path=None):
    """Grouped bar chart of PDR across different UAV counts."""
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
    ax.set_ylabel("Packet Delivery Ratio (R_P)")
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
# Figure 24 — Battery Life Curves
# ═══════════════════════════════════════════════════════════════════════════
def plot_battery_life(metrics_with, metrics_without, num_uavs,
                      save_path=None):
    """Remaining energy curves with vs without Algorithm 5."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f"Battery Life Comparison — {num_uavs} UAV(s)",
                 fontsize=14, fontweight="bold")

    for ax, met, label_pref in zip(
        axes, [metrics_with, metrics_without],
        ["With Charging", "Without Charging"]
    ):
        # Per‑UAV curves
        for uid, e_vals in met.per_uav_energy.items():
            t = met.round_times[:len(e_vals)]
            ax.plot(t, [e / 1e3 for e in e_vals], "-o",
                    color=COLORS[uid % len(COLORS)], lw=2, ms=3,
                    label=f"UAV {uid}")
        # Total
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
