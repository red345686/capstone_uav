"""
Entry point — sweep UAV counts 1-4, compare with/without alternating charging,
and generate all result plots including paper figures (20, 23, 24).
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import SimConfig
from simulation import run_simulation
from visualization import (
    plot_environment,
    plot_metrics,
    plot_uav_count_comparison,
    plot_aoi_heatmap,
    plot_pdr_barchart,
    plot_battery_life,
)


def run_experiments():
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
    os.makedirs(out, exist_ok=True)

    print("=" * 60)
    print("  UAV Sensor Data Collection Simulation")
    print("  (Paper-accurate equations: Eq 1, 4, 7-9, 12, 14-15,")
    print("   17, 20-22, 25, 33, 41, Algorithm 1-5)")
    print("  MGDC architecture: Sensor -> CN -> UAV buffer -> relay -> BS")
    print("=" * 60)

    results = {}
    all_met_w = {}
    all_met_wo = {}

    for n_uav in range(1, 5):
        print(f"\n{'=' * 60}")
        print(f"  Experiment: {n_uav} UAV(s)")
        print(f"{'=' * 60}")

        cfg = SimConfig(num_uavs=n_uav)

        # ── WITH alt. charging ────────────────────────────────────────
        print(f"\n--- With Alternating Charging ---")
        env_w, met_w, paths_w, clust_w = run_simulation(
            cfg, with_alternating_charging=True, verbose=True,
        )
        all_met_w[n_uav] = met_w

        # save environment plot
        rd = min(paths_w.keys()) if paths_w else 0
        if rd in paths_w and rd in clust_w:
            plot_environment(
                env_w, clusters=clust_w[rd], paths=paths_w[rd],
                round_num=rd,
                save_path=os.path.join(out, f"env_{n_uav}uav.png"),
                title=f"{n_uav} UAV(s) – Round {rd}",
            )

        # AoI heatmap  (Figure 20)
        plot_aoi_heatmap(
            met_w, env_w,
            save_path=os.path.join(out, f"aoi_heatmap_{n_uav}uav.png"),
            title=f"AoI Heatmap — {n_uav} UAV(s)",
        )

        # ── WITHOUT alt. charging ─────────────────────────────────────
        print(f"\n--- Without Alternating Charging ---")
        env_wo, met_wo, _, _ = run_simulation(
            cfg, with_alternating_charging=False, verbose=True,
        )
        all_met_wo[n_uav] = met_wo

        sw = met_w.get_summary()
        swo = met_wo.get_summary()

        # Average per-round delivery rate
        avg_del_w = (np.mean(met_w.per_round_delivered)
                     if met_w.per_round_delivered else 0)
        avg_del_wo = (np.mean(met_wo.per_round_delivered)
                      if met_wo.per_round_delivered else 0)

        results[n_uav] = {
            "avg_aoi":         sw["avg_aoi"],
            "final_aoi_w":     sw["final_aoi"],
            "final_aoi_wo":    swo["final_aoi"],
            "final_pdr":       sw["final_pdr"],
            "pdr_with":        sw["effective_pdr"],
            "pdr_without":     swo["effective_pdr"],
            "avg_del_w":       avg_del_w,
            "avg_del_wo":      avg_del_wo,
            "runtime_with":    sw["runtime"],
            "runtime_without": swo["runtime"],
            "rounds_with":     sw["total_rounds"],
            "rounds_without":  swo["total_rounds"],
            "energy_consumed_with":    sw["total_energy_consumed"],
            "energy_consumed_without": swo["total_energy_consumed"],
            "epp_with":        sw["energy_per_packet"],
            "epp_without":     swo["energy_per_packet"],
            "distance_with":   sw["total_distance"],
            "distance_without": swo["total_distance"],
            "expired_with":    sw["total_packets_expired"],
            "expired_without": swo["total_packets_expired"],
            "delivered_with":  sw["total_packets_delivered"],
            "delivered_without": swo["total_packets_delivered"],
        }

        # Per‑experiment metrics panels
        plot_metrics(
            [met_w, met_wo],
            ["With Charging", "Without Charging"],
            save_path=os.path.join(out, f"metrics_{n_uav}uav.png"),
        )

        # Battery life curves  (Figure 24)
        plot_battery_life(
            met_w, met_wo, n_uav,
            save_path=os.path.join(out, f"battery_life_{n_uav}uav.png"),
        )

    # ── Cross‑UAV comparison ──────────────────────────────────────────
    plot_uav_count_comparison(
        results, save_path=os.path.join(out, "uav_count_comparison.png"),
    )

    # PDR bar chart  (Figure 23)
    plot_pdr_barchart(
        results, save_path=os.path.join(out, "pdr_comparison.png"),
    )

    # ── Summary table ─────────────────────────────────────────────────
    print(f"\n{'=' * 120}")
    print("  RESULTS SUMMARY  (PDR = effective: delivered / (delivered + expired))")
    print(f"{'=' * 120}")
    hdr = (f"{'UAVs':>5} | {'AoI w/':>7} | {'AoI w/o':>7} | "
           f"{'PDR w/':>7} | {'PDR w/o':>7} | "
           f"{'Del/rnd w/':>10} | {'Del/rnd w/o':>11} | "
           f"{'Rounds':>6} | "
           f"{'E/pkt w/':>9} | {'E/pkt w/o':>10}")
    print(hdr)
    print("-" * len(hdr))
    for n in sorted(results):
        r = results[n]
        print(f"{n:>5} | {r['final_aoi_w']:>7.1f} | {r['final_aoi_wo']:>7.1f} | "
              f"{r['pdr_with']:>7.3f} | {r['pdr_without']:>7.3f} | "
              f"{r['avg_del_w']:>10.1f} | {r['avg_del_wo']:>11.1f} | "
              f"{r['rounds_with']:>3}/{r['rounds_without']:<3}| "
              f"{r['epp_with']:>7.1f} J | {r['epp_without']:>8.1f} J")

    print(f"\nPlots saved to: {os.path.abspath(out)}")


if __name__ == "__main__":
    run_experiments()
