"""
Metrics tracking — AoI, PDR, energy, battery fairness, composite objective,
per-BS delivery tracking.
"""
import numpy as np


class MetricsTracker:
    def __init__(self):
        # ── time-series ──────────────────────────────────────────────────
        self.aoi_history = []
        self.pdr_history = []
        self.energy_history = []
        self.energy_consumed_history = []
        self.round_times = []

        # ── per-UAV energy ───────────────────────────────────────────────
        self.per_uav_energy = {}

        # ── per-node AoI ─────────────────────────────────────────────────
        self.per_node_aoi = {}
        self.per_node_positions = {}

        # ── per-round delivery tracking ──────────────────────────────────
        self.per_round_delivered = []
        self.per_round_generated = []
        self.prev_delivered = 0
        self.prev_generated = 0

        # ── battery fairness tracking ────────────────────────────────────
        self.fairness_history = []          # Δ_batt per round
        self.rho_history = []               # list of per-sensor ρ_i arrays

        # ── composite objective ──────────────────────────────────────────
        self.composite_obj_history = []

        # ── per-BS delivery tracking ─────────────────────────────────────
        self.per_bs_delivered = {}          # {bs_id: total_packets}

        # ── search convergence ───────────────────────────────────────────
        self.search_cost_history = []       # F* per iteration

        # ── scalar accumulators ──────────────────────────────────────────
        self.packets_delivered = 0
        self.packets_generated = 0
        self.packets_expired = 0
        self.total_energy_consumed = 0.0
        self.total_distance = 0.0
        self.runtime = 0.0

    def record_energy_spent(self, energy_delta):
        self.total_energy_consumed += energy_delta

    def record_bs_delivery(self, bs_id, count):
        """Track packets delivered to a specific BS."""
        self.per_bs_delivered[bs_id] = self.per_bs_delivered.get(bs_id, 0) + count

    def update(self, env, round_num, current_time):
        """Snapshot all metrics at the end of a round."""
        # ── System AoI ──────────────────────────────────────────────────
        aoi_vals = []
        for n in env.sensor_nodes:
            if n.last_collected_round >= 0:
                aoi = round_num - n.last_collected_round
            else:
                aoi = round_num + 1
            aoi_vals.append(float(aoi))
            self.per_node_aoi[n.id] = aoi
            self.per_node_positions[n.id] = (n.x, n.y)

        self.aoi_history.append(np.mean(aoi_vals) if aoi_vals else 0.0)

        # ── PDR ─────────────────────────────────────────────────────────
        self.packets_generated = env.total_packets_generated
        self.packets_delivered = env.total_packets_delivered
        self.packets_expired = env.total_packets_expired
        pdr = (self.packets_delivered / self.packets_generated
               if self.packets_generated > 0 else 0.0)
        self.pdr_history.append(pdr)

        # ── Per-round delivery delta ────────────────────────────────────
        round_del = env.total_packets_delivered - self.prev_delivered
        round_gen = env.total_packets_generated - self.prev_generated
        self.per_round_delivered.append(round_del)
        self.per_round_generated.append(round_gen)
        self.prev_delivered = env.total_packets_delivered
        self.prev_generated = env.total_packets_generated

        # ── Remaining energy ────────────────────────────────────────────
        self.energy_history.append(sum(u.e_current for u in env.uavs))

        # ── Per-UAV remaining energy ────────────────────────────────────
        for u in env.uavs:
            self.per_uav_energy.setdefault(u.id, []).append(u.e_current)

        # ── Cumulative consumed ─────────────────────────────────────────
        self.energy_consumed_history.append(self.total_energy_consumed)

        # ── Total distance ──────────────────────────────────────────────
        self.total_distance = sum(u.total_distance for u in env.uavs)

        # ── Battery fairness (Eq 21-22) ─────────────────────────────────
        rho_vals = []
        for n in env.sensor_nodes:
            if n.e_max > 0:
                rho_vals.append(n.e_current / n.e_max)
            else:
                rho_vals.append(1.0)
        rho_arr = np.array(rho_vals)
        self.rho_history.append(rho_arr)
        delta_batt = float(np.max(rho_arr) - np.min(rho_arr)) if len(rho_arr) > 0 else 0.0
        self.fairness_history.append(delta_batt)

        # ── Composite objective (Eq 23) ─────────────────────────────────
        cfg = env.config
        avg_aoi = self.aoi_history[-1]
        e_total = self.total_energy_consumed
        # Normalise for composite: AoI in [0, max_rounds], energy in [0, N*E_max]
        aoi_norm = avg_aoi / max(cfg.max_rounds, 1)
        e_norm = e_total / max(cfg.num_uavs * cfg.E_n_max, 1)
        composite = (cfg.alpha_obj * aoi_norm
                     + cfg.beta_obj * e_norm
                     + cfg.gamma_obj * delta_batt)
        self.composite_obj_history.append(composite)

        self.round_times.append(current_time)
        self.runtime = current_time

    def get_summary(self):
        e_per_pkt = (self.total_energy_consumed / self.packets_delivered
                     if self.packets_delivered > 0 else 0.0)

        effective_total = self.packets_delivered + self.packets_expired
        effective_pdr = (self.packets_delivered / effective_total
                         if effective_total > 0 else 0.0)

        return {
            "avg_aoi":  np.mean(self.aoi_history) if self.aoi_history else 0,
            "final_aoi": self.aoi_history[-1] if self.aoi_history else 0,
            "avg_pdr":  np.mean(self.pdr_history) if self.pdr_history else 0,
            "final_pdr": self.pdr_history[-1] if self.pdr_history else 0,
            "effective_pdr": effective_pdr,
            "runtime":  self.runtime,
            "total_rounds": len(self.aoi_history),
            "total_packets_delivered": self.packets_delivered,
            "total_packets_generated": self.packets_generated,
            "total_packets_expired": self.packets_expired,
            "total_energy_consumed": self.total_energy_consumed,
            "energy_per_packet": e_per_pkt,
            "total_distance": self.total_distance,
            "final_fairness": self.fairness_history[-1] if self.fairness_history else 0,
            "avg_fairness": np.mean(self.fairness_history) if self.fairness_history else 0,
            "final_composite": self.composite_obj_history[-1] if self.composite_obj_history else 0,
            "per_bs_delivered": dict(self.per_bs_delivered),
        }
