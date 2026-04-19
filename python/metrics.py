"""
Metrics tracking — Age of Information (AoI, Eq 17), Packet Delivery Ratio,
remaining energy, per‑node AoI (for heatmap), per‑UAV energy (for battery curves).

AoI is computed in rounds (not raw seconds) so it stays bounded and meaningful.
PDR snapshot = cumulative delivered / cumulative generated.
Per‑round PDR = packets delivered this round / packets generated this round.
"""
import numpy as np


class MetricsTracker:
    def __init__(self):
        # ── time‑series ──────────────────────────────────────────────────
        self.aoi_history = []
        self.pdr_history = []
        self.energy_history = []          # total remaining energy per round
        self.energy_consumed_history = [] # cumulative energy consumed
        self.round_times = []

        # ── per‑UAV energy (for battery life curves — Figure 24) ────────
        self.per_uav_energy = {}          # {uav_id: [remaining_energy_per_round]}

        # ── per‑node AoI (for heatmap — Figure 20) ─────────────────────
        self.per_node_aoi = {}            # {node_id: aoi_at_end}
        self.per_node_positions = {}      # {node_id: (x, y)}

        # ── per‑round delivery tracking ─────────────────────────────────
        self.per_round_delivered = []     # packets delivered each round
        self.per_round_generated = []     # packets generated each round
        self.prev_delivered = 0           # for delta computation
        self.prev_generated = 0

        # ── scalar accumulators ──────────────────────────────────────────
        self.packets_delivered = 0
        self.packets_generated = 0
        self.packets_expired = 0
        self.total_energy_consumed = 0.0  # cumulative (incl. recharges)
        self.total_distance = 0.0
        self.runtime = 0.0

    # ──────────────────────────────────────────────────────────────────
    def record_energy_spent(self, energy_delta):
        """Called each round with the energy consumed (before recharging)."""
        self.total_energy_consumed += energy_delta

    def update(self, env, round_num, current_time):
        """Snapshot all metrics at the end of a round."""
        # ── System AoI  (Eq 17) — round‑based ────────────────────────
        # AoI = rounds since last delivery to BS for each sensor
        aoi_vals = []
        for n in env.sensor_nodes:
            if n.last_collected_round >= 0:
                aoi = round_num - n.last_collected_round
            else:
                aoi = round_num + 1   # never collected → age = all rounds
            aoi_vals.append(float(aoi))
            # store for heatmap
            self.per_node_aoi[n.id] = aoi
            self.per_node_positions[n.id] = (n.x, n.y)

        self.aoi_history.append(np.mean(aoi_vals) if aoi_vals else 0.0)

        # ── PDR (cumulative snapshot) ─────────────────────────────────
        self.packets_generated = env.total_packets_generated
        self.packets_delivered = env.total_packets_delivered
        self.packets_expired = env.total_packets_expired
        pdr = (self.packets_delivered / self.packets_generated
               if self.packets_generated > 0 else 0.0)
        self.pdr_history.append(pdr)

        # ── Per‑round delivery delta ─────────────────────────────────
        round_del = env.total_packets_delivered - self.prev_delivered
        round_gen = env.total_packets_generated - self.prev_generated
        self.per_round_delivered.append(round_del)
        self.per_round_generated.append(round_gen)
        self.prev_delivered = env.total_packets_delivered
        self.prev_generated = env.total_packets_generated

        # ── Remaining energy ─────────────────────────────────────────
        self.energy_history.append(sum(u.e_current for u in env.uavs))

        # ── Per‑UAV remaining energy ─────────────────────────────────
        for u in env.uavs:
            self.per_uav_energy.setdefault(u.id, []).append(u.e_current)

        # ── Cumulative consumed ──────────────────────────────────────
        self.energy_consumed_history.append(self.total_energy_consumed)

        # ── Total distance ───────────────────────────────────────────
        self.total_distance = sum(u.total_distance for u in env.uavs)

        self.round_times.append(current_time)
        self.runtime = current_time

    # ──────────────────────────────────────────────────────────────────
    def get_summary(self):
        e_per_pkt = (self.total_energy_consumed / self.packets_delivered
                     if self.packets_delivered > 0 else 0.0)

        # Effective PDR: delivered / (delivered + expired)
        # This measures how well the system utilises generated data
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
        }
