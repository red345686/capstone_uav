"""
Configuration parameters for the UAV sensor data collection simulation.
All values match the paper's Table 3.
"""
from dataclasses import dataclass


@dataclass
class SimConfig:
    """All tunable simulation parameters."""

    # ── Area ──────────────────────────────────────────────────────────────
    area_width: float = 1000.0          # metres
    area_height: float = 1000.0         # metres

    # ── Base Station ──────────────────────────────────────────────────────
    bs_x: float = 100.0
    bs_y: float = 100.0
    bs_z: float = 50.0                  # altitude (m)

    # ── UAVs ──────────────────────────────────────────────────────────────
    num_uavs: int = 3
    uav_altitude: float = 100.0         # H_n  (m)
    uav_speed: float = 10.0             # V_max  (m/s) — paper Table 3
    uav_speed_max: float = 10.0         # V_max
    uav_speed_min: float = 1.0          # minimum cruise speed

    # ── Sensor Nodes (editable subnet layout) ────────────────────────────
    num_subnets: int = 6
    nodes_per_subnet_min: int = 15
    nodes_per_subnet_max: int = 25
    sensor_e_max: float = 500.0         # E_i^max  (J) — high enough that UAV battery is the bottleneck
    safety_threshold_ratio: float = 0.2 # fraction of E_max for CN replacement

    # ── Data model ────────────────────────────────────────────────────────
    #   δ_T = C_S / s  where cache C_S = 5 MB, generation s = 512 KB/slot
    cache_size_bytes: int = 5 * 1024 * 1024        # C_S = 5 MB
    data_gen_bytes: int = 512 * 1024               # s = 512 KB per slot per sensor
    data_gen_rate: float = 1.0                     # 1 data‑unit per slot per sensor
    delta_T: int = 10                              # TTL in slots = C_S/s ≈ 10

    # ── Communication parameters ─────────────────────────────────────────
    B: float = 1e6                      # Hz   – bandwidth (1 MHz)
    sigma_sq_dbm: float = -110.0        # dBm  – noise power
    beta_0_db: float = -60.0            # dB   – reference channel gain at 1 m
    P_S_i: float = 0.1                  # W    – sensor transmit power
    psi: float = 4.88                   # LoS environment constant a (Eq 20)
    beta_env: float = 0.43              # LoS environment constant b (Eq 20)
    eta_LoS: float = 0.1                # dB   – additional LoS path loss
    eta_NLoS: float = 21.0              # dB   – additional NLoS path loss
    f_c: float = 915e6                  # Hz   – carrier frequency

    N_R: int = 10
    T_C: int = 5                        # swap period (rounds)
    F_R: int = 2                        # CN replacement trigger parameter
    G_u: float = 2.0                    # dBi  – UAV antenna gain
    P_T: float = 35.68                  # dBW  – transmit power
    G_t: float = 15.0                   # dBi  – ground antenna gain
    eta_comm: float = 0.6

    # ── Energy / propulsion model (Table 3) ──────────────────────────────
    epsilon: float = 0.95
    E_ele: float = 50e-9               # J/bit  – electronics energy
    eps_fs: float = 10e-12             # J/bit/m^2  – free‑space coeff
    eps_mp: float = 1.3e-15            # J/bit/m^4  – multipath coeff
    E_n_max: float = 1e6               # J   – UAV battery capacity
    P_0: float = 99.66                 # W   – blade profile power (Eq 25)
    P_1: float = 120.16                # W   – induced power in hover (Eq 25)
    U_tip: float = 120.0               # m/s – tip speed of rotor blade
    v_0: float = 2e-3                  # m/s – mean rotor induced velocity (2×10⁻³)
    z_0: float = 0.48                  # fuselage drag ratio
    rho_0: float = 1.225               # kg/m^3  – air density
    s_0: float = 0.05                  # rotor solidity
    A_rotor: float = 0.5               # m^2 – rotor disc area
    alpha_0: float = 2.0
    W_U: float = 3.0                   # kg  – UAV weight

    # ── Load balancing ───────────────────────────────────────────────────
    gamma_B: int = 10                   # node‑count imbalance threshold

    # ── IALNS path‑planning ──────────────────────────────────────────────
    ialns_iterations: int = 500
    ialns_temp_init: float = 100.0
    ialns_temp_decay: float = 0.995
    ialns_destroy_fraction: float = 0.3

    # ── Simulation control ───────────────────────────────────────────────
    max_rounds: int = 100
    round_duration: float = 120.0       # seconds per round — tight budget so
                                        # 1 UAV (1200m) can't visit all 6 CNs,
                                        # but 3-4 UAVs with shorter paths can
    max_hover_time: float = 15.0        # cap hover duration per CN (s)
    seed: int = 42

    # ── Derived helpers ──────────────────────────────────────────────────
    @property
    def packet_size_bits(self):
        """Size of one data-unit in bits (512 KB x 8)."""
        return self.data_gen_bytes * 8  # 4 194 304 bits

    @property
    def sigma_sq_watts(self):
        """Noise power in Watts (σ² from dBm → W)."""
        return 10.0 ** (self.sigma_sq_dbm / 10.0) * 1e-3

    @property
    def beta_0_linear(self):
        """Reference channel gain in linear scale (β₀ from dB)."""
        return 10.0 ** (self.beta_0_db / 10.0)
