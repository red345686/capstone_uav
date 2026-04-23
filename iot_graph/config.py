"""
Configuration parameters for the Multi-Hop IoT-UAV simulation with Battery Fairness.
Extends original MGDC parameters with multi-BS, IoT graph, and fairness settings.
"""
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class SimConfig:
    """All tunable simulation parameters."""

    # ── Area ──────────────────────────────────────────────────────────────
    area_width: float = 1000.0          # metres
    area_height: float = 1000.0         # metres

    # ── Base Stations (multiple) ─────────────────────────────────────────
    num_base_stations: int = 2          # L
    base_station_positions: List[Tuple[float, float, float]] = None  # (x,y,z) per BS
    n_chargers_per_bs: List[int] = None  # charging bays per BS
    w_c: float = 0.5                    # BS strength weight for chargers
    w_n: float = 0.5                    # BS strength weight for UAV assignment

    # ── UAVs ──────────────────────────────────────────────────────────────
    num_uavs: int = 3
    uav_altitude: float = 100.0         # H_n  (m)
    uav_speed: float = 10.0             # V_max  (m/s)
    uav_speed_max: float = 10.0         # V_max
    uav_speed_min: float = 1.0          # minimum cruise speed

    # ── Sensor Nodes (editable subnet layout) ────────────────────────────
    num_subnets: int = 6
    nodes_per_subnet_min: int = 15
    nodes_per_subnet_max: int = 25
    num_independent_nodes: int = 10     # S^ind — not in any subnet
    sensor_e_max: float = 500.0         # E_i^max  (J)
    safety_threshold_ratio: float = 0.2 # fraction of E_max for CN replacement
    sensor_buffer_capacity: int = 20    # C_i — max packets in buffer

    # ── Communication ranges ─────────────────────────────────────────────
    d_II_max: float = 150.0             # max IoT-to-IoT communication range (m)
    d_UU_th: float = 500.0              # UAV-to-UAV communication threshold (m)
    d_IU_max: float = 300.0             # max IoT-to-UAV range (m)
    d_prox: float = 200.0               # proximity radius for P classification (m)
    d_iso: float = 800.0                # isolation threshold for I classification (m)

    # ── Data model ────────────────────────────────────────────────────────
    cache_size_bytes: int = 5 * 1024 * 1024        # C_S = 5 MB
    data_gen_bytes: int = 512 * 1024               # s = 512 KB per slot per sensor
    data_gen_rate: float = 1.0                     # 1 data-unit per slot per sensor
    delta_T: int = 10                              # TTL in slots

    # ── Communication parameters ─────────────────────────────────────────
    B: float = 1e6                      # Hz   – bandwidth (1 MHz)
    sigma_sq_dbm: float = -110.0        # dBm  – noise power
    beta_0_db: float = -60.0            # dB   – reference channel gain at 1 m
    P_S_i: float = 0.1                  # W    – sensor transmit power
    P_U_n: float = 0.5                  # W    – UAV transmit power
    W_U_comm: float = 1.0               # constant path loss factor
    psi: float = 4.88                   # LoS environment constant a
    beta_env: float = 0.43              # LoS environment constant b
    eta_LoS: float = 0.1                # dB   – additional LoS path loss
    eta_NLoS: float = 21.0              # dB   – additional NLoS path loss
    los_epsilon: float = 0.9            # LoS probability threshold
    f_c: float = 915e6                  # Hz   – carrier frequency

    N_R: int = 10
    T_C: int = 5                        # swap period (rounds)
    F_R: int = 2                        # CN replacement trigger parameter
    G_u: float = 2.0                    # dBi  – UAV antenna gain
    P_T: float = 35.68                  # dBW  – transmit power
    G_t: float = 15.0                   # dBi  – ground antenna gain
    eta_comm: float = 0.6

    # ── Energy / propulsion model ────────────────────────────────────────
    epsilon: float = 0.95
    E_ele: float = 50e-9               # J/bit  – electronics energy
    eps_fs: float = 10e-12             # J/bit/m^2  – free-space coeff
    eps_mp: float = 1.3e-15            # J/bit/m^4  – multipath coeff
    E_n_max: float = 1e6               # J   – UAV battery capacity
    P_0: float = 99.66                 # W   – blade profile power
    P_1: float = 120.16                # W   – induced power in hover
    U_tip: float = 120.0               # m/s – tip speed of rotor blade
    v_0: float = 2e-3                  # m/s – mean rotor induced velocity
    z_0: float = 0.48                  # fuselage drag ratio
    rho_0: float = 1.225               # kg/m^3  – air density
    s_0: float = 0.05                  # rotor solidity
    A_rotor: float = 0.5               # m^2 – rotor disc area
    alpha_0: float = 2.0
    W_U: float = 3.0                   # kg  – UAV weight

    # ── Objective weights (Eq 23) ────────────────────────────────────────
    alpha_obj: float = 0.4              # AoI weight
    beta_obj: float = 0.3              # energy weight
    gamma_obj: float = 0.3             # battery fairness weight
    eta_fair: float = 0.1              # battery fairness regularisation

    # ── Load balancing ───────────────────────────────────────────────────
    gamma_B: int = 10                   # node-count imbalance threshold

    # ── MSCT ─────────────────────────────────────────────────────────────
    kappa_msct: float = 2.0            # MSCT budget factor

    # ── IALNS / JIALNS path-planning ─────────────────────────────────────
    ialns_iterations: int = 500
    ialns_temp_init: float = 100.0
    ialns_temp_decay: float = 0.995
    ialns_destroy_fraction: float = 0.3
    jialns_iterations: int = 300       # iterations for joint search
    jialns_R: int = 10                 # outer iterative loop repeats

    # ── Simulation control ───────────────────────────────────────────────
    max_rounds: int = 100
    round_duration: float = 120.0       # seconds per round
    max_hover_time: float = 15.0        # cap hover duration per CN (s)
    seed: int = 42

    def __post_init__(self):
        if self.base_station_positions is None:
            if self.num_base_stations == 1:
                self.base_station_positions = [(100.0, 100.0, 50.0)]
            elif self.num_base_stations == 2:
                self.base_station_positions = [
                    (100.0, 100.0, 50.0),
                    (900.0, 900.0, 50.0),
                ]
            else:
                # Spread evenly along diagonal
                self.base_station_positions = []
                for i in range(self.num_base_stations):
                    frac = (i + 1) / (self.num_base_stations + 1)
                    self.base_station_positions.append(
                        (frac * self.area_width, frac * self.area_height, 50.0)
                    )
        if self.n_chargers_per_bs is None:
            self.n_chargers_per_bs = [2] * self.num_base_stations

    # ── Derived helpers ──────────────────────────────────────────────────
    @property
    def packet_size_bits(self):
        """Size of one data-unit in bits (512 KB x 8)."""
        return self.data_gen_bytes * 8

    @property
    def sigma_sq_watts(self):
        """Noise power in Watts."""
        return 10.0 ** (self.sigma_sq_dbm / 10.0) * 1e-3

    @property
    def beta_0_linear(self):
        """Reference channel gain in linear scale."""
        return 10.0 ** (self.beta_0_db / 10.0)
