"""
Communication channel models for Multi-Hop IoT-UAV system.

Ground-to-Air (G2A), Air-to-Air (A2A), IoT-to-IoT (I2I),
UAV-to-UAV, UAV-to-BS, IoT-to-UAV LoS constraint.
"""
import numpy as np


# ═══════════════════════════════════════════════════════════════════════════
# Channel gain
# ═══════════════════════════════════════════════════════════════════════════
def channel_gain_g2a(d_horizontal, H, beta_0_linear):
    """Free-space channel power gain  h² = β₀ / (H² + d²)."""
    d_3d_sq = H ** 2 + d_horizontal ** 2
    return beta_0_linear / max(d_3d_sq, 1.0)


# ═══════════════════════════════════════════════════════════════════════════
# LoS probability
# ═══════════════════════════════════════════════════════════════════════════
def los_probability(d_horizontal, H, psi, beta_env):
    """P_LoS = 1 / (1 + ψ · exp(−β · (θ° − ψ)))"""
    if d_horizontal < 1e-6:
        return 1.0
    theta_deg = np.degrees(np.arctan2(H, d_horizontal))
    return 1.0 / (1.0 + psi * np.exp(-beta_env * (theta_deg - psi)))


# ═══════════════════════════════════════════════════════════════════════════
# Shannon capacity — G2A
# ═══════════════════════════════════════════════════════════════════════════
def shannon_capacity_g2a(d_horizontal, cfg):
    """G2A Shannon capacity Q = B · log₂(1 + SNR) [bits/s]."""
    h_sq = channel_gain_g2a(d_horizontal, cfg.uav_altitude, cfg.beta_0_linear)
    p_los = los_probability(d_horizontal, cfg.uav_altitude,
                            cfg.psi, cfg.beta_env)
    nlos_atten = 10.0 ** (-cfg.eta_NLoS / 10.0)
    h_sq_eff = h_sq * (p_los + (1.0 - p_los) * nlos_atten)
    snr = h_sq_eff * cfg.P_S_i / cfg.sigma_sq_watts
    return cfg.B * np.log2(1.0 + max(snr, 0.0))


# ═══════════════════════════════════════════════════════════════════════════
# Shannon capacity — A2A
# ═══════════════════════════════════════════════════════════════════════════
def shannon_capacity_a2a(d_3d, cfg):
    """A2A capacity between two UAVs at the same altitude."""
    d_3d = max(d_3d, 1.0)
    h_sq = cfg.beta_0_linear / (d_3d ** 2)
    snr = h_sq * cfg.P_U_n / cfg.sigma_sq_watts
    return cfg.B * np.log2(1.0 + max(snr, 0.0))


# ═══════════════════════════════════════════════════════════════════════════
# Energy Efficiency metric
# ═══════════════════════════════════════════════════════════════════════════
def energy_efficiency_metric(d_horizontal, cfg):
    """E_η = Q / P_R  [bits per Joule of received power]."""
    capacity = shannon_capacity_g2a(d_horizontal, cfg)
    h_sq = channel_gain_g2a(d_horizontal, cfg.uav_altitude, cfg.beta_0_linear)
    p_los = los_probability(d_horizontal, cfg.uav_altitude,
                            cfg.psi, cfg.beta_env)
    eta_factor = (p_los * 10.0 ** (-cfg.eta_LoS / 10.0)
                  + (1.0 - p_los) * 10.0 ** (-cfg.eta_NLoS / 10.0))
    P_received = cfg.P_S_i * h_sq * eta_factor
    if P_received < 1e-30:
        return 0.0
    return capacity / P_received


# ═══════════════════════════════════════════════════════════════════════════
# LoS verification for A2A
# ═══════════════════════════════════════════════════════════════════════════
def verify_a2a_los(uav_a, uav_b, cfg):
    """Return True if A2A link between two UAVs is viable."""
    d = np.hypot(uav_a[0] - uav_b[0], uav_a[1] - uav_b[1])
    if d < 1e-3:
        return True
    p_los = los_probability(d, cfg.uav_altitude, cfg.psi, cfg.beta_env)
    return p_los > 0.5


# ═══════════════════════════════════════════════════════════════════════════
# IoT-to-IoT energy (Eq 3-5)
# ═══════════════════════════════════════════════════════════════════════════
def iot_to_iot_energy(k_bits, distance, cfg):
    """Energy for IoT-to-IoT transmission + reception per packet.

    Eq 3-5: E_tx + E_rx. Returns 0 for same-subnet (handled externally).
    """
    d_cross = np.sqrt(cfg.eps_fs / cfg.eps_mp) if cfg.eps_mp > 0 else 1e6
    e_tx = cfg.E_ele * k_bits
    if distance < d_cross:
        e_tx += cfg.eps_fs * k_bits * distance ** 2
    else:
        e_tx += cfg.eps_mp * k_bits * distance ** 4
    e_rx = cfg.E_ele * k_bits
    return e_tx + e_rx


# ═══════════════════════════════════════════════════════════════════════════
# UAV-to-UAV transmission time (Eq 11)
# ═══════════════════════════════════════════════════════════════════════════
def uav_to_uav_tx_time(data_bits, distance, cfg):
    """Transmission time for UAV-to-UAV relay."""
    cap = shannon_capacity_a2a(distance, cfg)
    if cap <= 0:
        return float("inf")
    return data_bits / cap


# ═══════════════════════════════════════════════════════════════════════════
# UAV-to-BS transmission time (Eq 12)
# ═══════════════════════════════════════════════════════════════════════════
def uav_to_bs_tx_time(data_bits, distance, cfg):
    """Transmission time for UAV-to-BS delivery."""
    cap = shannon_capacity_g2a(distance, cfg)
    if cap <= 0:
        return float("inf")
    return data_bits / cap


# ═══════════════════════════════════════════════════════════════════════════
# IoT-to-UAV LoS constraint (Eq 10 / C5)
# ═══════════════════════════════════════════════════════════════════════════
def iot_to_uav_valid(d_horizontal, H, cfg):
    """Check if IoT-to-UAV link satisfies LoS constraint C5."""
    if d_horizontal > cfg.d_IU_max:
        return False
    p_los = los_probability(d_horizontal, H, cfg.psi, cfg.beta_env)
    return p_los >= cfg.los_epsilon
