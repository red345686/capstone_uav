"""
Communication channel models.

Ground-to-Air (G2A)
--------------------
Shannon capacity  (Eq 1):  Q_{i,n} = B · log₂(1 + h²·P_S / σ²)
Channel gain:               h² = β₀ / (H² + d²)
LoS probability  (Eq 20):  P_LoS = 1 / (1 + ψ·exp(-β·(θ° - ψ)))

Air-to-Air (A2A)
-----------------
Free-space channel at identical altitude.
Transfer time    (Eq 22):  t_A2A = data_bits / Q_A2A

Energy Efficiency (matching preference — related to Eq 41)
-----------------------------------------------------------
E_η = Q / P_R   [bits / Joule]
"""
import numpy as np


# ═══════════════════════════════════════════════════════════════════════════
# Channel gain
# ═══════════════════════════════════════════════════════════════════════════
def channel_gain_g2a(d_horizontal, H, beta_0_linear):
    """Free-space channel power gain  h² = β₀ / (H² + d²).

    Parameters
    ----------
    d_horizontal : float  - horizontal distance (m)
    H            : float  - UAV altitude (m)
    beta_0_linear: float  - reference gain in linear scale
    """
    d_3d_sq = H ** 2 + d_horizontal ** 2
    return beta_0_linear / max(d_3d_sq, 1.0)


# ═══════════════════════════════════════════════════════════════════════════
# LoS probability  (Eq 20)
# ═══════════════════════════════════════════════════════════════════════════
def los_probability(d_horizontal, H, psi, beta_env):
    """P_LoS = 1 / (1 + ψ · exp(−β · (θ° − ψ)))

    θ° = elevation angle in degrees = atan(H / d).
    When d → 0 the UAV is directly overhead → P_LoS = 1.
    """
    if d_horizontal < 1e-6:
        return 1.0
    theta_deg = np.degrees(np.arctan2(H, d_horizontal))
    return 1.0 / (1.0 + psi * np.exp(-beta_env * (theta_deg - psi)))


# ═══════════════════════════════════════════════════════════════════════════
# Shannon capacity — G2A  (Eq 1)
# ═══════════════════════════════════════════════════════════════════════════
def shannon_capacity_g2a(d_horizontal, cfg):
    r"""G2A Shannon capacity  Q = B · log₂(1 + SNR)  [bits/s].

    SNR = h²_eff · P_S / σ²
    h²_eff  = h² · [P_LoS + (1−P_LoS)·10^{−η_NLoS/10}]
    """
    h_sq = channel_gain_g2a(d_horizontal, cfg.uav_altitude, cfg.beta_0_linear)
    p_los = los_probability(d_horizontal, cfg.uav_altitude,
                            cfg.psi, cfg.beta_env)

    # Weighted effective gain (LoS + NLoS)
    nlos_atten = 10.0 ** (-cfg.eta_NLoS / 10.0)
    h_sq_eff = h_sq * (p_los + (1.0 - p_los) * nlos_atten)

    snr = h_sq_eff * cfg.P_S_i / cfg.sigma_sq_watts
    return cfg.B * np.log2(1.0 + max(snr, 0.0))


# ═══════════════════════════════════════════════════════════════════════════
# Shannon capacity — A2A  (Eq 22 context)
# ═══════════════════════════════════════════════════════════════════════════
def shannon_capacity_a2a(d_3d, cfg):
    """A2A capacity between two UAVs at the same altitude.

    Uses free-space channel gain  h² = β₀ / d².
    """
    d_3d = max(d_3d, 1.0)  # minimum practical distance 1 m
    h_sq = cfg.beta_0_linear / (d_3d ** 2)
    snr = h_sq * cfg.P_S_i / cfg.sigma_sq_watts
    return cfg.B * np.log2(1.0 + max(snr, 0.0))


# ═══════════════════════════════════════════════════════════════════════════
# Energy Efficiency  E_η  (receiver preference — Eq 41 context)
# ═══════════════════════════════════════════════════════════════════════════
def energy_efficiency_metric(d_horizontal, cfg):
    r"""E_η = Q / P_R  [bits per Joule of received power].

    P_R = P_S · h² · L    where L folds in LoS / NLoS attenuation.
    Higher E_η ⇒ receiver prefers this sender.
    """
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
# LoS verification for A2A  (Eq 20, 21)
# ═══════════════════════════════════════════════════════════════════════════
def verify_a2a_los(uav_a, uav_b, cfg):
    """Return True if A2A link between two UAVs is viable (LoS satisfied).

    Both UAVs fly at altitude H, so the link is horizontal.
    For horizontal links between UAVs at high altitude (100 m),
    LoS is almost always satisfied; we check via elevation model.
    """
    d = np.hypot(uav_a[0] - uav_b[0], uav_a[1] - uav_b[1])
    # Elevation angle for horizontal A2A ≈ 0 → use a small positive value
    # representing the fact that both drones are high above obstacles.
    # For same‑altitude links well above ground: treat as LoS.
    if d < 1e-3:
        return True
    # Conservative check using the G2A LoS model with H as effective altitude
    p_los = los_probability(d, cfg.uav_altitude, cfg.psi, cfg.beta_env)
    return p_los > 0.5
