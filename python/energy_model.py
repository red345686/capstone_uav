"""
UAV propulsion energy model.

P(V) = P_0·(1 + 3V²/U_tip²)
     + P_1·√(√(1 + V⁴/(4v₀⁴)) - V²/(2v₀²))
     + ½·z₀·P₀·s₀·A·V³
"""
import numpy as np


def propulsion_power(V, cfg):
    """Instantaneous propulsion power at airspeed *V* (m/s)."""
    # Blade profile
    term1 = cfg.P_0 * (1.0 + 3.0 * V**2 / cfg.U_tip**2)
    # Induced
    inner = np.sqrt(1.0 + V**4 / (4.0 * cfg.v_0**4)) - V**2 / (2.0 * cfg.v_0**2)
    term2 = cfg.P_1 * np.sqrt(max(inner, 0.0))
    # Parasitic drag
    term3 = 0.5 * cfg.z_0 * cfg.rho_0 * cfg.s_0 * cfg.A_rotor * V**3
    return term1 + term2 + term3


def hover_power(cfg):
    """Power while hovering (V = 0)."""
    return propulsion_power(0.0, cfg)


def flight_energy(distance, speed, cfg):
    """Energy (J) to fly *distance* metres at *speed* m/s."""
    if speed <= 0 or distance <= 0:
        return 0.0
    return propulsion_power(speed, cfg) * (distance / speed)


def hover_energy(duration, cfg):
    """Energy (J) to hover for *duration* seconds."""
    return hover_power(cfg) * duration


def sensor_tx_energy(bits, distance, cfg):
    """Sensor‑side energy to transmit *bits* over *distance* metres."""
    d_cross = np.sqrt(cfg.eps_fs / cfg.eps_mp) if cfg.eps_mp > 0 else 1e6
    e_elec = cfg.E_ele * bits
    if distance < d_cross:
        return e_elec + cfg.eps_fs * bits * distance**2
    return e_elec + cfg.eps_mp * bits * distance**4


def sensor_rx_energy(bits, cfg):
    """Sensor‑side energy to receive *bits*."""
    return cfg.E_ele * bits
