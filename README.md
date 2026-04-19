# UAV-Assisted Sensor Data Collection Simulation

Multi-phase simulation of UAV-based ground data collection (MGDC) from isolated wireless sensor networks. Implements convergence node selection, area division, relay pairing, IALNS path planning, and alternating charging with buffer-based relay delivery.

## Problem Statement

Sensor nodes are deployed in isolated subnets across a 1000 x 1000 m area. UAVs collect data from designated **convergence nodes**, relay via air-to-air (A2A) communication, and deliver to the base station. Packets are only counted as "delivered" when a UAV physically reaches the BS.

**Objectives**: Minimise Age of Information (AoI), maximise Packet Delivery Ratio (PDR), and extend operational lifetime through alternating charging.

## Simulation Phases

| Phase | Algorithm | Description |
|-------|-----------|-------------|
| 1 | Convergence Node Selection | Pick one CN per subnet; angular-zone replacement when depleted (Eq 33) |
| 2 | Area Division (K-Means) | Cluster CNs into UAV assignments with load balancing |
| 3 | Relay Matching (Gale-Shapley) | Pair sender UAVs with receivers for stable relay matching |
| 4 | Path Planning (IALNS) | Optimise flight tours with communication point insertion |
| 5 | Alternating Charging | Swap sender/receiver roles every T_C rounds; M/M/1/m queue recharging |

## Key Equations

| Equation | Description |
|----------|-------------|
| Q = B * log2(1 + h^2 * Ps / sigma^2) | Shannon capacity (G2A, Eq 1) |
| P_LoS = 1 / (1 + psi * exp(-beta * (theta - psi))) | LoS probability (Eq 20) |
| E_TX = l * E_elec + l * eps_fs * d^2 | Sensor TX energy (Eq 4) |
| P(V) = P0*(1+3V^2/Utip^2) + P1*sqrt(...) + drag | UAV propulsion power (Eq 25) |
| AoI_i = current_round - last_delivered_round | Age of Information (Eq 17) |
| PDR_eff = delivered / (delivered + expired) | Effective PDR |

## Parameters

| Parameter | Value | Parameter | Value |
|-----------|-------|-----------|-------|
| Area | 1000 x 1000 m | UAV battery | 1 MJ |
| UAVs | 1-4 | Round duration | 120 s |
| Subnets | 6 | Max hover/CN | 15 s |
| Sensors/subnet | 15-25 | TTL (delta_T) | 10 rounds |
| UAV altitude | 100 m | Charging period T_C | 5 rounds |
| UAV speed | 10 m/s | Bandwidth | 1 MHz |

All parameters editable in `python/config.py`.

## How to Run

```bash
cd paper2_sim
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r python/requirements.txt
python python/run.py
```

Results (18 plots) saved to `results/`.

## Results

| UAVs | AoI w/ | AoI w/o | ePDR w/ | ePDR w/o | Rounds w/ / w/o |
|------|--------|---------|---------|----------|-----------------|
| 1    | 25.6   | 25.6    | 0.634   | 0.634    | 61 / 61         |
| 2    | 13.2   | 8.0     | 0.855   | 0.854    | 100 / 41        |
| 3    | 8.0    | 12.1    | 0.879   | 0.876    | 100 / 61        |
| 4    | 8.2    | 8.4     | 0.836   | 0.875    | 100 / 69        |

**Key findings**: More UAVs reduce AoI (25.6 -> 8.0) and increase PDR (63% -> 88%). Alternating charging extends lifetime from 41-69 rounds to 100 rounds. Diminishing returns appear at 4 UAVs due to relay overhead.
