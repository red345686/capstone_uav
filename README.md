# UAV-Assisted Sensor Data Collection Simulation

A multi-phase simulation of UAV-based ground data collection (MGDC) from isolated wireless sensor networks. Implements convergence node selection, area division, relay pairing, IALNS path planning, and alternating charging with buffer-based relay delivery.

---

## Problem Statement

Wireless sensor nodes are deployed in isolated subnets across a 1000 m x 1000 m area. These subnets cannot communicate with each other or the base station directly. UAVs must fly to each subnet, collect data from a designated **convergence node (CN)**, relay it via air-to-air (A2A) communication, and deliver it to the base station (BS).

The challenge:
- **Minimise Age of Information (AoI)** - how stale is the collected data?
- **Maximise Packet Delivery Ratio (PDR)** - what fraction of sensor data reaches the BS?
- **Maximise operational runtime** - how long before UAVs run out of energy?

---

## Data Flow Architecture (MGDC)

```
Sensor Node --[intra-subnet TX]--> Convergence Node --[G2A uplink]--> UAV Buffer
                                                                        |
                                        +-------------------------------+
                                        |
                           +------------v-------------+
                           |   Role-based routing:    |
                           |                          |
                           |  SENDER: fly to Pn       |
                           |    --[A2A relay]-->       |
                           |  RECEIVER: fly to BS     |
                           |  SOLO: fly to BS         |
                           +-----------+--------------+
                                       |
                                       v
                             Base Station (delivery)
```

**Key rule**: Packets are ONLY counted as "delivered" when a UAV (receiver or solo) physically reaches the BS.

---

## Simulation Architecture

The simulation executes **5 phases per round** in a time-stepped loop:

```
+-------------------------------------------------------------+
|                    SIMULATION ROUND LOOP                     |
|                                                              |
|  Phase 1: Convergence Node Selection (Algorithm 1)           |
|     - Pick one node per subnet for external communication    |
|     - Angular-zone replacement when CN depletes (Eq 33)      |
|                                                              |
|  Phase 2: Mission Area Division (Algorithm 2)                |
|     - K-Means clustering of CNs into UAV assignments         |
|     - Load balancing within gamma_B threshold                |
|                                                              |
|  Phase 3: Relay Matching (Algorithm 3 / Gale-Shapley)        |
|     - Pair remote "sender" UAVs with proximal "receivers"    |
|     - Stable matching via deferred acceptance                |
|                                                              |
|  Phase 4: Path Planning (Algorithm 4 / IALNS)                |
|     - Optimise each UAV's flight tour                        |
|     - Insert communication point Pn for paired UAVs          |
|     - Adjust speeds for simultaneous rendezvous              |
|                                                              |
|  -> EXECUTE: fly paths (time-budgeted), collect data,        |
|              A2A relay at Pn, deliver at BS                   |
|                                                              |
|  Phase 5: Alternating Charging (Algorithm 5)                 |
|     - Swap sender/receiver roles every T_C rounds            |
|     - Recharge returning UAV at base station                 |
|     - M/M/1/m queue wait time modelling                      |
|                                                              |
|  -> CHECK: all UAVs depleted OR all sensors depleted -> stop |
+-------------------------------------------------------------+
```

---

## Key Equations

| Equation | Description | Reference |
|----------|-------------|-----------|
| Q = B * log2(1 + h^2 * Ps / sigma^2) | Shannon capacity (G2A) | Eq 1 |
| h^2 = beta_0 / (H^2 + d^2) | Free-space channel gain | — |
| P_LoS = 1 / (1 + psi * exp(-beta * (theta - psi))) | LoS probability | Eq 20 |
| E_TX = l * E_elec + l * eps_fs * d^2 | Sensor TX energy | Eq 4 |
| P(V) = P0*(1+3V^2/Utip^2) + P1*sqrt(...) + drag | UAV propulsion power | Eq 25 |
| AoI_i = round - last_delivered_round | Age of Information | Eq 17 |
| PDR_eff = delivered / (delivered + expired) | Effective packet delivery ratio | — |
| delta_T = C_S / s = 10 rounds | Packet TTL | — |

---

## Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Area | 1000 x 1000 m | Mission area |
| Base Station | (100, 100), z=50 m | Fixed ground station |
| UAVs | 1-4 | Quadrotor UAVs |
| UAV altitude | 100 m | Fixed flight altitude |
| UAV speed | 10 m/s | Max cruise speed |
| UAV battery | 10^6 J (1 MJ) | Battery capacity |
| Subnets | 6 | Gaussian-clustered sensor groups |
| Sensors/subnet | 15-25 | Random per subnet |
| Sensor energy | 500 J | Sensor battery capacity |
| Round duration | 120 s | Time budget per round |
| Max hover time | 15 s | Per CN visit |
| Bandwidth | 1 MHz | Communication channel |
| TTL (delta_T) | 10 rounds | Packet expiry |
| T_C | 5 rounds | Charging swap period |
| P0 | 99.66 W | Blade profile power |
| P1 | 120.16 W | Induced power |
| gamma_B | 10 | Load balancing threshold |

All parameters are editable in `python/config.py`.

---

## Project Structure

```
paper2_sim/
├── README.md
├── python/
│   ├── config.py                   # All simulation parameters
│   ├── environment.py              # Sensor nodes, subnets, UAVs, BS
│   ├── energy_model.py             # Propulsion power P(V), sensor TX energy
│   ├── communication.py            # Shannon capacity (G2A, A2A), LoS model
│   ├── phase1_convergence.py       # CN selection + angular-zone replacement
│   ├── phase2_division.py          # K-Means area division + load balancing
│   ├── phase3_matching.py          # Gale-Shapley relay matching
│   ├── phase4_pathplanning.py      # IALNS path optimisation
│   ├── phase5_charging.py          # Alternating charging + M/M/1/m queue
│   ├── simulation.py               # Main simulation loop (orchestrator)
│   ├── metrics.py                  # AoI, PDR, energy tracking
│   ├── visualization.py            # All plotting utilities
│   ├── run.py                      # Entry point - runs all experiments
│   └── requirements.txt            # numpy, matplotlib, scipy
└── results/                        # Generated plots (after running)
    ├── env_*uav.png                # Environment layout per UAV count
    ├── metrics_*uav.png            # Per-experiment metrics (6 panels)
    ├── aoi_heatmap_*uav.png        # AoI heatmap (Figure 20)
    ├── battery_life_*uav.png       # Battery life curves (Figure 24)
    ├── pdr_comparison.png          # PDR bar chart (Figure 23)
    └── uav_count_comparison.png    # Cross-UAV comparison (6 panels)
```

---

## How to Run

```bash
cd paper2_sim

# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/Mac

# Install dependencies
pip install -r python/requirements.txt

# Run all experiments (1-4 UAVs, with/without charging)
python python/run.py
```

Results are saved to the `results/` directory (18 plots total).

---

## Results

The simulation sweeps 1-4 UAVs and compares "with alternating charging" vs "without":

### Summary Table

| UAVs | AoI w/ | AoI w/o | ePDR w/ | ePDR w/o | Del/rnd w/ | Rounds w/ / w/o |
|------|--------|---------|---------|----------|------------|-----------------|
| 1    | 25.6   | 25.6    | 0.634   | 0.634    | 72.1       | 61 / 61         |
| 2    | 13.2   | 8.0     | 0.855   | 0.854    | 101.3      | 100 / 41        |
| 3    | 8.0    | 12.1    | 0.879   | 0.876    | 107.2      | 100 / 61        |
| 4    | 8.2    | 8.4     | 0.836   | 0.875    | 101.1      | 100 / 69        |

### Key Findings

1. **AoI decreases with more UAVs** (25.6 -> 8.0): With a 120s time budget, 1 UAV can only visit ~3 of 6 CNs per round. 3 UAVs cover all 6 CNs.

2. **PDR increases with more UAVs** (63.4% -> 87.9%): More coverage means fewer packets expire before collection.

3. **Charging extends operational lifetime**: With charging, 2+ UAVs sustain 100 rounds. Without, they deplete at 41-69 rounds.

4. **1 UAV: charging has no effect**: No sender-receiver pairs to swap, so both conditions are identical.

5. **Diminishing returns at 4 UAVs**: PDR drops slightly (0.879 -> 0.836) due to relay overhead with more pairs.

---

## Tracked Metrics

| Metric | Definition |
|--------|-----------|
| **System AoI** | Average (current_round - last_delivery_round) across all sensors |
| **Effective PDR** | packets_delivered / (packets_delivered + packets_expired) |
| **Runtime** | Total simulation time until termination |
| **Total energy consumed** | Cumulative propulsion + hover energy (including recharged energy) |
| **Energy per packet** | total_energy_consumed / total_packets_delivered |
| **Delivery per round** | Average packets delivered to BS per round |

---

## Customisation

```python
from config import SimConfig
from simulation import run_simulation

# Example: run with 5 UAVs and 10 subnets
cfg = SimConfig(num_uavs=5, num_subnets=10)
env, metrics, paths, clusters = run_simulation(cfg)
```

### Key parameters to experiment with:

| Parameter | Effect |
|-----------|--------|
| `num_uavs` | More UAVs -> better coverage, higher energy cost |
| `num_subnets` | More subnets -> longer tours, more data |
| `round_duration` | Lower -> tighter time budget, more UAV differentiation |
| `T_C` | Higher -> less frequent swaps, more uneven energy drain |
| `max_rounds` | Limits how long the simulation runs with charging |
| `E_n_max` | UAV battery capacity - lower values cause earlier depletion |
| `ialns_iterations` | More iterations -> better paths, longer compute time |
