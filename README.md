# TALOS NIO

Neural-Inertial Odometry for open XR hardware.

TALOS NIO is a drift-bounding tracking stack that combines:
- a fast inertial ESKF loop,
- a lightweight spectral neural corrector,
- and physics-based safety gates.

The goal is stable 6-DOF inertial tracking that stays bounded long enough to be a solid base for future visual fusion.

## What this repo contains

- `incremental_train.py` — incremental training + physical ESKF evaluation loop
- `SMLP.py` — SpectralMLP model
- `nymeria_loader.py` — Nymeria data loading/windowing pipeline
- `laid.py`, `npp.py`, `halo.py`, `bulwark.py` — physics and biomechanical guardrails
- `telemetry.py` — diagnostic dashboards

## Documentation

- Full technical spec and current code-synced behavior: [TALOS.md](TALOS.md)

## Quick Start

```bash
python cache_builder.py
python incremental_train.py
python plot_shelby.py
```

## Status

Active R&D / prototype codebase under rapid iteration.

---

Part of Project TALOS (GLR).
