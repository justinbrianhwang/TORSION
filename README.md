# TORSION

TORSION is a research scaffold for algorithm-aware semantic fault injection in autonomous driving intermediate representations. This CARLA-independent slice implements pure Python + NumPy object-set torsion while preserving the representation contract: object count, class, track ID, and bbox size stay intact while position, yaw, velocity direction, or confidence semantics are perturbed.

## Scaffold

Implemented now: `torsion/operators/object.py`, `torsion/operators/twist.py`, `torsion/operators/temporal.py`, `torsion/metrics/safety.py`, and `configs/object_torsion.yaml`. Phase 2/3 modules for cost maps, BEV features, injectors, scenarios, planning metrics, and representation metrics are present as TODO placeholders only. The `carla/` directory is an external junction and is not part of this repository slice.

## Representation Choices

`ObjectSet.v` is a 2D velocity vector `[vx, vy]` in the same frame as `x` and `y`; target selector `nearest_front` assumes +x is forward. Section 8.1 defines position/yaw/velocity magnitudes but not confidence, so confidence redistribution uses documented defaults of low `0.05`, medium `0.10`, and high `0.20`. Implausible deltas are rejected by configurable caps: 2 m position shift, 30 deg yaw, 30 deg velocity rotation, and 0.5 confidence delta by default.

Phase 2a adds the canonical spatial twist field in `torsion/operators/twist.py`: `theta(p) = alpha * (r / sigma) * exp(-0.5 * (r / sigma)^2)`, so the angle varies with distance instead of producing a uniform rotation. `scene_swirl_torsion` applies that same local angle to object position, yaw, and velocity. `temporal_curl_torsion` rotates target velocity by a linearly increasing window-local heading twist.

FAIR synthetic baselines now match on one common budget: `mean_realized_budget`, the active-window mean L2 displacement between the perturbed target actor prediction and the clean prediction over the prediction horizon. Section 8.1 position levels define the target realized budgets: low `0.2 m`, medium `0.5 m`, high `1.0 m`. Per-field magnitudes still define the base directions/caps, but the runner calibrates each first-class method to the common prediction-L2 budget before reporting safety metrics.

## Tests

Use the existing conda environment:

```powershell
conda run -n torsion pytest -q
```
