# TORSION

**A Semantic Fault Propagation Characterization framework for autonomous-driving systems.**

TORSION injects **semantic faults** as *excitation signals* into an autonomous-driving pipeline and
**empirically identifies how errors propagate** across its intermediate representations — which
representation boundaries *amplify* an error into a safety failure, which *attenuate* it, and **why**.

> Fault injection is the **tool**; error propagation is the **observation**; characterization is the
> **result**; (empirical) system identification is the **methodology**.

![Fault injection points in the autonomous-driving pipeline](assets/images/01-fault-injection-points.png)

*TORSION injects a semantic fault at the output of **object tracking**, **prediction**, or the **cost-map** (red injection points) and measures how the error propagates downstream toward a safety outcome. The **sensors and the vehicle are never modified** — only the intermediate software representations are.*

---

## Where in the autonomous-driving stack TORSION operates

A modern driving stack is a **cascade of representations**: raw sensors become a list of *tracked
objects*, which become *predicted trajectories*, which are rasterized into a *cost-map*, from which a
*planner* selects a trajectory that a *controller* executes. TORSION does not touch the sensors or the
car — it **injects a semantic fault at one chosen representation boundary** and measures how far, and
how strongly, that error travels down the rest of the cascade toward a safety outcome.

Each boundary has a measured **interface gain** (>1 amplifies an error, <1 attenuates it). Two
structural facts hold in **both synthetic and real (nuPlan)** data:

- **Rasterization boundaries attenuate** — projecting object/prediction state onto a grid is a
  many-to-one, kernel-limited map (local Jacobian ≈ 0.02–0.04 ≪ 1).
- **The cost-map is the most safety-critical representation interface** — injecting there degrades safety
  far more than at the object stage, at matched budget.

The per-boundary gains and their structural causes are tabulated in **[Key findings](#key-findings-honest--positive-and-negative)** below.

### Method: the fault is an *excitation signal*

TORSION treats the pipeline as an unknown dynamical system and the semantic fault as a **known
excitation** `u_t` injected into it. By measuring the **response** `y_t` (safety metrics, interface
gains) it performs *empirical system identification*: inferring which stage amplifies the signal and
which attenuates it — without ever assuming a parametric model of the internals.

![Semantic fault as an excitation signal for empirical system identification](assets/images/05-excitation-sysid.png)

---

## What is a "semantic fault"?

A fault that **preserves the structural contract** of a representation (object count / class / track-ID,
cost-value range, drivable topology, tensor shape) while **twisting only its meaning** — e.g. displacing a
tracked object's position / velocity / heading. This models realistic errors (tracking drift, calibration
error, occupancy / segmentation error) and, unlike Gaussian noise or bit-flips (which *break* the contract
and serve as baselines), the faulted representation stays a *valid* input, so its propagation can be
observed.

![Semantic fault (valid, contract-preserving) vs. corruption baseline (invalid)](assets/images/02-semantic-fault.png)

*Left — a semantic fault shifts the tracked object's position/velocity but keeps it a **single valid
object**; the representation contract holds, so the fault flows through the pipeline as a normal input.
Right — Gaussian corruption shatters the object and breaks the contract; it is used only as a matched
baseline.*

---

## Key findings (honest — positive and negative)

Measured on a shared synthetic closed-loop harness (30 seeds), a real learned model (InterFuser BEV),
real CARLA (450 episodes), and **real nuPlan** (real HD map + real agents, 10,440 runs).

**Measured interface gains** (6-stage pipeline, across fault magnitudes; >1 amplifies, <1 attenuates)

| Boundary | Gain | Behaviour | Response type |
|---|---|---|---|
| object → prediction | 1.3 – 1.9 | amplify | linear (gain CV 0.03) |
| prediction → cost-map | **0.004** | **attenuate** | linear |
| cost-map → plan | **6 – 9** | **amplify** | **nonlinear — switching (gain CV 0.26)** |
| plan → control | 2.5 | amplify | linear (gain CV 0.07) |

**Mechanisms — *why* each boundary behaves as it does**

| # | Boundary | Behaviour | Structural cause | Evidence |
|---|----------|-----------|------------------|----------|
| M1 | rasterization (→ cost-map) | attenuator | many-to-one grid projection | local Jacobian 0.02–0.04 ≪ 1 |
| M2 | cost-map → plan | amplifier | **sampling-argmin decision-boundary switching** | min-margin quartile: plan deviation ×27, argmin-flip ×24, Spearman ρ=0.52 |
| M3 | object → prediction | amplifier | constant-velocity integration over horizon | ∂pred/∂v = t; analytic = empirical; ∝ horizon |

![Interface gains: which boundaries amplify vs. attenuate](assets/images/03-interface-gains.png)

*The same input error is strongly **attenuated** by rasterization (gain ≈ 0.02–0.04), then sharply
**amplified and clipped** by the cost-map→planner argmin (a switching nonlinearity), while
object→prediction amplifies mildly and **linearly**.*

The cost-map→plan amplification is a **decision-boundary switch**, not a smooth gain: a tiny cost-map
perturbation can flip the planner's `argmin` from a straight trajectory to a hard swerve/brake.

![argmin decision-boundary switching produces a phantom brake](assets/images/04-argmin-switching.png)

*Mechanism M2: a tiny cost-map perturbation moves the global minimum from candidate 8 (straight) to
candidate 13 (hard swerve/brake) — a discontinuous "phantom brake". This switching behaviour is why the
cost-map is the most safety-critical interface.*

**System characterization**

- **Propagation response (amplitude sweep):** `cost→plan` is a *nonlinear* switching element; `plan→control`
  and `object→prediction` are *linear* elements. The pipeline is a cascade of linear transfer elements with the
  **planner (argmin) as the critical nonlinearity**.
- **Failure taxonomy (810 runs):** across every fault origin, **88.9%** of faults route through a planner
  switch; the dominant induced failure is a phantom / hard brake (76–85% of all faults); collision rate is
  highest for cost-map faults (11.9%) > object (5.2%) > prediction (0.4%).
- **Causal control:** softening the planner's selection (argmin → softmax, T=0.02) cuts switching by 36%
  (0.89 → 0.57) and drives object / cost-map collisions to **0** — *discrete selection amplifies*.
  **But this is a trade-off, not a free win:** prediction-origin collisions *rise* (0.6% → 2.8%, and to
  14.4% at T=0.10), because a softened planner no longer decisively avoids a genuinely mispredicted agent.

**Generalization to real data (nuPlan open-loop, real map + real agents)**

| Claim | Reproduces on real data? |
|-------|--------------------------|
| Rasterization attenuates (object→cost gain ≪ 1) | **Yes** — robust, all scenario categories |
| Cost-map is the most safety-critical interface | **Yes** — robust, all categories |
| Planner amplifies (cost→plan > 1) | Partial — car-following 2.80, but intersection 0.83 and lane-change 0.11 (**attenuates**) |
| Planner-switch is a "universal gateway" | **No** — argmin-flip collapses 0.89 (sparse) → 0.03–0.14 (dense real) |
| Directed > random in raw strength | **No** — the robust distinction is *consistency*, not strength |

> The **core characterization (rasterization = attenuator, cost-map = most-critical interface) generalizes
> to real data.** The planner-switch "gateway" is a mechanism observed in **sparse / argmin-planner** settings,
> not a universal law. These bounds are stated explicitly rather than overclaimed.

---

## Repository layout

```
torsion/
  operators/     object / cost-map / BEV / twist / temporal semantic-fault operators (contract-preserving)
  scenarios/     unified closed-loop pipeline, cost-map & sampling/potential-field planners, CV prediction
  analysis/      propagation metrics, mechanism (Jacobian / decision-margin), transfer response, failure taxonomy
  data/          nuPlan (.db) + nuScenes adapters, nuPlan HD-map (gpkg) road-prior, shared geometry
  metrics/       safety (min-TTC, collision), statistics (bootstrap CI)
scripts/         run_* drivers for each experiment
tests/           unit + integration tests
configs/         experiment configs
assets/images/   figures used in this README
```

Framing / positioning write-up: `TORSION_framing.md`.

---

## Reproducing

```bash
# environment (Python 3.12)
conda create -n torsion python=3.12 && conda activate torsion
pip install -e .            # numpy, etc.  (nuPlan path also needs: pip install shapely pyproj)

# tests
conda run -n torsion pytest -q

# a few experiments
python scripts/run_propagation_analysis.py --seeds 30                   # CIS / interface gains
python scripts/run_mechanism_analysis.py --seeds 30                     # M1 / M2 / M3
python scripts/run_transfer_function.py --seeds 25 --use-prediction     # linear/nonlinear response
python scripts/run_failure_taxonomy.py --seeds 30                       # propagation-path taxonomy
python scripts/run_planner_independence.py --seeds 20 --use-prediction  # sampling vs potential-field
python scripts/run_nuplan_propagation.py --n-frames 150                 # real-data generalization (nuPlan)
```

Large assets are intentionally not versioned: model weights, vendored `third_party/InterFuser`, and the
nuPlan / nuScenes datasets are obtained separately and placed locally.

---

## Status

Research code accompanying an in-progress paper. Target venues: IEEE T-ITS / T-IV.
Findings are reported with their honest scope; the metrics (interface gain, FAR, CIS, reach-safety) are
analysis tools defined for this framework, used to *observe* the reported phenomena — not claimed as general
laws.
