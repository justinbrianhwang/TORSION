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
  many-to-one, kernel-limited map (local Jacobian ≈ 0.024–0.035 ≪ 1).
- **The cost-map is the most safety-critical representation interface** — under a *matched perturbation
  budget*, a semantic fault injected there ends in a **collision 23.9%** of the time, vs. 10.6% at the
  object stage and 3.9% at prediction (high magnitude; CI-separated).

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
| object → prediction | 1.9 – 2.1 | amplify | linear (gain CV 0.03) |
| prediction → cost-map | **0.010 – 0.017** | **attenuate** | linear |
| cost-map → plan | **5.6 – 9.5** | **amplify** | **nonlinear — switching (gain CV 0.26)** |
| plan → control | 2.4 – 2.8 | amplify | linear (gain CV 0.07) |

**Mechanisms — *why* each boundary behaves as it does**

| # | Boundary | Behaviour | Structural cause | Evidence |
|---|----------|-----------|------------------|----------|
| M1 | rasterization (→ cost-map) | attenuator | many-to-one grid projection | local Jacobian 0.024–0.035 ≪ 1 |
| M2 | cost-map → plan | amplifier | **sampling-argmin decision-boundary switching** | min-margin quartile: plan deviation ×7.5, argmin-flip ×7.1, Spearman ρ=0.42 (6 scenarios, 5,400 frames) |
| M3 | object → prediction | amplifier | constant-velocity integration over horizon | ∂pred/∂v = t; analytic = empirical to 8e-14 |

![Interface gains: which boundaries amplify vs. attenuate](assets/images/03-interface-gains.png)

*The same input error is strongly **attenuated** by rasterization (two orders of magnitude), then sharply
**amplified and clipped** by the cost-map→planner argmin (a switching nonlinearity), while
object→prediction amplifies mildly and **linearly**.*

The cost-map→plan amplification is a **decision-boundary switch**, not a smooth gain: a tiny cost-map
perturbation can flip the planner's `argmin` from a straight trajectory to a hard swerve/brake.

![argmin decision-boundary switching produces a phantom brake](assets/images/04-argmin-switching.png)

*Mechanism M2: a tiny cost-map perturbation moves the global minimum from candidate 8 (straight) to
candidate 13 (hard swerve/brake) — a discontinuous "phantom brake". This switching behaviour is why the
cost-map is the most safety-critical interface.*

**Which interface is most safety-critical?** We rank by **collision rate under a matched perturbation
budget** — the outcome that matters — not by our own sensitivity metrics. At **high fault magnitude**
(the regime where collisions occur), across **6 scenarios, n=180/stage**:

| Injection stage | **Collision rate [95% CI]** | Reach-safety | CIS (high) | CIS (all mag.) |
|---|---|---|---|---|
| object | 10.6%  [6.5, 16.0] | *0.26* | 0.45 | 0.65 |
| prediction | 3.9%  [1.6, 7.8] | 0.18 | 0.17 | 0.26 |
| **cost-map** | **23.9%  [17.9, 30.8]** | 0.25 | **0.56** | **0.73** |

Cost-map's lower bound (17.9) exceeds object's upper bound (16.0) — **CI-separated**, not merely ordered.
Cost-map ranks first in **5 of 6 scenarios**; the exception is `oncoming_drift`, where *prediction* faults
dominate (short time-to-conflict makes misprediction the hazard).

**The ranking is robust to how the budget is matched.** Re-ranked under 4 different matching rules
(operator magnitude, delivered plan deviation, delivered cost perturbation, delivered control deviation) —
cost-map is first, CI-separated, in **all four**.

### 🔑 Directedness is what makes the critical interface *visible*

Under a **random** contract-preserving warp of **equal budget**, the ranking **disappears entirely**:

| | cost-map | object | prediction |
|---|---|---|---|
| **directed** (semantic fault) | **23.9%** | 10.6% | 3.9% |
| **random** (same budget) | 12.2% | *13.3%* | 8.3% |

The stages become statistically indistinguishable and cost-map isn't even first. Directedness matters
**only at the cost-map**: directed nearly doubles collisions there (23.9% vs 12.2%, Fisher p=0.006), while
at object and prediction it makes no difference (p=0.52, p=0.12).

> **The semantic fault is not a stylistic preference over Gaussian noise — it is what makes the critical
> interface observable at all.** An undirected robustness study of this pipeline, however carefully
> budgeted, would have concluded the three stages are equally critical.

> **Honest caveat:** the ranking is a **high-magnitude** statement — at low/medium magnitude faults
> essentially never collide, so there is no ranking to make. Reach-safety rate ranks object marginally
> first (0.26 vs 0.25 — a tie), which is correct: cost-map faults don't *arrive* more often, they're more
> *catastrophic on arrival*.

**System characterization**

- **Propagation response (amplitude sweep):** `cost→plan` is a *nonlinear* switching element; `plan→control`
  and `object→prediction` are *linear* elements. The pipeline is a cascade of linear transfer elements with the
  **planner (argmin) as the critical nonlinearity**.
- **Failure taxonomy (1,620 runs, 6 scenarios):** across every fault origin, **~92%** of faults route
  through a planner switch; the dominant induced failure is a phantom / hard brake (74–88%).
- **Causal control:** softening the planner's selection (argmin → softmax, T=0.02) cuts switching by 36%
  (0.89 → 0.57) and drives object / cost-map collisions to **0** — *discrete selection amplifies*.
  **But this is a trade-off, not a free win:** prediction-origin collisions *rise* (0.6% → 2.8%, and to
  14.4% at T=0.10), because a softened planner no longer decisively avoids a genuinely mispredicted agent.

**Generalization to real data (nuPlan open-loop, real map + real agents)**

| Claim | Reproduces on real data? |
|-------|--------------------------|
| Rasterization attenuates (object→cost gain ≪ 1) | **Yes** — 0.032–0.069, all categories |
| Cost-map is the most safety-critical interface (by collision rate) | **Yes** — all categories |
| Planner amplifies (cost→plan > 1) | **Yes** — 3.09 following, **3.74 intersection**, 2.60 lane-change (*strongest where densest*) |
| Switching pathway transfers | **Yes** — argmin-flip **0.237** (real) vs **0.258** (synthetic) |
| Margin governs switching (M2) | **Yes** — ρ = 0.219, p = 6e-58 |
| Directed > random in raw strength | **No** — the robust distinction is *consistency*, not strength |
| Switching is exclusive to argmin planners | **No** — potential-field gateway (0.30) *exceeds* sampling (0.23) |

### Benchmark validity: report the feasible set

Every planner-based metric here — cost→plan gain, argmin-flip rate, decision margin, the switching pathway
— presupposes the planner has a **genuine choice**. If the cost map admits no collision-free candidate, the
planner doesn't select; it falls back, and every one of those metrics silently changes meaning.

This is easy to violate on real data and hard to notice: an incomplete set of map layers, a slightly large
obstacle inflation, or a conservative collision threshold will each make the *entire* candidate set
infeasible exactly where it matters (dense intersections, tight gaps) while leaving open-road frames
untouched. The numbers aren't noisy — they're confidently wrong, and they fail in a **seductive direction**:
the planner stops switching, amplification appears to vanish, and you get a clean, plausible *"the
simulation result doesn't generalize to real data"* — a conclusion the field is primed to believe.

> The check is nearly free. We report **`n_feasible` = 10.7 / 17** candidates (11.6 at intersections),
> confirming the planner is *choosing*, not falling back. Our drivable surface composes nuPlan's
> `lanes_polygons` **and** `gen_lane_connectors_scaled_width_polygons` — the latter carries the area a
> vehicle traverses *through an intersection*, where the ego lies outside every lane polygon.

**A hypothesis we tested and rejected.** The obvious reason the gateway collapses on real data is *scene
density*: dense scenes make most planner candidates infeasible → the surviving minimum is well separated →
the argmin can't flip. We built a controlled dense scenario (`dense_traffic`: ego boxed in by a braking
lead + two flanking vehicles) and **the middle step fails.**

| | feasible candidates | decision margin | **argmin flip** |
|---|---|---|---|
| `dense_traffic` | **4.9 / 23** | 0.0060 | **0.29** |
| `leading_vehicle` | 5.3 / 23 | **0.0331** | **0.00** |
| sparse (avg) | 13.9 / 23 | 0.0043 | 0.25 |

Density constrains the candidate set as designed (4.9/23) and does raise the median margin (1.39×,
p=3e-7) — but **flips do not fall** (0.29 vs 0.25). The decisive contrast is `dense_traffic` vs
`leading_vehicle`: *equally constrained* (~5 feasible), yet only `leading_vehicle` never flips — its margin
is 5.5× larger. Across scenarios, flip rate tracks **margin** (ρ = −0.77), not feasible-candidate count
(ρ = +0.60, wrong sign).

> **Constraining the feasible set does not, by itself, separate the surviving candidates' costs.** The
> quantity governing argmin switching is the **decision margin**; scene density is not a reliable proxy for
> it. We withdraw the density explanation. M2 (margin governs switching) is untouched — and now holds
> across six scenarios.

**…and the margin mechanism is confirmed on real data.** We instrumented the nuPlan runner to log the same
margin: inverse margin predicts argmin switching on real logs at **Spearman ρ = 0.219** (p = 6e-58,
n = 5,220), just as in simulation (ρ = 0.42). **M2 replicates outside the synthetic harness.**

> **Actionable:** monitor the **planner's decision margin** — it predicts the failure mode and the planner
> already computes it, so exposing it is free. The intuitive proxy (scene density) is measurably *not*
> causal: the controlled experiment shows density doesn't suppress switching, and on real data the densest
> scenes (intersections) are where amplification is **strongest**.

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
Findings are reported with their honest scope. Interfaces are ranked by **collision rate**; the metrics we
define for this framework (interface gain, FAR, CIS, reach-safety) are analysis tools used to *observe* and
*explain* the phenomena — they are not claimed as general laws, and where they disagree with the collision
ranking we say so rather than reporting only the agreeing one.
