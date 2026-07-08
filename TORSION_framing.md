# TORSION — Framing v3 (methodology-centered)

> Supersedes v2. Written after fair, budget-matched, contract-respecting experiments across THREE
> representations (object-set, cost-map, and a real learned BEV feature via InterFuser). This is the
> project's working definition and contribution structure.

## 0. The core contribution — a new Fault Model

The biggest result of this research is NOT the discovery of a "twist" operator. It is the establishment of
**Semantic Fault** as a new **fault model**: a fault injected into an algorithm's *representation space* that
preserves structural validity (the contract) but twists the *meaning*, sitting between low-level dependability
faults (bit-flip, stuck-at, transient) and worst-case adversarial input perturbations. TORSION is the framework
that (a) defines this semantic fault model per representation, and (b) systematically analyzes how such faults
**propagate into safety failures**. Individual operators (rotation, displacement, inflation) are just instantiations
inside the model. This is a Fault-Injection contribution — the right home is dependability (DSN/ISSRE) — and it
extends naturally to any domain with structured intermediate representations and a safety/decision output
(robotics, VLM agents, medical AI).

## 1. What TORSION means (redefinition)

**TORSION is not about twisting geometry. It is about twisting the *semantic interpretation* of
structured driving representations while preserving their structural validity.**

- A geometric rotation/swirl is just ONE operator.
- A directed displacement is ANOTHER operator.
- A cost inflation/deflation is ANOTHER operator.

"TORSION" names the **framework** — semantic twisting under a preserved structural contract — not any single
operator. (This keeps the name; it changes what the name points at.)

## 2. The contribution is a METHODOLOGY, not a new operator

Empirically, the strongest operator is directed *displacement*, which is not itself novel. The novelty and the
paper live in the methodology, whose three pillars are:

1. **Representation-aware Fault Injection.** Faults are designed per representation (object-set, cost-map,
   BEV feature, trajectory), each with operators natural to that space — not one generic perturbation.
2. **Semantic Contract Preservation.** The representation stays a *valid* member of its contract
   (object count/class/track-id; cost value range; drivable/road topology; feature tensor shape). Only the
   *meaning* is twisted. Contract-violating faults (Gaussian, boundary-warping) are baselines, not the method.
3. **Cross-representation Error Propagation Analysis.** A unified protocol to inject budget-matched semantic
   faults at different representations and analyze, with comparable metrics, how the induced error propagates to
   planning/control — i.e. which representation is most safety-critical and why. (This is the flagship pillar and
   the least-developed so far.)

## 3. Robust empirical findings (honest, budget-matched)

- **Directed semantic displacement > geometric twist — confirmed in all three representations** (object-set,
  cost-map, and learned BEV features at two hook resolutions; the gap widens at higher BEV resolution). Geometric
  rotation is the weaker elegant cousin everywhere. Displacement is the effective operator.
- **Directedness matters.** Directed contract-preserving faults are more severe AND more consistent
  (near-deterministic per scenario) than random-direction faults at equal budget.
- **Consistency = interpretability signal.** Directed torsion produces reproducible, attributable degradation;
  random/Gaussian faults fail sporadically (high variance, tail events).
- **REAL-SIM validation (CARLA 0.9.16, 450 episodes).** A directed object-set semantic fault degrades REAL
  closed-loop safety (leading-vehicle: safe until a sharp threshold, then 100% collision; cut-in: graded rise).
  The consistency finding reproduces in real physics: directed = low-variance / threshold-like (min-TTC std
  ~0-0.3), random-direction = sporadic / high-variance (std ~0.74-0.81). Honest nuance: directed does NOT cause
  MORE collisions than random in the mid-range — random causes more there; the robust distinction is
  consistency/interpretability, not raw strength.
- **Honest caveats.** (a) "Contract violation ⇒ catastrophe" was a *budget artifact* — at matched budget,
  contract-violating faults are not more dangerous, only less consistent/valid. (b) The InterFuser BEV study is
  open-loop feature-sensitivity (no CARLA closed loop), so it complements rather than matches the closed-loop
  min-TTC numbers of object-set/cost-map. (c) translation is effective because a planner *responds* to shifted
  obstacle evidence — it is NOT a benign in-distribution transform (equivariance error is large).

## 4. Revised hypotheses

- **H1.** At equal budget on the same representation fields, a directed, contract-preserving semantic fault
  induces a *more severe and more consistent (near-deterministic)* safety degradation than random or
  contract-violating faults; random-fault failures are sporadic tail events.
- **H2 (flagship).** Semantic-fault sensitivity differs systematically by injection point (object-set vs
  cost-map vs BEV feature); the methodology can rank representation criticality with comparable metrics.
- **H3.** The effect is driven by *directed semantic displacement* and *contract preservation*, not by geometric
  form or perturbation magnitude.

## 5. Status & next

- Built & verified: object-set + cost-map closed-loop harnesses; fair baselines; statistics (bootstrap CI, tail,
  variance); canonical operators incl. twist/displace/inflation; InterFuser real-model BEV feature hook + fair
  twist-vs-displacement study.
- Under-built (the flagship): **Cross-representation Error Propagation Analysis** — a unified, comparable
  injection-point-sensitivity study (object-set vs cost-map in the shared closed loop; BEV as real-model
  complement), the scenario-wise failure heatmap, and an error-propagation trace per injection point.
- DONE: CARLA closed-loop real-sim validation (450 episodes, Town10HD) — see the real-sim bullet above.

## 6a. APEX FRAMING (v6 — ADVISOR CONFIRM #4, 2026-07-07) — System Identification via excitation

**Top-level definition (final):** TORSION uses **semantic faults as EXCITATION SIGNALS** to **empirically identify
the internal transfer/propagation characteristics** of the autonomous-driving pipeline. Roles:
- **Fault injection = the tool** (excitation design).
- **Error propagation = the observation target** (system response).
- **Characterization = the result** (per-interface gains, linear/nonlinear element map, critical interfaces).
- **(Empirical) System Identification = the methodology.**

Everything we built maps onto this: excitation DESIGN = contract-preserving semantic faults per representation;
excitation AMPLITUDE sweep = Task 1 linearity (amplitude-dependent gain); excitation LOCATION sweep =
injection-point/CIS; RESPONSE = propagation gains + safety; IDENTIFIED CHARACTERISTICS = M1/M2/M3 + the
linear-elements + argmin-nonlinear-switch map; PLANT VARIANTS = argmin/softmax/gradient planners (is the identified
characteristic plant-invariant?); REAL PLANT = nuPlan generalization.

**HONESTY CAVEAT (same principle as dropping "transfer function"):** say **"empirical / system-identification-
inspired characterization"** — we measure ε-finite-difference empirical gains and classify linear/nonlinear
response; we do NOT fit a formal PARAMETRIC model (no Laplace/state-space/frequency-domain transfer function).
Frame as "semantic faults as excitation to empirically identify the pipeline's propagation response." This keeps the
elevated framing while staying defensible. The SI frame also naturally ABSORBS the honest limits: "this element is
linear here, nonlinear there, scene-dependent" IS the identification result; C4's triple-qualification =
"identified characteristic varies with plant (planner) and input regime (scene density)."

**VENUE implication:** the excitation/SI frame fits **IEEE T-ITS / T-IV (control-/systems-flavored) as PRIMARY**;
DSN/ISSRE/RESS remain secondary (dependability-flavored).

## 6b. Contribution & framing (v5 — ADVISOR CONFIRM #3, 2026-07-07; SUPERSEDES v4 below)

**Positioning (final):** NOT a "fault-injection" paper — a **"System Identification + error-propagation
Characterization of the autonomous-driving pipeline, using semantic faults as system probes."**
**Title (confirmed):** *TORSION: A Semantic Fault Propagation Characterization Framework for Autonomous Driving
Systems* (fault injection → subtitle/tool). "Semantic Fault" MUST prefix "Characterization" (else scope too broad).

**Terminology caution (confirmed):** avoid the strong control-theory term **"transfer function"** — we measure
ε-finite-difference EMPIRICAL gains (no dynamics / Laplace / frequency domain). Use **"propagation response",
"empirical transfer behavior", or "propagation characteristics"**. (Code module `transfer_function.py` is internal.)

**Four contributions (confirmed structure):**
1. We **formulate semantic fault propagation across intermediate representations**.
2. We **characterize the autonomous-driving pipeline using semantic faults as system probes**.
3. We **identify critical propagation interfaces and explain WHY they amplify or attenuate faults** (M1 projection
   Jacobian ≪1 = attenuation; M2 argmin decision-boundary switching = amplification; M3 CV integration = amplification).
4. We **demonstrate that planner switching acts as a UNIVERSAL GATEWAY toward safety-critical failures**
   (Task-2 data: 76–85% of faults of every origin route through a planner switch; Task-3 causal: the hard argmin
   is the source — softening it cuts switches −36% and object/cost-map collisions to 0).

**Strict humility rule (keep to the end):** Interface Gain / FAR / CIS / reach-safety are **metrics WE define for
this analysis**. Claim "we define these and observe X within our framework" — NOT "this is a general law of AD
systems." The "consistent observed phenomenon within our framework, not a law" stance stays through the write-up.

**Advisor scores (v2):** Novelty 9.3, Technical Depth 9.7, Story 9.8, DSN-fit 9.8, T-ITS-fit 9.4.

**Two remaining experiments the advisor names (only these):**
- ① A **structurally different planner** (rule-based + learning-based) → "planner-independent" claim. NOTE:
  Task 3 (argmin↔softmax) is SELECTION-RULE variation within the SAME sampling family — it does NOT yet close
  this; a genuinely different planner (potential-field/gradient or learned) is still needed.
- ② **nuPlan open-loop** → generalization beyond the synthetic harness (larger setup cost).

---

## 6. Contribution (v4 — propagation-analysis centered; retained for history, superseded by 6b)

The paper's center is **cross-representation semantic-fault propagation analysis**, NOT the operator.

1. We formulate **semantic fault propagation across intermediate representations** in autonomous driving.
2. We introduce **TORSION**, a representation-aware fault-injection framework that **preserves structural
   contracts while perturbing semantic interpretation**.
3. We define **propagation metrics** that quantify amplification, attenuation, recovery, and safety-critical
   interface sensitivity (see §7).
4. We empirically show that **directed semantic displacement produces more consistent and interpretable safety
   degradation than random faults** across object-set, cost-map, and BEV representations.
5. We reveal a **cross-stage asymmetry**: late-stage cost-map faults propagate more directly to planning
   failures, whereas earlier object-set faults can be **attenuated or even inverted** by downstream modules.

Central research question: **"At which representation boundary does error get amplified into safety failure?"**
Target venues: DSN, ISSRE, Reliability Engineering & System Safety (RESS), IEEE T-ITS, IEEE T-IV.

## 7. Propagation metrics (to formalize — the analysis framework)

All measured against the common **realized-budget** denominator so they are comparable across representations.
Our unified pipeline already records per-stage state (object-set → cost-map → plan → control → safety), so most
are computable from existing traces.

- **Fault Amplification Ratio (FAR)** = downstream safety degradation / injected fault magnitude.
- **Fault Attenuation Ratio** = intermediate error reduction after a downstream module (per interface).
- **Propagation Depth** = number of modules a fault traverses before it measurably affects the safety metric.
- **Critical Interface Score (CIS)** = sensitivity of a representation boundary (fault → safety failure) — the
  formalized injection-point-sensitivity.
- **Recovery Time** = time to return to a nominal trajectory after the fault.

**MEASURED (Phase A, 30 seeds, high, directed displacement; from existing unified traces, no new experiments).**
The propagation metrics are now computed (`torsion/analysis/propagation.py`, `scripts/run_propagation_analysis.py`;
see `results_summary` §4.7 and `results/metrics/propagation_*.csv`). Headline numbers:
- **CIS asymmetry** — inject@cost-map ≫ inject@object: ALL 1.152 vs 0.125 (≈9×); leading_vehicle 1.739 vs 0.066 (≈26×).
- **Interface-gain map** — object→cost-map = attenuator (gain 0.004), cost-map→plan = amplifier (17.6–40.3×).
  So the object→cost-map rasterization boundary absorbs early faults; the cost-map→plan boundary amplifies them.
- **FAR** — object 0.009 vs cost-map 0.167 (≈19×); leading_vehicle object FAR = −0.040 (inversion: fault improves safety).
- **Reach-safety rate** (fair, position-unbiased form of depth) — object 0.33 vs cost-map 0.67 (ALL); leading 0.30 vs 1.00.
- **Recovery time** — object 3.03 s, cost-map 3.38 s.
- **Directedness × injection interaction** (new) — random faults collapse the asymmetry (cost-map CIS 1.152→0.048);
  directed faults are specifically dangerous at the cost-map interface.

Caveat: interface-gain absolute values depend on the per-stage normalization (a documented modeling choice) — claim
sign/ordering, report normalization sensitivity. CIS / reach-safety / recovery are in physical units and robust.
The gaussian baseline fails in the unified calibration (contract-limit overflow), so random_warp is the matched baseline.

**PHASE B — explicit PREDICTION stage (pipeline realism).** Chain is now
object-set → **prediction** (constant-velocity) → cost-map → plan → control → safety (`use_prediction`, default
False keeps all prior results; new outputs in `results/metrics/propagation_pred_*.csv`; 74 tests pass). Findings:
- **Prediction is a magnitude amplifier** (object→prediction gain 2.09 overall, 3.40 leading), and prediction-stage
  error scales linearly with the horizon (raw L2 2.26/4.67/7.55 at H=1/2/3 s ⇒ CV Jacobian ≈ H — a mechanism, not
  just a measurement). The next interface **prediction→cost rasterization attenuates** (0.004), same structural
  bottleneck as object→cost; cost→plan (argmin) remains the amplifier.
- **Critical interfaces are scenario-specific:** prediction dominates where the FUTURE trajectory decides safety
  (pedestrian-crossing CIS 1.02) and is negligible where current geometry dominates (leading-vehicle 0.03).
- **Realistic prediction corrects two Phase-A artifacts:** object-fault criticality was under-estimated without
  prediction (object CIS 0.125→0.684, reach-safety 0.33→0.91); and pedestrian-crossing was NOT fault-insensitive
  (Phase-A CIS ~0.01 was a prediction-missing artifact → with prediction all three interfaces are highly critical,
  1.23/1.02/0.95). This is exactly why a realistic multi-stage pipeline matters for the propagation claim.
Structural takeaway: **rasterization boundaries attenuate; the argmin planner interface amplifies; prediction
amplifies error magnitude but its safety-criticality is gated by the downstream rasterization and the scenario.**

**PHASE A+ — MECHANISM (measurement → explanation; the #1 reviewer demand).** Each interface's behavior is
explained by a structural quantity that predicts the measured gain (pure analysis, no core changes; 5 tests pass;
`torsion/analysis/mechanism.py`, `results/metrics/mechanism_*.csv`):
- **M1 rasterization = attenuator** because the cost-map is a kernel-smoothed, grid-quantized many-to-one
  projection of object position: local finite-difference Jacobian J = Δcost_L2/ε ≈ 0.020–0.040 ≪ 1 (near-linear),
  across object→cost and prediction→cost — explains the ~0.004 attenuation.
- **M2 cost→plan = amplifier** because the planner is a sampling ARGMIN with decision boundaries (NOT gradient —
  corrects the advisor's phrasing). Binning fault frames by clean decision-margin quartile: mean realized plan
  deviation 0.151→0.039→0.023→0.0056 m and argmin-flip rate 0.68→0.196→0.114→0.028 from smallest→largest margin
  (≈27× / ≈24×, monotonic; Spearman 0.52 ≫ Pearson 0.09 ⇒ hyperbolic 1/margin blow-up near the boundary). This is
  the causal explanation of the amplifier, and is Figure-worthy. Honest limits noted (Q4 flip 0.028≠0; Pearson weak).
- **M3 object→prediction = amplifier** because CV prediction integrates a velocity error over the horizon:
  ∂pred/∂v = t ⇒ horizon-mean Jacobian = H/2; analytic exactly matches empirical (0.5/1.0/1.5 at H=1/2/3 s) and raw
  prediction L2 ∝ H.
This is the propagation-theory core: attenuation = projection Jacobian ≪1; amplification = argmin boundary switching
and CV integration. Fault injection is now the experimental tool; the contribution is the propagation mechanism.

**TASK 1 — TRANSFER-FUNCTION characterization (system-identification framing; advisor-confirmed direction).**
Each interface's gain is measured as a function of injected magnitude (plan budget 0.05→2.0 m), classifying it as a
LINEAR transfer element (magnitude-independent gain) or a NONLINEAR element (magnitude-dependent gain). RAW physical
gain ratios are used (normalized gains telescope to the end-to-end gain by construction → trivial; avoided).
`torsion/analysis/transfer_function.py`, `results/metrics/transfer_function_*.csv`; 5 tests pass. Findings (CV of raw
gain across budgets):
- **cost→plan = the dominant NONLINEAR switching element** — nonlinear at ALL 3 injection points (CV 0.17/0.37/0.17);
  raw gain rises 9.5→12.7 with magnitude. Confirms M2 (sampling argmin) as a magnitude-dependent switching element.
- **plan→control (CV 0.03–0.08) and object→prediction (CV 0.054) are LINEAR transfer elements** across injections.
- Pipeline picture: **linear transfer elements (prediction CV-integration, tracking) with the planner (argmin) as
  the critical nonlinear switching element** — a clean control-systems characterization.
- **Honest nuance (do NOT overclaim "only planner is nonlinear"):** control→safety is ALSO nonlinear (CV 0.46/0.54,
  a collision-boundary SATURATION at the output) — but noisily estimated (safety_drop is censored/sparse; CV degenerates
  when mean gain≈0). And prediction→cost rasterization is ~linear with mild saturation at large magnitude.
Framing consequence: the "system-identification" claim RAISES the generality bar → showing this transfer-function
split is planner-invariant (a 2nd planner) becomes important, not optional.

**TASK 2 — data-grounded FAILURE TAXONOMY** (`torsion/analysis/failure_taxonomy.py`,
`results/metrics/failure_taxonomy_*.csv`; 810 runs; 4 tests pass). Representative paths extracted (not hand-drawn):
- **The planner argmin switch is a UNIVERSAL propagation gateway** — 76–85% of faults of EVERY origin
  (object/prediction/cost-map) reach the outcome via a planner_switch. This independently re-confirms M2 and the
  §4.10 transfer-function nonlinearity: the planner is the single amplification gate.
- **Dominant induced failure = phantom / hard emergency brake**, not collision — this contract-preserving planner
  responds to semantic faults by braking, not swerving (no lane-departure/off-road modes appear).
- **Origin → severity gradient**: collision rate cost-map 11.9% > object 5.2% > prediction 0.4%.
- **Honesty**: the advisor's three illustrative paths (object→prediction-drift→wrong-yield;
  cost-map→planner-switch→lane-departure; prediction→late-brake→collision) are NOT supported by the data
  (lane-departure/near-miss essentially absent; prediction→collision only 1 run). The data gives a cleaner unified
  story instead: "every fault → planner switch → mostly hard-brake, and only cost-map faults escalate to collision."
- **Consequence**: the absence of swerve/lane-departure is this planner's braking-policy artifact → a 2nd planner
  (Task 3) is needed to test whether the taxonomy/transfer-function split is planner-invariant.

**TASK 3 — planner-invariance & causal isolation (pluggable ARGMIN↔SOFTMAX selection).**
`CostMapPlannerConfig.selection_mode` ("argmin"|"softmax", τ); default argmin reproduces everything (full suite
93 passed, 1 skipped). `results/metrics/planner_invariance.csv`. Verified findings (seeds 20):
- **H1 (partly, with a twist):** the HARD ARGMIN causes the discrete planner-switch gateway and the collision
  escalation — softening it cuts planner_switch 0.889→0.57 (−36%) and drives object/cost-map collision to ZERO
  (0.05→0, 0.106→0). So soft selection is a genuine fault→collision MITIGATION (actionable planner-design lever).
  BUT cost→plan stays NONLINEAR in all modes — softmax does not linearize it; it converts the nonlinearity from
  discrete switching to continuous magnitude-dependence (CV 0.19→1.3, though the absolute CV is partly inflated by
  near-zero plan responses under soft selection). So argmin is the source of the discrete switching + collisions,
  NOT the sole source of all nonlinearity.
- **H2 (partly invariant):** the amplify/attenuate TOPOLOGY (cost→plan amplifies, rasterization attenuates,
  prediction amplifies) PERSISTS across selection modes → structure is planner-general; but absolute CIS
  magnitudes/ranking are planner-specific (softmax is globally far safer: cost-map CIS 0.957→0.007; at τ0.1 object
  CIS even exceeds cost-map).
- **Honest anomaly:** at τ0.1 only prediction-fault collisions rise (0.006→0.144) — soft-blended paths avoid a
  corrupted predicted occupancy less decisively in a few sparse pedestrian cases (n=20); not over-interpreted.
Net: propagation STRUCTURE is planner-general; the discrete planner nonlinearity + collision gateway is
argmin-specific and is a controllable safety lever — a stronger, more actionable result than plain invariance.

**REMAINING ① DONE — planner-independence via a STRUCTURALLY DIFFERENT planner (PotentialFieldPlanner:
continuous cost-gradient + lane attraction, no candidate enumeration / no argmin).** `build_planner` factory +
`planner_type` (default "sampling" reproduces everything; full suite 97 passed/1 skipped).
`results/metrics/planner_independence.csv`. Verified (seeds 20, sampling vs potential-field):
- **Planner-INDEPENDENT:** cost-map is the highest-CIS injection RANK under both; cost→plan is a NONLINEAR
  response under both; argmin-flips absent under potential-field (expected).
- **Architecture-SPECIFIC (refines C4 — important):** the planner-switch "universal gateway" is SAMPLING-ARGMIN
  specific — gateway rate 0.852→0.139 and cost→plan mean gain 15.4→0.816 (<1) under potential-field, which is also
  far safer (collisions →~0). Same direction as Task 3: DISCRETE (argmin) selection = dangerous amplifier;
  CONTINUOUS (softmax / gradient) selection = safer / attenuating.
- **Caveat:** cross-planner CIS ABSOLUTE magnitudes are not comparable (potential-field CIS inflated by small plan
  budgets — same small-denominator issue as softmax); compare ranks/verdicts only. Gradient planner gains/thresholds
  are tunable, so "less amplification" may be partly an over-damped setting.

**C4 HONEST RESTATEMENT (must use this scope):** "Regardless of FAULT ORIGIN (Task 2: 76–85%), planner selection
is the universal amplifying gateway to safety failure **for argmin/sampling-class planners** (the dominant AD
family); continuous-selection planners (softmax, gradient) attenuate this gateway." The universality is over fault
ORIGINS, NOT over planner ARCHITECTURES. The planner-INDEPENDENT claims are: (i) cost-map is the most safety-critical
interface, and (ii) the planner interface is nonlinear; plus the actionable dichotomy **discrete-selection = amplifier,
continuous-selection = safer**. Do NOT claim "planner switching is universal across all planners" — this experiment
refutes that.

**REMAINING ② DONE — REAL-DATA GENERALIZATION (nuPlan open-loop, real map + real agents).** Built a lightweight
nuPlan stack (shapely+pyproj, NO devkit): `torsion/data/{nuplan_adapter,nuplan_map,_geometry}.py` +
`scripts/run_nuplan_propagation.py`; 12 logs / 290 frames / 10,440 runs; real gpkg lane road-prior (4326→UTM) +
real-agent obstacles; two open-loop safety metrics (plan deviation; closest-approach + CV-TTC of the planned path to
real tracks). Full suite 108 passed/1 skipped. **HONEST reproduction verdict (positive AND negative):**
- **GENERALIZES (robust, real+synthetic):** (a) rasterization ATTENUATES — object→cost gain ≈0.014 (≪1) across all
  scenario categories; (c) cost-map injection is the MOST safety-critical interface — across FOLLOWING / INTERSECTION /
  LANE_CHANGE. These are the paper's CORE characterization claims and they hold on real data.
- **PARTIAL:** (b) planner AMPLIFICATION (cost→plan >1) reproduces in FOLLOWING (2.68, the synthetic analog) but
  weakens in dense INTERSECTION/LANE_CHANGE (<1) — the real cost landscape is richer.
- **DOES NOT GENERALIZE (must temper C4):** (d) the planner-switch GATEWAY — argmin-flip rate collapses from 0.89
  (sparse synthetic) to 0.02–0.23 (dense real). A single-agent fault rarely flips the global argmin among ~83 real
  agents. (e) directed>random does not reproduce (consistent with CARLA: the robust claim is CONSISTENCY, not raw
  strength). Effect sizes are also small on real data.
**Consequence for C4 (TRIPLE-qualified now):** the "planner-switch universal gateway" is argmin-specific
(planner-independence) AND sparse/synthetic-scene-specific (nuPlan). It must be presented as a MECHANISM observed in
sparse settings, NOT a universal law. The DEFENSIBLE, generalizing contributions are: **rasterization = attenuator,
cost-map = the most safety-critical representation interface** (hold in synthetic + real), plus the mechanistic WHY
(M1/M2/M3) and the actionable discrete-vs-continuous-selection safety dichotomy. This honest scoping is a strength,
not a weakness — the core characterization survives real data; the fragile claims are explicitly bounded.

## 8. Semantic fault → real-world cause (why these faults occur)

| TORSION fault | Real-world cause |
|---|---|
| object position displacement | tracking drift, calibration error, sensor-fusion delay |
| velocity / yaw displacement | temporal misalignment, motion-estimation error |
| cost-map shift | occupancy-prediction error, segmentation error, map-update delay |
| BEV feature displacement | feature-fusion drift, attention failure, domain shift |

Use these to GROUND the magnitude ranges (e.g. typical tracking drift ~0.1-0.5 m) so magnitudes are not arbitrary
(addresses the "arbitrary magnitude" threat, design §19).

## 9. Scale-up plan (to reach journal grade)

- **CARLA**: Town03 / Town05 / Town10HD; leading-vehicle, cut-in, pedestrian-crossing, intersection,
  obstacle-avoidance; weather variation; 30-50 seeds/cell. NOTE: launch a fresh CARLA per map at STARTUP
  (never runtime `load_world` — it crashes the server).
- **Planners**: rule-based/simple stack (have) + one learning-based planner.
- **Open-loop**: add nuScenes or nuPlan replay analysis. CARLA closed-loop + nuPlan open/closed-loop is much
  stronger than CARLA-only. (nuPlan open-loop is likely more tractable than driving InterFuser in CARLA 0.9.16.)
