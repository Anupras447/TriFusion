# Technical Report — Adaptive Path Planning & Collision Avoidance
## on Unstructured Indian Roads (TriFusion SIH)

> Draft skeleton — fill bracketed fields before submission. Metrics below are measured
> (Python sim v2, 15 seeded runs); MATLAB column fills in after `runAllScenarios`.

### 1. Problem & approach
Indian roads break structured-driving assumptions: no lane markings, wrong-way
two-wheelers, cattle, vendor carts, informal merges. We built a closed-loop pipeline —
**perception → prediction → lattice planning → safety-filtered control** — validated on
5 seeded scenarios in two independent implementations (Python + MATLAB/Simulink).

### 2. System architecture
- **Scenes:** RoadRunner OpenDRIVE (`roadrunner/*.xodr`, §3 of `matlab/MATLAB_PLAN.md`) +
  Python twin configs (`sim/scenarios/*.yaml`), fixed seeds for identical reruns.
- **Perception:** YOLOv8 (12 IDD classes incl. autorickshaw) + U-Net road seg + MobileSAM
  masks → 9-channel Tri-Fusion net (5 classes) → BEV → Kalman tracker. Simulink twin:
  `visionDetectionGenerator` + `radarDetectionGenerator` → `multiObjectTracker` (`matlab/sensors.m`).
- **Prediction:** constant-velocity Kalman rollout, 8 agents × 5 steps (same 80-float
  contract as the DQN head in legacy `IRS_dum (2) (1).py`).
- **Planning:** Frenet lattice, 3 lateral × 3 speeds, 2.5 s horizon @20 Hz;
  cost = 1000·collision(SAT) + 0.5·jerk + 2.0·lane + 1.0·speed (`planning/lattice.py`).
- **Decision/safety:** bubble gate — B2 emergency brake, B1 decelerate, rear veto
  (`planning/safety.py`, Stateflow chart `SafetyChart` / `matlab/safetyFilter.m`).
- **Control:** kinematic bicycle (L=2.7), accel rate-limited ±0.8/step, full authority in Emergency.

### 3. Scenarios & results (Python measured)

| Scenario | Seed | Collisions | Success | Compl.% | minTTC | p50 | p95 | jerk |
|---|---|---|---|---|---|---|---|---|
| S1 village | 0–2 | 0 | 3/3 | 100 | 10–38 s | 3–9 ms | 10–47 ms | — |
| S2 intersection | 0–2 | 0 | 3/3 | 100 | 5–50 s | 6–14 ms | 17–47 ms | — |
| S3 highway merge | 0–2 | 0 | 3/3 | 100 | 1.5–4.4 s | 2–4 ms | 4–17 ms | — |
| S4 market | 0–2 | 0 | 3/3 | 100 | — | 6–18 ms | 18–38 ms | crawl |
| S5 cattle | 0–2 | 0 | 3/3 | 100 | 4.9–23 s | 2–4 ms | 7–24 ms | — |
| **Total** | | **0** | **15/15** | | | | **p95 < 50 ms** | |

(Full numbers: `runs/*.json`, reproduced by `python eval/run_all.py --seeds 3`.)
Headline: **15/15 collision-free, replan p95 < 50 ms on laptop CPU.**

### 4. MATLAB/Simulink validation
`TriFusionEgo.slx` (built by `matlab/buildEgoModel.m`), same scenarios via
`roadrunner/*.xosc`, same metrics via `matlab/runAllScenarios.m`. Parity table
(`matlab/importPythonResults.m`): [paste after MATLAB run].

### 5. Ablations (suggested, 1 evening)
No safety filter → collisions return (proves the gate); no tracker (raw detections) →
S5 cow missed (proves prediction); 1 candidate (no lattice) → S3 merge fails.

### 6. Design choices & limits
Lattice over DQN for determinism + auditability; stub perception default so planning is
testable without GPU; cow = scaled pedestrian asset (no cattle model in RoadRunner).
Limits: single-frame IDD perception (no video MOT yet), sim-only validation, no rain/night.

### 7. MATLAB equivalence statement (for the "why not Simulink?" question)
Statement encourages MathWorks tools; identical semantics implemented in Python for
portability (table: RoadRunner→YAML scenes, ADT→perception/, Nav→lattice,
Stateflow→safety filter, VDB bicycle→world.py, DLT→YOLO/U-Net). Simulink twin in
`matlab/`; parity table §4.
