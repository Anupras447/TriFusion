# MATLAB / Simulink + RoadRunner — Compulsory-Track Technical Plan

**Status:** Plan + all source assets implemented in this repo. Final `Run` step needs a
machine with MATLAB R2023b+ (no local MATLAB on the build machine — scripts are written
against documented APIs and must be executed once by the team; see §7 checklist).

## 1. What the PS demands (and where each item lives)

| PS requirement | Deliverable in this repo | Tool |
|---|---|---|
| 2 detailed RoadRunner scenes (village + urban intersection) | `roadrunner/S1_village.xodr`, `roadrunner/S2_intersection.xodr` | RoadRunner / OpenDRIVE |
| All 5 scenarios testable | `roadrunner/S1..S5.xosc` (OpenSCENARIO 1.0, actors + triggers) | RoadRunner ScenarioBuilder |
| Perception: camera + LiDAR + radar, Indian road users | `matlab/sensors.m` (vision/radar generators + fusion) | Automated Driving Toolbox |
| Prediction + planning + decision logic | `matlab/buildEgoModel.m` → `TriFusionEgo.slx` | Navigation Toolbox + Stateflow* |
| Vehicle motion | bicycle block inside `TriFusionEgo.slx` | Vehicle Dynamics Blockset (fallback: kinematic equations in a MATLAB Function — no extra license) |
| Detection / trajectory prediction (deep) | Import YOLOv8n + U-Net ONNX via `importONNXNetwork` (weights already in Drive) | Deep Learning Toolbox |
| Metrics: replan latency, smoothness, completion | `matlab/runAllScenarios.m` + `eval/metrics.py` parity | — |
| Report + video | `report/TECH_REPORT.md`, `report/VIDEO_SHOTLIST.md` | — |

\* If no Stateflow license: `buildEgoModel.m` generates the same logic as a MATLAB Function
(`safetyFilter.m`) — decision table is identical, only the editor differs.

## 2. Toolbox checklist (install before §7)

MATLAB R2023b+, RoadRunner R2023b+, plus: Automated Driving Toolbox, Navigation Toolbox,
Sensor Fusion and Tracking Toolbox, Deep Learning Toolbox, (optional) Stateflow,
(optional) Vehicle Dynamics Blockset. Verify with `matlab/checkToolboxes.m`.

## 3. RoadRunner scene design (the 2 detailed scenes + 3 supporting)

Shared conventions: right-hand traffic, SI units, ego = `Sedan` asset, auto-rickshaw =
`TukTuk`-style box truck asset recolored, cow = `Pedestrian` asset scaled 1.6×
(RoadRunner has no cattle asset — documented workaround, same collision footprint as
`sim/` 1.2×2.0 m), vendor cart = `StreetCart` prop, pothole = road-surface decal +
speed-penalty zone, lane markings `none` wherever the scenario says "unmarked".

### Scene A — `S1_village.xodr` (DETAILED #1)
- 220 m rural 2-lane, width 4.5 m half, S-curves (line→spiral→arc), `roadMark type="none"`.
- Props: parked tractor (`<object type="vehicle">` static at s=60, lateral −3.2),
  pothole decals at s=95, roadside cow props at s=70/80.
- Dynamic (`S1_village.xosc`): 8 forward + 4 oncoming + 2 rear vehicles, 6 pedestrians;
  timed trigger t=8 s: cow entity `Trajectory` action from (−3.8) to (+3.8) at 1.2 m/s.
- Pass: reach s=200, 0 collisions, stops ≤ 3 s cumulative.

### Scene B — `S2_intersection.xodr` (DETAILED #2)
- 4-way unmarked junction at s=100 (`<junction>` with 4 connecting roads, no signals,
  no stop lines), 2 idling-auto props on corners.
- Dynamic (`S2_intersection.xosc`): cross-traffic events t=5 s (auto, left→right 5 m/s)
  and t=7 s (bike, right→left 6 m/s); 12 pedestrians, 14+8+4 vehicles.
- Pass: clear junction < 30 s, 0 collisions (creep-and-yield expected).

### Scenes C–E (supporting, same conventions)
- `S3_highway_merge.xodr`: 300 m 2-lane + entrance ramp (s=40–90); `.xosc`: truck
  `LaneChange` into ego lane t=6 s + tailgater spawn; pass TTC > 2 s.
- `S4_market.xodr`: 220 m street + 4 stall props both sides; `.xosc`: cart pullout t=10 s,
  ped crossings t=12/14 s; pass at crawl ≤ 3 m/s, 0 contacts.
- `S5_open_road.xodr`: 220 m open road + 3 roadside cow props; `.xosc`: cow sprint
  across t=7 s at 2.5 m/s, 20 m ahead; pass: full stop, gap > 2 m.

Seed rule: every `.xosc` fixes `RandomSeed=7` (matches Python seeds) so judge reruns are identical.

## 4. Simulink architecture (`TriFusionEgo.slx`, built by `buildEgoModel.m`)

```
[Scenario Reader] → [Sensor Fusion: camera+radar → tracks] → [Prediction: CV+KF, 8×5 steps]
      → [Lattice Planner: 9 candidates, cost + SAT] → [Safety Filter: bubbles/Stateflow]
      → [Bicycle Model: x,y,θ,v] → [Metrics: latency/jerk/TTC/completion] → [To Workspace]
```

- **Scenario Reader**: `roadrunner` API block, loads `.xodr` + `.xosc` per run.
- **Sensor Fusion** (`sensors.m`): `visionDetectionGenerator` (monocamera, 60° FOV) +
  `radarDetectionGenerator` (long-range) → `multiObjectTracker` (GNN, same 60 px/10-frame
  semantics as `perception/tracker.py`).
- **Prediction**: constant-velocity rollout of 8 nearest tracks × 5 steps — byte-identical
  contract to the Python DQN trajectory head (80 floats).
- **Lattice Planner** (MATLAB Function, port of `planning/lattice.py`): 3 lateral ×
  3 speeds, 2.5 s horizon, cost = 1000·collision + 0.5·jerk + 2.0·lane + 1.0·speed.
- **Safety Filter** (Stateflow chart `SafetyChart` / fallback `safetyFilter.m`): states
  `Cruise → Caution(B1) → Emergency(B2)`, rear-veto guard — same table as
  `planning/safety.py` and legacy `choose_bot_action`.
- **Bicycle Model**: `ẋ=v·cosθ, ẏ=v·sinθ, θ̇=v/L·tanδ` (L=2.7), accel rate-limited ±0.8/step
  except Emergency (mirrors Python rate limiter).
- **Metrics**: `replan_ms` via `tic/toc` around planner, jerk RMS, bumper-gap TTC,
  completion % → `runs_matlab/<S>_seed<N>.mat` + CSV with the same columns as `runs/*.json`.

## 5. Parity story (your killer slide)

`matlab/importPythonResults.m` loads `runs/*.json` next to the `.mat` files and prints one
table: Python vs Simulink on collisions / p95 latency / completion across 15 runs.
Claim: "same planner, same scenarios, two independent implementations, matching results."
Run Python first (done: 15/15), then MATLAB — any mismatch is a bug in the port, not the idea.

## 6. File map

```
roadrunner/S{1..5}_*.xodr + S{1..5}_*.xosc   RoadRunner scenes + scenarios
matlab/checkToolboxes.m      license preflight
matlab/buildEgoModel.m       generates TriFusionEgo.slx (Stateflow or fallback)
matlab/safetyFilter.m        decision logic (also used if no Stateflow)
matlab/sensors.m             camera+radar generators + tracker config
matlab/runAllScenarios.m     5 scenarios × 3 seeds → runs_matlab/ + table
matlab/importPythonResults.m Python↔Simulink parity table
report/TECH_REPORT.md        submission report draft (MATLAB mapping § included)
report/VIDEO_SHOTLIST.md     3-min demo shot list
```

## 7. Execution checklist (team machine with MATLAB — ~2 hours)

1. `checkToolboxes` → all green (or note Stateflow/VDB fallbacks in report).
2. RoadRunner: open each `.xodr`, verify geometry; import `.xosc`, press Play once per scenario.
3. `buildEgoModel` → open `TriFusionEgo.slx`, confirm 6 blocks + chart, sample time 0.05.
4. `importONNXNetwork` for YOLOv8n/U-Net (weights in Drive) — optional; stub flag if skipped.
5. `runAllScenarios` (seeds 0,1,2) → expect 15/15, 0 collisions (mirrors Python).
6. `importPythonResults` → paste parity table into report + PPT.
7. Record video per shotlist; export scene screenshots from RoadRunner for the report.
