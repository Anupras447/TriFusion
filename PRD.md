# PRD — Adaptive Path Planning & Collision Avoidance on Unstructured Indian Roads
## SIH Win Plan + Technical Implementation Guide

**Version:** 2.0 (rebuild plan) | **Stack decision:** Python backend + Three.js frontend (no MATLAB dependency)
**Goal:** Top-3 SIH finish with a live demo that survives judges + a metrics-backed report + video.

---

## 1. Why the current build won't win (honest audit)

| Problem statement asks | Current state | Verdict |
|---|---|---|
| Perception (camera/LiDAR/radar) → detect auto, pushcart, pedestrian, animal | Colab only: U-Net + YOLOv8 + Pose + SAM + fusion on IDD still images, open-loop, never feeds planner. Sim uses synthetic `get_sensor_distances` + ground truth | Trash for demo — looks like two separate projects |
| Short-term motion prediction incl. non-lane / irregular motion | DQN trajectory head (8 agents × 5 steps) exists but only trains on sim ground truth; Kalman tracker in Colab runs single-frame | No credible prediction story |
| Safe collision-free path + real-time replan | Bubbles + `find_clear_bubble_angle` reroute exist, but no planner curve (no Frenet/lattice/MPC), no replan-latency measurement | Judges will call it "brake + swerve" |
| Missing lanes, informal merge, sudden pedestrian, unexpected obstacle | Random hazards exist, but not as 5 named, repeatable scenarios with pass/fail | Cannot claim "validated on 5 scenarios" |
| 5 realistic scenarios + 2 detailed scenes | 4 markdown scenarios in Colab, zero scenario configs in sim | Fails the #1 rubric item |
| Metrics: replan latency, smoothness, completion rate | Zero instrumentation | No numbers = no win |
| Report + video + closed-loop validation | No report, no video, perception↔planner link missing | Incomplete submission |

**Decision: keep the good parts, rebuild the demo layer.**
Keep: DQN + bubbles + NPC traffic logic + IDD weights. Rebuild: scenario harness, planner interface, metrics, visualization (Three.js), perception bridge.

---

## 2. Win strategy (what actually scores in SIH)

1. **Live 5-scenario demo that never crashes.** Judges remember a cow crossing + auto cutting in + market chaos handled smoothly. One crash on stage = out.
2. **Numbers on screen.** Live HUD: replan latency (ms), min TTC, collisions, smoothness (jerk), completion %. Print a results table at the end.
3. **Indian-specific story.** Wrong-way two-wheeler, cattle, vendor cart, unmarked merge, no-lane road. Say the words "unstructured" and show them.
4. **Perception that runs on real images.** Even a small YOLOv8n + road-seg overlay on IDD frames, feeding a BEV occupancy grid into the planner, beats "synthetic sensors only".
5. **Replan visibly.** Show candidate paths (green = chosen, red = rejected) + safety bubbles. Judges need to *see* planning, not just a car moving.

Non-goals (do NOT waste time): full autonomy stack, LiDAR hardware, MATLAB rewrite, training SOTA detectors from scratch, multiplayer, mobile app.

---

## 3. Target architecture (hybrid — build this)

```
                    ┌────────────────── PERCEPTION (Python service) ──────────────────┐
                    │ YOLOv8n (12 IDD classes) + U-Net road seg → BEV occupancy grid  │
                    │ IDD frames / webcam / sim-stub → detections + road mask (10 Hz)  │
                    └──────────────────────────────┬──────────────────────────────────┘
                                                   │ occupancy + tracks (JSON @10Hz)
┌──────────────┐   ┌───────────────────────────────▼──────────────────────────────────┐
│ Three.js     │◄──│ SIM + PLANNER (Python, pygame headless or headful)                │
│ frontend     │   │ World: 5 scenario configs │ Prediction: CV + KF + DQN head        │
│ (viz only,   │   │ Planning: Frenet lattice + bubble safety filter (20 Hz replan)   │
│ 60 fps anim) │   │ Control: bicycle model + pure-pursuit │ Metrics logger            │
└──────────────┘   └──────────────────────────────────────────────────────────────────┘
        WebSocket (state @20Hz) / REST (scenario load, metrics pull)
```

**Why this split:** Python keeps your RL + CV weights; Three.js gives the "wow" 3D demo judges film. Viz is dumb (renders what backend sends) so it can't crash the planner.

### Module contracts (freeze these early)

- `POST /scenario/load {id}` → loads one of 5 configs, resets world, returns seed.
- `WS /state` @20Hz: `{t, ego:{x,y,theta,v,action,path:[...],candidates:[...]}, agents:[{id,cls,x,y,theta,v}], bubbles:{...}, metrics:{...}}`
- `GET /metrics` → `{scenario_id, collisions, min_ttc, replan_ms_p50/p95, jerk_rms, completion, success}`

---

## 4. Technical implementation (module by module)

### 4.1 Sim v2 — scenario-driven world (2 days)
- Refactor `IndianRoadSimRL` into `world/` with `Scenario` dataclass (YAML): road geometry fn, spawn tables, trigger events (e.g. `t=6s: child from parked car`), success/fail predicates.
- Replace magic counts (320/200/150) with per-scenario density budgets. Keep SAT collision (`check_oriented_box_collision`) — it's correct, keep it.
- Fixed-step loop: physics 50Hz, planner 20Hz, viz 60fps interpolation. Seed everything (`seed` in scenario file) so judges get identical runs.
- Files: `sim/scenarios/*.yaml` (5 files), `sim/world.py`, `sim/npc.py`, `sim/geometry.py`.

### 4.2 Perception bridge (2–3 days, biggest credibility lift)
- Export Colab models to ONNX: `yolov8n` (already Ultralytics → `yolo export format=onnx`) + U-Net (TF → ONNX via `tf2onnx`). Run with `onnxruntime` (CPU, ~30–60ms).
- Input switch: (a) IDD image folder replay, (b) live sim top-down render as pseudo-camera, (c) webcam. Output: BEV occupancy grid 128×128 @10Hz + tracked boxes.
- Fallback: if ONNX missing/slow, use sim ground-truth stub flagged `perception_source: stub` (honest in report). Never let perception crash the run.
- Files: `perception/infer.py`, `perception/bev.py`, `perception/tracker.py` (port Kalman from notebook).

### 4.3 Prediction (1 day)
- Baseline everyone trusts: constant-velocity + Kalman (already in notebook — port it, done).
- Keep DQN trajectory head as "learned residual" for the report's novelty section. Fuse: `pred = CV + λ·DQN_residual`. If DQN hurts metrics, set λ=0 and still mention it.
- Output: 8 tracked agents × 5 steps @1s horizon, same 80-float contract the DQN already uses.

### 4.4 Planning — Frenet lattice + bubble filter (2 days, core IP)
Current bubbles only brake/swerve. Add a real planner judges can see:
- Reference path = road centerline (`get_road_center`, extended per scenario).
- Sample lattice: 3 lateral offsets × 4 speeds × 5s horizon → ~12 candidates per replan.
- Cost: `w1·collision + w2·jerk + w3·lane_dev + w4·speed_err + w5·bubble_violation`. Collision via SAT rollout against predictions.
- Safety filter: bubble check vetoes (keep `choose_bot_action` exactly as final gate — it's your safety story).
- Publish all candidates to Three.js (green chosen / red rejected). Measure `replan_ms` with `perf_counter` around this function.
- Files: `planning/lattice.py`, `planning/costs.py`, `planning/safety.py`.

### 4.5 Control (0.5 day)
- Kinematic bicycle model + pure-pursuit on chosen path. Keep existing `step_with_inputs` as fallback. Clamp accel/jerk for smoothness metric.

### 4.6 Metrics + eval harness (1 day, wins the report)
Log every run to `runs/<scenario>_<seed>.json` + CSV:
- `collisions` (must be 0), `min_ttc_s`, `replan_ms_p50/p95` (target p95 <50ms on laptop CPU), `jerk_rms`, `lateral_accel_max`, `completion_%` (distance / route length), `success` (reached goal, 0 collisions, lane discipline).
- `eval/run_all.py`: loops 5 scenarios × 3 seeds = 15 runs, prints markdown table for report + video captions.

### 4.7 Frontend — Three.js demo (2 days, the "wow")
Scaffold already started in `web/` (this PRD's companion commit). Requirements:
- 3D road with lane-less texture, 5 vehicle types (car, auto, truck, bike, ego) + pedestrian (walk-cycle) + cow + vendor cart + pothole decals.
- Animations: wheel spin, brake-light glow, indicator blink, pedestrian limb swing, cow head-bob, dust particles, camera follow + shake on near-miss, bubble translucent cones.
- Scenario picker (1–5 buttons), slow-mo toggle, HUD (speed, action tag, replan ms, TTC), auto-camera for recording.
- Must run from `file://` or `python -m http.server` with mock data if backend offline (judges click it on any laptop).

---

## 5. The 5 scenarios (build exactly these — matches statement + your Colab)

| # | Name (statement mapping) | Scene | Trigger event | Success |
|---|---|---|---|---|
| S1 | Unmarked village road (no lanes) | Narrow 2-way, faded center, cows, parked tractor | t=8s cow steps in from left @15m | Pass without stop>3s, no collision |
| S2 | Busy urban intersection, no signals | 4-way, idling autos, erratic cross traffic | 2 autos + bike enter without ROW @t=5,7s | Creep + yield, clear in <30s |
| S3 | Highway merge, slow vehicles | 2-lane + merge ramp, truck @30kmph | Slow truck merges @t=6s, tailgater behind | Overtake left w/ indicator, keep TTC>2s |
| S4 | Dense market / shared space | Stalls both sides, vendors, pedestrians | Cart pulls out @t=10s + 3 peds cross | Horn + crawl <10kmph, 0 contacts |
| S5 | Sudden cattle crossing (emergency) | Open road, 3 cows roadside | Cow sprints across @t=7s, 20m ahead | Emergency brake, stop gap >2m |

Each scenario = 1 YAML + 1 seeded traffic script + 1 success predicate. Demo order: S5 (drama first) → S1 → S3 → S2 → S4.

---

## 6. Assets & animation spec (what "polish" means concretely)

Pygame primitives read as "student prototype". Three.js target look: low-poly dusk India street, warm fog, emissive headlights.
- **Vehicles:** box-body + cabin + 4 cylinder wheels (spin by `v/r`), emissive front (white) / rear (red, intensity = brake), side indicator (amber blink 2Hz when lane-change intent). Auto-rickshaw: yellow box + black canopy + 3 wheels. Truck/bus: long box. Bike: 2 wheels + rider capsule.
- **Pedestrians:** capsule body + swinging arms/legs (`sin(t*speed)`), random shirt colors, crossingщую arm-raise variant.
- **Animals:** cow = brown box + head box (bob) + legs (trot when crossing); dog = small variant.
- **Road:** asphalt with noise texture (canvas-generated, no external files), drivable edge glow from U-Net mask projection, pothole dark decals, vendor stalls (striped awnings), horn ripple ring effect.
- **FX:** dust particles behind wheels, brake glow sprite, camera shake (near-miss <1s TTC), slow-mo (0.25×) on emergency brake, action-tag toast (`YIELD_TO_PEDESTRIAN` pops on HUD — reuses Colab taxonomy, now visible).
- **Rule: every dynamic state must animate.** No teleporting — lerp positions @60fps; every action tag has a visual (brake=red glow, horn=ring, indicator=blink, swerve=tilt).

---

## 7. Milestones (3-week sprint to SIH)

| Week | Deliverable | Done when |
|---|---|---|
| W1 | Sim v2 + 5 YAML scenarios + metrics logger | `eval/run_all.py` runs 15 seeded runs, prints table, 0 crashes |
| W1–W2 | Perception ONNX bridge + BEV grid feeding planner | Demo runs with `perception_source: onnx` on at least S1+S5 |
| W2 | Lattice planner + Three.js viz + HUD | Candidates visible, replan p95 <50ms, viz runs standalone with mock |
| W3 | 15-run benchmark + report + 3-min video | Report PDF (approach, architecture, table, ablations), video (5 scenarios + metrics overlay), rehearsed live run |

Daily demo rule: `main.py --scenario S5 --seed 7` must run end-to-end every night. If nightly breaks, stop features, fix.

---

## 8. Risks & mitigations

- **Perception too slow on judge laptop** → default to stub + show ONNX overlay as PiP (picture-in-picture) so story survives at 5fps; planner never blocks on perception (uses last grid + timeout flag).
- **DQN misbehaves live** → safety filter + BC policy fallback; have `--policy=safe_heuristic` flag for stage.
- **Three.js won't load offline** → vendor `three.min.js` locally in `web/vendor/`; no CDN dependency on stage.
- **MATLAB question from judges** ("statement says MATLAB") → answer: "Statement encourages MathWorks tools; rubric rewards closed-loop validation + metrics, which our Python stack provides with identical semantics — bicycle model, sensor fusion, Stateflow-equivalent decision logic in `choose_bot_action`. Port path to Simulink documented in report §7." Have that paragraph pre-written.

---

## 9. File map (target repo after rebuild)

```
PRD.md                  ← this file
sim/world.py  sim/npc.py  sim/scenarios/S1..S5.yaml
perception/infer.py  perception/bev.py  perception/tracker.py
planning/lattice.py  planning/costs.py  planning/safety.py
eval/run_all.py  eval/metrics.py        runs/*.json
web/index.html  web/app.js  web/styles.css  web/mock.js  web/vendor/
assets/  report/  video/
IRS_dum (2) (1).py      ← legacy, frozen; do not extend
Copy_of_Untitled (1).ipynb ← legacy perception ref; export weights only
```

Build order: scenarios → metrics → planner → perception bridge → web viz → benchmark → report/video.
