# Agents

## Overview

This project implements adaptive path planning and collision avoidance for autonomous
vehicles on unstructured Indian roads (SIH problem statement). The current stack is
Python-based (pygame simulation + PyTorch DQN + TensorFlow/Ultralytics perception in
`Copy_of_Untitled (1).ipynb`), not MATLAB/Simulink + RoadRunner.

Two subsystems exist side by side and are not yet wired end-to-end:

1. `IRS_dum (2) (1).py` — pygame closed-loop simulator + RL planner + safety bubbles.
2. `Copy_of_Untitled (1).ipynb` — real-image perception pipeline trained/tested on IDD.

## A. Closed-Loop Simulator + Planner (`IRS_dum (2) (1).py`)

### 1. RL Decision Agent (`MultiAgentPredictiveDQN`)

- **Type**: Dueling Double DQN with Prioritized Experience Replay.
- **Observation**: 35-dim vector (ego lateral offset, heading, velocity + 4 nearest
  forward vehicles x4, 2 nearest oncoming x4, 2 nearest pedestrians x4).
- **Actions (6)**: brake, steer-left, accelerate, decelerate, steer-right, cruise.
- **Architecture**: Input 35→256, 2x ResBlock (LayerNorm + SiLU), value stream
  256→128→1, advantage stream 256→128→6, trajectory head 256→128→80.
- **Trajectory head**: predicts 8 entities (4 forward + 2 oncoming + 2 pedestrians)
  x 5 steps x 2 coords = 80 values; trained with MSE alongside Huber Q-loss.
- **Training**: Double-DQN targets, gradient clipping, soft target updates,
  Behavioral Cloning pre-training (CrossEntropy) on human CSV data, replay capacity
  120,000, bubble-rate adaptation from batch stats and collisions.

### 2. Adaptive Safety Bubble System

| Bubble | Zone | Trigger | Response |
|--------|------|---------|----------|
| Bubble 1 (Blue) | Outer awareness | Moderate proximity | Gentle decel + `find_clear_bubble_angle` reroute |
| Bubble 2 (Red) | Inner critical | Close proximity | Emergency brake + reroute |
| Rear bubble | Behind ego | Approaching rear vehicle | Veto lane-change actions |

- Sizes scale with ego speed and nearby density; SAT-based
  `check_oriented_box_collision` is used for all checks.
- Learned params (`forward_rate`, `rear_rate`, `contraction_rate`) persist to
  `learned_bubble_config.json`.

### 3. NPC / Environment Agents

| Agent | Count | Behavior |
|-------|-------|----------|
| Forward traffic | 320 | Lane-follow, signal obedience, overtake, lane change |
| Oncoming vehicles | 200 | Opposite-lane driving, lateral safety checks |
| Rear vehicles | 150 | Follow + overtake logic |
| Pedestrians | 80 | Crossing / walking / wandering, spawn-recycle |
| Animals (cow/dog) | 25 | Static hazards |
| Double-parked vehicles | 22 | Static obstacles |
| Hazards (pothole/speed-breaker) | 45 | Speed penalties |
| Traffic signals | 6 | State cycling, gates ego + NPC motion |
| Merge zones + curved segments | — | Sinusoidal `get_road_center`, funnel logic |

### 4. Decision Override (`choose_bot_action`)

```
if Bubble 2: emergency brake + reroute
elif Bubble 1: decelerate + reroute
elif Rear bubble + lane-change action: veto
else: DQN action
apply lane re-centering
```

### 5. Human-in-the-Loop Rollback

- Bot collision → rewind to state from ~3 s prior → forced human control (TAB toggles
  bot/human, P pause, ENTER restart, arrows + space drive) until past collision point.
- Every step logs telemetry (`Y_Pos,Heading,Vx,Front_Dist,Left_Dist,Right_Dist,Action`)
  to `IndianRoadSim_Logs/clean_driving_dataset.csv` (currently header-only), flushed
  every 100 rows for replay/BC retraining.

## B. Real-Image Perception Pipeline (`Copy_of_Untitled (1).ipynb`)

Dataset: IDD `idd20kII` (~5.9 GB tar) — ~10,098 image↔polygon pairs indexed,
~8,583 after strict `_leftImg8bit` filtering, split 6,866 train / 1,717 val, plus
optional custom-video frame extraction path (`real_world_dataset/`, 256x256 @ 2 fps).
YOLO conversion keeps 12 classes (person, rider, motorcycle, bicycle, autorickshaw,
car, truck, bus, traffic sign, traffic light, pole, billboard) with `idd_yolo.yaml`.

### 6. Road Segmentation Agent (U-Net)

- Deep U-Net 64→128→256→512 + dropout, BCE+Dice loss, 256x256 input.
- Binary road/drivable mask, threshold 0.5/0.6 + morphological close; weights at
  `unet_road_segmentation.weights.h5` (92.7 MB, Drive-loaded, train flag off by default).

### 7. Object / Pose / Segmentation Agents (YOLO + SAM)

- `yolov8n.pt` (general detector), `yolov8x.pt` low-conf (0.10) distant
  vehicle/light detector, `yolov8x-pose.pt` (conf 0.05–0.15) with convex-hull fallback
  mask from keypoints, `mobile_sam.pt` mask refinement from YOLO boxes.
- Helpers: `detect_indicators` (amber HSV hotspot inside vehicle mask),
  `detect_traffic_light_color` (red/green/yellow HSV ratio), per-class coloring
  (person blue, vehicle green, indicator orange).

### 8. Tri-Model Fusion Agent (`adaptive_model`)

- 9-channel input: RGB + U-Net road + YOLO+SAM vehicle + YOLO-Pose person + 3x traffic
  light channels + indicator + animal mask → 1 conv block → 5-class softmax
  (0 background, 1 road, 2 vehicle, 3 person, 4 animal).
- Weights at `fusion_adaptive_model.weights.h5` (389 MB); 1-epoch / 2000-image train
  mode exists but default run is load + test. Heavy-augment variant (imgaug dust /
  motion-blur / rain / elastic) defined but not executed.

### 9. BEV + Tracking Agents

- `transform_to_bev`: perspective warp tuned for vertical parallelism, object-aware
  source rect.
- `KalmanFilter` (constant-velocity, 4-state) + `Track` / `ObjectTracker`
  (centroid match <60 px, max_age 10, min_hits 2) over BEV mask contours; outputs
  relative distance (`PIXELS_PER_METER_BEV=10`), speed (x30 fps), size table and
  VEH/PED/ANML overlays. Single-frame demo only — no multi-frame closed loop yet.

## C. Behavior Taxonomy (Colab markdown, not code)

~27 action tags (e.g. `EMERGENCY_BRAKE`, `YIELD_TO_PEDESTRIAN`, `YIELD_TO_ANIMAL`,
`ADAPT_TO_NO_LANES`, `EXPECT_WRONG_WAY_DRIVER`, `NEGOTIATE_SHARED_SPACE`,
`REACT_TO_POTHOLE`, `NAVIGATE_CROWDED_STREET_WITH_VENDORS`, `SOUND_HORN_WARNING`)
mapped to 4 demo scenarios (child from parked cars, aggressive rear motorcycle,
unmarked auto-rickshaw intersection, vendor-cart alley) + 5 rules (divert, sudden
stop+horn, overtake protocol, yield rear, yield lateral). No state-machine /
planner consumes these tags yet.

## Data Flow (as built)

```
IDD tar / custom video → pairs → U-Net + YOLO/Pose + SAM → 9-channel geometry
    → fusion seg → BEV → Kalman tracker → viz tables (open-loop, Colab only)

pygame sim → synthetic sensors + 35-dim obs → DQN + bubbles → kinematic step
    → CSV telemetry → replay buffer → DQN/BC retrain → checkpoint + bubble JSON
```

Perception ↔ planner link is missing: the simulator uses synthetic
`get_sensor_distances` / ground-truth states, never the U-Net/YOLO/fusion outputs.
