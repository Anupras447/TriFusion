# SIH — Adaptive Path Planning on Unstructured Indian Roads (sim v2)

Python stack, zero MATLAB dependency. Status: **15/15 eval runs pass, 0 collisions, replan p95 < 50 ms.**

## Quickstart

```bash
pip install -r requirements.txt
python eval/run_all.py --seeds 3     # full benchmark → markdown table + runs/*.json
python main.py --scenario S5 --seed 7 --headless   # one run, JSON to stdout
python main.py --scenario S5 --seed 7              # pygame debug viz (needs screen)
```

## 3D demo (judges)

```bash
cd web && python -m http.server 8000
# open http://localhost:8000 — scenario buttons S1–S5, slow-mo, chase/top camera
```

Works offline except the Three.js CDN import (vendor it into `web/vendor/` before stage).

## Repo map

| Path | What |
|---|---|
| `PRD.md` | win plan + architecture + MATLAB mapping |
| `AGENTS.md` | agent inventory (sim + perception) |
| `sim/world.py`, `sim/geometry.py` | deterministic world, SAT collisions, bicycle ego |
| `sim/scenarios/S1–S5.yaml` | the 5 validation scenarios (seeded) |
| `planning/lattice.py`, `planning/safety.py` | Frenet lattice + bubble safety veto |
| `perception/infer.py`, `perception/tracker.py` | stub (flagged) + Kalman tracker, ONNX-ready |
| `eval/run_all.py`, `eval/metrics.py` | 5×N harness, latency/smoothness/TTC table |
| `web/` | Three.js demo (mock default, WS live optional) |
| `IRS_dum (2) (1).py` | legacy DQN sim — frozen, do not extend |
| `Copy_of_Untitled (1).ipynb` | legacy perception ref — export weights only |

## Demo script (3 min)

S5 (cow emergency, drama first) → S1 (village) → S3 (merge overtake) → S2 (intersection) → S4 (market crawl). Seeds fixed per scenario; nightly rule: `main.py --scenario S5 --seed 7` must run green.
