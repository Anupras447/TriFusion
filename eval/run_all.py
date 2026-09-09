"""Nightly harness: 5 scenarios x N seeds, headless. Prints markdown table for report/video."""
import argparse
import glob
import json
import os
import yaml

from sim.world import ScenarioWorld
from planning.safety import safe_action
from perception.infer import infer
from perception.tracker import Tracker
from eval.metrics import summarize, markdown_table

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys as _sys
if BASE not in _sys.path:
    _sys.path.insert(0, BASE)


def run_one(scenario_path, seed):
    with open(scenario_path) as f:
        cfg = yaml.safe_load(f)
    world = ScenarioWorld(cfg, seed=seed)
    tracker = Tracker()
    replan_ms, done, info = [], False, {}
    while not done:
        dets, _src = infer(world)
        preds = tracker.update(dets)
        act = safe_action(world, preds)
        replan_ms.append(act["replan_ms"])
        done, info = world.step(steer=act["steer"], accel=act["accel"], action_tag=act["action_tag"])
        if world.step_count >= world.max_steps:
            break
    completion = max(0.0, min(1.0, world.ego["y"] / world.goal_y)) * 100
    success = (not world.collision) and world.ego["y"] >= world.goal_y
    log = {"scenario": str(cfg["id"]), "seed": int(seed),
           "collisions": int(1 if world.collision else 0),
           "success": bool(success), "completion_pct": float(completion),
           "min_ttc": float(world.min_ttc if world.min_ttc != float("inf") else 99.9),
           "replan_ms": [float(x) for x in replan_ms],
           "jerk_rms": float((world.jerk_acc / max(1, world.jerk_n)) ** 0.5),
           "max_lat_accel": float(world.max_lat_accel), "steps": int(world.step_count)}
    return summarize(log), log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", default=os.path.join(BASE, "runs"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    paths = sorted(glob.glob(os.path.join(BASE, "sim", "scenarios", "*.yaml")))
    rows = []
    for p in paths:
        for s in range(args.seeds):
            row, log = run_one(p, seed=s)
            rows.append(row)
            with open(os.path.join(args.out, f"{row['scenario']}_seed{s}.json"), "w") as f:
                json.dump(log, f, indent=2)
    print(markdown_table(rows))
    ok = sum(1 for r in rows if r["success"])
    print(f"\n{ok}/{len(rows)} runs successful, "
          f"collisions={sum(r['collisions'] for r in rows)}")


if __name__ == "__main__":
    main()
