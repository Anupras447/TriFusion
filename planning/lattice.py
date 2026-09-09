"""Frenet lattice planner (PRD §4.4). Samples lateral×speed candidates, costs them,
publishes all candidates for the Three.js viz (green=chosen, red=rejected)."""
import time
import numpy as np

from sim.geometry import check_oriented_box_collision

LAT_OFFSETS = [-2.5, 0.0, 2.5]
TARGET_SPEEDS = [3.0, 7.0, 11.0]
HORIZON = 2.5  # s
DT = 0.25
STEPS = 10
# cheap prefilter box (m): skip SAT when clearly far
PREFILTER_DX = 3.5
PREFILTER_DY = 7.0


def rollout(x0, y0, th0, v0, lat_target, v_target, road_center_fn, steps=12):
    pts = []
    x, y, th, v = x0, y0, th0, v0
    for _ in range(steps):
        v += np.clip(v_target - v, -4 * DT, 2 * DT)
        rc = road_center_fn(y + 5)
        steer = np.clip((rc + lat_target - x) * 0.15 - th, -0.5, 0.5)
        th += steer * DT * 2
        x += np.sin(th) * v * DT
        y += np.cos(th) * v * DT
        pts.append((x, y, th, v))
    return pts


def plan(world, predictions):
    """Returns dict(chosen, candidates, replan_ms, action_tag). predictions: list of
    {x,y,vx,vy,w,l} future-agnostic current states — CV rollout done internally."""
    t0 = time.perf_counter()
    e = world.ego
    rc0 = world.road_center(e["y"])
    cands = []
    # prefilter predictions once: only agents that could matter this horizon
    near = [p for p in predictions
            if -8 < p["y"] - e["y"] < 45 and abs(p["x"] - e["x"]) < 8]
    for lat in LAT_OFFSETS:
        for vt in TARGET_SPEEDS:
            # lat = absolute lateral offset from road center: want x = rc(y) + lat
            pts = rollout(e["x"], e["y"], e["theta"], e["v"], lat, vt, world.road_center,
                          steps=STEPS)
            cost, collides = cost_of(pts, near, world, vt)
            cands.append({"lat": lat, "speed": vt, "path": [(p[0], p[1]) for p in pts],
                          "cost": cost, "collides": collides})
    cands.sort(key=lambda c: c["cost"])
    chosen = cands[0]
    ms = (time.perf_counter() - t0) * 1000
    tag = ("EMERGENCY_BRAKE" if world.bubble_profile()["b2"]
           else "DECREASE_SPEED" if world.bubble_profile()["b1"]
           else "OVERTAKE_LEFT" if chosen["lat"] < -1 else
           "OVERTAKE_RIGHT" if chosen["lat"] > 1 else "MAINTAIN_LANE")
    return {"chosen": chosen, "candidates": cands, "replan_ms": ms, "action_tag": tag}


def cost_of(pts, predictions, world, v_target):
    W_COL, W_JERK, W_LANE, W_SPEED = 1000.0, 0.5, 2.0, 1.0
    cost, collides = 0.0, False
    prev_th = world.ego["theta"]
    for i, (x, y, th, v) in enumerate(pts):
        rc = world.road_center(y)
        cost += W_LANE * abs(x - rc) * 0.1 + W_SPEED * abs(v - v_target) * 0.1
        cost += W_JERK * abs(th - prev_th)
        prev_th = th
        t = (i + 1) * DT
        for p in predictions:
            px = p["x"] + p.get("vx", 0) * t
            py = p["y"] + p.get("vy", 0) * t
            dx, dy = px - x, py - y
            if abs(dx) > PREFILTER_DX or abs(dy) > PREFILTER_DY:
                continue  # cheap reject before SAT
            if check_oriented_box_collision((x, y), (1.7, 4.2), np.degrees(th),
                                           (px, py), (p.get("w", 1.8), p.get("l", 4.2))):
                cost += W_COL
                collides = True
                break
    return cost, collides
