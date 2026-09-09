"""Scenario-driven world loop (PRD §4.1). Deterministic, headless, numpy-only.

Entity dicts use plain python types so the WS bridge / web viz can JSON-encode them.
Units: meters, seconds. Ego moves +y.(dt=0.05 default → 20Hz planner-ready).
"""
import copy
import random
import numpy as np

from .geometry import check_oriented_box_collision, road_center_sine, ttc

DIMENSIONS = {  # w, l per kind
    "car": (1.8, 4.2), "auto": (1.4, 3.0), "truck": (2.4, 8.0),
    "bus": (2.5, 9.0), "bike": (0.7, 2.0), "ped": (0.6, 0.6),
    "cow": (1.2, 2.0), "cart": (1.6, 2.4), "stall": (2.0, 2.0),
    "parked": (1.8, 4.2), "tractor": (2.0, 4.5), "ego": (1.7, 4.2),
}


class ScenarioWorld:
    def __init__(self, cfg, seed=7):
        self.cfg = copy.deepcopy(cfg)
        self.seed = seed
        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(seed)
        self.dt = 0.05
        self.t = 0.0
        self.step_count = 0
        self.half_w = float(cfg.get("road_half_width", 6.5))
        self.segments = [tuple(s) for s in cfg.get("segments", [])]
        self.road_length = float(cfg.get("road_length", 220.0))
        self.goal_y = float(cfg.get("goal_y", 200.0))
        self.max_steps = int(cfg.get("max_steps", 1200))
        self.events = sorted(copy.deepcopy(cfg.get("events", [])), key=lambda e: e["t"])
        self.event_idx = 0
        ego_start = cfg.get("ego_start", {})
        c0 = self.road_center(0.0)
        self.ego = {"x": c0 + float(ego_start.get("lateral", 1.5)),
                    "y": 0.0, "theta": 0.0, "v": float(ego_start.get("speed", 6.0))}
        self.prev_accel = 0.0
        self.vehicles, self.peds, self.animals, self.statics = [], [], [], []
        self._spawn_traffic(cfg.get("traffic", {}))
        for a in cfg.get("animals", []):
            self.animals.append({"x": self.road_center(a["y"]) + a["lateral"], "y": float(a["y"]),
                                 "vx": 0.0, "vy": 0.0, "kind": a.get("kind", "cow"),
                                 "behavior": a.get("behavior", "idle")})
        for s in cfg.get("static_obstacles", []):
            self.statics.append({"x": self.road_center(s["y"]) + s["lateral"], "y": float(s["y"]),
                                 "w": float(s.get("w", 2.0)), "l": float(s.get("l", 4.2)),
                                 "kind": s.get("kind", "parked")})
        self.collision = False
        self.min_ttc = float("inf")
        self.jerk_acc = 0.0
        self.jerk_n = 0
        self.max_lat_accel = 0.0
        self.action_tag = "MAINTAIN_LANE"

    # ---------- helpers ----------
    def road_center(self, y):
        return float(road_center_sine(y, self.segments))

    def _spawn_traffic(self, tr):
        def lat():
            return self.rng.uniform(-self.half_w + 1.0, self.half_w - 1.0)
        ex0 = self.ego["x"]
        for _ in range(int(tr.get("forward", 6))):
            y = self.rng.uniform(15, self.road_length)
            x = self.road_center(y) + lat()
            if y < 30 and abs(x - ex0) < 3.0:  # spawn exclusion: never block ego at reset
                y = 30 + self.rng.uniform(0, 20)
                x = self.road_center(y) + lat()
            kind = self.np_rng.choice(["car", "auto", "truck", "bike"], p=[0.4, 0.3, 0.1, 0.2])
            self.vehicles.append({"x": x, "y": y,
                                  "v": float(self.rng.uniform(4, 8)), "dir": 1, "kind": str(kind)})
        for _ in range(int(tr.get("oncoming", 2))):
            y = self.rng.uniform(30, self.road_length)
            self.vehicles.append({"x": self.road_center(y) + self.rng.uniform(-self.half_w + 0.8, -1.5),
                                  "y": y, "v": float(self.rng.uniform(4, 7)), "dir": -1, "kind": "car"})
        for _ in range(int(tr.get("rear", 2))):
            self.vehicles.append({"x": self.road_center(-30) + self.rng.uniform(0.5, 3.5),
                                  "y": float(self.rng.uniform(-55, -25)),
                                  "v": float(self.rng.uniform(5, 8)), "dir": 1, "kind": "car"})
        for _ in range(int(tr.get("pedestrians", 4))):
            y = self.rng.uniform(10, self.road_length)
            x = self.road_center(y) + lat()
            if y < 20 and abs(x - ex0) < 2.5:
                y = 20 + self.rng.uniform(0, 15)
                x = self.road_center(y) + lat()
            self.peds.append({"x": x, "y": y,
                              "vx": 0.0, "vy": 0.0, "kind": "ped",
                              "mode": "wander", "phase": self.rng.uniform(0, 6.28)})

    def _fire_events(self):
        while self.event_idx < len(self.events) and self.t >= self.events[self.event_idx]["t"]:
            e = self.events[self.event_idx]
            self.event_idx += 1
            typ = e["type"]
            if typ in ("cow_cross", "ped_cross", "cross_traffic", "cart_pullout"):
                kind = {"cow_cross": "cow", "ped_cross": "ped",
                        "cross_traffic": e.get("kind", "auto"), "cart_pullout": "cart"}[typ]
                self.animals.append({"x": self.road_center(e["y"]) + e["from_lateral"], "y": float(e["y"]),
                                     "vx": float(np.sign(e["to_lateral"] - e["from_lateral"]) * e.get("speed", 1.2)),
                                     "vy": 0.0, "kind": kind, "behavior": "cross",
                                     "target_x": self.road_center(e["y"]) + e["to_lateral"]})
            elif typ == "merge_vehicle":
                self.vehicles.append({"x": self.road_center(e["y"]) + e["lateral"], "y": float(e["y"]),
                                      "v": float(e.get("speed", 8.0)), "dir": 1,
                                      "kind": e.get("kind", "truck"),
                                      "merge_target": e.get("target_lateral", 2.5)})
            elif typ == "tailgater":
                self.vehicles.append({"x": self.road_center(e["y"]) + e["lateral"], "y": float(e["y"]),
                                      "v": float(e.get("speed", 14.0)), "dir": 1, "kind": "car"})

    # ---------- perception-ish outputs ----------
    def sensor_distances(self):
        ey, ex = self.ego["y"], self.ego["x"]
        front, left, right = 60.0, 15.0, 15.0
        ents = ([(v["y"], v["x"]) for v in self.vehicles] + [(p["y"], p["x"]) for p in self.peds] +
                [(a["y"], a["x"]) for a in self.animals] + [(s["y"], s["x"]) for s in self.statics])
        for ay, ax in ents:
            dy, dx = ay - ey, ax - ex
            if 0 < dy < front and abs(dx) < 1.8:
                front = dy
            if -10 < dy < 10:
                if -left < dx < 0:
                    left = abs(dx)
                if 0 < dx < right:
                    right = dx
        return round(front, 2), round(left, 2), round(right, 2)

    def bubble_profile(self):
        v = max(0.0, self.ego["v"])
        b1_dist = float(np.exp(0.3 * v) - 1.0 + 5.0)
        b2_dist = float(2.0 * v + 1.25)
        b1 = b2 = False
        ey, ex = self.ego["y"], self.ego["x"]
        cands = ([(v_["y"], v_["x"], 2.0) for v_ in self.vehicles] +
                 [(p["y"], p["x"], 0.3) for p in self.peds] +
                 [(a["y"], a["x"], 0.8) for a in self.animals] +
                 [(s["y"], s["x"], s["l"] / 2) for s in self.statics])
        for ay, ax, hl in cands:
            ry, lat = ay - ey, abs(ax - ex)
            if 0 < ry <= b2_dist + hl and lat <= 1.1:
                b1 = b2 = True
                break
            elif 0 < ry <= b1_dist + hl and lat <= 1.1:
                b1 = True
        return {"b1": b1, "b2": b2, "b1_dist": b1_dist, "b2_dist": b2_dist}

    # ---------- main step ----------
    def step(self, steer=0.0, accel=0.0, action_tag="MAINTAIN_LANE"):
        """Bicycle-lite ego update + NPC update + collisions. Returns (done, info)."""
        self._fire_events()
        e = self.ego
        accel = float(np.clip(accel, -6.0, 3.0))
        jerk = (accel - self.prev_accel) / self.dt
        self.jerk_acc += jerk ** 2
        self.jerk_n += 1
        self.prev_accel = accel
        e["v"] = float(np.clip(e["v"] + accel * self.dt, 0.0, 18.0))
        e["theta"] = float(np.clip(e["theta"] + steer * self.dt * (1 + e["v"] * 0.05), -0.5, 0.5))
        self.max_lat_accel = max(self.max_lat_accel, abs(steer) * max(1.0, e["v"]))
        e["y"] += np.cos(e["theta"]) * e["v"] * self.dt
        e["x"] += np.sin(e["theta"]) * e["v"] * self.dt
        # soft road-edge clamp (rumble, not wall — lets metrics punish lane_dev instead)
        rc = self.road_center(e["y"])
        e["x"] = float(np.clip(e["x"], rc - self.half_w - 1.0, rc + self.half_w + 1.0))
        self.action_tag = action_tag
        self._update_npc()
        self.t += self.dt
        self.step_count += 1
        # collisions (SAT, ego vs everything)
        ew, el = DIMENSIONS["ego"]
        self.collision_kind = None
        for v in self.vehicles:
            w, l = DIMENSIONS.get(v["kind"], (1.8, 4.2))
            if check_oriented_box_collision((e["x"], e["y"]), (ew, el), np.degrees(e["theta"]),
                                           (v["x"], v["y"]), (w, l)):
                self.collision = True
                self.collision_kind = v["kind"]
        for group, dims in ((self.peds, (0.6, 0.6)), (self.animals, (1.2, 2.0))):
            for o in group:
                if check_oriented_box_collision((e["x"], e["y"]), (ew, el), np.degrees(e["theta"]),
                                               (o["x"], o["y"]), dims):
                    self.collision = True
                    self.collision_kind = o["kind"]
        for s in self.statics:
            if check_oriented_box_collision((e["x"], e["y"]), (ew, el), np.degrees(e["theta"]),
                                           (s["x"], s["y"]), (s["w"], s["l"])):
                self.collision = True
                self.collision_kind = s.get("kind", "static")
        # min TTC vs ahead agents (bumper gap; ignore lateral passes)
        for v in self.vehicles:
            if v["dir"] == 1 and abs(v["x"] - e["x"]) < 1.5:
                other_l = DIMENSIONS.get(v["kind"], (1.8, 4.2))[1]
                bumper = (v["y"] - e["y"]) - (el / 2 + other_l / 2)
                if bumper > 0.5:
                    rel = e["v"] - v["v"]
                    if rel > 1e-6:
                        self.min_ttc = min(self.min_ttc, bumper / rel)
        done = self.collision or e["y"] >= self.goal_y or self.step_count >= self.max_steps
        return done, {"t": self.t, "collision": self.collision}

    def _update_npc(self):
        e = self.ego
        for v in self.vehicles:
            # car-following: slow if leader ahead close (vehicles AND ego)
            ahead = [o["v"] for o in self.vehicles if o is not v and 0 < o["y"] - v["y"] < 10
                     and abs(o["x"] - v["x"]) < 2.0 and o["dir"] == v["dir"]]
            if v["dir"] == 1 and 0 < e["y"] - v["y"] < 10 and abs(e["x"] - v["x"]) < 2.0:
                ahead.append(e["v"])  # NPCs brake for ego (no more rear-ending a stopped ego)
            if v["dir"] == -1 and 0 < v["y"] - e["y"] < 12 and abs(e["x"] - v["x"]) < 2.2:
                ahead.append(0.0)  # oncoming halts rather than ramming ego head-on
            target = v["v"]
            if ahead:
                target = min(target, min(ahead) * 0.9)
            if v.get("merge_target") is not None:
                rc = self.road_center(v["y"])
                want = rc + v["merge_target"]
                v["x"] += np.clip(want - v["x"], -1.5 * self.dt, 1.5 * self.dt)
            v["v"] = target
            v["y"] += v["dir"] * v["v"] * self.dt
        for p in self.peds:
            p["phase"] += self.dt * 2
            p["x"] += np.sin(p["phase"]) * 0.15 * self.dt
            p["y"] += 0.9 * self.dt  # drift ahead so crowds clear instead of blocking forever
        for a in self.animals:
            a["x"] += a.get("vx", 0.0) * self.dt
            a["y"] += a.get("vy", 0.0) * self.dt
            if a.get("behavior") == "cross" and abs(a["x"] - a.get("target_x", a["x"])) < 0.3:
                a["vx"] = 0.0

    def snapshot(self):
        """JSON-safe state for WS bridge / web viz."""
        f, l, r = self.sensor_distances()
        b = self.bubble_profile()
        return {"t": round(self.t, 2), "ego": {**self.ego, "action_tag": self.action_tag},
                "vehicles": self.vehicles, "peds": self.peds, "animals": self.animals,
                "statics": self.statics, "sensors": {"front": f, "left": l, "right": r},
                "bubbles": b, "collision": self.collision}
