"""Constant-velocity Kalman tracker ported from the Colab (4-state x,y,vx,vy)."""
import numpy as np


class KF:
    def __init__(self, dt=0.05):
        # dt = sim step (20Hz): velocities stay in m/s so planner predictions are correct
        self.x = np.zeros((4, 1))
        self.P = np.eye(4) * 1000.
        self.A = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], float)
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], float)
        self.Q = np.eye(4) * 0.1
        self.R = np.eye(2) * 1.0

    def predict(self):
        self.x = self.A @ self.x
        self.P = self.A @ self.P @ self.A.T + self.Q
        return self.x

    def update(self, z):
        z = np.asarray(z).reshape(-1, 1)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = self.P - K @ self.H @ self.P


class Tracker:
    def __init__(self, max_age=10, dist=6.0):
        self.tracks = {}
        self.next_id = 0
        self.max_age = max_age
        self.dist = dist

    def update(self, dets):
        for tr in self.tracks.values():
            tr["kf"].predict()
            tr["age"] += 1
        used = set()
        for d in dets:
            best, bd = -1, self.dist
            z = np.array([d["x"], d["y"]])
            for tid, tr in self.tracks.items():
                if tid in used:
                    continue
                dd = float(np.linalg.norm(z - tr["kf"].x[:2].flatten()))
                if dd < bd:
                    bd, best = dd, tid
            if best >= 0:
                self.tracks[best]["kf"].update(z)
                self.tracks[best].update({"age": 0, "det": d})
                used.add(best)
            else:
                kf = KF()
                kf.x[:2] = z.reshape(-1, 1)
                self.tracks[self.next_id] = {"kf": kf, "age": 0, "det": d}
                self.next_id += 1
        for tid in [t for t, tr in self.tracks.items() if tr["age"] > self.max_age]:
            del self.tracks[tid]
        out = []
        for tr in self.tracks.values():
            d = tr["det"]
            vx, vy = tr["kf"].x[2:4].flatten()
            out.append({**d, "vx": float(vx), "vy": float(vy)})
        return out
