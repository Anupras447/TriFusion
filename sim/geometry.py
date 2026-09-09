"""Geometry helpers ported from IRS_dum (2) (1).py — SAT collision, road center, sensors."""
import numpy as np


def get_corners(center, dim, angle_deg):
    cx, cy = center
    w, l = dim
    rad = np.radians(angle_deg)
    cos_a, sin_a = np.cos(rad), np.sin(rad)
    hw, hl = w / 2.0, l / 2.0
    out = []
    for lx, ly in ((-hw, -hl), (hw, -hl), (hw, hl), (-hw, hl)):
        out.append(np.array([cx + lx * cos_a - ly * sin_a,
                             cy + lx * sin_a + ly * cos_a]))
    return out


def check_oriented_box_collision(c1, d1, a1, c2, d2, a2=0.0):
    """Separating Axis Theorem test. c=(x,y), d=(w,l), a=degrees."""
    boxes = [get_corners(c1, d1, a1), get_corners(c2, d2, a2)]
    for b in range(2):
        for i in range(4):
            p1, p2 = boxes[b][i], boxes[b][(i + 1) % 4]
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            n = np.linalg.norm(normal)
            if n < 1e-9:
                continue
            normal = normal / n
            projs = [[np.dot(c, normal) for c in box] for box in boxes]
            if max(projs[0]) < min(projs[1]) or max(projs[1]) < min(projs[0]):
                return False
    return True


def road_center_sine(y, segments, fallback_amp=1.0, fallback_freq=80.0):
    for seg in segments:
        if seg[0] <= y <= seg[1]:
            local = y - seg[0]
            amp, freq = seg[2], seg[3]
            return amp * np.sin(local / freq) + (amp * 0.5) * np.cos(local / (freq * 0.5))
    return fallback_amp * np.sin(y / fallback_freq)


def ttc(ego_y, ego_v, agent_y, agent_v, min_gap=0.1):
    """Longitudinal time-to-collision for same-lane follower/leader."""
    gap = agent_y - ego_y
    if gap <= min_gap:
        return 0.0
    rel = ego_v - agent_v
    if rel <= 1e-6:
        return float("inf")
    return gap / rel
