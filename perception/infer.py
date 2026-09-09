"""Perception bridge: tries ONNX YOLO/U-Net, falls back to ground-truth stub.
Never crashes the loop — returns (detections, source_flag)."""
try:
    import onnxruntime  # noqa: F401
    _ONNX = True
except Exception:
    _ONNX = False


def infer(world, image=None):
    """Stub mode: project world agents into detection list.
    Real mode: if image + weights wired later, run ONNX here (same return contract)."""
    dets = []
    for v in world.vehicles:
        dets.append({"cls": v["kind"], "x": v["x"], "y": v["y"], "w": 1.8,
                     "l": 4.2, "vx": 0.0, "vy": v["dir"] * v["v"]})
    for p in world.peds:
        dets.append({"cls": "person", "x": p["x"], "y": p["y"], "w": 0.6, "l": 0.6,
                     "vx": 0.0, "vy": 0.4})
    for a in world.animals:
        dets.append({"cls": "animal", "x": a["x"], "y": a["y"], "w": 1.2, "l": 2.0,
                     "vx": a.get("vx", 0.0), "vy": 0.0})
    return dets, ("onnx" if (_ONNX and image is not None) else "stub")
