"""Safety filter — final gate before actuation (PRD §4.4). Keeps the legacy
choose_bot_action semantics: B2=b Emergency brake, B1=decelerate, rear=veto."""
from .lattice import plan


def safe_action(world, predictions):
    res = plan(world, predictions)
    b = world.bubble_profile()
    ch = res["chosen"]
    ex = world.ego["x"]
    if ch["collides"]:
        # every candidate collides → hard stop, hold lane (never drive into a hit)
        res.update({"accel": -4.0, "steer": 0.0, "action_tag": "EMERGENCY_BRAKE"})
        return res
    # desired accel toward chosen speed, rate-limited for smoothness
    # (emergency paths return early above with full authority)
    raw = float(max(-6.0, min(3.0, (ch["speed"] - world.ego["v"]) * 0.8)))
    prev = float(world.prev_accel)
    accel = float(max(prev - 0.8, min(prev + 0.8, raw)))
    # lateral steer toward path
    look = ch["path"][2] if len(ch["path"]) > 2 else ch["path"][-1]
    steer = float(max(-0.5, min(0.5, (look[0] - ex) * 0.4 - world.ego["theta"])))
    tag = res["action_tag"]
    if b["b2"]:
        accel, steer, tag = -6.0, steer * 1.5, "EMERGENCY_BRAKE"
    elif b["b1"]:
        accel, tag = min(accel, -1.5), "DECREASE_SPEED"
    # rear veto: if something fast behind and we want a big lateral jump, hold lane
    for v in world.vehicles:
        if -12 < v["y"] - world.ego["y"] < 0 and abs(v["x"] - ex) < 2.0 and v["v"] > world.ego["v"] + 2:
            if abs(ch["lat"]) > 1.0:
                steer, tag = 0.0, "MAINTAIN_LANE"
            break
    res.update({"accel": accel, "steer": steer, "action_tag": tag})
    return res
