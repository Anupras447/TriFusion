"""Metrics: replan latency, smoothness (jerk), TTC, completion (PRD §4.6)."""
import numpy as np


def summarize(log):
    lat = log.get("replan_ms", [0])
    return {
        "scenario": log.get("scenario", "?"),
        "seed": log.get("seed", 0),
        "collisions": int(log.get("collisions", 0)),
        "success": bool(log.get("success", False)),
        "completion_pct": round(log.get("completion_pct", 0.0), 1),
        "min_ttc_s": round(float(log.get("min_ttc", 0.0)), 2),
        "replan_ms_p50": round(float(np.median(lat)), 2),
        "replan_ms_p95": round(float(np.percentile(lat, 95)), 2),
        "jerk_rms": round(float(log.get("jerk_rms", 0.0)), 2),
        "max_lat_accel": round(float(log.get("max_lat_accel", 0.0)), 2),
        "steps": int(log.get("steps", 0)),
    }


def markdown_table(rows):
    hdr = "| Scenario | Seed | Collisions | Success | Completion% | minTTC(s) | replan p50(ms) | replan p95(ms) | jerk_rms |"
    sep = "|---|---|---|---|---|---|---|---|---|"
    lines = [hdr, sep]
    for r in rows:
        lines.append(f"| {r['scenario']} | {r['seed']} | {r['collisions']} | "
                     f"{'YES' if r['success'] else 'NO'} | {r['completion_pct']} | "
                     f"{r['min_ttc_s']} | {r['replan_ms_p50']} | {r['replan_ms_p95']} | {r['jerk_rms']} |")
    return "\n".join(lines)
