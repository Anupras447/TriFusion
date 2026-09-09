# Demo Video Shotlist — 3 minutes (compulsory deliverable)

Record at 1080p, HUD on, metrics overlay burned in. Total ~3:00.

| # | Time | Shot | Caption |
|---|---|---|---|
| 1 | 0:00–0:15 | Title + architecture diagram (PRD §3) | "Perception → prediction → planning → control, 20 Hz" |
| 2 | 0:15–0:50 | **S5 cattle** (Three.js, slow-mo on brake) | "Cow sprint at 20 m — emergency stop, gap > 2 m" |
| 3 | 0:50–1:10 | **S1 village** (RoadRunner viewport) | "Unmarked road, tractor + potholes, no lane lines" |
| 4 | 1:10–1:30 | **S3 merge** (top cam, candidate paths) | "Truck cuts in — green chosen vs red rejected" |
| 5 | 1:30–1:50 | **S2 intersection** | "No signals — creep, yield to auto + bike" |
| 6 | 1:50–2:10 | **S4 market** | "Shared space crawl ≤ 3 m/s, cart + crossings" |
| 7 | 2:10–2:40 | Metrics table fullscreen (15/15, p95 < 50 ms) | "Zero collisions across 15 seeded runs" |
| 8 | 2:40–3:00 | Parity slide + team + "RoadRunner + Simulink twin" | Close on Simulink block diagram |

Recording: `cd web && python -m http.server 8000`, open page, press S5, enable slow-mo
for the brake moment, capture with OBS/Xbox Game Bar. RoadRunner clips: Play each
`.xosc` once, screen-capture. Export stills for report §3.
