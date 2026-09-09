/* Mock world: generates the same snapshot contract as sim/world.py snapshot().
   Used when no live backend is connected. Scenario densities mirror sim/scenarios/. */
(function () {
  const cfgs = {
    S1: { name: "S1 — Unmarked village road", halfW: 4.5, fwd: 8, on: 4, rear: 2, peds: 6, cows: 1, event: { t: 8, kind: "cow" } },
    S2: { name: "S2 — Urban intersection, no signals", halfW: 6.5, fwd: 14, on: 8, rear: 4, peds: 12, cows: 0, event: { t: 5, kind: "auto" } },
    S3: { name: "S3 — Highway merge, slow vehicles", halfW: 7.5, fwd: 10, on: 0, rear: 6, peds: 0, cows: 0, event: { t: 6, kind: "truck" } },
    S4: { name: "S4 — Dense market shared space", halfW: 5.0, fwd: 6, on: 2, rear: 2, peds: 16, cows: 0, event: { t: 10, kind: "cart" } },
    S5: { name: "S5 — Sudden cattle crossing", halfW: 6.0, fwd: 4, on: 2, rear: 1, peds: 2, cows: 3, event: { t: 7, kind: "cow" } },
  };
  let R = 7;
  const rnd = () => (R = (R * 16807) % 2147483647) / 2147483647;

  function makeAgent(kind, x, y, v, dir) { return { kind, x, y, v: v || 5, dir: dir || 1, w: 1.8, l: 4.2 }; }

  window.MockSim = {
    cfgs,
    reset(sid) {
      R = 7 + sid.charCodeAt(1);
      const c = cfgs[sid], A = { veh: [], peds: [], anim: [] };
      for (let i = 0; i < c.fwd; i++) A.veh.push(makeAgent(["car", "auto", "bike"][i % 3], (rnd() - .5) * c.halfW, 20 + rnd() * 160, 4 + rnd() * 4, 1));
      for (let i = 0; i < c.on; i++) A.veh.push(makeAgent("car", -c.halfW / 2 + rnd() * 2, 40 + rnd() * 140, 4 + rnd() * 3, -1));
      for (let i = 0; i < c.rear; i++) A.veh.push(makeAgent("car", 1 + rnd() * 2, -50 + rnd() * 20, 5 + rnd() * 3, 1));
      for (let i = 0; i < c.peds; i++) A.peds.push({ kind: "ped", x: (rnd() - .5) * c.halfW * 2, y: 20 + rnd() * 160, ph: rnd() * 6 });
      for (let i = 0; i < c.cows; i++) A.anim.push({ kind: "cow", x: -c.halfW - 1, y: 70 + i * 15, vx: 0, cross: false });
      return { t: 0, ego: { x: 1, y: 0, theta: 0, v: sid === "S3" ? 12 : sid === "S5" ? 10 : 6, action_tag: "MAINTAIN_LANE" },
        agents: A, collisions: 0, eventFired: false, cfg: c };
    },
    step(S, dt) {
      S.t += dt;
      const ev = S.cfg.event;
      if (!S.eventFired && S.t >= ev.t) {  // trigger event agent
        S.eventFired = true;
        if (ev.kind === "cow") S.agents.anim.push({ kind: "cow", x: S.cfg.halfW + 1, y: S.ego.y + 25, vx: -2.5, cross: true });
        if (ev.kind === "auto") S.agents.veh.push(makeAgent("auto", -8, S.ego.y + 20, 5, 1));
        if (ev.kind === "truck") S.agents.veh.push(makeAgent("truck", 6, S.ego.y + 10, 8, 1));
        if (ev.kind === "cart") S.agents.anim.push({ kind: "cart", x: S.cfg.halfW, y: S.ego.y + 18, vx: -1.2, cross: true });
      }
      // ego: cruise + brake for ahead agents + steer to freer side
      let nearest = null, nd = 1e9;
      const all = S.agents.veh.concat(S.agents.anim);
      for (const a of all) { const dy = a.y - S.ego.y, dx = Math.abs(a.x - S.ego.x); if (dy > 0 && dy < nd && dx < 2) { nd = dy; nearest = a; } }
      let want = 8, tag = "MAINTAIN_LANE";
      if (nearest) {
        if (nd < 6) { want = 0; tag = "EMERGENCY_BRAKE"; }
        else if (nd < 14) { want = 2; tag = nearest.kind === "ped" ? "YIELD_TO_PEDESTRIAN" : nearest.kind === "cow" ? "YIELD_TO_ANIMAL" : "DECREASE_SPEED"; }
        if (nd < 14 && nearest.x > S.ego.x) { S.ego.x -= 1.5 * dt; tag = "OVERTAKE_LEFT"; }
        else if (nd < 14 && nearest.x <= S.ego.x) { S.ego.x += 1.5 * dt; tag = "OVERTAKE_RIGHT"; }
      }
      S.ego.v += Math.max(-6 * dt, Math.min(3 * dt, want - S.ego.v));
      S.ego.y += S.ego.v * dt;
      S.ego.action_tag = tag;
      for (const v of S.agents.veh) v.y += v.dir * v.v * dt;
      for (const p of S.agents.peds) { p.ph += dt * 6; p.x += Math.sin(p.ph) * .2 * dt; p.y += .9 * dt; }
      for (const a of S.agents.anim) { a.x += (a.vx || 0) * dt; }
      // candidate lattice paths for viz (3 lateral x current speed)
      const cands = [-2.5, 0, 2.5].map(L => ({ lat: L, pts: Array.from({ length: 10 }, (_, i) => [S.ego.x + (L - (S.ego.x - 0)) * .1 * (i + 1), S.ego.y + (i + 1) * 2.5]) }));
      return { t: S.t, ego: S.ego, veh: S.agents.veh, peds: S.agents.peds, anim: S.agents.anim,
        cands, collisions: S.collisions, replanMs: 2 + Math.random() * 6,
        ttc: nearest ? Math.max(0, nd / Math.max(.5, S.ego.v)) : 99, braking: want < S.ego.v - .5 };
    }
  };
})();
