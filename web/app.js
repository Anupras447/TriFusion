/* SIH 3D demo — procedural low-poly Indian street, all assets generated in code.
   Mock mode by default; if ws://localhost:8765 serves snapshots, switches to LIVE. */
import * as THREE from 'three';

const $ = id => document.getElementById(id);
let renderer;
try {
  renderer = new THREE.WebGLRenderer({ antialias: true });
} catch (e) { $('nogl').hidden = false; throw e; }
renderer.setSize(innerWidth, innerHeight);
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
$('app').appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0d1420);
scene.fog = new THREE.Fog(0x0d1420, 60, 220);
scene.add(new THREE.HemisphereLight(0x8fb4ff, 0x33261a, 0.9));
const sun = new THREE.DirectionalLight(0xffd9a0, 1.1);
sun.position.set(-30, 60, -40); scene.add(sun);

let camMode = 'chase';
const cam = new THREE.PerspectiveCamera(60, innerWidth / innerHeight, 0.1, 500);
addEventListener('resize', () => { cam.aspect = innerWidth / innerHeight; cam.updateProjectionMatrix(); renderer.setSize(innerWidth, innerHeight); });

// ---------- materials / factories ----------
const M = {
  road: new THREE.MeshStandardMaterial({ color: 0x2b2f36, roughness: .95 }),
  ground: new THREE.MeshStandardMaterial({ color: 0x1a2230, roughness: 1 }),
  dash: new THREE.MeshBasicMaterial({ color: 0xd8c95a }),
  ego: new THREE.MeshStandardMaterial({ color: 0x18c8e8, roughness: .4, metalness: .3 }),
  car: new THREE.MeshStandardMaterial({ color: 0xc7ccd6, roughness: .5, metalness: .2 }),
  auto: new THREE.MeshStandardMaterial({ color: 0xf2b705, roughness: .6 }),
  truck: new THREE.MeshStandardMaterial({ color: 0x7a4a2b, roughness: .7 }),
  bike: new THREE.MeshStandardMaterial({ color: 0xd23c3c, roughness: .5 }),
  glass: new THREE.MeshStandardMaterial({ color: 0x10151f, roughness: .2, metalness: .6 }),
  tyre: new THREE.MeshStandardMaterial({ color: 0x0a0a0a, roughness: 1 }),
  skin: [0xe8b88a, 0xc98d5f, 0x8a5a34].map(c => new THREE.MeshStandardMaterial({ color: c })),
  shirt: [0x2e9e5b, 0x3c6ed2, 0xd23c8e, 0xe07b2a].map(c => new THREE.MeshStandardMaterial({ color: c })),
  cow: new THREE.MeshStandardMaterial({ color: 0x8a6a4a, roughness: .9 }),
  cart: new THREE.MeshStandardMaterial({ color: 0x6a4a8a, roughness: .8 }),
  stall: new THREE.MeshStandardMaterial({ color: 0xb03a2a, roughness: .8 }),
  brake: new THREE.MeshBasicMaterial({ color: 0xff2a1a }),
  brakeDim: new THREE.MeshBasicMaterial({ color: 0x551210 }),
  head: new THREE.MeshBasicMaterial({ color: 0xfff6c8 }),
  amber: new THREE.MeshBasicMaterial({ color: 0xffa500 }),
  amberOff: new THREE.MeshBasicMaterial({ color: 0x4a3208 }),
};
const wheelGeo = new THREE.CylinderGeometry(0.35, 0.35, 0.3, 12);
wheelGeo.rotateZ(Math.PI / 2);
function wheel(x, y, z) { const m = new THREE.Mesh(wheelGeo, M.tyre); m.position.set(x, y, z); return m; }

function makeVehicle(kind, isEgo) {
  const g = new THREE.Group();
  const bodyM = isEgo ? M.ego : (kind === 'auto' ? M.auto : kind === 'truck' ? M.truck : kind === 'bike' ? M.bike : M.car);
  const L = kind === 'truck' ? 8 : kind === 'bike' ? 2.0 : kind === 'auto' ? 3.0 : 4.2;
  const W = kind === 'bike' ? 0.7 : kind === 'auto' ? 1.4 : 1.8;
  const body = new THREE.Mesh(new THREE.BoxGeometry(W, 0.7, L), bodyM);
  body.position.y = 0.65; g.add(body);
  if (kind !== 'bike') {
    const cab = new THREE.Mesh(new THREE.BoxGeometry(W * 0.85, 0.55, L * 0.4), M.glass);
    cab.position.set(0, 1.2, kind === 'truck' ? L * 0.28 : -L * 0.05); g.add(cab);
  } else {
    const rider = new THREE.Mesh(new THREE.CapsuleGeometry(0.28, 0.7, 3, 8), M.shirt[1]);
    rider.position.set(0, 1.3, 0); g.add(rider);
  }
  const wheels = [];
  const wz = L / 2 - 0.7, wx = W / 2;
  const nW = kind === 'auto' ? [[0, wz], [-wx, -wz], [wx, -wz]] : [[-wx, wz], [wx, wz], [-wx, -wz], [wx, -wz]];
  for (const [x, z] of nW) { const w = wheel(x, 0.35, z); wheels.push(w); g.add(w); }
  const hl = [], tl = [], ind = [];
  for (const s of [-1, 1]) {
    const h = new THREE.Mesh(new THREE.BoxGeometry(0.25, 0.18, 0.1), M.head);
    h.position.set(s * (W / 2 - 0.2), 0.7, L / 2 + 0.03); g.add(h); hl.push(h);
    const t = new THREE.Mesh(new THREE.BoxGeometry(0.25, 0.18, 0.1), M.brakeDim.clone());
    t.position.set(s * (W / 2 - 0.2), 0.7, -L / 2 - 0.03); g.add(t); tl.push(t);
    const im = new THREE.Mesh(new THREE.BoxGeometry(0.12, 0.12, 0.08), M.amberOff.clone());
    im.position.set(s * (W / 2 + 0.02), 0.95, L * 0.2); g.add(im); ind.push(im);
  }
  g.userData = { wheels, tl, ind, L, W, kind, isEgo, blink: Math.random() * 6 };
  return g;
}

function makePed(i) {
  const g = new THREE.Group();
  const torso = new THREE.Mesh(new THREE.CapsuleGeometry(0.22, 0.6, 3, 8), M.shirt[i % 4]);
  torso.position.y = 1.05; g.add(torso);
  const head = new THREE.Mesh(new THREE.SphereGeometry(0.18, 10, 8), M.skin[i % 3]);
  head.position.y = 1.7; g.add(head);
  const limbs = [];
  for (const s of [-1, 1]) {
    const arm = new THREE.Mesh(new THREE.BoxGeometry(0.12, 0.55, 0.12), M.shirt[i % 4]);
    arm.position.set(s * 0.32, 1.05, 0); g.add(arm); limbs.push({ m: arm, ph: s * 1.5 });
    const leg = new THREE.Mesh(new THREE.BoxGeometry(0.14, 0.6, 0.14), M.skin[0]);
    leg.position.set(s * 0.12, 0.35, 0); g.add(leg); limbs.push({ m: leg, ph: s * 1.5 + 3.14 });
  }
  g.userData = { limbs, ph: Math.random() * 6 };
  return g;
}

function makeCow() {
  const g = new THREE.Group();
  const body = new THREE.Mesh(new THREE.BoxGeometry(1.1, 0.9, 2.0), M.cow);
  body.position.y = 1.0; g.add(body);
  const head = new THREE.Mesh(new THREE.BoxGeometry(0.55, 0.55, 0.7), M.cow);
  head.position.set(0, 1.25, 1.25); g.add(head);
  const legs = [];
  for (const [x, z] of [[-0.4, 0.7], [0.4, 0.7], [-0.4, -0.7], [0.4, -0.7]]) {
    const l = new THREE.Mesh(new THREE.BoxGeometry(0.22, 0.8, 0.22), M.cow);
    l.position.set(x, 0.4, z); g.add(l); legs.push(l);
  }
  g.userData = { head, legs, ph: Math.random() * 6 };
  return g;
}

// ---------- static world ----------
const ground = new THREE.Mesh(new THREE.PlaneGeometry(400, 600), M.ground);
ground.rotation.x = -Math.PI / 2; ground.position.y = -0.05; scene.add(ground);
const road = new THREE.Mesh(new THREE.PlaneGeometry(16, 600), M.road);
road.rotation.x = -Math.PI / 2; scene.add(road);
const dashes = [];
for (let i = 0; i < 60; i++) {
  const d = new THREE.Mesh(new THREE.PlaneGeometry(0.25, 2), M.dash);
  d.rotation.x = -Math.PI / 2; d.position.set(0, 0.01, -280 + i * 10); scene.add(d); dashes.push(d);
}
// stalls + potholes decoration
for (let i = 0; i < 8; i++) {
  const s = new THREE.Mesh(new THREE.BoxGeometry(2, 2, 2), M.stall);
  s.position.set(i % 2 ? 9 : -9, 1, 40 + i * 25); scene.add(s);
  const awn = new THREE.Mesh(new THREE.BoxGeometry(2.6, 0.15, 2.6), M.dash);
  awn.position.set(s.position.x, 2.2, s.position.z); scene.add(awn);
}
for (let i = 0; i < 10; i++) {
  const p = new THREE.Mesh(new THREE.CircleGeometry(0.5 + (i % 3) * 0.3, 10),
    new THREE.MeshBasicMaterial({ color: 0x0c0d10 }));
  p.rotation.x = -Math.PI / 2; p.position.set((i * 37 % 9) - 4.5, 0.012, 20 + i * 22); scene.add(p);
}

// bubbles (safety cones) + candidate path lines + dust
function bubbleMesh(color, op) {
  const m = new THREE.Mesh(new THREE.PlaneGeometry(3.4, 14),
    new THREE.MeshBasicMaterial({ color, transparent: true, opacity: op, side: THREE.DoubleSide, depthWrite: false }));
  m.rotation.x = -Math.PI / 2; return m;
}
const bub1 = bubbleMesh(0x2a7fff, 0.10), bub2 = bubbleMesh(0xff2a1a, 0.14);
scene.add(bub1, bub2);
const pathLines = [];
for (let i = 0; i < 3; i++) {
  const geo = new THREE.BufferGeometry().setFromPoints(new Array(10).fill(0).map(() => new THREE.Vector3()));
  const line = new THREE.Line(geo, new THREE.LineBasicMaterial({ color: 0xff4444, transparent: true, opacity: 0.8 }));
  scene.add(line); pathLines.push(line);
}
const dustGeo = new THREE.BufferGeometry();
const dustN = 120, dustPos = new Float32Array(dustN * 3), dustVel = [];
for (let i = 0; i < dustN; i++) { dustPos.set([0, -10, 0], i * 3); dustVel.push(0); }
dustGeo.setAttribute('position', new THREE.BufferAttribute(dustPos, 3));
const dust = new THREE.Points(dustGeo, new THREE.PointsMaterial({ color: 0x9a8a6a, size: 0.35, transparent: true, opacity: 0.6 }));
scene.add(dust);
let dustI = 0;

// horn ripple rings
const rings = [];
function hornRing(x, y, z) {
  const m = new THREE.Mesh(new THREE.RingGeometry(0.5, 0.7, 24),
    new THREE.MeshBasicMaterial({ color: 0xffd75c, transparent: true, opacity: 0.9, side: THREE.DoubleSide }));
  m.position.set(x, y, z); scene.add(m); rings.push({ m, r: 0.5 });
}

// ---------- entity pools ----------
const ego = makeVehicle('car', true); scene.add(ego);
scene.add(new THREE.SpotLight(0xfff2c0, 60, 40, 0.5, 0.5)); // headlight throw (follows ego)
const pools = { veh: [], peds: [], anim: [] };
function getPooled(pool, maker) {
  for (const o of pool) if (!o.visible) { o.visible = true; return o; }
  const o = maker(); scene.add(o); pool.push(o); return o;
}

// ---------- sim plumbing (mock or live WS) ----------
let S = window.MockSim.reset('S5'), sid = 'S5', live = null, playing = true, slowmo = false;
try {
  const ws = new WebSocket('ws://localhost:8765');
  ws.onmessage = ev => { live = JSON.parse(ev.data); $('srcBadge').textContent = 'LIVE'; };
  ws.onerror = () => { try { ws.close(); } catch (e) { /* mock */ } };
} catch (e) { /* mock mode */ }

const btns = $('scenBtns');
for (const id of Object.keys(window.MockSim.cfgs)) {
  const b = document.createElement('button');
  b.textContent = id; b.onclick = () => { sid = id; S = window.MockSim.reset(id); $('scenName').textContent = window.MockSim.cfgs[id].name; };
  btns.appendChild(b);
}
$('btnPlay').onclick = e => { playing = !playing; e.target.textContent = playing ? 'Pause' : 'Play'; };
$('btnSlow').onclick = e => { slowmo = !slowmo; e.target.textContent = 'Slow-mo: ' + (slowmo ? 'on' : 'off'); e.target.classList.toggle('on', slowmo); };
$('btnCam').onclick = e => { camMode = camMode === 'chase' ? 'top' : 'chase'; e.target.textContent = 'Camera: ' + camMode; };

let shake = 0, lastHorn = 0;
const clock = new THREE.Clock();
function frame() {
  requestAnimationFrame(frame);
  const rawDt = Math.min(clock.getDelta(), 0.05);
  const dt = playing ? rawDt * (slowmo ? 0.25 : 1) : 0;
  const snap = live || (dt > 0 ? window.MockSim.step(S, dt) : null);
  if (!snap) { renderer.render(scene, cam); return; }
  const t = snap.t, e = snap.ego;

  // ego transform + animation
  ego.position.set(e.x, 0, -e.y); ego.rotation.y = -e.theta;
  const u = ego.userData;
  for (const w of u.wheels) w.rotation.x += e.v * dt * 2.5;
  const braking = snap.braking || /BRAKE|DECREASE|YIELD/.test(e.action_tag);
  for (const tl of u.tl) tl.material.color.setHex(braking ? 0xff2a1a : 0x551210);
  u.blink += dt * 6;
  const wantInd = /OVERTAKE_LEFT|CHANGE_LANE_LEFT/.test(e.action_tag) ? 0 : /OVERTAKE_RIGHT|CHANGE_LANE_RIGHT/.test(e.action_tag) ? 1 : -1;
  u.ind.forEach((m, i) => m.material.color.setHex(wantInd === i && (u.blink % 2 < 1) ? 0xffa500 : 0x4a3208));
  if (/HORN|YIELD|EMERGENCY/.test(e.action_tag) && t - lastHorn > 1.2) { lastHorn = t; hornRing(e.x, 1.5, -e.y); }

  // bubbles follow ego
  bub1.position.set(e.x, 0.02, -e.y - 7); bub2.position.set(e.x, 0.03, -e.y - 3.5);
  bub1.material.opacity = /DECREASE|EMERGENCY/.test(e.action_tag) ? 0.28 : 0.10;
  bub2.material.opacity = /EMERGENCY/.test(e.action_tag) ? 0.4 : 0.12;
  shake = /EMERGENCY/.test(e.action_tag) ? 0.5 : Math.max(0, shake - dt * 2);

  // agents
  for (const o of [...pools.veh, ...pools.peds, ...pools.anim]) o.visible = false;
  for (const v of snap.veh || []) {
    const m = getPooled(pools.veh, () => makeVehicle(v.kind || 'car', false));
    m.position.set(v.x, 0, -v.y); m.rotation.y = (v.dir || 1) > 0 ? 0 : Math.PI;
    for (const w of m.userData.wheels) w.rotation.x += (v.v || 4) * dt * 2.5;
  }
  let pi = 0;
  for (const p of snap.peds || []) {
    const m = getPooled(pools.peds, () => makePed(pi++));
    m.position.set(p.x, 0, -p.y);
    m.userData.ph += dt * 8;
    for (const l of m.userData.limbs) l.m.rotation.x = Math.sin(m.userData.ph + l.ph) * 0.6;
  }
  for (const a of snap.anim || []) {
    if (a.kind === 'cart') { const m = getPooled(pools.anim, makeCow); m.position.set(a.x, 0, -a.y); continue; }
    const m = getPooled(pools.anim, makeCow);
    m.position.set(a.x, 0, -a.y);
    m.userData.ph += dt * (a.vx ? 7 : 2);
    m.userData.head.position.y = 1.25 + Math.sin(m.userData.ph) * (a.vx ? 0.12 : 0.05);
    m.userData.legs.forEach((l, i) => l.rotation.x = a.vx ? Math.sin(m.userData.ph + i * 1.6) * 0.5 : 0);
  }

  // candidate paths: chosen green, rest red
  (snap.cands || []).forEach((c, i) => {
    const line = pathLines[i]; if (!line) return;
    const pts = c.pts.map(([x, y]) => new THREE.Vector3(x, 0.15, -y));
    line.geometry.setFromPoints(pts);
    line.material.color.setHex(i === 1 ? 0x39ff6a : 0xff4444);
  });

  // dust + rings
  if (e.v > 3 && Math.random() < 0.5) {
    dustPos.set([e.x + (Math.random() - .5), 0.3, -e.y + 2], dustI * 3); dustVel[dustI] = 1; dustI = (dustI + 1) % dustN;
    dustGeo.attributes.position.needsUpdate = true;
  }
  for (let i = 0; i < dustN; i++) if (dustVel[i] > 0) { dustPos[i * 3 + 1] += dt; dustPos[i * 3 + 2] += dt * 2; if (dustPos[i * 3 + 1] > 2) dustVel[i] = 0; }
  dustGeo.attributes.position.needsUpdate = true;
  for (let i = rings.length - 1; i >= 0; i--) {
    const r = rings[i]; r.r += dt * 8; r.m.scale.setScalar(r.r); r.m.material.opacity = Math.max(0, 0.9 - r.r * 0.15);
    if (r.m.material.opacity <= 0) { scene.remove(r.m); rings.splice(i, 1); }
  }

  // camera
  const shx = shake > 0 ? (Math.random() - .5) * shake : 0;
  if (camMode === 'chase') cam.position.lerp(new THREE.Vector3(e.x + shx, 9, -e.y + 13), 0.08);
  else cam.position.lerp(new THREE.Vector3(e.x, 70, -e.y + 1), 0.08);
  cam.lookAt(e.x, 0, -e.y - 8);

  // HUD
  $('hV').textContent = e.v.toFixed(1); $('hT').textContent = t.toFixed(1);
  $('hR').textContent = (snap.replanMs || 0).toFixed(1);
  $('hTTC').textContent = (snap.ttc > 90 ? '–' : snap.ttc.toFixed(1));
  $('hC').textContent = snap.collisions || 0;
  const toast = $('tagToast'); toast.textContent = e.action_tag;
  toast.classList.toggle('hot', /EMERGENCY|BRAKE/.test(e.action_tag));

  renderer.render(scene, cam);
}
frame();
