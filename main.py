"""Single-scenario runner: headless eval + optional pygame viz + WS state stream stub."""
import argparse
import json
import os
import yaml

BASE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser(description="SIH sim v2 — scenario runner")
    ap.add_argument("--scenario", default="S5")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--dump", default="", help="write final snapshot JSON to this path")
    args = ap.parse_args()

    from sim.world import ScenarioWorld
    from planning.safety import safe_action
    from perception.infer import infer
    from perception.tracker import Tracker

    scen = os.path.join(BASE, "sim", "scenarios", f"{args.scenario}.yaml")
    with open(scen) as f:
        cfg = yaml.safe_load(f)
    world = ScenarioWorld(cfg, seed=args.seed)
    tracker = Tracker()
    use_viz = not args.headless
    screen = None
    if use_viz:
        import pygame
        pygame.init()
        screen = pygame.display.set_mode((640, 480))
        pygame.display.set_caption(f"SIH sim v2 — {cfg['id']} {cfg['name']}")
    clock = None
    done = False
    while not done:
        dets, _ = infer(world)
        preds = tracker.update(dets)
        act = safe_action(world, preds)
        done, _info = world.step(steer=act["steer"], accel=act["accel"], action_tag=act["action_tag"])
        if use_viz:
            import pygame
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    done = True
            screen.fill((18, 20, 26))
            snap = world.snapshot()
            import pygame as pg
            # minimal top-down debug viz: ego cyan, vehicles white, peds yellow, animals orange
            def dot(x, y, color, r=4):
                sx = int(320 + (x - snap["ego"]["x"]) * 18)
                sy = int(430 - (y - snap["ego"]["y"]) * 6)
                pg.draw.circle(screen, color, (sx, sy), r)
            dot(snap["ego"]["x"], snap["ego"]["y"], (0, 255, 255), 6)
            for v in snap["vehicles"]:
                dot(v["x"], v["y"], (240, 240, 240))
            for p in snap["peds"]:
                dot(p["x"], p["y"], (255, 220, 80), 3)
            for a in snap["animals"]:
                dot(a["x"], a["y"], (255, 140, 40), 4)
            font = pg.font.SysFont("consolas", 16)
            hud = f"{cfg['id']} t={snap['t']:.1f}s v={snap['ego']['v']:.1f} {snap['ego']['action_tag']}"
            screen.blit(font.render(hud, True, (255, 255, 255)), (10, 10))
            pg.display.flip()
            if clock is None:
                import pygame as _pg
                clock = _pg.time.Clock()
            clock.tick(40)
        if world.step_count >= world.max_steps:
            break
    print(json.dumps({"scenario": cfg["id"], "seed": args.seed, "steps": world.step_count,
                      "collision": world.collision, "ego_y": round(world.ego["y"], 1),
                      "goal_y": world.goal_y, "action_tag": world.action_tag}, indent=2))
    if args.dump:
        with open(args.dump, "w") as f:
            json.dump(world.snapshot(), f, indent=2)


if __name__ == "__main__":
    main()
