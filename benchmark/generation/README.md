# Benchmark Generation

Scripts that generate the CARLA-F and B2D-C route files from the 220 Bench2Drive
routes in `benchmark/data/bench2drive`. Run them from the repository root.

| Script | Purpose |
|---|---|
| `generate_carla_f.py` | **CARLA-F** navigation-following routes (Algorithm 1, App. A.1.1). Keeps only the start point and town of each source route, rebuilds a follow-road route on the CARLA OpenDRIVE graph, and takes the earliest waypoint with the most feasible meta-commands as the trigger. Each feasible meta-command (`turn_left`, `turn_right`, `turn_straight` = go straight, `lane_change_left`, `lane_change_right`, and `lane_follow` with probability 0.2 unless it is the only one) yields one route whose ground-truth path performs it; further navigation instructions are chained from the maneuver endpoint, after a target-speed instruction derived from the OpenDRIVE speed limit. Background traffic is disabled and `--force-all-green-traffic-lights` keeps all lights green. |
| `generate_b2d_c.py` | **B2D-C** navigation-scene conflict routes (App. A.2). Keeps each Bench2Drive route and its scenario unchanged and adds a target-speed instruction at the start plus an unsafe instruction, sampled from a per-scenario pool, that fires when the scenario becomes active. |
| `verify_routes.py` | Densifies every adjacent waypoint pair with `GlobalRoutePlanner.trace_route()`, as the leaderboard evaluator does, and reports routes whose planner path turns into a detour. |
| `build_conflict_meta_commands.py` | Builds `team_code/conflict_instruction_meta_commands.json`, the table that maps every B2D-C / B2D-Adv instruction text to the meta-command indexing the world-model action anchor. A text is mapped by the words that name a maneuver (left / right turn, left / right lane change, straight through an intersection), ignoring any requested speed; any other text is `follow_road`. |

Shared modules: `route_builder.py` (route reconstruction, trigger selection),
`actionability.py` (feasible meta-commands), `instructions.py` (instruction pools,
target speed), `opendrive.py` (speed limits, map loading), `planner_route_tools.py`
(planner-safe waypoint anchors), `geometry.py`, `xml_builder.py`.

## Setup

CARLA-F generation and route verification need the CARLA 0.9.15 Python API and
OpenDRIVE maps (read from `$CARLA_ROOT/CarlaUE4/Content/Carla/Maps`, or
`--xodr-root`), but no running CARLA server. `generate_b2d_c.py` only parses XML.

```bash
pip install carla==0.9.15 networkx
export CARLA_ROOT=/path/to/carla0915
export PYTHONPATH=$CARLA_ROOT/PythonAPI/carla:$PYTHONPATH  # agents.navigation.global_route_planner
```

## Regenerate

```bash
python benchmark/generation/generate_carla_f.py benchmark/data/bench2drive \
    --output benchmark/data/generated/carla_f \
    --force-all-green-traffic-lights --seed 42

python benchmark/generation/generate_b2d_c.py benchmark/data/bench2drive \
    --output benchmark/data/generated/b2d_c --seed 42
```

All other options keep their defaults: 130 m route horizon, 3 m waypoint step,
10 m lane-change preparation, at most 3 chained navigation instructions, and at
least 30 m between chained triggers. Sampling is deterministic per route for a
fixed `--seed`.

## Verify

```bash
python benchmark/generation/verify_routes.py benchmark/data/generated/carla_f
```

A pathological route is densified into a detour by the evaluator even if its XML
polyline looks correct. `--output-json` writes the per-segment report.

## Released routes

The released `benchmark/data/carla_f` (210 routes) and `benchmark/data/b2d_c`
(150 routes) are curated subsets of the generated routes, so regeneration
reproduces the procedure, not necessarily the exact released files.
