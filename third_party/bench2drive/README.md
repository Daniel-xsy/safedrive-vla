# Bench2Drive

`leaderboard/`, `scenario_runner/`, `leaderboard/data/weather.xml` and
`tools/efficiency_smoothness_benchmark.py` are taken from
[Bench2Drive](https://github.com/Thinklab-SJTU/Bench2Drive) as distributed with
[SimLingo](https://github.com/RenzKa/simlingo) (commit `743b243`). The leaderboard
and the scenario runner are MIT-licensed (their `LICENSE` files); the rest of
Bench2Drive is CC BY-NC-ND 4.0 (`LICENSE`). Only what the leaderboard needs is
kept: the scenario runner's examples, tests, metrics and OpenSCENARIO schemas
are removed.

Modified files, which add the route attributes `disable_bg_vehicle` (no
background traffic) and `force_all_green_traffic_lights` (all traffic lights stay
green) used by the CARLA-F routes:

- `leaderboard/leaderboard/utils/route_parser.py`
- `leaderboard/leaderboard/scenarios/route_scenario.py`
- `scenario_runner/srunner/scenarioconfigs/route_scenario_configuration.py`
