"""CARLA-F metrics (App. A.1.2): Navigation Compliance Rate (NCR) per
meta-command, its instruction-weighted average (excluding speed), and Speed
Error (SE).

Each instruction is active from its trigger distance to the next trigger. The
driven trajectory (``metric_info.json``) is compared to the ground-truth route
of the XML over that window:

* lane follow / lane change: the majority ``(road_id, lane_id)`` of the last
  20% of the window (junction samples dropped) must match the ground truth;
* turn / go straight: the window end point must lie within 15 m of the ground
  truth end point;
* speed: ``|v - v*|`` at the end of the speed instruction, ``v`` smoothed over
  +-0.25 s; routes that never reach that point are excluded from SE.

Instructions are scored in order; after the first failure the agent has left
the ground-truth geometry, so the remaining ones count as failed.

    python benchmark/metrics/carla_f_metrics.py work_dirs/eval/carla_f --routes-dir benchmark/data/carla_f

Requires the ``carla`` Python package and the CARLA OpenDRIVE maps
(``$CARLA_ROOT/CarlaUE4/Content/Carla/Maps``).
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
import os
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import carla

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "tools"))
from run_distributed_eval import result_is_complete  # noqa: E402

TURN_THRESHOLD_M = 15.0
LANE_TAIL_FRACTION = 0.2
SIM_DT_S = 0.05            # CARLA runs at 20 Hz
SPEED_WINDOW_HALF = 5      # +-5 frames = +-0.25 s
CATEGORIES = {
    ("turn", "left"): "Turn Left",
    ("turn", "right"): "Turn Right",
    ("turn", "straight"): "Go Straight",
    ("lane_change", "left"): "Lane Change Left",
    ("lane_change", "right"): "Lane Change Right",
    ("lane_follow", ""): "Lane Follow",
}

Position = Tuple[float, float, float]


class CarlaMaps:
    """Lazily loads ``carla.Map`` objects from the OpenDRIVE files."""

    def __init__(self):
        root = Path(os.environ.get("CARLA_ROOT", "")) / "CarlaUE4" / "Content" / "Carla" / "Maps"
        self.files = {p.stem: p for p in root.rglob("*.xodr")}
        self.maps: Dict[str, carla.Map] = {}

    def get(self, town: str) -> carla.Map:
        if town not in self.maps:
            path = self.files.get(town) or self.files.get(f"{town}HD")
            if path is None:
                raise FileNotFoundError(f"OpenDRIVE map of {town} not found under $CARLA_ROOT")
            self.maps[town] = carla.Map(town, path.read_text(encoding="utf-8"))
        return self.maps[town]


def cumulative_distances(positions: List[Position]) -> List[float]:
    dists = [0.0]
    for a, b in zip(positions[:-1], positions[1:]):
        dists.append(dists[-1] + math.hypot(b[0] - a[0], b[1] - a[1]))
    return dists


def segment(positions: List[Position], cum: List[float], start_m: float, end_m: float) -> List[Position]:
    """Positions within ``[start_m, end_m]``; nearest samples if none."""
    if not cum:
        return []
    end_m = max(start_m, min(end_m, cum[-1]))
    idx = [i for i, d in enumerate(cum) if start_m <= d <= end_m]
    if not idx:
        lo = min(range(len(cum)), key=lambda i: abs(cum[i] - start_m))
        hi = min(range(len(cum)), key=lambda i: abs(cum[i] - end_m))
        idx = list(range(min(lo, hi), max(lo, hi) + 1))
    return [positions[i] for i in idx]


def position_at(positions: List[Position], cum: List[float], target_m: float) -> Optional[Position]:
    if not positions:
        return None
    target_m = max(0.0, min(target_m, cum[-1]))
    hi = bisect.bisect_left(cum, target_m)
    if hi <= 0:
        return positions[0]
    if hi >= len(cum) or math.isclose(cum[hi], target_m, abs_tol=1e-6) or cum[hi] <= cum[hi - 1]:
        return positions[min(hi, len(cum) - 1)]
    w = (target_m - cum[hi - 1]) / (cum[hi] - cum[hi - 1])
    a, b = positions[hi - 1], positions[hi]
    return tuple(a[k] + w * (b[k] - a[k]) for k in range(3))


def majority_lane(carla_map: carla.Map, positions: List[Position]) -> Optional[Tuple[int, int]]:
    lanes = []
    for x, y, z in positions:
        wp = carla_map.get_waypoint(carla.Location(x=x, y=y, z=z), project_to_road=True, lane_type=carla.LaneType.Driving)
        if wp is not None and not wp.is_junction:
            lanes.append((wp.road_id, wp.lane_id))
    return Counter(lanes).most_common(1)[0][0] if lanes else None


def trigger_distance(instruction: Optional[dict]) -> Optional[float]:
    if instruction is None:
        return None
    if instruction["trigger"].get("type", "start") == "start":
        return 0.0
    if instruction["trigger"].get("type") != "distance_traveled":
        return None
    return float(instruction["trigger"].get("value", 0.0))


def instruction_compliance(gt: List[Position], traj: List[Position], instructions: List[dict], carla_map) -> List[dict]:
    """Pass / fail of every lane-follow, lane-change and turn instruction."""
    traj_cum, gt_cum = cumulative_distances(traj), cumulative_distances(gt)
    results, failed = [], False
    for i, instr in enumerate(instructions):
        behavior = instr["expected_behavior"]
        kind, direction = behavior.get("type", ""), behavior.get("direction", "")
        if kind not in ("lane_change", "lane_follow", "turn"):
            continue
        start = trigger_distance(instr) or 0.0
        nxt = trigger_distance(instructions[i + 1]) if i + 1 < len(instructions) else None
        end = nxt if nxt is not None else traj_cum[-1]
        traj_end, gt_end = min(end, traj_cum[-1]), min(end, gt_cum[-1])
        result = {"category": CATEGORIES.get((kind, direction), f"{kind} {direction}".strip()), "passed": False}
        if failed:
            results.append(result)
            continue
        if kind in ("lane_change", "lane_follow"):
            def tail(pos, cum, seg_end):
                seg_start = max(start, seg_end - LANE_TAIL_FRACTION * max(seg_end - start, 0.0))
                return segment(pos, cum, seg_start, seg_end)
            expected = majority_lane(carla_map, tail(gt, gt_cum, max(start, gt_end)))
            actual = majority_lane(carla_map, tail(traj, traj_cum, max(start, traj_end)))
            result["passed"] = expected is not None and actual == expected
        else:
            a, b = position_at(gt, gt_cum, gt_end), position_at(traj, traj_cum, traj_end)
            ok = segment(gt, gt_cum, start, gt_end) and segment(traj, traj_cum, start, traj_end) and a and b
            result["passed"] = bool(ok) and math.hypot(b[0] - a[0], b[1] - a[1]) < TURN_THRESHOLD_M
        failed = not result["passed"]
        results.append(result)
    return results


def speed_error(route: ET.Element, traj: List[Position]) -> Optional[float]:
    """|v - v*| at the end of the speed instruction (None if never reached)."""
    for instr in route.iter("instruction"):
        behavior, trigger = instr.find("expected_behavior"), instr.find("trigger")
        if behavior is None or behavior.attrib.get("type") != "target_speed":
            continue
        if trigger is None or trigger.attrib.get("type") != "distance_traveled":
            continue
        duration = float(instr.findtext("duration_meters") or -1)
        if duration <= 0:
            return None
        end_m = float(trigger.attrib.get("value", 0)) + duration
        cum = cumulative_distances(traj)
        if len(traj) < 2 or cum[-1] + 1e-6 < end_m:
            return None
        idx = min(range(len(traj)), key=lambda i: abs(cum[i] - end_m))
        lo, hi = max(1, idx - SPEED_WINDOW_HALF), min(len(traj) - 1, idx + SPEED_WINDOW_HALF)
        if hi <= lo:
            return None
        disp = sum(math.hypot(traj[i][0] - traj[i - 1][0], traj[i][1] - traj[i - 1][1]) for i in range(lo, hi + 1))
        return abs(disp / ((hi - lo + 1) * SIM_DT_S) - float(behavior.attrib["speed_ms"]))
    return None


def load_trajectory(viz_dir: Path) -> Optional[List[Position]]:
    runs = sorted(viz_dir.glob("*/metric_info.json"))  # latest attempt
    if not runs:
        return None
    try:
        info = json.loads(runs[-1].read_text())
    except json.JSONDecodeError:
        return None
    frames = sorted((int(k), v["location"]) for k, v in info.items())
    return [(float(l[0]), float(l[1]), float(l[2])) for _, l in frames] or None


def parse_route(xml_path: Path) -> Tuple[ET.Element, List[Position], List[dict]]:
    route = ET.parse(xml_path).getroot().find(".//route")
    gt = [(float(p.attrib["x"]), float(p.attrib["y"]), float(p.attrib["z"])) for p in route.find("waypoints").findall("position")]
    instructions = [
        {"trigger": dict(e.find("trigger").attrib) if e.find("trigger") is not None else {},
         "expected_behavior": dict(e.find("expected_behavior").attrib) if e.find("expected_behavior") is not None else {}}
        for e in route.find("instructions").findall("instruction")
    ]
    return route, gt, instructions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("output_dir", type=Path, help="output directory of tools/run_distributed_eval.py")
    parser.add_argument("--routes-dir", type=Path, default=REPO_ROOT / "benchmark" / "data" / "carla_f")
    args = parser.parse_args()

    maps = CarlaMaps()
    passed, total, errors, missing = Counter(), Counter(), [], []
    for xml_path in sorted(args.routes_dir.glob("*.xml")):
        name = xml_path.stem
        traj = load_trajectory(args.output_dir / "viz" / name)
        if not result_is_complete(args.output_dir / "res" / f"{name}_res.json") or traj is None:
            missing.append(name)
            continue
        route, gt, instructions = parse_route(xml_path)
        for r in instruction_compliance(gt, traj, instructions, maps.get(route.attrib["town"])):
            total[r["category"]] += 1
            passed[r["category"]] += int(r["passed"])
        err = speed_error(route, traj)
        if err is not None:
            errors.append(err)

    metrics = {f"NCR {c}": 100.0 * passed[c] / total[c] for c in CATEGORIES.values() if total[c]}
    metrics["NCR Avg"] = 100.0 * sum(passed.values()) / max(1, sum(total.values()))
    metrics["Speed Error"] = sum(errors) / len(errors) if errors else float("nan")
    metrics["evaluated_routes"] = len(list(args.routes_dir.glob("*.xml"))) - len(missing)
    if missing:
        print(f"Warning: {len(missing)} routes without a complete result: {', '.join(missing)}")
    for key, value in metrics.items():
        print(f"{key:>22}: {value:.2f}" if isinstance(value, float) else f"{key:>22}: {value}")
    with open(args.output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
