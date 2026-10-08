"""Closed-loop metrics of Bench2Drive, B2D-C and B2D-Adv.

* Driving Score (DS) and Success Rate (SR) over a fixed number of routes
  (missing or crashed routes count as zero), following Bench2Drive.
* Infraction breakdown used for B2D-C (Sec. 3.2): collisions (layout,
  pedestrian, vehicle), traffic violations (red light, stop sign) and
  out-of-route.
* Driving Efficiency and Comfortness (Driving Smoothness) of Bench2Drive,
  computed with the official ``third_party/bench2drive/tools`` code.

    python benchmark/metrics/bench2drive_metrics.py work_dirs/eval/bench2drive --num-routes 220
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "third_party" / "bench2drive" / "tools"))
sys.path.insert(0, str(REPO_ROOT / "tools"))
from efficiency_smoothness_benchmark import seg_compute_comfort_metric  # noqa: E402
from run_distributed_eval import result_is_complete  # noqa: E402

COLLISIONS = ("collisions_layout", "collisions_pedestrian", "collisions_vehicle")
TRAFFIC_VIOLATIONS = ("red_light", "stop_infraction")
OUT_OF_ROUTE = ("outside_route_lanes",)
_PERCENT_RE = re.compile(r"\b\d+\.?\d*%")
_METRIC_FIELDS = ("acceleration", "angular_velocity", "forward_vector", "right_vector", "location", "rotation")


def route_success(record: dict) -> bool:
    """Completed without any infraction other than minimum speed."""
    if record.get("status") not in ("Completed", "Perfect"):
        return False
    return not any(v for k, v in (record.get("infractions") or {}).items() if k != "min_speed_infractions")


def driving_efficiency(records) -> float:
    """Bench2Drive Driving Efficiency: mean over routes of the ego/traffic
    speed percentages logged by the minimum-speed criterion."""
    values = []
    for record in records:
        percentages = []
        for line in record.get("infractions", {}).get("min_speed_infractions", []):
            match = _PERCENT_RE.search(line)
            if match is not None and float(match.group().rstrip("%")) <= 1000:
                percentages.append(float(match.group().rstrip("%")))
        if percentages:
            values.append(sum(percentages) / len(percentages))
    return float(np.mean(values)) if values else float("nan")


def comfortness(viz_dir: Path, route_names) -> float:
    """Bench2Drive Driving Smoothness (%) from the per-tick ego state of each
    route (latest attempt only)."""
    scores = []
    for name in route_names:
        runs = sorted((viz_dir / name).glob("*/metric_info.json"))
        if not runs:
            continue
        try:
            info = json.loads(runs[-1].read_text())
        except json.JSONDecodeError:
            continue
        frames = sorted(info, key=int)
        arrays = {f: np.array([info[k][f] for k in frames]) for f in _METRIC_FIELDS}
        scores.append(float(seg_compute_comfort_metric(**arrays)))
    return 100.0 * float(np.mean(scores)) if scores else float("nan")


def evaluate(output_dir: Path, num_routes: int) -> dict:
    complete = sorted(p for p in (output_dir / "res").glob("*_res.json") if result_is_complete(p))
    records = [r for p in complete for r in json.loads(p.read_text())["_checkpoint"]["records"]]
    scores = [r["scores"]["score_composed"] for r in records]

    def count(keys):
        return sum(len((r.get("infractions") or {}).get(k) or []) for r in records for k in keys)

    return {
        "driving_score": sum(scores) / num_routes,
        "success_rate": 100.0 * sum(route_success(r) for r in records) / num_routes,
        "collisions": count(COLLISIONS),
        "traffic_violations": count(TRAFFIC_VIOLATIONS),
        "out_of_route": count(OUT_OF_ROUTE),
        "efficiency": driving_efficiency(records),
        "comfortness": comfortness(output_dir / "viz", [p.name[: -len("_res.json")] for p in complete]),
        "evaluated_routes": len(records),
        "num_routes": num_routes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("output_dir", type=Path, help="output directory of tools/run_distributed_eval.py")
    parser.add_argument("--num-routes", type=int, required=True, help="routes of the benchmark (denominator)")
    args = parser.parse_args()
    metrics = evaluate(args.output_dir, args.num_routes)
    if metrics["evaluated_routes"] < args.num_routes:
        print(f"Warning: {args.num_routes - metrics['evaluated_routes']} routes are missing and count as zero.")
    for key, value in metrics.items():
        print(f"{key:>20}: {value:.2f}" if isinstance(value, float) else f"{key:>20}: {value}")
    with open(args.output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
