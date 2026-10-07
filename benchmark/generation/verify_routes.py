#!/usr/bin/env python3
"""Offline validation for benchmark route XMLs.

The leaderboard evaluator densifies every adjacent pair of XML waypoints with
``GlobalRoutePlanner.trace_route()``. This script runs the same call and flags
routes whose planner path expands far beyond the XML polyline (a hidden
detour), before running full CARLA rollouts.

    python benchmark/generation/verify_routes.py benchmark/data/generated/carla_f
"""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple

from opendrive import CarlaMapCache
from planner_route_tools import (
    PlannerCache,
    analyze_route_positions,
    iter_xml_files,
    route_has_pathology,
)


Position3D = Tuple[float, float, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify benchmark route XMLs with GlobalRoutePlanner.trace_route()."
    )
    parser.add_argument(
        "input_path",
        type=Path,
        help="One XML file or a directory of XML files to analyze.",
    )
    parser.add_argument(
        "--compare-dir",
        type=Path,
        default=None,
        help="Optional second directory to compare against the primary input.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to write the full report JSON.",
    )
    parser.add_argument(
        "--xodr-root",
        type=Path,
        action="append",
        default=None,
        help="Optional OpenDRIVE root. Can be supplied multiple times.",
    )
    return parser.parse_args()


def _load_route(xml_path: Path) -> Tuple[str, List[Position3D]]:
    root = ET.parse(xml_path).getroot()
    route = next(root.iter("route"))
    town = route.attrib["town"]
    positions = [
        (
            float(position.attrib["x"]),
            float(position.attrib["y"]),
            float(position.attrib["z"]),
        )
        for position in route.find("waypoints").iter("position")
    ]
    return town, positions


def _route_report_dict(xml_path: Path, planner_cache: PlannerCache) -> Dict[str, object]:
    town, positions = _load_route(xml_path)
    report = analyze_route_positions(town, positions, planner_cache)
    failing_segments = list(report.failing_segments)
    worst_segment = max(
        report.segments,
        key=lambda segment: (
            segment.planner_length_m - segment.euclidean_length_m,
            segment.expansion_ratio,
            segment.planner_point_count,
        ),
        default=None,
    )

    return {
        "xml_path": str(xml_path),
        "town": town,
        "raw_length_m": round(report.raw_length_m, 3),
        "planner_length_m": round(report.planner_length_m, 3),
        "planner_to_raw_ratio": (
            round(report.planner_length_m / report.raw_length_m, 3)
            if report.raw_length_m > 1e-6
            else None
        ),
        "pathological": route_has_pathology(report),
        "failing_segment_count": len(failing_segments),
        "failing_segments": [
            {
                "index": segment.index,
                "start": [round(v, 3) for v in segment.start],
                "end": [round(v, 3) for v in segment.end],
                "euclidean_length_m": round(segment.euclidean_length_m, 3),
                "planner_length_m": round(segment.planner_length_m, 3),
                "planner_point_count": segment.planner_point_count,
                "expansion_ratio": round(segment.expansion_ratio, 3),
            }
            for segment in failing_segments
        ],
        "worst_segment": None
        if worst_segment is None
        else {
            "index": worst_segment.index,
            "start": [round(v, 3) for v in worst_segment.start],
            "end": [round(v, 3) for v in worst_segment.end],
            "euclidean_length_m": round(worst_segment.euclidean_length_m, 3),
            "planner_length_m": round(worst_segment.planner_length_m, 3),
            "planner_point_count": worst_segment.planner_point_count,
            "expansion_ratio": round(worst_segment.expansion_ratio, 3),
        },
    }


def _analyze_collection(path: Path, planner_cache: PlannerCache) -> Dict[str, Dict[str, object]]:
    reports: Dict[str, Dict[str, object]] = {}
    for xml_path in iter_xml_files(path):
        reports[xml_path.stem] = _route_report_dict(xml_path, planner_cache)
    return reports


def _summarize_collection(name: str, reports: Dict[str, Dict[str, object]]) -> None:
    total = len(reports)
    failures = [route for route, report in reports.items() if report["pathological"]]
    print(f"{name}: {len(failures)}/{total} pathological routes")
    if not failures:
        return

    ranked = sorted(
        failures,
        key=lambda route: (
            reports[route]["worst_segment"]["planner_length_m"]
            - reports[route]["worst_segment"]["euclidean_length_m"],
            reports[route]["worst_segment"]["expansion_ratio"],
        ),
        reverse=True,
    )
    for route in ranked[:15]:
        worst = reports[route]["worst_segment"]
        print(
            "  "
            f"{route}: raw={reports[route]['raw_length_m']:.3f}m "
            f"planner={reports[route]['planner_length_m']:.3f}m "
            f"failing_segments={reports[route]['failing_segment_count']} "
            f"worst_seg={worst['index']} "
            f"seg_raw={worst['euclidean_length_m']:.3f}m "
            f"seg_planner={worst['planner_length_m']:.3f}m "
            f"ratio={worst['expansion_ratio']:.2f}"
        )


def _compare_reports(
    baseline: Dict[str, Dict[str, object]],
    candidate: Dict[str, Dict[str, object]],
) -> Dict[str, object]:
    baseline_failures = {route for route, report in baseline.items() if report["pathological"]}
    candidate_failures = {route for route, report in candidate.items() if report["pathological"]}
    common_routes = set(baseline) & set(candidate)
    return {
        "baseline_route_count": len(baseline),
        "candidate_route_count": len(candidate),
        "common_route_count": len(common_routes),
        "baseline_failure_count": len(baseline_failures),
        "candidate_failure_count": len(candidate_failures),
        "fixed_routes": sorted(baseline_failures - candidate_failures),
        "regressions": sorted(candidate_failures - baseline_failures),
        "remaining_failures": sorted(baseline_failures & candidate_failures),
        "missing_in_candidate": sorted(set(baseline) - set(candidate)),
        "new_in_candidate": sorted(set(candidate) - set(baseline)),
    }


def main() -> None:
    args = parse_args()
    xodr_roots = (
        None if args.xodr_root is None else [path.expanduser().resolve() for path in args.xodr_root]
    )
    map_cache = CarlaMapCache(xodr_search_roots=xodr_roots)
    planner_cache = PlannerCache(map_cache)

    primary_reports = _analyze_collection(args.input_path.resolve(), planner_cache)
    _summarize_collection(str(args.input_path), primary_reports)

    output: Dict[str, object] = {
        "primary": {
            "path": str(args.input_path.resolve()),
            "reports": primary_reports,
        }
    }

    if args.compare_dir is not None:
        compare_reports = _analyze_collection(args.compare_dir.resolve(), planner_cache)
        _summarize_collection(str(args.compare_dir), compare_reports)
        comparison = _compare_reports(primary_reports, compare_reports)
        output["comparison"] = {
            "baseline_path": str(args.input_path.resolve()),
            "candidate_path": str(args.compare_dir.resolve()),
            **comparison,
        }
        output["candidate"] = {
            "path": str(args.compare_dir.resolve()),
            "reports": compare_reports,
        }
        print("Comparison:")
        print(
            "  "
            f"fixed={len(comparison['fixed_routes'])} "
            f"regressions={len(comparison['regressions'])} "
            f"remaining={len(comparison['remaining_failures'])} "
            f"missing_in_candidate={len(comparison['missing_in_candidate'])} "
            f"new_in_candidate={len(comparison['new_in_candidate'])}"
        )

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(output, indent=2), encoding="utf-8")
        print(f"Wrote JSON report: {args.output_json}")


if __name__ == "__main__":
    main()
