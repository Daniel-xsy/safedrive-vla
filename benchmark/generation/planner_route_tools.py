"""
Planner-based validation and export helpers for generated routes.

The leaderboard evaluator does not use the XML polyline directly: it
re-interpolates every adjacent pair of XML positions with
``GlobalRoutePlanner.trace_route``. These helpers mirror that call so route XML
can be checked offline before running expensive CARLA evaluations, and so the
generator only exports planner-safe anchors.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from opendrive import CarlaMapCache

try:
    import carla
except ImportError as exc:
    carla = None
    CARLA_IMPORT_ERROR = exc
else:
    CARLA_IMPORT_ERROR = None

try:
    from agents.navigation.global_route_planner import GlobalRoutePlanner
except ImportError as exc:
    GlobalRoutePlanner = None
    GRP_IMPORT_ERROR = exc
else:
    GRP_IMPORT_ERROR = None


Position3D = Tuple[float, float, float]

DEFAULT_HOP_RESOLUTION = 1.0
DEFAULT_MAX_LENGTH_RATIO = 2.5
DEFAULT_MAX_EXTRA_LENGTH_M = 20.0
DEFAULT_MAX_TRACE_POINTS = 400
DEFAULT_TOTAL_LENGTH_RATIO = 1.5
DEFAULT_TOTAL_EXTRA_LENGTH_M = 20.0


@dataclass(frozen=True)
class PlannerSegmentReport:
    """Planner diagnostics for one adjacent route segment."""

    index: int
    start: Position3D
    end: Position3D
    euclidean_length_m: float
    planner_length_m: float
    planner_point_count: int

    @property
    def expansion_ratio(self) -> float:
        if self.euclidean_length_m <= 1e-6:
            return float("inf") if self.planner_length_m > 1e-6 else 1.0
        return self.planner_length_m / self.euclidean_length_m


@dataclass(frozen=True)
class PlannerRouteReport:
    """Planner diagnostics for a full XML route."""

    raw_length_m: float
    planner_length_m: float
    segments: Tuple[PlannerSegmentReport, ...]

    @property
    def failing_segments(self) -> Tuple[PlannerSegmentReport, ...]:
        return tuple(segment for segment in self.segments if is_pathological_segment(segment))


def _require_carla_planner() -> None:
    if carla is None:
        raise RuntimeError("CARLA Python API is required.") from CARLA_IMPORT_ERROR
    if GlobalRoutePlanner is None:
        raise RuntimeError(
            "CARLA's GlobalRoutePlanner is required; add $CARLA_ROOT/PythonAPI/carla "
            "to PYTHONPATH."
        ) from GRP_IMPORT_ERROR


def _location_from_position(position: Position3D) -> "carla.Location":
    return carla.Location(x=float(position[0]), y=float(position[1]), z=float(position[2]))


def _distance_positions(a: Position3D, b: Position3D) -> float:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    dz = a[2] - b[2]
    return (dx * dx + dy * dy + dz * dz) ** 0.5


def _trace_length(trace: Sequence[Tuple["carla.Waypoint", object]]) -> float:
    if len(trace) < 2:
        return 0.0
    total = 0.0
    for idx in range(len(trace) - 1):
        total += trace[idx][0].transform.location.distance(trace[idx + 1][0].transform.location)
    return total


def is_pathological_pair(
    planner_length_m: float,
    baseline_length_m: float,
    planner_point_count: int,
    *,
    max_length_ratio: float = DEFAULT_MAX_LENGTH_RATIO,
    max_extra_length_m: float = DEFAULT_MAX_EXTRA_LENGTH_M,
    max_trace_points: int = DEFAULT_MAX_TRACE_POINTS,
) -> bool:
    """Return whether a planner segment expands far beyond its baseline."""
    if planner_point_count > max_trace_points:
        return True
    limit = max(baseline_length_m * max_length_ratio, baseline_length_m + max_extra_length_m)
    return planner_length_m > limit


def is_pathological_segment(
    segment: PlannerSegmentReport,
    *,
    max_length_ratio: float = DEFAULT_MAX_LENGTH_RATIO,
    max_extra_length_m: float = DEFAULT_MAX_EXTRA_LENGTH_M,
    max_trace_points: int = DEFAULT_MAX_TRACE_POINTS,
) -> bool:
    return is_pathological_pair(
        segment.planner_length_m,
        segment.euclidean_length_m,
        segment.planner_point_count,
        max_length_ratio=max_length_ratio,
        max_extra_length_m=max_extra_length_m,
        max_trace_points=max_trace_points,
    )


class PlannerCache:
    """Cache ``GlobalRoutePlanner`` instances per town."""

    def __init__(
        self,
        map_cache: CarlaMapCache,
        hop_resolution: float = DEFAULT_HOP_RESOLUTION,
    ) -> None:
        _require_carla_planner()
        self._map_cache = map_cache
        self._hop_resolution = hop_resolution
        self._planners: Dict[str, "GlobalRoutePlanner"] = {}

    def get_planner(self, town: str) -> "GlobalRoutePlanner":
        planner = self._planners.get(town)
        if planner is not None:
            return planner
        planner = GlobalRoutePlanner(self._map_cache.get_map(town), self._hop_resolution)
        self._planners[town] = planner
        return planner


def analyze_route_positions(
    town: str,
    positions: Sequence[Position3D],
    planner_cache: PlannerCache,
) -> PlannerRouteReport:
    """Analyze a route exactly as the evaluator re-interpolates XML positions."""
    planner = planner_cache.get_planner(town)
    raw_length_m = 0.0
    planner_length_m = 0.0
    segments: List[PlannerSegmentReport] = []

    for idx in range(len(positions) - 1):
        start = positions[idx]
        end = positions[idx + 1]
        euclidean_length_m = _distance_positions(start, end)
        raw_length_m += euclidean_length_m
        trace = planner.trace_route(_location_from_position(start), _location_from_position(end))
        planner_segment_length_m = _trace_length(trace)
        planner_length_m += planner_segment_length_m
        segments.append(
            PlannerSegmentReport(
                index=idx,
                start=start,
                end=end,
                euclidean_length_m=euclidean_length_m,
                planner_length_m=planner_segment_length_m,
                planner_point_count=len(trace),
            )
        )

    return PlannerRouteReport(
        raw_length_m=raw_length_m,
        planner_length_m=planner_length_m,
        segments=tuple(segments),
    )


def route_has_pathology(
    report: PlannerRouteReport,
    *,
    total_length_ratio: float = DEFAULT_TOTAL_LENGTH_RATIO,
    total_extra_length_m: float = DEFAULT_TOTAL_EXTRA_LENGTH_M,
) -> bool:
    if report.failing_segments:
        return True
    total_limit = max(
        report.raw_length_m * total_length_ratio,
        report.raw_length_m + total_extra_length_m,
    )
    return report.planner_length_m > total_limit


def _compute_cumulative_lengths(positions: Sequence[Position3D]) -> List[float]:
    cumulative = [0.0]
    for idx in range(1, len(positions)):
        cumulative.append(cumulative[-1] + _distance_positions(positions[idx - 1], positions[idx]))
    return cumulative


def _terminal_junction_start_index(junction_flags: Sequence[bool]) -> Optional[int]:
    """Return the first index of a terminal suffix of junction waypoints."""
    if not junction_flags or not junction_flags[-1]:
        return None

    index = len(junction_flags) - 1
    while index >= 0 and junction_flags[index]:
        index -= 1
    start_index = index + 1
    if start_index <= 0:
        return None
    return start_index


def _validate_anchor_pair(
    planner: "GlobalRoutePlanner",
    positions: Sequence[Position3D],
    cumulative_lengths: Sequence[float],
    start_index: int,
    end_index: int,
    *,
    max_length_ratio: float = DEFAULT_MAX_LENGTH_RATIO,
    max_extra_length_m: float = DEFAULT_MAX_EXTRA_LENGTH_M,
    max_trace_points: int = DEFAULT_MAX_TRACE_POINTS,
) -> bool:
    if end_index <= start_index:
        return True
    start = positions[start_index]
    end = positions[end_index]
    expected_length_m = cumulative_lengths[end_index] - cumulative_lengths[start_index]
    trace = planner.trace_route(_location_from_position(start), _location_from_position(end))
    planner_length_m = _trace_length(trace)
    return not is_pathological_pair(
        planner_length_m,
        expected_length_m,
        len(trace),
        max_length_ratio=max_length_ratio,
        max_extra_length_m=max_extra_length_m,
        max_trace_points=max_trace_points,
    )


def _find_best_split_index(
    planner: "GlobalRoutePlanner",
    positions: Sequence[Position3D],
    cumulative_lengths: Sequence[float],
    junction_flags: Sequence[bool],
    start_index: int,
    end_index: int,
    *,
    max_length_ratio: float = DEFAULT_MAX_LENGTH_RATIO,
    max_extra_length_m: float = DEFAULT_MAX_EXTRA_LENGTH_M,
    max_trace_points: int = DEFAULT_MAX_TRACE_POINTS,
) -> Optional[int]:
    if end_index - start_index <= 1:
        return None

    internal = list(range(start_index + 1, end_index))
    preferred = [idx for idx in internal if not junction_flags[idx]]
    candidates = preferred + [idx for idx in internal if junction_flags[idx]]
    best_index: Optional[int] = None
    best_score: Optional[Tuple[int, float, int]] = None

    for candidate in candidates:
        left_ok = _validate_anchor_pair(
            planner,
            positions,
            cumulative_lengths,
            start_index,
            candidate,
            max_length_ratio=max_length_ratio,
            max_extra_length_m=max_extra_length_m,
            max_trace_points=max_trace_points,
        )
        right_ok = _validate_anchor_pair(
            planner,
            positions,
            cumulative_lengths,
            candidate,
            end_index,
            max_length_ratio=max_length_ratio,
            max_extra_length_m=max_extra_length_m,
            max_trace_points=max_trace_points,
        )
        valid_count = int(left_ok) + int(right_ok)
        midpoint_bias = abs((candidate - start_index) - (end_index - candidate))
        score = (-valid_count, midpoint_bias, candidate)
        if best_score is None or score < best_score:
            best_score = score
            best_index = candidate
        if left_ok and right_ok and candidate in preferred:
            return candidate

    return best_index


def build_planner_safe_export_positions(
    town: str,
    route_waypoints: Sequence["carla.Waypoint"],
    planner_cache: PlannerCache,
    *,
    max_length_ratio: float = DEFAULT_MAX_LENGTH_RATIO,
    max_extra_length_m: float = DEFAULT_MAX_EXTRA_LENGTH_M,
    max_trace_points: int = DEFAULT_MAX_TRACE_POINTS,
) -> Tuple[Position3D, ...]:
    """Convert a CARLA waypoint chain into evaluator-safe XML anchor positions.

    Waypoints inside junctions are dropped first: neighbouring connector lanes
    are close in space, so ``trace_route`` can snap such a pair onto a
    different branch and expand it into a long detour. Pairs that still
    expand are split with intermediate waypoints of the original chain. If a
    pair still fails, a terminal junction tail is cut off; otherwise a
    ``ValueError`` is raised.
    """
    if not route_waypoints:
        return tuple()

    positions = [
        (
            float(waypoint.transform.location.x),
            float(waypoint.transform.location.y),
            float(waypoint.transform.location.z),
        )
        for waypoint in route_waypoints
    ]
    if len(positions) <= 2:
        return tuple(positions)

    cumulative_lengths = _compute_cumulative_lengths(positions)
    junction_flags = [bool(waypoint.is_junction) for waypoint in route_waypoints]
    planner = planner_cache.get_planner(town)

    anchor_indices: List[int] = [0]
    for idx in range(1, len(route_waypoints) - 1):
        if not junction_flags[idx]:
            anchor_indices.append(idx)
    anchor_indices.append(len(route_waypoints) - 1)

    # Refine any remaining bad anchor pairs by splitting them using the original
    # CARLA waypoint chain as the source of additional anchors.
    while True:
        changed = False
        refined_indices: List[int] = [anchor_indices[0]]
        for idx in range(len(anchor_indices) - 1):
            start_index = anchor_indices[idx]
            end_index = anchor_indices[idx + 1]
            if _validate_anchor_pair(
                planner,
                positions,
                cumulative_lengths,
                start_index,
                end_index,
                max_length_ratio=max_length_ratio,
                max_extra_length_m=max_extra_length_m,
                max_trace_points=max_trace_points,
            ):
                refined_indices.append(end_index)
                continue

            split_index = _find_best_split_index(
                planner,
                positions,
                cumulative_lengths,
                junction_flags,
                start_index,
                end_index,
                max_length_ratio=max_length_ratio,
                max_extra_length_m=max_extra_length_m,
                max_trace_points=max_trace_points,
            )
            if split_index is None or split_index in (start_index, end_index):
                refined_indices.append(end_index)
                continue
            refined_indices.extend([split_index, end_index])
            changed = True

        deduped_indices: List[int] = []
        for anchor_index in refined_indices:
            if not deduped_indices or anchor_index != deduped_indices[-1]:
                deduped_indices.append(anchor_index)
        anchor_indices = deduped_indices
        if not changed:
            break

    while True:
        failing_pair: Optional[Tuple[int, int]] = None
        for idx in range(len(anchor_indices) - 1):
            start_index = anchor_indices[idx]
            end_index = anchor_indices[idx + 1]
            if _validate_anchor_pair(
                planner,
                positions,
                cumulative_lengths,
                start_index,
                end_index,
                max_length_ratio=max_length_ratio,
                max_extra_length_m=max_extra_length_m,
                max_trace_points=max_trace_points,
            ):
                continue
            failing_pair = (start_index, end_index)
            break

        if failing_pair is None:
            break

        terminal_junction_start = _terminal_junction_start_index(junction_flags)
        if terminal_junction_start is not None and failing_pair[1] >= terminal_junction_start:
            anchor_indices = [idx for idx in anchor_indices if idx < terminal_junction_start]
            if not anchor_indices or anchor_indices[-1] != terminal_junction_start - 1:
                anchor_indices.append(terminal_junction_start - 1)
            if len(anchor_indices) >= 2:
                continue

        raise ValueError(
            "Could not construct planner-safe export anchors for route in "
            f"{town}: segment {failing_pair[0]}->{failing_pair[1]} remains pathological."
        )

    return tuple(positions[idx] for idx in anchor_indices)


def iter_xml_files(path: Path) -> Iterable[Path]:
    """Yield XML files from a single file or recursively from a directory."""
    resolved = Path(path)
    if resolved.is_file():
        yield resolved
        return
    for xml_path in sorted(resolved.rglob("*.xml")):
        if xml_path.is_file():
            yield xml_path
