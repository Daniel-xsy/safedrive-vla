"""
Route reconstruction from CARLA waypoints.

This module rebuilds driving routes from a start waypoint by following road
successors, executing lane changes, and tracing junction branches, so that
the ground-truth route physically performs the sampled meta-command. It also
provides actionability sampling and trigger selection on rebuilt routes and
XML helpers for emitting waypoint and scenario elements.

Key types:
    * ``RebuiltRoute`` — an immutable snapshot of a rebuilt route with
      positions, cumulative distances, and the underlying CARLA waypoints.
    * ``ActionSuffixResult`` — the waypoint list produced by executing a
      single navigation action (turn / lane change / lane follow).
"""

import math
import random
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from geometry import (
    _can_change_lane,
    _compute_turn_category,
    _is_same_direction_lane,
    _normalize_yaw_delta_deg,
)
from actionability import (
    _build_actionable_navigation_categories,
)
from instructions import (
    OPTIONAL_LANE_FOLLOW_PROBABILITY,
)

try:
    import carla
except ImportError:
    carla = None


# ---------------------------------------------------------------------------
# Type aliases & constants
# ---------------------------------------------------------------------------

Position3D = Tuple[float, float, float]

# Distance after a lane hop before the lane change is considered "settled".
LANE_CHANGE_SETTLE_DISTANCE_M = 15.0


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RebuiltRoute:
    """Immutable snapshot of a route rebuilt from CARLA waypoints.

    Attributes:
        waypoints: CARLA waypoint objects along the route.
        positions: Corresponding ``(x, y, z)`` tuples.
        cumulative_distances_m: Cumulative distance from the first position.
        total_length_m: Total route length in metres.
    """
    waypoints: Tuple["carla.Waypoint", ...]
    positions: Tuple[Position3D, ...]
    cumulative_distances_m: Tuple[float, ...]
    total_length_m: float


@dataclass
class ActionSuffixResult:
    """Result of executing a navigation action from a trigger waypoint.

    Attributes:
        waypoints: The full waypoint list (trigger → action → post-action follow).
        action_end_index: Index in *waypoints* where the action completes.
    """
    waypoints: List["carla.Waypoint"]
    action_end_index: int


@dataclass(frozen=True)
class RebuiltActionabilitySample:
    """Feasible navigation actions at a single point on a rebuilt route.

    ``score`` is the actionability score, i.e. the number of feasible actions
    other than ``lane_follow``.
    """
    index: int
    distance_m: float
    actions: Tuple[str, ...]
    scored_actions: Tuple[str, ...]
    score: int
    is_junction: bool


@dataclass(frozen=True)
class RebuiltTrigger:
    """A selected trigger point on a rebuilt route.

    Attributes:
        index: Position index in the rebuilt route.
        distance_m: Cumulative distance from route start.
        actions / scored_actions: Feasible actions at this point.
        is_junction: Whether the point is inside a junction.
        phrasing_mode: ``"approach"`` or ``"at_junction"``.
    """
    index: int
    distance_m: float
    actions: Tuple[str, ...]
    scored_actions: Tuple[str, ...]
    is_junction: bool
    phrasing_mode: str


# ---------------------------------------------------------------------------
# Low-level waypoint helpers
# ---------------------------------------------------------------------------

def _waypoint_position(waypoint: "carla.Waypoint") -> Position3D:
    """Extract an ``(x, y, z)`` tuple from a CARLA waypoint."""
    location = waypoint.transform.location
    return (float(location.x), float(location.y), float(location.z))


def _distance_positions(a: Position3D, b: Position3D) -> float:
    """Euclidean distance between two 3-D position tuples."""
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    dz = a[2] - b[2]
    return (dx * dx + dy * dy + dz * dz) ** 0.5


def _distance_waypoints(a: "carla.Waypoint", b: "carla.Waypoint") -> float:
    """Distance between two CARLA waypoints using their built-in method."""
    return a.transform.location.distance(b.transform.location)


def _append_waypoint_if_new(
    route_waypoints: List["carla.Waypoint"],
    waypoint: "carla.Waypoint",
) -> None:
    """Append *waypoint* to the list only if it is not a duplicate of the last entry."""
    if not route_waypoints:
        route_waypoints.append(waypoint)
        return
    prev = route_waypoints[-1]
    if _distance_waypoints(prev, waypoint) <= 1e-3:
        return
    route_waypoints.append(waypoint)


def _truncate_waypoints_before_terminal_junction(
    route_waypoints: Sequence["carla.Waypoint"],
) -> List["carla.Waypoint"]:
    """Trim a route that ends inside a junction back to the junction entrance.

    This is intended for terminal follow tails with no further chained
    instruction. If the route never enters a junction, or starts inside one,
    the input is returned unchanged.
    """
    if not route_waypoints or not route_waypoints[-1].is_junction:
        return list(route_waypoints)

    first_junction_index = next(
        (idx for idx, waypoint in enumerate(route_waypoints) if waypoint.is_junction),
        None,
    )
    if first_junction_index is None or first_junction_index <= 0:
        return list(route_waypoints)
    return list(route_waypoints[:first_junction_index])


# ---------------------------------------------------------------------------
# Road-following scoring
# ---------------------------------------------------------------------------

def _follow_score(
    current_waypoint: "carla.Waypoint",
    candidate: "carla.Waypoint",
) -> Tuple[float, int, int, float]:
    """Score a candidate successor for road-following continuity.

    Lower is better.  The tuple components are:
        1. Absolute yaw delta (degrees).
        2. 0 if same road, 1 otherwise.
        3. Absolute lane-id delta.
        4. Distance between waypoints.
    """
    yaw_delta = abs(
        _normalize_yaw_delta_deg(
            candidate.transform.rotation.yaw - current_waypoint.transform.rotation.yaw
        )
    )
    same_road_penalty = 0 if candidate.road_id == current_waypoint.road_id else 1
    lane_delta = abs(abs(candidate.lane_id) - abs(current_waypoint.lane_id))
    return (
        yaw_delta,
        same_road_penalty,
        lane_delta,
        _distance_waypoints(current_waypoint, candidate),
    )


def _select_follow_road_successor(
    current_waypoint: "carla.Waypoint",
    candidates: Sequence["carla.Waypoint"],
) -> "carla.Waypoint":
    """Pick the best road-following successor from *candidates*."""
    if not candidates:
        raise ValueError("Cannot select follow-road successor from empty candidate list.")
    return min(candidates, key=lambda wp: _follow_score(current_waypoint, wp))


# ---------------------------------------------------------------------------
# Junction branch tracing
# ---------------------------------------------------------------------------

def _trace_candidate_turn(
    approach_waypoint: "carla.Waypoint",
    candidate: "carla.Waypoint",
    step_m: float,
    max_scan_distance_m: float = 40.0,
) -> str:
    """Follow *candidate* through a junction and classify the resulting turn.

    Returns ``"turn_left"``, ``"turn_right"``, or ``"turn_straight"``.
    """
    traversed = _distance_waypoints(approach_waypoint, candidate)
    branch_waypoint = candidate

    while traversed < max_scan_distance_m:
        next_candidates = branch_waypoint.next(step_m)
        if not next_candidates:
            break
        next_waypoint = _select_follow_road_successor(branch_waypoint, next_candidates)
        traversed += _distance_waypoints(branch_waypoint, next_waypoint)
        branch_waypoint = next_waypoint
        if not branch_waypoint.is_junction and traversed >= step_m:
            break

    return _compute_turn_category(approach_waypoint, branch_waypoint)


def _select_turn_successor(
    approach_waypoint: "carla.Waypoint",
    current_waypoint: "carla.Waypoint",
    candidates: Sequence["carla.Waypoint"],
    desired_action: str,
    step_m: float,
) -> Optional["carla.Waypoint"]:
    """Find a candidate whose traced turn matches *desired_action*.

    Returns ``None`` if no candidate produces the desired turn category.
    """
    matching = [
        candidate
        for candidate in candidates
        if _trace_candidate_turn(approach_waypoint, candidate, step_m) == desired_action
    ]
    if not matching:
        return None
    return min(matching, key=lambda wp: _follow_score(current_waypoint, wp))


# ---------------------------------------------------------------------------
# Cumulative distances & route finalisation
# ---------------------------------------------------------------------------

def _compute_cumulative_distances(positions: Sequence[Position3D]) -> Tuple[float, ...]:
    """Build a tuple of cumulative distances from the first position."""
    cumulative = [0.0]
    for idx in range(1, len(positions)):
        cumulative.append(cumulative[-1] + _distance_positions(positions[idx - 1], positions[idx]))
    return tuple(cumulative)


def _finalize_route(route_waypoints: Sequence["carla.Waypoint"]) -> RebuiltRoute:
    """Wrap raw waypoints into a ``RebuiltRoute``."""
    positions = [_waypoint_position(wp) for wp in route_waypoints]
    cumulative_distances = _compute_cumulative_distances(positions)
    total_length = cumulative_distances[-1] if cumulative_distances else 0.0
    return RebuiltRoute(
        waypoints=tuple(route_waypoints),
        positions=tuple(positions),
        cumulative_distances_m=cumulative_distances,
        total_length_m=total_length,
    )


# ---------------------------------------------------------------------------
# Route reconstruction: follow / turn / lane-change / dispatcher
# ---------------------------------------------------------------------------

def _rebuild_follow_route(
    start_waypoint: "carla.Waypoint",
    max_distance_m: float,
    step_m: float,
) -> RebuiltRoute:
    """Rebuild a route by following the road forward from *start_waypoint*."""
    route_waypoints = [start_waypoint]
    current_waypoint = start_waypoint
    traveled = 0.0

    while traveled + 1e-6 < max_distance_m:
        next_candidates = current_waypoint.next(step_m)
        if not next_candidates:
            break
        next_waypoint = _select_follow_road_successor(current_waypoint, next_candidates)
        segment_distance = _distance_waypoints(current_waypoint, next_waypoint)
        if segment_distance <= 1e-3:
            break
        traveled += segment_distance
        _append_waypoint_if_new(route_waypoints, next_waypoint)
        current_waypoint = next_waypoint

    return _finalize_route(route_waypoints)


def _rebuild_turn_suffix(
    start_waypoint: "carla.Waypoint",
    desired_action: str,
    max_distance_m: float,
    step_m: float,
) -> Optional[ActionSuffixResult]:
    """Trace a turn action through a junction, returning ``None`` on failure.

    The algorithm:
        1. Follow the road until a junction branch point is reached.
        2. Select the branch that matches *desired_action*.
        3. Continue through the junction until an exit waypoint is found.
        4. Validate the overall turn category matches *desired_action*.
        5. Extend the route past the junction exit up to *max_distance_m*.
    """
    route_waypoints = [start_waypoint]
    current_waypoint = start_waypoint
    traveled = 0.0
    approach_waypoint: Optional["carla.Waypoint"] = None
    entered_junction = False
    first_exit_waypoint: Optional["carla.Waypoint"] = None
    action_end_index: Optional[int] = None
    branch_selected = False

    while traveled + 1e-6 < max_distance_m:
        next_candidates = current_waypoint.next(step_m)
        if not next_candidates:
            break

        near_junction = current_waypoint.is_junction or any(
            c.is_junction for c in next_candidates
        )
        if near_junction and approach_waypoint is None:
            approach_waypoint = current_waypoint

        if near_junction and len(next_candidates) > 1 and not branch_selected:
            selected_waypoint = _select_turn_successor(
                approach_waypoint or current_waypoint,
                current_waypoint,
                next_candidates,
                desired_action,
                step_m,
            )
            if selected_waypoint is None:
                return None
            branch_selected = True
        else:
            selected_waypoint = _select_follow_road_successor(current_waypoint, next_candidates)

        segment_distance = _distance_waypoints(current_waypoint, selected_waypoint)
        if segment_distance <= 1e-3:
            break
        if traveled + segment_distance > max_distance_m + 1e-6:
            break

        traveled += segment_distance
        previous_waypoint = current_waypoint
        current_waypoint = selected_waypoint
        _append_waypoint_if_new(route_waypoints, current_waypoint)

        if current_waypoint.is_junction:
            entered_junction = True
        if entered_junction and not current_waypoint.is_junction and previous_waypoint.is_junction:
            first_exit_waypoint = current_waypoint
            action_end_index = len(route_waypoints) - 1
            break

    if approach_waypoint is None:
        return None

    # Validate that the turn actually matches the desired action.
    validation_waypoint = first_exit_waypoint or route_waypoints[-1]
    if _compute_turn_category(approach_waypoint, validation_waypoint) != desired_action:
        return None

    if action_end_index is None:
        action_end_index = len(route_waypoints) - 1

    # Extend past the junction exit.
    while traveled + 1e-6 < max_distance_m:
        next_candidates = current_waypoint.next(step_m)
        if not next_candidates:
            break
        selected_waypoint = _select_follow_road_successor(current_waypoint, next_candidates)
        segment_distance = _distance_waypoints(current_waypoint, selected_waypoint)
        if segment_distance <= 1e-3 or traveled + segment_distance > max_distance_m + 1e-6:
            break
        traveled += segment_distance
        current_waypoint = selected_waypoint
        _append_waypoint_if_new(route_waypoints, current_waypoint)

    return ActionSuffixResult(
        waypoints=route_waypoints,
        action_end_index=action_end_index,
    )


def _rebuild_lane_change_suffix(
    start_waypoint: "carla.Waypoint",
    direction: str,
    max_distance_m: float,
    step_m: float,
    prep_distance_m: float,
) -> Optional[ActionSuffixResult]:
    """Execute a lane change, returning ``None`` if the manoeuvre is infeasible.

    The algorithm:
        1. Check lane changeability at the start.
        2. Drive forward *prep_distance_m* (preparation phase).
        3. Hop to the adjacent lane.
        4. Drive forward until settled (``LANE_CHANGE_SETTLE_DISTANCE_M``).
        5. Continue filling up to *max_distance_m*.
    """
    if not _can_change_lane(start_waypoint, direction):
        return None

    route_waypoints = [start_waypoint]
    current_waypoint = start_waypoint
    traveled = 0.0
    prep_target = min(prep_distance_m, max_distance_m)

    # Phase 1: preparation — drive straight, abort if a junction appears.
    while traveled + 1e-6 < prep_target:
        if current_waypoint.is_junction:
            return None
        next_candidates = current_waypoint.next(step_m)
        if not next_candidates or any(c.is_junction for c in next_candidates):
            return None
        next_waypoint = _select_follow_road_successor(current_waypoint, next_candidates)
        segment_distance = _distance_waypoints(current_waypoint, next_waypoint)
        if segment_distance <= 1e-3 or traveled + segment_distance > max_distance_m + 1e-6:
            break
        traveled += segment_distance
        current_waypoint = next_waypoint
        _append_waypoint_if_new(route_waypoints, current_waypoint)

    # Phase 2: lane hop.
    if not _can_change_lane(current_waypoint, direction):
        return None

    adjacent_waypoint = (
        current_waypoint.get_left_lane()
        if direction == "left"
        else current_waypoint.get_right_lane()
    )
    if adjacent_waypoint is None:
        return None
    if adjacent_waypoint.lane_type != carla.LaneType.Driving:
        return None
    if not _is_same_direction_lane(current_waypoint, adjacent_waypoint):
        return None

    lane_change_distance = _distance_waypoints(current_waypoint, adjacent_waypoint)
    if traveled + lane_change_distance > max_distance_m + 1e-6:
        return None

    traveled += lane_change_distance
    current_waypoint = adjacent_waypoint
    _append_waypoint_if_new(route_waypoints, current_waypoint)
    post_change_traveled = 0.0
    action_end_index: Optional[int] = None

    # Phase 3: post-change settlement.
    while traveled + 1e-6 < max_distance_m:
        next_candidates = current_waypoint.next(step_m)
        if not next_candidates:
            break
        next_waypoint = _select_follow_road_successor(current_waypoint, next_candidates)
        segment_distance = _distance_waypoints(current_waypoint, next_waypoint)
        if segment_distance <= 1e-3 or traveled + segment_distance > max_distance_m + 1e-6:
            break
        traveled += segment_distance
        post_change_traveled += segment_distance
        current_waypoint = next_waypoint
        _append_waypoint_if_new(route_waypoints, current_waypoint)
        if action_end_index is None and post_change_traveled >= LANE_CHANGE_SETTLE_DISTANCE_M - 1e-6:
            action_end_index = len(route_waypoints) - 1

    if action_end_index is None:
        action_end_index = len(route_waypoints) - 1

    return ActionSuffixResult(
        waypoints=route_waypoints,
        action_end_index=action_end_index,
    )


def _rebuild_action_suffix(
    start_waypoint: "carla.Waypoint",
    action: str,
    max_distance_m: float,
    step_m: float,
    lane_change_prep_m: float,
) -> Optional[ActionSuffixResult]:
    """Dispatch to the appropriate action-specific suffix builder.

    Supports ``lane_follow``, ``lane_change_left``, ``lane_change_right``,
    ``turn_left``, ``turn_right``, and ``turn_straight``.
    """
    if action == "lane_follow":
        follow_route = _rebuild_follow_route(start_waypoint, max_distance_m, step_m)
        return ActionSuffixResult(
            waypoints=list(follow_route.waypoints),
            action_end_index=len(follow_route.waypoints) - 1,
        )
    if action == "lane_change_left":
        return _rebuild_lane_change_suffix(
            start_waypoint, "left", max_distance_m, step_m, lane_change_prep_m
        )
    if action == "lane_change_right":
        return _rebuild_lane_change_suffix(
            start_waypoint, "right", max_distance_m, step_m, lane_change_prep_m
        )
    if action in ("turn_left", "turn_right", "turn_straight"):
        return _rebuild_turn_suffix(start_waypoint, action, max_distance_m, step_m)
    raise ValueError(f"Unsupported action: {action}")


# ---------------------------------------------------------------------------
# Actionability on rebuilt routes
# ---------------------------------------------------------------------------

def _build_actionability_samples(
    route: RebuiltRoute,
    carla_map: "carla.Map",
) -> List[RebuiltActionabilitySample]:
    """Sample actionability at every position of a rebuilt route."""
    samples: List[RebuiltActionabilitySample] = []
    for index, position in enumerate(route.positions):
        distance_m = route.cumulative_distances_m[index]
        remaining_distance_m = max(route.total_length_m - distance_m, 0.0)
        actions = _build_actionable_navigation_categories(
            carla_map,
            position,
            max_turn_scan_distance_m=remaining_distance_m,
        )
        scored_actions = tuple(a for a in actions if a != "lane_follow")
        waypoint = carla_map.get_waypoint(
            carla.Location(x=position[0], y=position[1], z=position[2]),
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        samples.append(
            RebuiltActionabilitySample(
                index=index,
                distance_m=distance_m,
                actions=tuple(actions),
                scored_actions=scored_actions,
                score=len(scored_actions),
                is_junction=bool(waypoint.is_junction) if waypoint is not None else False,
            )
        )
    return samples


# ---------------------------------------------------------------------------
# Trigger & action selection on rebuilt routes
# ---------------------------------------------------------------------------

def _select_output_actions(actions: Sequence[str], rng: random.Random) -> List[str]:
    """De-duplicate *actions*, keep non-lane_follow, optionally add lane_follow.

    ``lane_follow`` is kept only when it is the sole feasible action or, with
    probability ``OPTIONAL_LANE_FOLLOW_PROBABILITY``, next to other actions.
    """
    unique_actions: List[str] = []
    for action in actions:
        if action not in unique_actions:
            unique_actions.append(action)

    retained = [a for a in unique_actions if a != "lane_follow"]
    if "lane_follow" in unique_actions:
        if not retained or rng.random() < OPTIONAL_LANE_FOLLOW_PROBABILITY:
            retained.append("lane_follow")
    return retained


def _select_trigger(
    samples: Sequence[RebuiltActionabilitySample],
    min_trigger_distance_m: float = 5.0,
) -> Optional[RebuiltTrigger]:
    """Pick the highest-scored sample at or past *min_trigger_distance_m*.

    Ties are broken by taking the earliest sample. Returns ``None`` if no
    eligible sample exists.
    """
    valid_samples = [s for s in samples if s.distance_m >= min_trigger_distance_m]
    if not valid_samples:
        return None

    best_score = max(s.score for s in valid_samples)
    winner = next(s for s in valid_samples if s.score == best_score)

    return RebuiltTrigger(
        index=winner.index,
        distance_m=winner.distance_m,
        actions=winner.actions,
        scored_actions=winner.scored_actions,
        is_junction=winner.is_junction,
        phrasing_mode="at_junction" if winner.is_junction else "approach",
    )


# ---------------------------------------------------------------------------
# XML helpers for rebuilt routes
# ---------------------------------------------------------------------------

def _build_waypoints_element(positions: Sequence[Position3D]) -> ET.Element:
    """Create a ``<waypoints>`` XML element from a sequence of positions."""
    if not positions:
        raise ValueError("Cannot build <waypoints> from an empty route.")

    waypoints_elem = ET.Element("waypoints")
    for x, y, z in positions:
        ET.SubElement(
            waypoints_elem,
            "position",
            {
                "x": f"{x:.2f}",
                "y": f"{y:.2f}",
                "z": f"{z:.2f}",
            },
        )
    return waypoints_elem


def _build_default_scenarios_from_positions(positions: Sequence[Position3D]) -> ET.Element:
    """Create a ``<scenarios>`` element with a FreeRide trigger from *positions*.

    Computes the initial yaw from the first two positions.
    """
    if not positions:
        raise ValueError("Cannot build scenarios from an empty route.")

    x0, y0, z0 = positions[0]
    yaw_deg = 0.0
    if len(positions) >= 2:
        x1, y1, _ = positions[1]
        dy = y1 - y0
        dx = x1 - x0
        if abs(dx) > 1e-6 or abs(dy) > 1e-6:
            yaw_deg = math.degrees(math.atan2(dy, dx))

    scenarios_elem = ET.Element("scenarios")
    scenario_elem = ET.SubElement(
        scenarios_elem,
        "scenario",
        {"name": "FreeRide_1", "type": "FreeRide"},
    )
    ET.SubElement(
        scenario_elem,
        "trigger_point",
        {
            "x": f"{x0:.1f}",
            "y": f"{y0:.1f}",
            "z": f"{z0:.1f}",
            "yaw": f"{yaw_deg:.1f}",
        },
    )
    return scenarios_elem
