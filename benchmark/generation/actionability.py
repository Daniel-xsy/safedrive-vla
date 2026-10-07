"""
Feasible meta-commands at a route point.

Answers "which navigation actions can be executed from this waypoint?" from
the CARLA road topology: ``lane_follow`` is always feasible, lane changes need
a same-direction adjacent driving lane, and ``turn_left`` / ``turn_right`` /
``turn_straight`` (go straight) are found by scanning the branches of the
upcoming junction. The number of feasible actions other than ``lane_follow``
is the actionability score used for trigger selection.
"""

from typing import List, Set, Tuple

from geometry import (
    _can_change_lane,
    _compute_turn_category,
)

try:
    import carla
except ImportError:
    carla = None


# ---------------------------------------------------------------------------
# Turn scanning (BFS along CARLA topology)
# ---------------------------------------------------------------------------

def _scan_turn_actions(
    start_waypoint: "carla.Waypoint",
    scan_distance_m: float = 45.0,
    step_m: float = 2.0,
) -> Set[str]:
    """Scan the upcoming road topology and return feasible ``turn_*`` actions.

    Uses a BFS-style frontier expansion: at each step, follow road
    successors forward.  When the topology branches near a junction,
    trace each branch to its exit and classify the resulting yaw delta
    as ``turn_left``, ``turn_right``, or ``turn_straight``.
    """
    turn_actions: Set[str] = set()
    frontier = [start_waypoint]
    visited: Set[Tuple[int, int, int, int]] = set()
    max_steps = max(1, int(scan_distance_m / step_m))

    for _ in range(max_steps):
        if not frontier:
            break

        next_frontier: List["carla.Waypoint"] = []
        for waypoint in frontier:
            key = (
                waypoint.road_id,
                waypoint.section_id,
                waypoint.lane_id,
                int(round(waypoint.s * 10.0)),
            )
            if key in visited:
                continue
            visited.add(key)

            next_candidates = waypoint.next(step_m)
            if not next_candidates:
                continue

            # Detect branching near a junction.
            is_branch_near_junction = len(next_candidates) > 1 and (
                waypoint.is_junction or any(c.is_junction for c in next_candidates)
            )
            if is_branch_near_junction:
                for candidate in next_candidates:
                    branch_waypoint = candidate
                    branch_traversed = step_m
                    while branch_traversed < scan_distance_m:
                        next_branch = branch_waypoint.next(step_m)
                        if not next_branch:
                            break
                        branch_waypoint = next_branch[0]
                        branch_traversed += step_m
                        if not branch_waypoint.is_junction:
                            break
                    turn_actions.add(_compute_turn_category(start_waypoint, branch_waypoint))
                continue

            candidate = next_candidates[0]
            if waypoint.is_junction or candidate.is_junction:
                branch_waypoint = candidate
                branch_traversed = step_m
                while branch_traversed < scan_distance_m:
                    next_branch = branch_waypoint.next(step_m)
                    if not next_branch:
                        break
                    branch_waypoint = next_branch[0]
                    branch_traversed += step_m
                    if not branch_waypoint.is_junction:
                        break
                turn_actions.add(_compute_turn_category(start_waypoint, branch_waypoint))

            next_frontier.append(candidate)

        frontier = next_frontier

    return turn_actions


def _build_actionable_navigation_categories(
    carla_map: "carla.Map",
    ego_position: Tuple[float, float, float],
    max_turn_scan_distance_m: float = 45.0,
) -> List[str]:
    """Query the CARLA map at *ego_position* and return all feasible navigation actions.

    Always includes ``"lane_follow"``; may also include lane-change and turn
    actions depending on the local road topology.
    """
    actions: List[str] = ["lane_follow"]
    ego_waypoint = carla_map.get_waypoint(
        carla.Location(x=ego_position[0], y=ego_position[1], z=ego_position[2]),
        project_to_road=True,
        lane_type=carla.LaneType.Driving,
    )
    if ego_waypoint is None:
        return actions

    if _can_change_lane(ego_waypoint, "left"):
        actions.append("lane_change_left")
    if _can_change_lane(ego_waypoint, "right"):
        actions.append("lane_change_right")

    turn_actions = _scan_turn_actions(ego_waypoint, scan_distance_m=max_turn_scan_distance_m)
    for turn_action in ("turn_left", "turn_right", "turn_straight"):
        if turn_action in turn_actions:
            actions.append(turn_action)

    return actions
