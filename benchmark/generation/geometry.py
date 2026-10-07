"""
Geometry and lane helpers for route analysis.

Pure-computation utilities for XML waypoint parsing, yaw normalization, turn
classification, and CARLA lane queries. None of these functions perform I/O or
modify external state.
"""

from typing import List, Tuple

try:
    import carla
except ImportError:
    carla = None


# ---------------------------------------------------------------------------
# XML waypoint extraction
# ---------------------------------------------------------------------------

def _get_waypoint_positions(waypoints_elem) -> List[Tuple[float, float, float]]:
    """Parse ``<position x=... y=... z=...>`` children of *waypoints_elem*.

    Args:
        waypoints_elem: An ``xml.etree.ElementTree.Element`` whose children
            are ``<position>`` tags with *x*, *y*, *z* attributes.

    Returns:
        A list of ``(x, y, z)`` float tuples in document order.
    """
    positions = []
    for pos in waypoints_elem.findall("position"):
        positions.append(
            (
                float(pos.attrib["x"]),
                float(pos.attrib["y"]),
                float(pos.attrib["z"]),
            )
        )
    return positions


# ---------------------------------------------------------------------------
# Yaw / turn classification
# ---------------------------------------------------------------------------

def _normalize_yaw_delta_deg(delta_deg: float) -> float:
    """Normalize an angle delta to the range ``[-180, 180]`` degrees."""
    while delta_deg > 180.0:
        delta_deg -= 360.0
    while delta_deg < -180.0:
        delta_deg += 360.0
    return delta_deg


def _compute_turn_category(
    current_waypoint: "carla.Waypoint",
    next_waypoint: "carla.Waypoint",
    threshold_deg: float = 35.0,
) -> str:
    """Classify the turn between two CARLA waypoints.

    Returns:
        ``"turn_straight"``, ``"turn_left"``, or ``"turn_right"`` depending
        on the yaw delta relative to *threshold_deg*.
    """
    yaw_delta = _normalize_yaw_delta_deg(
        next_waypoint.transform.rotation.yaw - current_waypoint.transform.rotation.yaw
    )
    if abs(yaw_delta) < threshold_deg:
        return "turn_straight"
    if yaw_delta < 0.0:
        return "turn_left"
    return "turn_right"


# ---------------------------------------------------------------------------
# CARLA lane helpers
# ---------------------------------------------------------------------------

def _is_same_direction_lane(
    source_waypoint: "carla.Waypoint",
    candidate_waypoint: "carla.Waypoint",
    max_yaw_delta_deg: float = 45.0,
) -> bool:
    """Check whether two waypoints face roughly the same direction."""
    yaw_delta = _normalize_yaw_delta_deg(
        candidate_waypoint.transform.rotation.yaw - source_waypoint.transform.rotation.yaw
    )
    return abs(yaw_delta) < max_yaw_delta_deg


def _can_change_lane(ego_waypoint: "carla.Waypoint", direction: str) -> bool:
    """Check whether a lane change in *direction* is feasible.

    Verifies that the CARLA lane-change flag permits the manoeuvre, the
    adjacent lane exists, is of type ``Driving``, and faces the same
    direction.

    Args:
        ego_waypoint: Current CARLA waypoint.
        direction: ``"left"`` or ``"right"``.

    Returns:
        ``True`` if the lane change is valid.
    """
    if direction == "left":
        if ego_waypoint.lane_change not in (carla.LaneChange.Left, carla.LaneChange.Both):
            return False
        adjacent = ego_waypoint.get_left_lane()
    elif direction == "right":
        if ego_waypoint.lane_change not in (carla.LaneChange.Right, carla.LaneChange.Both):
            return False
        adjacent = ego_waypoint.get_right_lane()
    else:
        raise ValueError(f"Unsupported lane-change direction: {direction}")

    if adjacent is None:
        return False
    if adjacent.lane_type != carla.LaneType.Driving:
        return False
    return _is_same_direction_lane(ego_waypoint, adjacent)
