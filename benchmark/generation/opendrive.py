"""
OpenDRIVE speed-limit lookup and CARLA map loading.

Provides:
    * ``OpenDriveSpeedLimitResolver`` – parses ``.xodr`` files to look up the
      posted speed limit for a given ``(town, road_id, s)`` triple. It is the
      source of the target speed of the CARLA-F speed instruction.
    * ``CarlaMapCache`` – lazily builds ``carla.Map`` objects from ``.xodr``
      files, so route generation does not need a running CARLA server.
    * Helper functions for locating ``.xodr`` files in common CARLA
      installation directories.
"""

import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

try:
    import carla
except ImportError as exc:
    carla = None
    CARLA_IMPORT_ERROR = exc
else:
    CARLA_IMPORT_ERROR = None


# ---------------------------------------------------------------------------
# Unit conversion
# ---------------------------------------------------------------------------

def _convert_speed_to_kmh(value: float, unit: str) -> float:
    """Convert an OpenDRIVE speed value to km/h.

    Recognised units: ``km/h``, ``kmh``, ``kph``, ``""`` (default km/h),
    ``mph``, ``m/s``, ``mps``.
    """
    normalized = unit.strip().lower()
    if normalized in {"km/h", "kmh", "kph", ""}:
        return value
    if normalized == "mph":
        return value * 1.609344
    if normalized in {"m/s", "mps"}:
        return value * 3.6
    raise ValueError(f"Unsupported OpenDRIVE speed unit: {unit}")


# ---------------------------------------------------------------------------
# XODR file discovery
# ---------------------------------------------------------------------------

def _default_xodr_search_roots() -> List[Path]:
    """Build a de-duplicated, existence-checked list of OpenDRIVE search roots.

    Looks at ``CARLA_XODR_ROOT``, ``CARLA_ROOT``, the CARLA client map cache
    (``~/carlaCache``), and ``/opt/carla``.
    """
    roots: List[Path] = []

    carla_xodr_root = os.environ.get("CARLA_XODR_ROOT")
    if carla_xodr_root:
        roots.append(Path(carla_xodr_root).expanduser())

    carla_root = os.environ.get("CARLA_ROOT")
    if carla_root:
        base = Path(carla_root).expanduser()
        roots.extend(
            [
                base / "CarlaUE4" / "Content" / "Carla" / "Maps",
                base / "Carla" / "Maps",
                base / "Maps",
            ]
        )

    home = Path.home()
    roots.extend(
        [
            home / "carlaCache" / "0.9.15" / "Carla" / "Maps",
            Path("/opt/carla/CarlaUE4/Content/Carla/Maps"),
        ]
    )

    deduped: List[Path] = []
    seen: Set[Path] = set()
    for root in roots:
        resolved = root.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.exists() and resolved.is_dir():
            deduped.append(resolved)
    return deduped


def _resolve_xodr_path(town: str, search_roots: List[Path]) -> Optional[Path]:
    """Find the ``.xodr`` file for *town* under the given search roots.

    Tries several naming conventions (``Town.xodr``, ``Town_Opt.xodr``,
    ``TownHD.xodr``) both in an ``OpenDrive/`` subdirectory and inside a
    per-town folder. The ``HD`` variant lets a bare ``Town10`` resolve to the
    stock ``Town10HD.xodr``.
    """
    name_variants = [town]
    if not town.endswith("HD"):
        name_variants.append(f"{town}HD")

    candidates: List[Path] = []
    for root in search_roots:
        for name in name_variants:
            candidates.extend(
                [
                    root / "OpenDrive" / f"{name}.xodr",
                    root / "OpenDrive" / f"{name}_Opt.xodr",
                    root / name / "OpenDrive" / f"{name}.xodr",
                    root / name / "OpenDrive" / f"{name}_Opt.xodr",
                ]
            )

    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


# ---------------------------------------------------------------------------
# OpenDRIVE speed-limit resolver
# ---------------------------------------------------------------------------

class OpenDriveSpeedLimitResolver:
    """Look up the posted speed limit from OpenDRIVE ``.xodr`` maps.

    Speed definitions (``<road><type><speed>``) are read once per town and
    cached in memory. Roads without their own ``<speed>`` element inherit
    from a linked predecessor or successor road (common for junction
    connector roads).
    """

    def __init__(self, xodr_search_roots: Optional[List[Path]] = None) -> None:
        self._xodr_search_roots = (
            _default_xodr_search_roots()
            if not xodr_search_roots
            else [p.resolve() for p in xodr_search_roots]
        )
        self._maps: Dict[str, Dict[int, List[Tuple[float, float]]]] = {}

    def get_speed_limit_kmh(self, town: str, road_id: int, s: float) -> float:
        """Return the speed limit in km/h at longitudinal position *s* on *road_id*.

        Raises ``KeyError`` if no speed data exists for the road.
        """
        road_sections = self._load_map(town)
        sections = road_sections.get(int(road_id))
        if not sections:
            raise KeyError(
                f"No OpenDRIVE speed definition found for road {road_id} in town {town}."
            )

        active_speed_kmh = sections[0][1]
        for section_start_s, section_speed_kmh in sections:
            if s + 1e-6 < section_start_s:
                break
            active_speed_kmh = section_speed_kmh
        return active_speed_kmh

    # -- internal ------------------------------------------------------------

    def _load_map(self, town: str) -> Dict[int, List[Tuple[float, float]]]:
        """Parse the ``.xodr`` for *town* and cache speed sections per road."""
        if town in self._maps:
            return self._maps[town]

        xodr_path = _resolve_xodr_path(town, self._xodr_search_roots)
        if xodr_path is None:
            roots = ", ".join(str(root) for root in self._xodr_search_roots)
            raise FileNotFoundError(
                f"Could not find OpenDRIVE map for {town}. Searched under: {roots}"
            )

        root = ET.parse(xodr_path).getroot()
        road_sections: Dict[int, List[Tuple[float, float]]] = {}
        # Roads without speed definitions — will try to inherit from linked roads.
        roads_without_speed: Dict[int, List[int]] = {}

        for road_elem in root.findall("road"):
            road_id_str = road_elem.attrib.get("id")
            if road_id_str is None:
                continue
            rid = int(road_id_str)

            sections: List[Tuple[float, float]] = []
            for type_elem in road_elem.findall("type"):
                speed_elem = type_elem.find("speed")
                if speed_elem is None:
                    continue
                section_start_s = float(type_elem.attrib.get("s", "0"))
                speed_max = float(speed_elem.attrib["max"])
                speed_unit = speed_elem.attrib.get("unit", "km/h")
                sections.append(
                    (section_start_s, _convert_speed_to_kmh(speed_max, speed_unit))
                )

            if sections:
                sections.sort(key=lambda item: item[0])
                road_sections[rid] = sections
            else:
                linked_ids: List[int] = []
                link_elem = road_elem.find("link")
                if link_elem is not None:
                    for tag in ("predecessor", "successor"):
                        elem = link_elem.find(tag)
                        if elem is not None and elem.attrib.get("elementType") == "road":
                            try:
                                linked_ids.append(int(elem.attrib["elementId"]))
                            except (KeyError, ValueError):
                                pass
                roads_without_speed[rid] = linked_ids

        # Inherit speed for roads without own definition (e.g. junction roads).
        for rid, linked_ids in roads_without_speed.items():
            for linked_id in linked_ids:
                if linked_id in road_sections:
                    road_sections[rid] = road_sections[linked_id]
                    break

        self._maps[town] = road_sections
        return road_sections


# ---------------------------------------------------------------------------
# CARLA map cache
# ---------------------------------------------------------------------------

class CarlaMapCache:
    """Lazily load ``carla.Map`` objects from ``.xodr`` files on disk.

    This avoids repeated parsing of the same OpenDRIVE file when multiple
    routes reference the same town.
    """

    def __init__(self, xodr_search_roots: Optional[List[Path]] = None) -> None:
        if carla is None:
            raise RuntimeError(
                "CARLA Python API is required for map-aware action sampling."
            ) from CARLA_IMPORT_ERROR
        self._xodr_search_roots = (
            _default_xodr_search_roots()
            if not xodr_search_roots
            else [p.resolve() for p in xodr_search_roots]
        )
        self._maps: Dict[str, "carla.Map"] = {}

    def get_map(self, town: str) -> "carla.Map":
        """Return the ``carla.Map`` for *town*, loading from disk if needed."""
        if town in self._maps:
            return self._maps[town]

        xodr_path = _resolve_xodr_path(town, self._xodr_search_roots)
        if xodr_path is None:
            roots = ", ".join(str(root) for root in self._xodr_search_roots)
            raise FileNotFoundError(
                f"Could not find OpenDRIVE map for {town}. "
                f"Searched under: {roots}"
            )

        map_obj = carla.Map(town, xodr_path.read_text(encoding="utf-8"))
        self._maps[town] = map_obj
        return map_obj
