"""SafeDriveVLA agent for language-instruction benchmarks (CARLA-F).

The navigation signal is the active natural-language instruction of the route
XML (``ROUTES``), shown as ``Command: <instruction>.``; the route-planner
command and target waypoints are hidden from the model. Instructions are
triggered by the distance travelled (or by a scenario becoming active). Once
the ego completes a turn or lane-change instruction, the prompt falls back to
``follow the road``.

The action anchor of world-model dreaming is indexed by the meta-command
parsed from the instruction text (``instruction_meta_commands.json``, the
paraphrase pools of the benchmark). Target-speed instructions name no maneuver
and use ``follow_road``, the anchor their frames get in training.
"""

from __future__ import annotations

import json
import math
import os
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import carla
import py_trees
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

from safedrive_vla.constants import META_COMMAND_TO_INDEX
from team_code.agent_safedrive import SafeDriveAgent

INSTRUCTION_META_COMMANDS = Path(__file__).resolve().parent / "instruction_meta_commands.json"
NAVIGATION_COMMANDS = {1, 2, 3, 5, 6}   # route-planner commands of maneuver instructions
LANE_FOLLOW_COMMAND = 4
COMPLETION_MIN_PROGRESS_M = 3.0
SCENARIO_ACTIVE_DISTANCE_M = 30.0
_SPEED_RE = re.compile(r"(-?\d+(?:\.\d+)?)\s*m\s*/\s*s", re.IGNORECASE)
_WS_RE = re.compile(r"\s+")


def get_entry_point():
    return "SafeDriveLanguageAgent"


def _normalize(text: str) -> str:
    return _WS_RE.sub(" ", text.strip().lower())


@dataclass
class Instruction:
    id: int
    text: str
    command_id: int
    trigger_type: str            # start | distance_traveled | scenario_active
    trigger_value: float = 0.0
    scenario_name: str = ""
    duration_meters: float = -1.0
    # runtime state
    is_active: bool = False
    start_distance: float = 0.0
    planner_command_seen: bool = False


class SafeDriveLanguageAgent(SafeDriveAgent):
    COMPLETION_MONITOR = True
    # instruction text -> meta-command; text outside the table: no anchor
    META_COMMAND_TABLE = INSTRUCTION_META_COMMANDS
    UNKNOWN_META_COMMAND: Optional[str] = None

    def setup(self, path_to_conf_file):
        super().setup(path_to_conf_file)
        self.instructions: List[Instruction] = []
        self.current_instruction: Optional[Instruction] = None
        self.completed_ids = set()
        self.distance_traveled = 0.0
        self.last_location: Optional[carla.Location] = None
        self.scenario_variables: Dict[str, str] = {}
        self.scenario_trigger_points: Dict[str, carla.Location] = {}
        self._hero = None
        self._parse_route_xml(os.environ.get("ROUTES", ""))
        with open(self.META_COMMAND_TABLE) as f:
            self.instruction_meta_commands = {_normalize(k): v for k, v in json.load(f).items()}
        self.instruction = "follow the road"

    def _parse_route_xml(self, path: str) -> None:
        route = ET.parse(path).getroot().find(".//route")
        for elem in route.find("instructions").findall("instruction"):
            trigger = elem.find("trigger")
            self.instructions.append(Instruction(
                id=int(elem.attrib.get("id", 0)),
                text=elem.findtext("text") or "follow the road",
                command_id=int(elem.findtext("command_id") or LANE_FOLLOW_COMMAND),
                trigger_type=trigger.attrib.get("type", "start") if trigger is not None else "start",
                trigger_value=float(trigger.attrib.get("value", 0.0)) if trigger is not None else 0.0,
                scenario_name=trigger.attrib.get("scenario_name", "") if trigger is not None else "",
                duration_meters=float(elem.findtext("duration_meters") or -1.0),
            ))
        self.instructions.sort(key=lambda x: x.id)
        scenarios = route.find("scenarios")
        for i, elem in enumerate(scenarios.findall("scenario") if scenarios is not None else []):
            name = elem.attrib.get("name", f"scenario_{i}")
            self.scenario_variables[name] = f"ScenarioRouteNumber{i}"
            point = elem.find("trigger_point")
            if point is not None:
                self.scenario_trigger_points[name] = carla.Location(
                    x=float(point.attrib.get("x", 0)), y=float(point.attrib.get("y", 0)), z=float(point.attrib.get("z", 0)))

    # ------------------------------------------------------------------
    # instruction selection
    # ------------------------------------------------------------------
    def _ego_location(self) -> Optional[carla.Location]:
        if self._hero is None:
            self._hero = CarlaDataProvider.get_hero_actor()
        if self._hero is None:
            return None
        location = CarlaDataProvider.get_location(self._hero)
        if location is None:
            self._hero = CarlaDataProvider.get_hero_actor()
            location = CarlaDataProvider.get_location(self._hero) if self._hero is not None else None
        return location

    def _scenario_active(self, instr: Instruction, ego_location: carla.Location) -> bool:
        if not instr.scenario_name:
            return False
        variable = self.scenario_variables.get(instr.scenario_name)
        try:
            if variable and py_trees.blackboard.Blackboard().get(variable):
                return True
        except Exception:
            pass
        # Fall back to the proximity of the scenario's actors.
        world = CarlaDataProvider.get_world()
        max_distance = instr.trigger_value if instr.trigger_value > 0 else SCENARIO_ACTIVE_DISTANCE_M
        trigger_location = self.scenario_trigger_points.get(instr.scenario_name)
        for actor in world.get_actors().filter("vehicle.*"):
            if actor.attributes.get("role_name", "") != "scenario":
                continue
            location = actor.get_location()
            if location is None or location.z < -10.0:
                continue
            if trigger_location is not None and location.distance(trigger_location) <= max_distance:
                return True
            if location.distance(ego_location) <= max_distance:
                return True
        return False

    def _active_instruction(self, ego_location: carla.Location) -> Optional[Instruction]:
        """The last triggered instruction that is neither completed nor expired."""
        active = None
        for instr in self.instructions:
            if instr.id in self.completed_ids:
                continue
            if instr.trigger_type == "start":
                triggered = True
            elif instr.trigger_type == "distance_traveled":
                triggered = self.distance_traveled >= instr.trigger_value
            elif instr.trigger_type == "scenario_active":
                triggered = self._scenario_active(instr, ego_location)
            else:
                raise ValueError(f"Unsupported instruction trigger: {instr.trigger_type}")
            if triggered:
                if not instr.is_active:
                    instr.is_active = True
                    instr.start_distance = self.distance_traveled
                if instr.duration_meters > 0 and self.distance_traveled > instr.start_distance + instr.duration_meters:
                    triggered = False
            if triggered:
                active = instr
        return active

    def on_route_command(self, current_command: int) -> None:
        """A maneuver instruction is completed once the route planner has gone
        through its command and is back to lane following."""
        instr = self.current_instruction
        if not self.COMPLETION_MONITOR or instr is None or instr.command_id not in NAVIGATION_COMMANDS:
            return
        if current_command == instr.command_id:
            instr.planner_command_seen = True
        progress = self.distance_traveled - instr.start_distance
        if instr.planner_command_seen and current_command == LANE_FOLLOW_COMMAND and progress >= COMPLETION_MIN_PROGRESS_M:
            self.completed_ids.add(instr.id)
            instr.is_active = False
            self.current_instruction = None
            self.instruction = "follow the road"

    def tick(self, input_data):
        location = self._ego_location()
        if location is not None:
            if self.last_location is not None:
                self.distance_traveled += location.distance(self.last_location)
            self.last_location = location
            self.current_instruction = self._active_instruction(location)
        instr = self.current_instruction
        self.instruction = instr.text.rstrip(". ") if instr is not None else "follow the road"
        return super().tick(input_data)

    def anchor_meta_command(self) -> int:
        """Meta-command parsed from the instruction text."""
        text = (self.current_instruction.text if self.current_instruction is not None else self.instruction).strip()
        meta = self.instruction_meta_commands.get(_normalize(text))
        if meta is None and _SPEED_RE.search(text):
            meta = "follow_road"
        if meta is None:
            meta = self.UNKNOWN_META_COMMAND
        return META_COMMAND_TO_INDEX[meta] if meta is not None else -1
