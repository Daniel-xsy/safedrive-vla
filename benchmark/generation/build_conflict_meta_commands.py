#!/usr/bin/env python3
"""
Build the instruction -> meta-command table of B2D-C and B2D-Adv
(``team_code/conflict_instruction_meta_commands.json``).

On these benchmarks the action anchor of world-model dreaming is indexed by the
meta-command parsed from the active instruction text, so no route-planner
information reaches the model. Every distinct instruction text of the route
files is mapped by the words that name a maneuver, ignoring any requested speed:

    turn_left                 "left turn" / "turn left" / "take a|the left"
    turn_right                "right turn" / "turn right" / "take a|the right"
    lane_change_left|right    "enter|move to|change to|switch to the left|right lane"
    go_straight_intersection  "straight" or "through" ... "intersection" or "junction"
    follow_road               everything else (speed-only, keep-lane, a turn
                              with no direction, ...)

Left is checked first, since "make a left turn right away" contains "turn right".
"do not change lanes" is a negation and matches no lane-change rule.

    python benchmark/generation/build_conflict_meta_commands.py
"""

import argparse
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROUTE_DIRS = (REPO_ROOT / "benchmark" / "data" / "b2d_c", REPO_ROOT / "benchmark" / "data" / "b2d_adv")
DEFAULT_OUTPUT = REPO_ROOT / "team_code" / "conflict_instruction_meta_commands.json"
# same order as safedrive_vla.constants.META_COMMANDS
META_COMMANDS = ("follow_road", "go_straight_intersection", "turn_left", "turn_right",
                 "lane_change_left", "lane_change_right")

LEFT = re.compile(r"\b(left turn|turn left|take (a|the) left)\b")
RIGHT = re.compile(r"\b(right turn|turn right|take (a|the) right)\b")
LANE_LEFT = re.compile(r"\b(enter|move (in)?to|change (in)?to|switch to) the left lane\b")
LANE_RIGHT = re.compile(r"\b(enter|move (in)?to|change (in)?to|switch to) the right lane\b")
STRAIGHT_JUNCTION = re.compile(r"\b(straight|strsight|through)\b.*\b(intersection|junction)\b")


def meta_command(text: str) -> str:
    t = re.sub(r"\s+", " ", text.strip().lower())
    if LEFT.search(t):
        return "turn_left"
    if RIGHT.search(t):
        return "turn_right"
    if LANE_LEFT.search(t):
        return "lane_change_left"
    if LANE_RIGHT.search(t):
        return "lane_change_right"
    if STRAIGHT_JUNCTION.search(t):
        return "go_straight_intersection"
    return "follow_road"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the B2D-C / B2D-Adv instruction -> meta-command table.")
    parser.add_argument("--route-dirs", type=Path, nargs="+", default=list(DEFAULT_ROUTE_DIRS))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    texts = set()
    for route_dir in args.route_dirs:
        for xml_path in sorted(route_dir.glob("*.xml")):
            for instruction in ET.parse(xml_path).iter("instruction"):
                text = (instruction.findtext("text") or "").strip()
                if text:
                    texts.add(text)
    table = {}
    for meta in META_COMMANDS:
        for text in sorted(t for t in texts if meta_command(t) == meta):
            table[text] = meta
    args.output.write_text(json.dumps(table, indent=2))
    print(f"wrote {len(table)} instruction texts to {args.output}")
    for meta in META_COMMANDS:
        print(f"  {meta:26s} {sum(m == meta for m in table.values())}")


if __name__ == "__main__":
    main()
