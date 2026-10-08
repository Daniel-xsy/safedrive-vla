"""SafeDriveVLA agent for the navigation-scene conflict (B2D-C) and the
adversarial-instruction (B2D-Adv) benchmarks.

As in the CARLA-F agent, the only navigation signal is the active language
instruction, and the action anchor is indexed by the meta-command parsed from
the instruction text; no route-planner information reaches the model or the
world model. An instruction stays active for as long as its trigger holds (the
unsafe instruction is injected when the safety-critical scenario starts).

The instructions of these benchmarks are mapped to meta-commands by the words
that name a maneuver, ignoring any requested speed
(``conflict_instruction_meta_commands.json``): a left / right turn, a lane
change to the left / right lane, or going straight through an intersection or
junction. Any other instruction, including speed-only ones, is ``follow_road``.
The table is built by ``benchmark/generation/build_conflict_meta_commands.py``.
"""

from pathlib import Path

from team_code.agent_safedrive_carla_f import SafeDriveLanguageAgent


def get_entry_point():
    return "SafeDriveConflictAgent"


class SafeDriveConflictAgent(SafeDriveLanguageAgent):
    COMPLETION_MONITOR = False
    META_COMMAND_TABLE = Path(__file__).resolve().parent / "conflict_instruction_meta_commands.json"
    UNKNOWN_META_COMMAND = "follow_road"
