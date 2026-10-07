"""Vocabulary shared by the dataset, the model, and the CARLA agents."""

from __future__ import annotations

from typing import Dict, Tuple

# Meta-commands (Sec. 4.1, App. C.2). The order defines the meta-command index
# used everywhere, including the rows of the action-anchor table.
META_COMMANDS: Tuple[str, ...] = (
    "follow_road",
    "go_straight_intersection",
    "turn_left",
    "turn_right",
    "lane_change_left",
    "lane_change_right",
)
META_COMMAND_TO_INDEX: Dict[str, int] = {name: i for i, name in enumerate(META_COMMANDS)}

# CARLA route-planner command (``RoadOption`` value, also the ``command`` field
# of the PDM-Lite measurement files) -> meta-command.
COMMAND_TO_META_COMMAND: Dict[int, str] = {
    1: "turn_left",
    2: "turn_right",
    3: "go_straight_intersection",
    4: "follow_road",
    5: "lane_change_left",
    6: "lane_change_right",
}

# Driving modes (Sec. 4.1) and the matching VLM tokens.
DRIVING_MODES: Tuple[str, ...] = ("strict", "cautious", "fallback")
DRIVING_MODE_TO_INDEX: Dict[str, int] = {name: i for i, name in enumerate(DRIVING_MODES)}
MODE_TOKENS: Tuple[str, ...] = ("<STRICT>", "<CAUTIOUS>", "<FALLBACK>")

# Beginning-of-action delimiter; the path head reads its hidden state.
ACTIONS_TOKEN = "<ACTIONS>"
# Target-waypoint placeholder; its embedding is replaced by the target-point encoder.
TARGET_POINT_TOKEN = "<TARGET_POINT>"
# World-token block spliced into the prompt; each <WM_LATENT> embedding is
# replaced by one projected dreamed latent.
WORLD_BEGIN_TOKEN = "<world_begin>"
WORLD_TOKEN = "<WM_LATENT>"
WORLD_END_TOKEN = "<world_end>"

# Special tokens added to the VLM vocabulary, in this exact order: the order
# fixes the token ids (and embedding rows) of the released checkpoints. The
# first ten are inherited from SimLingo.
SPECIAL_TOKENS = [
    "<SAFETY>",
    "<INSTRUCTION_FOLLOWING>",
    TARGET_POINT_TOKEN,
    "<WAYPOINTS>",
    "<WAYPOINTS_DIFF>",
    "<ORG_WAYPOINTS>",
    "<ORG_WAYPOINTS_DIFF>",
    "<WAYPOINT_LAST>",
    "<ROUTE>",
    "<ROUTE_DIFF>",
    ACTIONS_TOKEN,
    "<RESERVED_0>",
    "<RESERVED_1>",
    *MODE_TOKENS,
    WORLD_BEGIN_TOKEN,
    WORLD_END_TOKEN,
    WORLD_TOKEN,
]

# Prompt text for the world-token block and the task suffix. Changing either
# changes the model input, so keep them identical between training and the agent.
TASK_PROMPT = "Predict the actions."


def world_block_text(num_world_tokens: int) -> str:
    return f" Lookahead: {WORLD_BEGIN_TOKEN}{WORLD_TOKEN * num_world_tokens}{WORLD_END_TOKEN}."
