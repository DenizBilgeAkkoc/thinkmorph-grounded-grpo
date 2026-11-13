"""Conservative checks using original-problem facts, never reference solutions."""
from hashlib import sha256
from pathlib import Path
import re


def verified_grid(constraints, image):
    if constraints is None:
        return None
    if set(constraints) != {"image_sha256", "source", "grid"} or constraints["source"] != "verified_original_image":
        raise ValueError("Only verified original-image grid constraints are accepted")
    if sha256(Path(image).read_bytes()).hexdigest() != constraints["image_sha256"]:
        raise ValueError("Original constraint/image hash mismatch")
    grid = constraints["grid"]
    if set(grid) != {"rows", "cols", "start", "goal", "holes"}:
        raise ValueError("Invalid grid fields")
    if any(type(grid[k]) is not int or grid[k] < 1 for k in ("rows", "cols")):
        raise ValueError("Invalid grid dimensions")
    for cell in [grid["start"], grid["goal"], *grid["holes"]]:
        if not isinstance(cell, list) or len(cell) != 2 or any(type(x) is not int for x in cell):
            raise ValueError("Invalid grid cell")
        if not (0 <= cell[0] < grid["rows"] and 0 <= cell[1] < grid["cols"]):
            raise ValueError("Grid cell out of range")
    if grid["start"] in grid["holes"] or grid["goal"] in grid["holes"]:
        raise ValueError("Start/goal cannot be a hole")
    return grid


def check_path(moves, grid):
    if not isinstance(moves, list) or any(x not in ("U", "D", "L", "R") for x in moves):
        raise ValueError("Moves must be an explicit list of U,D,L,R")
    position = tuple(grid["start"])
    visited = [list(position)]
    for i, move in enumerate(moves):
        dr, dc = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}[move]
        position = position[0] + dr, position[1] + dc
        visited.append(list(position))
        if not (0 <= position[0] < grid["rows"] and 0 <= position[1] < grid["cols"]):
            return {"legal": False, "reaches_goal": False, "reason": "off_grid", "at_move": i + 1, "visited": visited}
        if list(position) in grid["holes"]:
            return {"legal": False, "reaches_goal": False, "reason": "hole", "at_move": i + 1, "visited": visited}
    return {"legal": True, "reaches_goal": list(position) == grid["goal"], "visited": visited,
        "reason": "goal_reached" if list(position) == grid["goal"] else "stops_before_goal"}


def parse_moves(answer):
    match = re.fullmatch(r"\s*\\boxed\{([^{}]*)\}\s*", answer)
    value = match.group(1) if match else answer
    value = re.sub(r"\s+", "", value).upper()
    if not re.fullmatch(r"[UDLR](?:,[UDLR])*", value):
        raise ValueError("Unparseable path; do not infer or repair moves")
    return value.split(",")
