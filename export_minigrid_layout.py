"""
Freeze a MiniGrid layout to JSON so the Gazebo run can mirror it exactly.

Run this on the machine where the text experiment already runs, the one that
has gymnasium and minigrid installed. The robot machine only reads the JSON, so
it never needs those packages.

    python export_minigrid_layout.py \
        --env MiniGrid-SimpleCrossingS9N1-v0 \
        --seed 0 \
        --output layout_simplecrossing_s9n1_seed0.json

layout_simplecrossing_s9n1_seed0.json is already generated, so you only need
this if you want a second layout or a different seed.
"""

import argparse
import json
from collections import deque

from minigrid_wrapper import ACTION_TO_DELTA, MiniGridCardinalWrapper


def shortest_path_length(wrapper, start, goal):
    """
    BFS distance from start to goal over free cells.
    """
    if start == goal:
        return 0

    queue = deque([(start, 0)])
    visited = {start}

    while queue:
        (row, col), distance = queue.popleft()

        for delta_row, delta_col in ACTION_TO_DELTA.values():
            neighbour = (row + delta_row, col + delta_col)

            blocked, _ = wrapper.is_blocked(neighbour[0], neighbour[1])

            if blocked or neighbour in visited:
                continue

            if neighbour == goal:
                return distance + 1

            visited.add(neighbour)
            queue.append((neighbour, distance + 1))

    return None


def main():
    parser = argparse.ArgumentParser(
        description="Freeze a MiniGrid layout to JSON for the Gazebo run."
    )
    parser.add_argument("--env", default="MiniGrid-SimpleCrossingS9N1-v0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output",
        default="layout_simplecrossing_s9n1_seed0.json",
    )
    args = parser.parse_args()

    wrapper = MiniGridCardinalWrapper(env_name=args.env, seed=args.seed)
    wrapper.reset(seed=args.seed)

    state = wrapper.get_state()

    layout = {
        "source": (
            f"{args.env}, seed {args.seed}, via MiniGridCardinalWrapper.get_state()"
        ),
        "env_name": args.env,
        "seed": args.seed,
        "grid_size": list(state["grid_size"]),
        "start_cell": list(state["agent"]),
        "start_facing": state["facing_name"],
        "goal_cell": list(state["goal"]),
        "obstacle_cells": [list(cell) for cell in state["obstacle_cells"]],
        "optimal_path_length": shortest_path_length(
            wrapper,
            state["agent"],
            state["goal"],
        ),
    }

    with open(args.output, "w", encoding="utf-8") as file:
        json.dump(layout, file, indent=2)

    print(wrapper.render_text())
    print()
    print(json.dumps(layout, indent=2))
    print()
    print(f"Wrote {args.output}")

    wrapper.close()


if __name__ == "__main__":
    main()
