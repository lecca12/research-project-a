from collections import deque

import gymnasium as gym
import minigrid


DIRECTION_NAMES = {
    0: "east",
    1: "south",
    2: "west",
    3: "north",
}

DIRECTION_SYMBOLS = {
    0: ">",
    1: "v",
    2: "<",
    3: "^",
}

# Project convention:
# 0=north, 1=east, 2=south, 3=west
ACTION_TO_DELTA = {
    0: (-1, 0),
    1: (0, 1),
    2: (1, 0),
    3: (0, -1),
}

ACTION_NAMES = {
    0: "north",
    1: "east",
    2: "south",
    3: "west",
}

RELATIVE_ACTION_NAMES = {
    0: "forward",
    1: "right",
    2: "backward",
    3: "left",
}


class MiniGridCardinalWrapper:
    """
    Thin MiniGrid wrapper using the same direct cardinal-move convention
    as the original GridWorld project.

    This intentionally does not use MiniGrid's native turn-left/turn-right/
    move-forward action semantics. Instead, actions directly attempt to move
    north/east/south/west so that the main experimental difference between
    allocentric and egocentric prompts remains the action frame.
    """

    def __init__(self, env_name="MiniGrid-Empty-8x8-v0", seed=0):
        self.env_name = env_name
        self.seed = seed
        self.env = gym.make(env_name, render_mode="rgb_array")
        self.obs = None
        self.info = None

    def reset(self, seed=None):
        if seed is None:
            seed = self.seed

        self.seed = seed
        self.obs, self.info = self.env.reset(seed=seed)
        return self.get_state()

    @property
    def unwrapped(self):
        return self.env.unwrapped

    def close(self):
        self.env.close()

    def get_state(self):
        env = self.unwrapped
        grid = env.grid

        agent_xy = tuple(env.agent_pos)
        agent_rc = (agent_xy[1], agent_xy[0])
        facing = int(env.agent_dir)

        goal_pos = None
        wall_cells = []
        obstacle_cells = []

        for y in range(grid.height):
            for x in range(grid.width):
                cell = grid.get(x, y)

                if cell is None:
                    continue

                rc = (y, x)

                if cell.type == "goal":
                    goal_pos = rc
                elif cell.type == "wall":
                    wall_cells.append(rc)
                else:
                    obstacle_cells.append((rc, cell.type))

        return {
            "grid_size": (grid.height, grid.width),
            "agent": agent_rc,
            "goal": goal_pos,
            "facing": facing,
            "facing_name": DIRECTION_NAMES[facing],
            "wall_cells": wall_cells,
            "obstacle_cells": obstacle_cells,
        }

    def is_blocked(self, row, col):
        env = self.unwrapped
        grid = env.grid

        if row < 0 or row >= grid.height or col < 0 or col >= grid.width:
            return True, "wall"

        cell = grid.get(col, row)

        if cell is None:
            return False, None

        if cell.type == "goal":
            return False, None

        return True, cell.type

    def _is_free(self, row, col):
        blocked, _ = self.is_blocked(row, col)
        return not blocked

    def _neighbors(self, pos):
        row, col = pos
        neighbors = []

        for action, (dr, dc) in ACTION_TO_DELTA.items():
            next_row = row + dr
            next_col = col + dc

            if self._is_free(next_row, next_col):
                neighbors.append((action, (next_row, next_col)))

        return neighbors

    def _shortest_distance(self, start, goal):
        if start == goal:
            return 0

        queue = deque([(start, 0)])
        visited = {start}

        while queue:
            current, dist = queue.popleft()

            for _, neighbor in self._neighbors(current):
                if neighbor in visited:
                    continue

                if neighbor == goal:
                    return dist + 1

                visited.add(neighbor)
                queue.append((neighbor, dist + 1))

        return None

    def get_optimal_actions(self):
        """
        Return all cardinal actions that preserve a shortest path.

        This mirrors the original GridWorld logic:
        an action is optimal if moving to that neighbour reduces the
        BFS distance to the goal by exactly one.
        """
        state = self.get_state()
        start = state["agent"]
        goal = state["goal"]

        if goal is None:
            return set()

        if start == goal:
            return set()

        shortest = self._shortest_distance(start, goal)
        if shortest is None:
            return set()

        optimal = set()

        for action, next_pos in self._neighbors(start):
            next_dist = self._shortest_distance(next_pos, goal)

            if next_dist is not None and 1 + next_dist == shortest:
                optimal.add(action)

        return optimal

    def get_optimal_action_names(self):
        return [ACTION_NAMES[action] for action in sorted(self.get_optimal_actions())]

    def get_optimal_relative_action_names(self):
        """
        Convert optimal cardinal actions into egocentric action names
        relative to the current MiniGrid facing direction.
        """
        facing = self.unwrapped.agent_dir

        # MiniGrid facing -> project cardinal action index.
        # MiniGrid: 0=east, 1=south, 2=west, 3=north
        # Project:  0=north, 1=east, 2=south, 3=west
        facing_to_cardinal = {
            0: 1,
            1: 2,
            2: 3,
            3: 0,
        }

        cardinal_facing = facing_to_cardinal[facing]

        relative_names = []
        for action in sorted(self.get_optimal_actions()):
            relative_action = (action - cardinal_facing) % 4
            relative_names.append(RELATIVE_ACTION_NAMES[relative_action])

        return relative_names

    def step_cardinal(self, action):
        """
        Apply direct cardinal action:
        0=north, 1=east, 2=south, 3=west.

        Returns:
            state, reward, terminated, truncated, info
        """
        if action not in ACTION_TO_DELTA:
            raise ValueError(f"Unknown action: {action}")

        env = self.unwrapped

        old_xy = tuple(env.agent_pos)
        old_row, old_col = old_xy[1], old_xy[0]

        dr, dc = ACTION_TO_DELTA[action]
        new_row = old_row + dr
        new_col = old_col + dc

        blocked, blocked_type = self.is_blocked(new_row, new_col)

        hit_wall = blocked and blocked_type == "wall"
        hit_obstacle = blocked and blocked_type != "wall"

        if not blocked:
            env.agent_pos = (new_col, new_row)

        # Update facing to match attempted absolute action.
        # MiniGrid facing: 0=east, 1=south, 2=west, 3=north.
        action_to_facing = {
            0: 3,
            1: 0,
            2: 1,
            3: 2,
        }
        env.agent_dir = action_to_facing[action]

        state = self.get_state()
        terminated = state["agent"] == state["goal"]
        truncated = False

        reward = 10 if terminated else (-2 if blocked else -1)

        info = {
            "hit_wall": hit_wall,
            "hit_obstacle": hit_obstacle,
            "blocked_type": blocked_type,
            "action_name": ACTION_NAMES[action],
        }

        return state, reward, terminated, truncated, info

    def relative_to_cardinal(self, relative_action):
        """
        Convert egocentric relative action to project cardinal action index.

        relative_action:
            0=forward, 1=right, 2=backward, 3=left
        """
        facing = self.unwrapped.agent_dir

        facing_to_cardinal = {
            0: 1,
            1: 2,
            2: 3,
            3: 0,
        }

        cardinal_facing = facing_to_cardinal[facing]
        return (cardinal_facing + relative_action) % 4

    def render_text(self):
        env = self.unwrapped
        grid = env.grid
        agent_xy = tuple(env.agent_pos)
        facing = int(env.agent_dir)

        rows = []

        for y in range(grid.height):
            row = []

            for x in range(grid.width):
                if (x, y) == agent_xy:
                    row.append(DIRECTION_SYMBOLS[facing])
                    continue

                cell = grid.get(x, y)

                if cell is None:
                    row.append(".")
                elif cell.type == "goal":
                    row.append("G")
                elif cell.type == "wall":
                    row.append("X")
                else:
                    row.append("?")

            rows.append(" ".join(row))

        return "\n".join(rows)

    def make_allocentric_description(self):
        state = self.get_state()

        return f"""You are navigating a MiniGrid environment.

Grid size: {state["grid_size"][0]} x {state["grid_size"][1]}
Agent position: {state["agent"]}
Goal position: {state["goal"]}
Wall cells: {state["wall_cells"]}

Coordinate system:
- Row 0 is the top (north), and row numbers increase downward (south).
- Column 0 is the left (west), and column numbers increase to the right (east).

Choose exactly one action from:
north, east, south, west

Rules:
- Do not move into a wall or blocked cell.
- Do not move outside the grid.
- Choose the best next move on a shortest valid path to the goal.

Answer with one word only:
north, east, south, or west
"""

    def make_egocentric_description(self):
        state = self.get_state()

        return f"""You are navigating a MiniGrid environment.

Grid size: {state["grid_size"][0]} x {state["grid_size"][1]}
Agent position: {state["agent"]}
Goal position: {state["goal"]}
Wall cells: {state["wall_cells"]}

The agent is currently facing {state["facing_name"]}.

Relative actions:
- forward: move in the direction you are facing
- right: turn right relative to your current facing and move
- backward: turn around and move
- left: turn left relative to your current facing and move

Choose exactly one action from:
forward, right, backward, left

Rules:
- Do not move into a wall or blocked cell.
- Do not move outside the grid.
- Choose the best next move on a shortest valid path to the goal.

Answer with one word only:
forward, right, backward, or left
"""


def main():
    wrapper = MiniGridCardinalWrapper(
        env_name="MiniGrid-Empty-8x8-v0",
        seed=0,
    )

    wrapper.reset(seed=0)

    print("\nFULL GRID")
    print("=" * 60)
    print(wrapper.render_text())

    print("\nOPTIMAL ACTIONS")
    print("=" * 60)
    print("Optimal allocentric actions:", wrapper.get_optimal_action_names())
    print("Optimal egocentric actions:", wrapper.get_optimal_relative_action_names())

    print("\nALLOCENTRIC DESCRIPTION")
    print("=" * 60)
    print(wrapper.make_allocentric_description())

    print("\nEGOCENTRIC DESCRIPTION")
    print("=" * 60)
    print(wrapper.make_egocentric_description())

    print("\nTEST CARDINAL STEP")
    print("=" * 60)
    state, reward, terminated, truncated, info = wrapper.step_cardinal(1)
    print("Action: east")
    print("Reward:", reward)
    print("Terminated:", terminated)
    print("Info:", info)
    print(wrapper.render_text())

    print("\nOPTIMAL ACTIONS AFTER STEP")
    print("=" * 60)
    print("Optimal allocentric actions:", wrapper.get_optimal_action_names())
    print("Optimal egocentric actions:", wrapper.get_optimal_relative_action_names())

    wrapper.close()


if __name__ == "__main__":
    main()