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
    Thin wrapper over official Farama MiniGrid environments.

    The wrapper uses direct cardinal actions rather than MiniGrid's native
    turn-left, turn-right, and move-forward action semantics. This preserves
    the experimental comparison between allocentric and egocentric action
    frames while keeping the underlying Farama environment unchanged.

    Project cardinal convention:
        0 = north
        1 = east
        2 = south
        3 = west
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

        self.seed = int(seed)
        self.obs, self.info = self.env.reset(seed=self.seed)
        return self.get_state()

    @property
    def unwrapped(self):
        return self.env.unwrapped

    def close(self):
        self.env.close()

    def _is_outer_boundary_cell(self, row, col):
        """
        Return True when a cell lies on the outer frame of the MiniGrid map.
        """
        grid = self.unwrapped.grid

        return (
            row == 0
            or row == grid.height - 1
            or col == 0
            or col == grid.width - 1
        )

    def get_state(self):
        """
        Return the current state using ordinary Python integers.

        The prompt-facing obstacle list includes interior walls and other
        blocking objects exactly once. Outer boundary walls are omitted from
        the obstacle list because the grid dimensions already define the
        boundary.
        """
        env = self.unwrapped
        grid = env.grid

        # Explicit int casts prevent np.int64 values leaking into prompts/JSON.
        agent_x = int(env.agent_pos[0])
        agent_y = int(env.agent_pos[1])
        agent_rc = (agent_y, agent_x)

        facing = int(env.agent_dir)

        goal_pos = None
        obstacle_cells = []

        for y in range(grid.height):
            for x in range(grid.width):
                cell = grid.get(x, y)

                if cell is None:
                    continue

                row = int(y)
                col = int(x)
                rc = (row, col)

                if cell.type == "goal":
                    goal_pos = rc
                    continue

                if cell.type == "wall":
                    # Do not enumerate the outer boundary in the prompt.
                    if self._is_outer_boundary_cell(row, col):
                        continue

                    # Interior walls are obstacle-like cells.
                    obstacle_cells.append(rc)
                    continue

                # Any other non-empty, non-goal object is obstacle-like for
                # this navigation task.
                obstacle_cells.append(rc)

        # Preserve deterministic ordering and prevent accidental duplicates.
        obstacle_cells = sorted(set(obstacle_cells))

        return {
            "grid_size": (int(grid.height), int(grid.width)),
            "agent": agent_rc,
            "goal": goal_pos,
            "facing": facing,
            "facing_name": DIRECTION_NAMES[facing],
            "obstacle_cells": obstacle_cells,
        }

    def is_blocked(self, row, col):
        """
        Determine whether a target cell is blocked.

        Returns:
            (blocked, blocked_type)

        blocked_type:
            "boundary"  - outside the grid or an outer-frame wall
            "obstacle"  - an interior wall or another blocking object
            None        - free cell or goal
        """
        env = self.unwrapped
        grid = env.grid

        row = int(row)
        col = int(col)

        if row < 0 or row >= grid.height or col < 0 or col >= grid.width:
            return True, "boundary"

        cell = grid.get(col, row)

        if cell is None:
            return False, None

        if cell.type == "goal":
            return False, None

        if cell.type == "wall":
            if self._is_outer_boundary_cell(row, col):
                return True, "boundary"

            return True, "obstacle"

        return True, "obstacle"

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
                neighbors.append(
                    (
                        action,
                        (int(next_row), int(next_col)),
                    )
                )

        return neighbors

    def _shortest_distance(self, start, goal):
        """
        Compute BFS distance from start to goal.

        Returns None when the goal is unreachable.
        """
        if start == goal:
            return 0

        queue = deque([(start, 0)])
        visited = {start}

        while queue:
            current, distance = queue.popleft()

            for _, neighbour in self._neighbors(current):
                if neighbour in visited:
                    continue

                if neighbour == goal:
                    return distance + 1

                visited.add(neighbour)
                queue.append((neighbour, distance + 1))

        return None

    def get_optimal_actions(self):
        """
        Return all cardinal actions that preserve a shortest path.

        An action is optimal when moving to its neighbouring cell reduces
        the BFS distance to the goal by exactly one. This accepts every
        equally optimal first action rather than selecting one shortest path.
        """
        state = self.get_state()
        start = state["agent"]
        goal = state["goal"]

        if goal is None or start == goal:
            return set()

        shortest_distance = self._shortest_distance(start, goal)

        if shortest_distance is None:
            return set()

        optimal_actions = set()

        for action, next_pos in self._neighbors(start):
            next_distance = self._shortest_distance(next_pos, goal)

            if (
                next_distance is not None
                and 1 + next_distance == shortest_distance
            ):
                optimal_actions.add(action)

        return optimal_actions

    def get_optimal_action_names(self):
        return [
            ACTION_NAMES[action]
            for action in sorted(self.get_optimal_actions())
        ]

    def get_optimal_relative_action_names(self):
        """
        Convert optimal cardinal actions into egocentric action names
        relative to the current facing direction.
        """
        facing = int(self.unwrapped.agent_dir)

        # MiniGrid facing:
        # 0=east, 1=south, 2=west, 3=north
        #
        # Project cardinal:
        # 0=north, 1=east, 2=south, 3=west
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
            relative_names.append(
                RELATIVE_ACTION_NAMES[relative_action]
            )

        return relative_names

    def step_cardinal(self, action):
        """
        Apply a direct cardinal action.

        Actions:
            0 = north
            1 = east
            2 = south
            3 = west

        Returns:
            state, reward, terminated, truncated, info
        """
        if action not in ACTION_TO_DELTA:
            raise ValueError(f"Unknown action: {action}")

        env = self.unwrapped

        old_col = int(env.agent_pos[0])
        old_row = int(env.agent_pos[1])

        dr, dc = ACTION_TO_DELTA[action]
        new_row = old_row + dr
        new_col = old_col + dc

        blocked, blocked_type = self.is_blocked(new_row, new_col)

        hit_wall = blocked and blocked_type == "boundary"
        hit_obstacle = blocked and blocked_type == "obstacle"

        if not blocked:
            env.agent_pos = (int(new_col), int(new_row))

        # Update facing to match the attempted absolute action.
        # MiniGrid facing:
        # 0=east, 1=south, 2=west, 3=north
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
            "hit_wall": bool(hit_wall),
            "hit_obstacle": bool(hit_obstacle),
            "blocked_type": blocked_type,
            "action_name": ACTION_NAMES[action],
        }

        return state, reward, terminated, truncated, info

    def relative_to_cardinal(self, relative_action):
        """
        Convert an egocentric relative action into a cardinal action.

        Relative actions:
            0 = forward
            1 = right
            2 = backward
            3 = left
        """
        if relative_action not in RELATIVE_ACTION_NAMES:
            raise ValueError(
                f"Unknown relative action: {relative_action}"
            )

        facing = int(self.unwrapped.agent_dir)

        facing_to_cardinal = {
            0: 1,
            1: 2,
            2: 3,
            3: 0,
        }

        cardinal_facing = facing_to_cardinal[facing]
        return (cardinal_facing + relative_action) % 4

    def render_text(self):
        """
        Render the complete MiniGrid state as text.

        Symbols:
            . = free cell
            X = outer boundary wall
            # = interior wall
            G = goal
            L = lava
            ? = another object
            > v < ^ = agent facing direction
        """
        env = self.unwrapped
        grid = env.grid

        agent_x = int(env.agent_pos[0])
        agent_y = int(env.agent_pos[1])
        agent_xy = (agent_x, agent_y)

        facing = int(env.agent_dir)
        rows = []

        for y in range(grid.height):
            row_symbols = []

            for x in range(grid.width):
                if (x, y) == agent_xy:
                    row_symbols.append(DIRECTION_SYMBOLS[facing])
                    continue

                cell = grid.get(x, y)

                if cell is None:
                    row_symbols.append(".")
                elif cell.type == "goal":
                    row_symbols.append("G")
                elif cell.type == "wall":
                    if self._is_outer_boundary_cell(y, x):
                        row_symbols.append("X")
                    else:
                        row_symbols.append("#")
                elif cell.type == "lava":
                    row_symbols.append("L")
                else:
                    row_symbols.append("?")

            rows.append(" ".join(row_symbols))

        return "\n".join(rows)

    def make_allocentric_description(self):
        state = self.get_state()

        return f"""You are navigating a MiniGrid environment.

Grid size: {state["grid_size"][0]} x {state["grid_size"][1]}
Agent position: {state["agent"]}
Goal position: {state["goal"]}
Obstacle cells: {state["obstacle_cells"]}

Coordinate system:
- Row 0 is the top (north), and row numbers increase downward (south).
- Column 0 is the left (west), and column numbers increase to the right (east).

Choose exactly one action from:
north, east, south, west

Rules:
- Do not move into an obstacle cell.
- Do not move outside the grid or into the outer boundary.
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
Obstacle cells: {state["obstacle_cells"]}

Coordinate system:
- Row 0 is the top (north), and row numbers increase downward (south).
- Column 0 is the left (west), and column numbers increase to the right (east).

The agent is currently facing {state["facing_name"]}.

Relative actions:
- forward: move in the direction you are facing
- right: turn right relative to your current facing and move
- backward: turn around and move
- left: turn left relative to your current facing and move

Choose exactly one action from:
forward, right, backward, left

Rules:
- Do not move into an obstacle cell.
- Do not move outside the grid or into the outer boundary.
- Choose the best next move on a shortest valid path to the goal.

Answer with one word only:
forward, right, backward, or left
"""


def main():
    wrapper = MiniGridCardinalWrapper(
        env_name="MiniGrid-SimpleCrossingS9N1-v0",
        seed=0,
    )

    wrapper.reset(seed=0)

    print("\nFULL GRID")
    print("=" * 60)
    print(wrapper.render_text())

    print("\nSTATE")
    print("=" * 60)
    print(wrapper.get_state())

    print("\nOPTIMAL ACTIONS")
    print("=" * 60)
    print(
        "Optimal allocentric actions:",
        wrapper.get_optimal_action_names(),
    )
    print(
        "Optimal egocentric actions:",
        wrapper.get_optimal_relative_action_names(),
    )

    print("\nALLOCENTRIC DESCRIPTION")
    print("=" * 60)
    print(wrapper.make_allocentric_description())

    print("\nEGOCENTRIC DESCRIPTION")
    print("=" * 60)
    print(wrapper.make_egocentric_description())

    wrapper.close()


if __name__ == "__main__":
    main()