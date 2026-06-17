from collections import deque
import random

import numpy as np

DIRECTION_NAMES = {0: "east", 1: "south", 2: "west", 3: "north"}
DIRECTION_SYMBOLS = {0: ">", 1: "v", 2: "<", 3: "^"}

# Project convention: 0=north, 1=east, 2=south, 3=west
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


class CustomMiniGridWrapper:
    """
    Controlled MiniGrid-style wrapper using the same direct cardinal action
    convention as the original GridWorld project.

    Cells:
      0 = free
      1 = obstacle/wall
    """

    def __init__(self, size=8, obstacle_density=0.1, max_steps=None):
        self.size = size
        self.obstacle_density = obstacle_density
        self.max_steps = max_steps if max_steps is not None else size ** 2

        self.grid = None
        self.agent = None
        self.goal = None
        self.facing = None
        self.step_count = 0
        self.seed = None

    def reset(self, seed=0, state=None):
        self.step_count = 0

        if state is not None:
            self.seed = state.get("seed")
            self.grid = np.array(state["grid"], dtype=np.int64)
            self.agent = tuple(state["agent"])
            self.goal = tuple(state["goal"])
            self.facing = int(state["facing"])
            return self.get_state()

        self.seed = seed
        rng = random.Random(seed)

        while True:
            self.grid = np.zeros((self.size, self.size), dtype=np.int64)

            num_obstacles = int(self.size * self.size * self.obstacle_density)

            all_cells = [(r, c) for r in range(self.size) for c in range(self.size)]
            rng.shuffle(all_cells)

            obstacle_cells = all_cells[:num_obstacles]
            for r, c in obstacle_cells:
                self.grid[r, c] = 1

            free_cells = [(r, c) for r in range(self.size) for c in range(self.size) if self.grid[r, c] == 0]

            if len(free_cells) < 2:
                continue

            self.agent = rng.choice(free_cells)
            goal_candidates = [cell for cell in free_cells if cell != self.agent]
            self.goal = rng.choice(goal_candidates)

            self.facing = rng.choice([0, 1, 2, 3])

            if self._shortest_distance(self.agent, self.goal) is not None:
                break

        return self.get_state()

    def export_state(self):
        return {
            "seed": self.seed,
            "size": self.size,
            "obstacle_density": self.obstacle_density,
            "grid": self.grid.tolist(),
            "agent": list(self.agent),
            "goal": list(self.goal),
            "facing": int(self.facing),
            "max_steps": self.max_steps,
        }

    def get_state(self):
        return {
            "grid_size": (self.size, self.size),
            "agent": self.agent,
            "goal": self.goal,
            "facing": int(self.facing),
            "facing_name": DIRECTION_NAMES[self.facing],
            "obstacle_cells": self.get_obstacle_cells(),
        }

    def get_obstacle_cells(self):
        cells = []
        for r in range(self.size):
            for c in range(self.size):
                if self.grid[r, c] == 1:
                    cells.append((r, c))
        return cells

    def is_blocked(self, row, col):
        if row < 0 or row >= self.size or col < 0 or col >= self.size:
            return True, "wall"

        if self.grid[row, col] == 1:
            return True, "obstacle"

        return False, None

    def _is_free(self, row, col):
        blocked, _ = self.is_blocked(row, col)
        return not blocked

    def _neighbors(self, pos):
        row, col = pos
        neighbors = []

        for action, (dr, dc) in ACTION_TO_DELTA.items():
            nr, nc = row + dr, col + dc
            if self._is_free(nr, nc):
                neighbors.append((action, (nr, nc)))

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
        if self.agent == self.goal:
            return set()

        shortest = self._shortest_distance(self.agent, self.goal)

        if shortest is None:
            return set()

        optimal = set()

        for action, next_pos in self._neighbors(self.agent):
            next_dist = self._shortest_distance(next_pos, self.goal)

            if next_dist is not None and 1 + next_dist == shortest:
                optimal.add(action)

        return optimal

    def get_optimal_action_names(self):
        return [ACTION_NAMES[a] for a in sorted(self.get_optimal_actions())]

    def get_optimal_relative_action_names(self):
        # Internal facing follows MiniGrid convention:
        # 0=east, 1=south, 2=west, 3=north
        facing_to_cardinal = {
            0: 1,
            1: 2,
            2: 3,
            3: 0,
        }

        cardinal_facing = facing_to_cardinal[self.facing]

        relative_names = []
        for action in sorted(self.get_optimal_actions()):
            rel_action = (action - cardinal_facing) % 4
            relative_names.append(RELATIVE_ACTION_NAMES[rel_action])

        return relative_names

    def relative_to_cardinal(self, relative_action):
        facing_to_cardinal = {
            0: 1,
            1: 2,
            2: 3,
            3: 0,
        }

        cardinal_facing = facing_to_cardinal[self.facing]
        return (cardinal_facing + relative_action) % 4

    def step_cardinal(self, action):
        if action not in ACTION_TO_DELTA:
            raise ValueError(f"Unknown action: {action}")

        self.step_count += 1

        row, col = self.agent
        dr, dc = ACTION_TO_DELTA[action]
        nr, nc = row + dr, col + dc

        blocked, blocked_type = self.is_blocked(nr, nc)

        hit_wall = blocked and blocked_type == "wall"
        hit_obstacle = blocked and blocked_type == "obstacle"

        if not blocked:
            self.agent = (nr, nc)

        action_to_facing = {
            0: 3,  # north
            1: 0,  # east
            2: 1,  # south
            3: 2,  # west
        }
        self.facing = action_to_facing[action]

        terminated = self.agent == self.goal
        truncated = self.step_count >= self.max_steps and not terminated

        reward = 10 if terminated else (-2 if blocked else -1)

        info = {
            "hit_wall": hit_wall,
            "hit_obstacle": hit_obstacle,
            "blocked_type": blocked_type,
            "action_name": ACTION_NAMES[action],
        }

        return self.get_state(), reward, terminated, truncated, info

    def render_text(self):
        rows = []

        for r in range(self.size):
            row = []

            for c in range(self.size):
                if (r, c) == self.agent:
                    row.append(DIRECTION_SYMBOLS[self.facing])
                elif (r, c) == self.goal:
                    row.append("G")
                elif self.grid[r, c] == 1:
                    row.append("X")
                else:
                    row.append(".")

            rows.append(" ".join(row))

        return "\n".join(rows)

    def make_allocentric_description(self):
        state = self.get_state()

        return f"""You are navigating a MiniGrid-style grid world.

Grid size: {self.size} x {self.size}
Agent position: {state["agent"]}
Goal position: {state["goal"]}
Obstacle cells: {state["obstacle_cells"] if state["obstacle_cells"] else "None"}

Coordinate system:
- Row 0 is the top (north), and row numbers increase downward (south).
- Column 0 is the left (west), and column numbers increase to the right (east).

Choose exactly one action from:
north, east, south, west

Rules:
- Do not move into an obstacle.
- Do not move outside the grid.
- Choose the best next move on a shortest valid path to the goal.

Answer with one word only:
north, east, south, or west
"""

    def make_egocentric_description(self):
        state = self.get_state()

        return f"""You are navigating a MiniGrid-style grid world.

Grid size: {self.size} x {self.size}
Agent position: {state["agent"]}
Goal position: {state["goal"]}
Obstacle cells: {state["obstacle_cells"] if state["obstacle_cells"] else "None"}

The agent is currently facing {state["facing_name"]}.

Relative actions:
- forward: move in the direction you are facing
- right: turn right relative to your current facing and move
- backward: turn around and move
- left: turn left relative to your current facing and move

Choose exactly one action from:
forward, right, backward, left

Rules:
- Do not move into an obstacle.
- Do not move outside the grid.
- Choose the best next move on a shortest valid path to the goal.

Answer with one word only:
forward, right, backward, or left
"""


def generate_custom_minigrid_states(size, obstacle_density, num_states, start_seed=0):
    states = []

    for seed in range(start_seed, start_seed + num_states):
        env = CustomMiniGridWrapper(
            size=size,
            obstacle_density=obstacle_density,
            max_steps=size ** 2,
        )
        env.reset(seed=seed)
        states.append(env.export_state())

    return states


def main():
    env = CustomMiniGridWrapper(size=8, obstacle_density=0.3)
    env.reset(seed=0)

    print("\nCUSTOM MINIGRID STATE")
    print("=" * 80)
    print(env.render_text())

    print("\nSTATE")
    print("=" * 80)
    print(env.get_state())

    print("\nOPTIMAL ACTIONS")
    print("=" * 80)
    print("Allocentric:", env.get_optimal_action_names())
    print("Egocentric:", env.get_optimal_relative_action_names())

    print("\nALLOCENTRIC DESCRIPTION")
    print("=" * 80)
    print(env.make_allocentric_description())

    print("\nEGOCENTRIC DESCRIPTION")
    print("=" * 80)
    print(env.make_egocentric_description())


if __name__ == "__main__":
    main()