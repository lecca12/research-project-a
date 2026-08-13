"""
GazeboGridEnv: the evaluation surface for the Gazebo transfer study.

This subclasses your existing GazeboGridWorld and changes nothing about how the
robot moves. It only makes the state, legality and prompt text match the text
experiment exactly, and it adds the BFS oracle that the text experiment uses to
score each step.

Copy this file to: gazebo_turtlebot/gazebo_turtlebot/gazebo_grid_env.py

What it adds on top of GazeboGridWorld:

1. Outer-ring boundary, so a 9 x 9 layout has a 7 x 7 traversable interior and
   the cell coordinates match MiniGrid exactly.
2. Prompt text copied verbatim from MiniGridCardinalWrapper.
3. BFS oracle: get_optimal_actions / get_optimal_action_names /
   get_optimal_relative_action_names.
4. render_text(), so traces look like the text-experiment traces.
5. Dead-reckoned cell and facing, so odometry drift cannot corrupt legality.
"""

import json
from collections import deque
from pathlib import Path

from gazebo_turtlebot.gazebo_grid_world import GazeboGridWorld


# Project cardinal convention, identical to minigrid_wrapper.py.
# 0 = north, 1 = east, 2 = south, 3 = west
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

CARDINAL_ORDER = ["north", "east", "south", "west"]

DIRECTION_SYMBOLS = {
    "east": ">",
    "south": "v",
    "west": "<",
    "north": "^",
}


def load_layout(path):
    """
    Load a frozen MiniGrid layout.

    Cells are converted back to tuples. This matters: the prompt prints the
    obstacle list directly, and a list of lists would render as [[2, 2], ...]
    instead of [(2, 2), ...] and break the match with the text experiment.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))

    return {
        "env_name": data["env_name"],
        "seed": int(data["seed"]),
        "grid_size": tuple(data["grid_size"]),
        "start_cell": tuple(data["start_cell"]),
        "start_facing": data["start_facing"],
        "goal_cell": tuple(data["goal_cell"]),
        "obstacle_cells": [tuple(cell) for cell in data["obstacle_cells"]],
        "optimal_path_length": int(data["optimal_path_length"]),
    }


class GazeboGridEnv(GazeboGridWorld):
    """
    Grid environment backed by the TurtleBot3 in Gazebo.

    The agent cell and facing are dead-reckoned on the waypoint graph. Odometry
    is read after every move and compared against the expected cell, but it is
    never used to decide legality. This is deliberate: the drive controller
    stops when the remaining distance is within DISTANCE_TOLERANCE, so each move
    lands slightly short, and after enough moves an odometry-derived cell index
    would silently jump by one.
    """

    def __init__(self, adapter, layout, cell_size=0.50):
        super().__init__(
            adapter=adapter,
            rows=layout["grid_size"][0],
            cols=layout["grid_size"][1],
            cell_size=cell_size,
            start_cell=layout["start_cell"],
            goal_cell=layout["goal_cell"],
            obstacle_cells=layout["obstacle_cells"],
        )

        self.layout = layout
        self.start_facing = layout["start_facing"]

        # Authoritative discrete state.
        self.cell = tuple(layout["start_cell"])
        self.facing = layout["start_facing"]

        self.max_step_error_metres = 0.0

        # Fixed reference frame, set once at initialisation and never
        # re-anchored, so cumulative drift stays measurable.
        self.initial_origin_x = None
        self.initial_origin_y = None

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def initialise_from_current_pose(self):
        """
        Anchor the grid on the robot's current pose and reset discrete state.

        The robot's current pose becomes start_cell. With MiniGrid coordinates
        the start cell is (1, 1), so the computed origin (cell (0, 0)) sits one
        cell north-west of the robot, inside the boundary ring. That point is
        never driven to; it is only the reference for grid_to_world. Do not
        "correct" it.
        """
        super().initialise_from_current_pose()

        odometry_facing = self.facing

        self.cell = tuple(self.start_cell)
        self.facing = self.start_facing

        self.initial_origin_x = self.origin_x
        self.initial_origin_y = self.origin_y

        if odometry_facing != self.start_facing:
            # TODO: the robot is not facing the direction the layout expects.
            # The egocentric prompt and the relative-action mapping both depend
            # on this, so it must match before the episode starts.
            # Either spawn the robot with the right yaw, or rotate it once here
            # with self.adapter.controller.rotate_to_heading(target_yaw).
            raise RuntimeError(
                f"Robot is facing {odometry_facing}, but the layout starts "
                f"facing {self.start_facing}. Rotate the robot before running."
            )

    # ------------------------------------------------------------------
    # Legality: MiniGrid semantics
    # ------------------------------------------------------------------

    def is_blocked(self, row, col):
        """
        Treat the outer ring as boundary, matching MiniGrid.

        In MiniGrid a 9 x 9 map is a solid wall frame around a 7 x 7 interior,
        and cell coordinates are stated in full-grid terms. Row 0, row 8,
        column 0 and column 8 are therefore not traversable.

        GazeboGridWorld routes both get_legal_actions() and execute_action()
        through this method, so overriding it here is enough.
        """
        row = int(row)
        col = int(col)

        outside = row < 0 or row >= self.rows or col < 0 or col >= self.cols

        on_ring = row == 0 or row == self.rows - 1 or col == 0 or col == self.cols - 1

        if outside or on_ring:
            return True, "boundary"

        if self.is_obstacle(row, col):
            return True, "obstacle"

        return False, None

    def get_agent_cell(self):
        """
        Return the dead-reckoned cell, not the odometry-derived cell.
        """
        return self.cell

    def get_odometry_cell(self):
        """
        Return the cell implied by raw odometry. Diagnostic only.
        """
        pose = self.adapter.get_state()
        return self.world_to_grid(pose["x"], pose["y"])

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    def get_state(self):
        """
        Return state with the same keys the text experiment uses.

        "facing" stays a cardinal string so the existing gazebo_policy helpers
        keep working; "facing_name" is the same value under the name the
        egocentric prompt expects.
        """
        pose = self.adapter.get_state()

        return {
            "grid_size": (self.rows, self.cols),
            "agent": tuple(self.cell),
            "goal": tuple(self.goal_cell),
            "facing": self.facing,
            "facing_name": self.facing,
            "obstacle_cells": sorted(self.obstacle_cells),
            "pose": pose,
        }

    # ------------------------------------------------------------------
    # BFS oracle, copied from MiniGridCardinalWrapper
    # ------------------------------------------------------------------

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
                neighbors.append((action, (int(next_row), int(next_col))))

        return neighbors

    def _shortest_distance(self, start, goal):
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
        Return every cardinal action that stays on a shortest path.
        """
        start = tuple(self.cell)
        goal = tuple(self.goal_cell)

        if goal is None or start == goal:
            return set()

        shortest_distance = self._shortest_distance(start, goal)

        if shortest_distance is None:
            return set()

        optimal_actions = set()

        for action, next_pos in self._neighbors(start):
            next_distance = self._shortest_distance(next_pos, goal)

            if next_distance is not None and 1 + next_distance == shortest_distance:
                optimal_actions.add(action)

        return optimal_actions

    def get_optimal_action_names(self):
        return [ACTION_NAMES[action] for action in sorted(self.get_optimal_actions())]

    def get_optimal_relative_action_names(self):
        cardinal_facing = CARDINAL_ORDER.index(self.facing)

        relative_names = []

        for action in sorted(self.get_optimal_actions()):
            relative_action = (action - cardinal_facing) % 4
            relative_names.append(RELATIVE_ACTION_NAMES[relative_action])

        return relative_names

    def get_legal_action_names_in_project_order(self):
        """
        Legal cardinal actions in north, east, south, west order.

        The text experiment builds the shield's legal-action list by sorting
        cardinal action numbers, which gives north, east, south, west. Sorting
        the names alphabetically instead gives east, north, south, west, which
        changes the shield reprompt text.
        """
        legal = set(self.get_legal_actions())

        return [
            ACTION_NAMES[action]
            for action in sorted(
                index for index, name in ACTION_NAMES.items() if name in legal
            )
        ]

    # ------------------------------------------------------------------
    # Movement
    # ------------------------------------------------------------------

    def execute_action(self, action):
        """
        Execute one cardinal action, refusing illegal ones before any motion.

        Adds dead reckoning and an odometry cross-check to the inherited
        behaviour. The refusal path is unchanged: the robot does not move and
        does not rotate.
        """
        result = super().execute_action(action)

        if not result["executed"]:
            # Facing is untouched on a refusal. Under the legality shield and
            # the reprompt control this branch is never reached, because an
            # illegal action is never returned for execution.
            result["dead_reckoned_cell"] = tuple(self.cell)
            result["odom_cell_divergence"] = None
            return result

        self.cell = tuple(result["intended_cell"])
        self.facing = result["action"]

        # Measure how far this single move fell short of one clean cell step,
        # not how far the robot has drifted from the ideal grid overall. The
        # drive controller stops a few centimetres early every time, so the
        # cumulative offset grows steadily and would trip a threshold near the
        # end of a good episode. The per-step error only grows when something
        # has actually gone wrong, such as the robot being obstructed.
        expected_dx, expected_dy = self.expected_displacement(action)

        actual_dx = result["after_pose"]["x"] - result["before_pose"]["x"]
        actual_dy = result["after_pose"]["y"] - result["before_pose"]["y"]

        step_error = (
            (actual_dx - expected_dx) ** 2 + (actual_dy - expected_dy) ** 2
        ) ** 0.5

        self.max_step_error_metres = max(
            self.max_step_error_metres,
            step_error,
        )

        result["dead_reckoned_cell"] = tuple(self.cell)
        result["odom_cell_before_reanchor"] = tuple(result["final_cell"])
        result["step_error_metres"] = round(step_error, 4)
        result["cumulative_offset_metres"] = round(
            self.cumulative_offset_metres(),
            4,
        )

        # A move that lands more than half a cell away from where it should
        # have is not a rounding problem, it is a failed move.
        if step_error > 0.5 * self.cell_size:
            raise RuntimeError(
                f"Move '{action}' finished {step_error:.3f} m away from the "
                f"expected position for cell {self.cell}. Stopping before the "
                "discrete state becomes unreliable."
            )

        # Re-anchor the grid on where the robot actually is, so the small
        # shortfall on each move does not accumulate into the readback.
        self.reanchor_origin()

        return result

    def expected_displacement(self, action):
        """
        The world-frame displacement one clean move should produce.
        """
        delta_row, delta_col = {
            name: ACTION_TO_DELTA[index] for index, name in ACTION_NAMES.items()
        }[action]

        return (
            delta_col * self.cell_size,
            -delta_row * self.cell_size,
        )

    def reanchor_origin(self):
        """
        Redefine the grid origin so the current cell maps to the current pose.
        """
        pose = self.adapter.get_state()

        self.origin_x = pose["x"] - self.cell[1] * self.cell_size
        self.origin_y = pose["y"] + self.cell[0] * self.cell_size

    def cumulative_offset_metres(self):
        """
        How far the robot has drifted from the ideal grid since the episode
        started, measured in the fixed frame set at initialisation.

        Diagnostic only. Nothing depends on this value, because the discrete
        state is dead-reckoned. It is logged so the size of the drift over a
        real episode is a measured number rather than a guess.
        """
        if self.initial_origin_x is None:
            return 0.0

        pose = self.adapter.get_state()

        target_x = self.initial_origin_x + self.cell[1] * self.cell_size
        target_y = self.initial_origin_y - self.cell[0] * self.cell_size

        return ((pose["x"] - target_x) ** 2 + (pose["y"] - target_y) ** 2) ** 0.5

    def step_cardinal(self, action):
        """
        Apply a cardinal action given as a number, matching the text
        experiment's step_cardinal signature.

        Returns:
            state, reward, terminated, truncated, info
        """
        if action not in ACTION_TO_DELTA:
            raise ValueError(f"Unknown action: {action}")

        result = self.execute_action(ACTION_NAMES[action])

        blocked = result["blocked"]
        blocked_type = result["blocked_type"]

        state = self.get_state()
        terminated = state["agent"] == state["goal"]
        truncated = False

        reward = 10 if terminated else (-2 if blocked else -1)

        info = {
            "hit_wall": bool(blocked and blocked_type == "boundary"),
            "hit_obstacle": bool(blocked and blocked_type == "obstacle"),
            "blocked_type": blocked_type,
            "action_name": ACTION_NAMES[action],
            "execution": result,
        }

        return state, reward, terminated, truncated, info

    def reached_goal(self):
        return tuple(self.cell) == tuple(self.goal_cell)

    # ------------------------------------------------------------------
    # Rendering and prompts
    # ------------------------------------------------------------------

    def render_text(self):
        """
        Render the layout as text, using the text experiment's symbols.

        . = free cell
        X = outer boundary wall
        # = interior wall
        G = goal
        > v < ^ = agent facing direction
        """
        rows = []

        for row in range(self.rows):
            row_symbols = []

            for col in range(self.cols):
                cell = (row, col)

                if cell == tuple(self.cell):
                    row_symbols.append(DIRECTION_SYMBOLS[self.facing])
                    continue

                on_ring = (
                    row == 0 or row == self.rows - 1 or col == 0 or col == self.cols - 1
                )

                if on_ring:
                    row_symbols.append("X")
                elif cell == tuple(self.goal_cell):
                    row_symbols.append("G")
                elif self.is_obstacle(row, col):
                    row_symbols.append("#")
                else:
                    row_symbols.append(".")

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
