import math


CARDINAL_ACTIONS = {
    "north": (-1, 0),
    "east": (0, 1),
    "south": (1, 0),
    "west": (0, -1),
}


class GazeboGridWorld:
    """
    Discrete grid abstraction over a continuous Gazebo/TurtleBot world.

    The grid uses the same row/column convention as the MiniGrid experiments:

        row increases southward
        column increases eastward

    Gazebo uses:
        +x = east
        +y = north

    Therefore:
        col -> +x
        row -> -y
    """

    def __init__(
        self,
        adapter,
        rows=5,
        cols=5,
        cell_size=0.50,
        start_cell=(4, 0),
        goal_cell=(0, 4),
        obstacle_cells=None,
    ):
        self.adapter = adapter

        self.rows = int(rows)
        self.cols = int(cols)

        self.cell_size = float(cell_size)

        self.start_cell = tuple(start_cell)
        self.goal_cell = tuple(goal_cell)

        self.obstacle_cells = set(
            obstacle_cells or []
        )

        self.origin_x = None
        self.origin_y = None

        self.facing = "east"

    def initialise_from_current_pose(self):
        """
        Treat the robot's current Gazebo pose as start_cell.

        This avoids depending on Gazebo absolute spawn coordinates.
        """

        state = self.adapter.get_state()

        self.origin_x = (
            state["x"]
            - self.start_cell[1] * self.cell_size
        )

        self.origin_y = (
            state["y"]
            + self.start_cell[0] * self.cell_size
        )

        self.facing = self.yaw_to_facing(
            state["yaw"]
        )

    def require_initialised(self):
        if self.origin_x is None or self.origin_y is None:
            raise RuntimeError(
                "GazeboGridWorld is not initialised. "
                "Call initialise_from_current_pose() first."
            )

    def grid_to_world(self, row, col):
        self.require_initialised()

        x = (
            self.origin_x
            + col * self.cell_size
        )

        y = (
            self.origin_y
            - row * self.cell_size
        )

        return x, y

    def world_to_grid(self, x, y):
        self.require_initialised()

        col = round(
            (x - self.origin_x)
            / self.cell_size
        )

        row = round(
            (self.origin_y - y)
            / self.cell_size
        )

        return int(row), int(col)

    def yaw_to_facing(self, yaw):
        if yaw is None:
            return self.facing

        headings = {
            "east": 0.0,
            "north": math.pi / 2.0,
            "west": math.pi,
            "south": -math.pi / 2.0,
        }

        best_name = None
        best_error = None

        for name, target in headings.items():
            error = abs(
                self.normalize_angle(
                    yaw - target
                )
            )

            if (
                best_error is None
                or error < best_error
            ):
                best_error = error
                best_name = name

        return best_name

    @staticmethod
    def normalize_angle(angle):
        while angle > math.pi:
            angle -= 2.0 * math.pi

        while angle < -math.pi:
            angle += 2.0 * math.pi

        return angle

    def get_agent_cell(self):
        state = self.adapter.get_state()

        return self.world_to_grid(
            state["x"],
            state["y"],
        )

    def get_state(self):
        pose = self.adapter.get_state()

        agent_cell = self.world_to_grid(
            pose["x"],
            pose["y"],
        )

        facing = self.yaw_to_facing(
            pose["yaw"]
        )

        self.facing = facing

        return {
            "agent": agent_cell,
            "goal": self.goal_cell,
            "facing": facing,
            "pose": pose,
        }

    def is_inside_grid(self, row, col):
        return (
            0 <= row < self.rows
            and 0 <= col < self.cols
        )

    def is_obstacle(self, row, col):
        return (
            row,
            col,
        ) in self.obstacle_cells

    def is_blocked(self, row, col):
        if not self.is_inside_grid(
            row,
            col,
        ):
            return True, "boundary"

        if self.is_obstacle(
            row,
            col,
        ):
            return True, "obstacle"

        return False, None

    def get_legal_actions(self):
        row, col = self.get_agent_cell()

        legal = []

        for action, (
            delta_row,
            delta_col,
        ) in CARDINAL_ACTIONS.items():
            next_row = row + delta_row
            next_col = col + delta_col

            blocked, _ = self.is_blocked(
                next_row,
                next_col,
            )

            if not blocked:
                legal.append(action)

        return legal

    def execute_action(self, action):
        action = action.lower().strip()

        if action not in CARDINAL_ACTIONS:
            raise ValueError(
                f"Unknown action '{action}'."
            )

        row, col = self.get_agent_cell()

        delta_row, delta_col = (
            CARDINAL_ACTIONS[action]
        )

        next_row = row + delta_row
        next_col = col + delta_col

        blocked, blocked_type = self.is_blocked(
            next_row,
            next_col,
        )

        if blocked:
            return {
                "executed": False,
                "blocked": True,
                "blocked_type": blocked_type,
                "action": action,
                "from_cell": (row, col),
                "to_cell": (next_row, next_col),
            }

        before_pose = self.adapter.get_state()

        after_pose = self.adapter.execute(
            action
        )

        final_cell = self.world_to_grid(
            after_pose["x"],
            after_pose["y"],
        )

        self.facing = action

        return {
            "executed": True,
            "blocked": False,
            "blocked_type": None,
            "action": action,
            "from_cell": (row, col),
            "intended_cell": (
                next_row,
                next_col,
            ),
            "final_cell": final_cell,
            "before_pose": before_pose,
            "after_pose": after_pose,
        }

    def reached_goal(self):
        return (
            self.get_agent_cell()
            == self.goal_cell
        )

    def make_allocentric_description(self):
        state = self.get_state()

        return f"""You are navigating a grid world.

Grid size: {self.rows} x {self.cols}
Agent position: {state["agent"]}
Goal position: {state["goal"]}
Obstacle cells: {sorted(self.obstacle_cells)}

Coordinate system:
- Row 0 is the top (north), and row numbers increase downward (south).
- Column 0 is the left (west), and column numbers increase to the right (east).

Choose exactly one action from:
north, east, south, west

Rules:
- Do not move into an obstacle cell.
- Do not move outside the grid.
- Choose the best next move toward the goal.

Answer with one word only:
north, east, south, or west
"""

    def make_egocentric_description(self):
        state = self.get_state()

        return f"""You are navigating a grid world.

Grid size: {self.rows} x {self.cols}
Agent position: {state["agent"]}
Goal position: {state["goal"]}
Facing direction: {state["facing"]}
Obstacle cells: {sorted(self.obstacle_cells)}

Coordinate system:
- Row 0 is the top (north), and row numbers increase downward (south).
- Column 0 is the left (west), and column numbers increase to the right (east).

Choose exactly one action from:
forward, right, backward, left

Interpret those actions relative to the current facing direction.

Rules:
- Do not move into an obstacle cell.
- Do not move outside the grid.
- Choose the best next move toward the goal.

Answer with one word only:
forward, right, backward, or left
"""
