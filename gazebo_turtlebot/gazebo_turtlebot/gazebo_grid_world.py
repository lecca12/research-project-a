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

    Grid convention:
        - row increases southward
        - column increases eastward

    Gazebo convention:
        - +x = east
        - +y = north

    Therefore:
        - increasing column -> increasing x
        - increasing row -> decreasing y
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
            tuple(cell)
            for cell in (
                obstacle_cells or []
            )
        )

        self.origin_x = None
        self.origin_y = None

        self.facing = "east"

    def initialise_from_current_pose(self):
        """
        Treat the robot's current Gazebo pose as start_cell.

        This means we do not depend on a particular absolute Gazebo
        spawn coordinate. Whatever pose the robot currently has becomes
        the world-space location corresponding to start_cell.
        """
        state = self.adapter.get_state()

        self.origin_x = (
            state["x"]
            - self.start_cell[1]
            * self.cell_size
        )

        self.origin_y = (
            state["y"]
            + self.start_cell[0]
            * self.cell_size
        )

        self.facing = self.yaw_to_facing(
            state["yaw"]
        )

    def require_initialised(self):
        if (
            self.origin_x is None
            or self.origin_y is None
        ):
            raise RuntimeError(
                "GazeboGridWorld is not initialised. "
                "Call initialise_from_current_pose() first."
            )

    def grid_to_world(
        self,
        row,
        col,
    ):
        """
        Convert a discrete grid cell into the corresponding Gazebo
        world-space coordinate.
        """
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

    def world_to_grid(
        self,
        x,
        y,
    ):
        """
        Convert a continuous Gazebo pose into the nearest discrete
        grid cell.
        """
        self.require_initialised()

        col = round(
            (x - self.origin_x)
            / self.cell_size
        )

        row = round(
            (self.origin_y - y)
            / self.cell_size
        )

        return (
            int(row),
            int(col),
        )

    @staticmethod
    def normalize_angle(angle):
        while angle > math.pi:
            angle -= (
                2.0 * math.pi
            )

        while angle < -math.pi:
            angle += (
                2.0 * math.pi
            )

        return angle

    def yaw_to_facing(
        self,
        yaw,
    ):
        """
        Convert a continuous yaw angle into the nearest cardinal
        facing direction.
        """
        if yaw is None:
            return self.facing

        headings = {
            "east": 0.0,
            "north": (
                math.pi / 2.0
            ),
            "west": math.pi,
            "south": (
                -math.pi / 2.0
            ),
        }

        best_name = None
        best_error = None

        for (
            name,
            target,
        ) in headings.items():
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

    def get_agent_cell(self):
        """
        Return the robot's current discrete grid cell.
        """
        state = (
            self.adapter.get_state()
        )

        return self.world_to_grid(
            state["x"],
            state["y"],
        )

    def get_state(self):
        """
        Return the current experiment state.
        """
        pose = (
            self.adapter.get_state()
        )

        agent_cell = (
            self.world_to_grid(
                pose["x"],
                pose["y"],
            )
        )

        facing = (
            self.yaw_to_facing(
                pose["yaw"]
            )
        )

        self.facing = facing

        return {
            "agent": agent_cell,
            "goal": self.goal_cell,
            "facing": facing,
            "pose": pose,
        }

    def is_inside_grid(
        self,
        row,
        col,
    ):
        return (
            0 <= row < self.rows
            and 0 <= col < self.cols
        )

    def is_obstacle(
        self,
        row,
        col,
    ):
        return (
            row,
            col,
        ) in self.obstacle_cells

    def is_blocked(
        self,
        row,
        col,
    ):
        """
        Return:

            (True, "boundary")
            (True, "obstacle")
            (False, None)
        """
        if not self.is_inside_grid(
            row,
            col,
        ):
            return (
                True,
                "boundary",
            )

        if self.is_obstacle(
            row,
            col,
        ):
            return (
                True,
                "obstacle",
            )

        return (
            False,
            None,
        )

    def get_legal_actions(self):
        """
        Return currently legal cardinal actions as strings.

        Example:
            ["north", "east", "west"]
        """
        row, col = (
            self.get_agent_cell()
        )

        legal = []

        for (
            action,
            (
                delta_row,
                delta_col,
            ),
        ) in CARDINAL_ACTIONS.items():
            next_row = (
                row + delta_row
            )

            next_col = (
                col + delta_col
            )

            blocked, _ = (
                self.is_blocked(
                    next_row,
                    next_col,
                )
            )

            if not blocked:
                legal.append(
                    action
                )

        return legal

    def relative_to_cardinal(
        self,
        relative_action,
    ):
        """
        Convert an egocentric relative action to a cardinal action.

        Relative action encoding:
            0 = forward
            1 = right
            2 = backward
            3 = left

        Example:
            facing north + right -> east
            facing east + right  -> south
        """
        facing_order = [
            "north",
            "east",
            "south",
            "west",
        ]

        state = self.get_state()

        facing = state["facing"]

        facing_index = (
            facing_order.index(
                facing
            )
        )

        cardinal_index = (
            facing_index
            + int(
                relative_action
            )
        ) % 4

        return facing_order[
            cardinal_index
        ]

    def cardinal_to_relative_name(
        self,
        cardinal_action,
    ):
        """
        Convert a cardinal action into the corresponding egocentric
        action name from the robot's current facing direction.

        Returns one of:
            forward
            right
            backward
            left
        """
        facing_order = [
            "north",
            "east",
            "south",
            "west",
        ]

        relative_names = [
            "forward",
            "right",
            "backward",
            "left",
        ]

        state = self.get_state()

        facing = state["facing"]

        facing_index = (
            facing_order.index(
                facing
            )
        )

        action_index = (
            facing_order.index(
                cardinal_action
            )
        )

        relative_index = (
            action_index
            - facing_index
        ) % 4

        return relative_names[
            relative_index
        ]

    def get_relative_legal_actions(self):
        """
        Return legal actions using the current egocentric vocabulary.
        """
        return [
            self.cardinal_to_relative_name(
                action
            )
            for action
            in self.get_legal_actions()
        ]

    def execute_action(
        self,
        action,
    ):
        """
        Execute one cardinal action if it is legal.

        Illegal actions are rejected before the TurtleBot moves.
        """
        action = (
            action.lower().strip()
        )

        if (
            action
            not in CARDINAL_ACTIONS
        ):
            raise ValueError(
                f"Unknown action "
                f"'{action}'. "
                "Expected north, east, "
                "south, or west."
            )

        row, col = (
            self.get_agent_cell()
        )

        (
            delta_row,
            delta_col,
        ) = CARDINAL_ACTIONS[
            action
        ]

        next_row = (
            row + delta_row
        )

        next_col = (
            col + delta_col
        )

        (
            blocked,
            blocked_type,
        ) = self.is_blocked(
            next_row,
            next_col,
        )

        if blocked:
            return {
                "executed": False,
                "blocked": True,
                "blocked_type": (
                    blocked_type
                ),
                "action": action,
                "from_cell": (
                    row,
                    col,
                ),
                "to_cell": (
                    next_row,
                    next_col,
                ),
            }

        before_pose = (
            self.adapter.get_state()
        )

        after_pose = (
            self.adapter.execute(
                action
            )
        )

        final_cell = (
            self.world_to_grid(
                after_pose["x"],
                after_pose["y"],
            )
        )

        self.facing = action

        return {
            "executed": True,
            "blocked": False,
            "blocked_type": None,
            "action": action,
            "from_cell": (
                row,
                col,
            ),
            "intended_cell": (
                next_row,
                next_col,
            ),
            "final_cell": (
                final_cell
            ),
            "before_pose": (
                before_pose
            ),
            "after_pose": (
                after_pose
            ),
        }

    def reached_goal(self):
        return (
            self.get_agent_cell()
            == self.goal_cell
        )

    def make_allocentric_description(
        self,
    ):
        """
        Allocentric prompt matching the existing grid-navigation format.
        """
        state = (
            self.get_state()
        )

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

    def make_egocentric_description(
        self,
    ):
        """
        Egocentric prompt matching the existing relative-action format.
        """
        state = (
            self.get_state()
        )

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