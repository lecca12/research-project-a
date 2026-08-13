import argparse
import json
import math
import time
from pathlib import Path

import rclpy

from gazebo_msgs.srv import (
    DeleteEntity,
    SpawnEntity,
)

from llm_policy import make_openai_policy_fn

from minigrid_wrapper import (
    MiniGridCardinalWrapper,
    ACTION_NAMES,
    ACTION_TO_DELTA,
)

# Reuse the already-validated Farama experiment logic.
# The existing Farama runner itself is NOT modified.
from run_farama_minigrid_experiment import (
    choose_action,
    get_prompt,
    get_legal_cardinal_actions,
)

from gazebo_turtlebot.gazebo_adapter import (
    GazeboAdapter,
)

from gazebo_turtlebot.grid_action_controller import (
    CARDINAL_HEADINGS,
    CELL_SIZE_METRES,
)

from gazebo_turtlebot.gazebo_waypoint_executor import (
    GazeboWaypointExecutor,
)


MAX_STEPS_BY_ENV = {
    "MiniGrid-SimpleCrossingS9N1-v0": 9 ** 2,
    "MiniGrid-SimpleCrossingS9N2-v0": 9 ** 2,
    "MiniGrid-SimpleCrossingS9N3-v0": 9 ** 2,
    "MiniGrid-FourRooms-v0": 19 ** 2,
}


PHYSICAL_WALL_MODEL_NAME = (
    "farama_physical_walls"
)

WALL_HEIGHT_METRES = 0.30

GAZEBO_SERVICE_TIMEOUT_SEC = 30.0


def make_json_safe(obj):
    if hasattr(obj, "item"):
        return obj.item()

    if isinstance(obj, tuple):
        return [
            make_json_safe(item)
            for item in obj
        ]

    if isinstance(obj, list):
        return [
            make_json_safe(item)
            for item in obj
        ]

    if isinstance(obj, set):
        return [
            make_json_safe(item)
            for item in sorted(obj)
        ]

    if isinstance(obj, dict):
        return {
            key: make_json_safe(value)
            for key, value in obj.items()
        }

    return obj


def save_result(
    result,
    output_path,
):
    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = (
        output_path.with_suffix(
            ".tmp"
        )
    )

    with temporary_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            make_json_safe(result),
            file,
            indent=2,
        )

    temporary_path.replace(
        output_path
    )


def align_robot_to_facing(
    adapter,
    facing_name,
):
    """
    Match the TurtleBot's physical heading to the exact initial
    MiniGrid facing direction.
    """
    target_yaw = (
        CARDINAL_HEADINGS[
            facing_name
        ]
    )

    adapter.controller.get_logger().info(
        "Aligning physical robot to "
        f"MiniGrid facing: {facing_name}"
    )

    adapter.controller.rotate_to_heading(
        target_yaw
    )

    adapter.controller.stop()

    time.sleep(0.2)


def expected_world_position(
    anchor_x,
    anchor_y,
    anchor_cell,
    target_cell,
):
    """
    Convert a MiniGrid cell into an absolute Gazebo coordinate.

    MiniGrid:
        row increases south
        column increases east

    Gazebo:
        +x = east
        +y = north
    """
    start_row, start_col = (
        anchor_cell
    )

    row, col = target_cell

    x = (
        anchor_x
        + (col - start_col)
        * CELL_SIZE_METRES
    )

    y = (
        anchor_y
        - (row - start_row)
        * CELL_SIZE_METRES
    )

    return x, y


def physical_position_error(
    robot_state,
    expected_x,
    expected_y,
):
    return math.hypot(
        robot_state["x"]
        - expected_x,
        robot_state["y"]
        - expected_y,
    )


def get_outer_wall_cells(
    grid_size,
):
    """
    Return all cells belonging to MiniGrid's outer wall frame.
    """
    rows, cols = grid_size

    cells = set()

    for col in range(cols):
        cells.add(
            (0, col)
        )

        cells.add(
            (rows - 1, col)
        )

    for row in range(rows):
        cells.add(
            (row, 0)
        )

        cells.add(
            (row, cols - 1)
        )

    return cells


def make_wall_link_sdf(
    row,
    col,
    x,
    y,
    interior,
):
    """
    Create one static wall-cell link.

    Outer boundary walls and interior obstacles are visually different,
    but their collision geometry is identical.
    """
    z = (
        WALL_HEIGHT_METRES / 2.0
    )

    if interior:
        ambient = (
            "0.55 0.20 0.20 1"
        )

        diffuse = (
            "0.65 0.25 0.25 1"
        )

    else:
        ambient = (
            "0.35 0.35 0.35 1"
        )

        diffuse = (
            "0.45 0.45 0.45 1"
        )

    link_name = (
        f"wall_r{row}_c{col}"
    )

    return f"""
    <link name="{link_name}">
      <pose>
        {x:.6f} {y:.6f} {z:.6f} 0 0 0
      </pose>

      <collision name="collision">
        <geometry>
          <box>
            <size>
              {CELL_SIZE_METRES:.6f}
              {CELL_SIZE_METRES:.6f}
              {WALL_HEIGHT_METRES:.6f}
            </size>
          </box>
        </geometry>
      </collision>

      <visual name="visual">
        <geometry>
          <box>
            <size>
              {CELL_SIZE_METRES:.6f}
              {CELL_SIZE_METRES:.6f}
              {WALL_HEIGHT_METRES:.6f}
            </size>
          </box>
        </geometry>

        <material>
          <ambient>{ambient}</ambient>
          <diffuse>{diffuse}</diffuse>
        </material>
      </visual>
    </link>
"""


def make_physical_world_sdf(
    grid_state,
    anchor_x,
    anchor_y,
    anchor_cell,
):
    """
    Create one Gazebo model containing every physical wall cell.

    This includes:
        - MiniGrid's entire outer boundary
        - every interior obstacle supplied in the exact Farama state
    """
    grid_size = tuple(
        grid_state[
            "grid_size"
        ]
    )

    interior_cells = set(
        tuple(cell)
        for cell in grid_state[
            "obstacle_cells"
        ]
    )

    outer_cells = (
        get_outer_wall_cells(
            grid_size
        )
    )

    all_wall_cells = (
        outer_cells
        | interior_cells
    )

    links = []

    for row, col in sorted(
        all_wall_cells
    ):
        x, y = (
            expected_world_position(
                anchor_x=anchor_x,
                anchor_y=anchor_y,
                anchor_cell=anchor_cell,
                target_cell=(
                    row,
                    col,
                ),
            )
        )

        links.append(
            make_wall_link_sdf(
                row=row,
                col=col,
                x=x,
                y=y,
                interior=(
                    (row, col)
                    in interior_cells
                ),
            )
        )

    link_text = "".join(
        links
    )

    sdf = f"""<?xml version="1.0"?>
<sdf version="1.6">
  <model name="{PHYSICAL_WALL_MODEL_NAME}">
    <static>true</static>

{link_text}

  </model>
</sdf>
"""

    metadata = {
        "model_name": (
            PHYSICAL_WALL_MODEL_NAME
        ),
        "outer_wall_cells": (
            sorted(
                outer_cells
            )
        ),
        "interior_wall_cells": (
            sorted(
                interior_cells
            )
        ),
        "all_wall_cells": (
            sorted(
                all_wall_cells
            )
        ),
        "num_outer_wall_cells": (
            len(
                outer_cells
            )
        ),
        "num_interior_wall_cells": (
            len(
                interior_cells
            )
        ),
        "num_total_wall_cells": (
            len(
                all_wall_cells
            )
        ),
    }

    return sdf, metadata


class GazeboPhysicalWorld:
    """
    Manage the physical MiniGrid wall model through Gazebo ROS services.

    All wall cells are spawned as one static Gazebo model so setup and
    cleanup require only one entity.
    """

    def __init__(
        self,
        node,
    ):
        self.node = node

        self.spawn_client = (
            node.create_client(
                SpawnEntity,
                "/spawn_entity",
            )
        )

        self.delete_client = (
            node.create_client(
                DeleteEntity,
                "/delete_entity",
            )
        )

        self.spawned = False

    def wait_for_services(
        self,
    ):
        self.node.get_logger().info(
            "Waiting for Gazebo "
            "/spawn_entity service..."
        )

        if not self.spawn_client.wait_for_service(
            timeout_sec=(
                GAZEBO_SERVICE_TIMEOUT_SEC
            )
        ):
            raise RuntimeError(
                "Gazebo /spawn_entity service "
                "is unavailable. "
                "Make sure Gazebo started with "
                "GazeboRosFactory."
            )

        self.node.get_logger().info(
            "Waiting for Gazebo "
            "/delete_entity service..."
        )

        if not self.delete_client.wait_for_service(
            timeout_sec=(
                GAZEBO_SERVICE_TIMEOUT_SEC
            )
        ):
            raise RuntimeError(
                "Gazebo /delete_entity service "
                "is unavailable."
            )

    def _wait_for_future(
        self,
        future,
        operation_name,
    ):
        rclpy.spin_until_future_complete(
            self.node,
            future,
            timeout_sec=(
                GAZEBO_SERVICE_TIMEOUT_SEC
            ),
        )

        if not future.done():
            raise RuntimeError(
                f"{operation_name} timed out."
            )

        response = future.result()

        if response is None:
            raise RuntimeError(
                f"{operation_name} returned "
                "no response."
            )

        return response

    def delete_existing_model(
        self,
        quiet=True,
    ):
        """
        Delete a stale physical-wall model if one exists.

        Failure is ignored when quiet=True because the normal case before
        the first spawn is simply that no model exists yet.
        """
        request = (
            DeleteEntity.Request()
        )

        request.name = (
            PHYSICAL_WALL_MODEL_NAME
        )

        future = (
            self.delete_client.call_async(
                request
            )
        )

        response = (
            self._wait_for_future(
                future,
                "DeleteEntity",
            )
        )

        if response.success:
            self.node.get_logger().info(
                "Deleted existing physical "
                "Farama wall model."
            )

            self.spawned = False

            time.sleep(0.2)

        elif not quiet:
            self.node.get_logger().warning(
                "Could not delete physical "
                "wall model: "
                f"{response.status_message}"
            )

        return response.success

    def spawn(
        self,
        sdf,
    ):
        """
        Spawn the complete physical MiniGrid wall model.
        """
        self.wait_for_services()

        # Makes rerunning the script safer if a previous run left the
        # same named model behind.
        self.delete_existing_model(
            quiet=True
        )

        request = (
            SpawnEntity.Request()
        )

        request.name = (
            PHYSICAL_WALL_MODEL_NAME
        )

        request.xml = sdf

        request.robot_namespace = ""

        request.reference_frame = (
            "world"
        )

        request.initial_pose.position.x = (
            0.0
        )

        request.initial_pose.position.y = (
            0.0
        )

        request.initial_pose.position.z = (
            0.0
        )

        request.initial_pose.orientation.w = (
            1.0
        )

        self.node.get_logger().info(
            "Spawning physical "
            "MiniGrid walls..."
        )

        future = (
            self.spawn_client.call_async(
                request
            )
        )

        response = (
            self._wait_for_future(
                future,
                "SpawnEntity",
            )
        )

        if not response.success:
            raise RuntimeError(
                "Gazebo failed to spawn "
                "physical MiniGrid walls: "
                f"{response.status_message}"
            )

        self.spawned = True

        self.node.get_logger().info(
            "Physical MiniGrid walls "
            "spawned successfully."
        )

        time.sleep(0.5)

    def cleanup(
        self,
    ):
        """
        Remove the physical wall model.

        Cleanup failures are logged rather than replacing the experiment
        result with another exception.
        """
        if not self.spawned:
            return

        try:
            request = (
                DeleteEntity.Request()
            )

            request.name = (
                PHYSICAL_WALL_MODEL_NAME
            )

            future = (
                self.delete_client.call_async(
                    request
                )
            )

            response = (
                self._wait_for_future(
                    future,
                    "DeleteEntity cleanup",
                )
            )

            if response.success:
                self.node.get_logger().info(
                    "Physical MiniGrid walls "
                    "removed."
                )

                self.spawned = False

            else:
                self.node.get_logger().warning(
                    "Gazebo wall cleanup failed: "
                    f"{response.status_message}"
                )

        except Exception as exc:
            self.node.get_logger().warning(
                "Gazebo wall cleanup raised "
                f"an error: {exc}"
            )


def run_episode(
    env_name,
    seed,
    policy_type,
    mode,
    policy_fn,
    max_steps,
    cleanup_walls=True,
    verbose=True,
):
    # ================================================================
    # EXACT FARAMA LOGICAL ENVIRONMENT
    # ================================================================

    env = MiniGridCardinalWrapper(
        env_name=env_name,
        seed=seed,
    )

    env.reset(
        seed=seed
    )

    initial_grid_state = (
        env.get_state()
    )

    # ================================================================
    # GAZEBO EXECUTION BACKEND
    # ================================================================

    adapter = GazeboAdapter()

    waypoint_executor = (
        GazeboWaypointExecutor(
            adapter
        )
    )

    physical_world = (
        GazeboPhysicalWorld(
            adapter.controller
        )
    )

    step_logs = []

    reached_goal = False

    retry_budget_exhausted = (
        False
    )

    physical_world_metadata = (
        None
    )

    try:
        # ------------------------------------------------------------
        # Align physical orientation with the exact Farama state.
        # ------------------------------------------------------------

        align_robot_to_facing(
            adapter,
            initial_grid_state[
                "facing_name"
            ],
        )

        initial_robot_state = (
            adapter.get_state()
        )

        # The TurtleBot's current pose represents the centre of the
        # exact MiniGrid starting cell.
        anchor_x = (
            initial_robot_state[
                "x"
            ]
        )

        anchor_y = (
            initial_robot_state[
                "y"
            ]
        )

        anchor_cell = tuple(
            initial_grid_state[
                "agent"
            ]
        )

        # ============================================================
        # CONSTRUCT THE PHYSICAL COPY OF THE FARAMA WORLD
        # ============================================================

        (
            physical_world_sdf,
            physical_world_metadata,
        ) = make_physical_world_sdf(
            grid_state=(
                initial_grid_state
            ),
            anchor_x=(
                anchor_x
            ),
            anchor_y=(
                anchor_y
            ),
            anchor_cell=(
                anchor_cell
            ),
        )

        print()
        print("=" * 80)
        print(
            "SPAWNING PHYSICAL FARAMA WORLD"
        )
        print("=" * 80)

        print(
            "Outer wall cells:",
            physical_world_metadata[
                "num_outer_wall_cells"
            ],
        )

        print(
            "Interior obstacle cells:",
            physical_world_metadata[
                "num_interior_wall_cells"
            ],
        )

        print(
            "Total physical wall cells:",
            physical_world_metadata[
                "num_total_wall_cells"
            ],
        )

        physical_world.spawn(
            physical_world_sdf
        )

        # ============================================================
        # EPISODE INFORMATION
        # ============================================================

        print()
        print("=" * 80)
        print(
            "GAZEBO + FARAMA REPLAY"
        )
        print("=" * 80)

        print(
            "Environment:",
            env_name,
        )

        print(
            "Seed:",
            seed,
        )

        print(
            "Policy:",
            policy_type,
        )

        print(
            "Mode:",
            mode,
        )

        print(
            "Grid size:",
            initial_grid_state[
                "grid_size"
            ],
        )

        print(
            "Start:",
            initial_grid_state[
                "agent"
            ],
        )

        print(
            "Goal:",
            initial_grid_state[
                "goal"
            ],
        )

        print(
            "Initial facing:",
            initial_grid_state[
                "facing_name"
            ],
        )

        print(
            "Interior obstacles:",
            initial_grid_state[
                "obstacle_cells"
            ],
        )

        print("=" * 80)

        # ============================================================
        # EPISODE LOOP
        # ============================================================

        for step in range(
            max_steps
        ):
            state_before = (
                env.get_state()
            )

            if (
                state_before["agent"]
                == state_before["goal"]
            ):
                reached_goal = True
                break

            # --------------------------------------------------------
            # EXACT benchmark prompt.
            # --------------------------------------------------------

            prompt = get_prompt(
                env,
                mode,
            )

            grid_text = (
                env.render_text()
            )

            legal_actions = (
                get_legal_cardinal_actions(
                    env
                )
            )

            if verbose:
                print()
                print(
                    "-" * 80
                )

                print(
                    f"STEP {step}"
                )

                print(
                    "-" * 80
                )

                print(
                    grid_text
                )

                print(
                    "Agent:",
                    state_before[
                        "agent"
                    ],
                )

                print(
                    "Facing:",
                    state_before[
                        "facing_name"
                    ],
                )

                print(
                    "Legal cardinal:",
                    [
                        ACTION_NAMES[
                            action
                        ]
                        for action
                        in sorted(
                            legal_actions
                        )
                    ],
                )

            # ========================================================
            # EXACT VALIDATED SHIELD / CONTROL LOGIC
            # ========================================================

            (
                parsed_action,
                action_metadata,
            ) = choose_action(
                env=env,
                mode=mode,
                policy_type=(
                    policy_type
                ),
                prompt=prompt,
                short_policy_fn=(
                    policy_fn
                ),
                reasoning_policy_fn=None,
                max_reprompts=2,
            )

            if verbose:
                print(
                    "Raw answers:",
                    action_metadata[
                        "all_raw_answers"
                    ],
                )

                print(
                    "Model-visible legal actions:",
                    action_metadata[
                        "legal_action_names"
                    ],
                )

                print(
                    "Reprompts:",
                    action_metadata[
                        "shield_reprompts"
                    ],
                )

            # --------------------------------------------------------
            # Retry budget exhausted.
            # No physical action occurs.
            # --------------------------------------------------------

            if parsed_action is None:
                retry_budget_exhausted = (
                    True
                )

                step_logs.append(
                    {
                        "step": step,
                        "grid_state_before": (
                            state_before
                        ),
                        "grid_text": (
                            grid_text
                        ),
                        "prompt": (
                            prompt
                        ),
                        "raw_model_answer": (
                            action_metadata[
                                "raw_model_answer"
                            ]
                        ),
                        "all_raw_answers": (
                            action_metadata[
                                "all_raw_answers"
                            ]
                        ),
                        "legal_action_names": (
                            action_metadata[
                                "legal_action_names"
                            ]
                        ),
                        "shield_used": (
                            action_metadata[
                                "shield_used"
                            ]
                        ),
                        "shield_reprompts": (
                            action_metadata[
                                "shield_reprompts"
                            ]
                        ),
                        "parse_failure": True,
                        "executed": False,
                    }
                )

                print()
                print(
                    "Retry budget exhausted."
                )

                print(
                    "No physical action executed."
                )

                break

            # --------------------------------------------------------
            # Final physical safety check.
            # --------------------------------------------------------

            if (
                parsed_action
                not in legal_actions
            ):
                raise RuntimeError(
                    "Safety violation: "
                    "choose_action returned "
                    "an illegal action."
                )

            action_name = (
                ACTION_NAMES[
                    parsed_action
                ]
            )

            print(
                "Executed cardinal action:",
                action_name,
            )

            # ========================================================
            # DETERMINE THE TARGET GRID CELL
            # ========================================================

            (
                current_row,
                current_col,
            ) = state_before[
                "agent"
            ]

            (
                delta_row,
                delta_col,
            ) = ACTION_TO_DELTA[
                parsed_action
            ]

            target_cell = (
                current_row
                + delta_row,
                current_col
                + delta_col,
            )

            (
                expected_x,
                expected_y,
            ) = expected_world_position(
                anchor_x=(
                    anchor_x
                ),
                anchor_y=(
                    anchor_y
                ),
                anchor_cell=(
                    anchor_cell
                ),
                target_cell=(
                    target_cell
                ),
            )

            # ========================================================
            # PHYSICAL EXECUTION
            # ========================================================

            robot_before = (
                adapter.get_state()
            )

            robot_after = (
                waypoint_executor.execute_to_waypoint(
                    action=(
                        action_name
                    ),
                    target_x=(
                        expected_x
                    ),
                    target_y=(
                        expected_y
                    ),
                )
            )

            # ========================================================
            # UPDATE EXACT FARAMA LOGICAL STATE
            # ========================================================

            (
                next_state,
                reward,
                terminated,
                truncated,
                info,
            ) = env.step_cardinal(
                parsed_action
            )

            if (
                tuple(
                    next_state[
                        "agent"
                    ]
                )
                != tuple(
                    target_cell
                )
            ):
                raise RuntimeError(
                    "Logical state mismatch: "
                    f"expected {target_cell}, "
                    f"got "
                    f"{next_state['agent']}."
                )

            # ========================================================
            # PHYSICAL / LOGICAL POSITION CHECK
            # ========================================================

            position_error = (
                physical_position_error(
                    robot_state=(
                        robot_after
                    ),
                    expected_x=(
                        expected_x
                    ),
                    expected_y=(
                        expected_y
                    ),
                )
            )

            if verbose:
                print(
                    "Logical next cell:",
                    next_state[
                        "agent"
                    ],
                )

                print(
                    "Logical facing:",
                    next_state[
                        "facing_name"
                    ],
                )

                print(
                    "Robot pose:",
                    (
                        f"x="
                        f"{robot_after['x']:.3f}, "
                        f"y="
                        f"{robot_after['y']:.3f}, "
                        f"yaw="
                        f"{robot_after['yaw_degrees']:.1f}"
                    ),
                )

                print(
                    "Expected waypoint:",
                    (
                        f"x={expected_x:.3f}, "
                        f"y={expected_y:.3f}"
                    ),
                )

                print(
                    "Position error:",
                    f"{position_error:.3f} m",
                )

            # ========================================================
            # STEP LOG
            # ========================================================

            step_logs.append(
                {
                    "step": (
                        step
                    ),
                    "grid_state_before": (
                        state_before
                    ),
                    "grid_text": (
                        grid_text
                    ),
                    "prompt": (
                        prompt
                    ),
                    "raw_model_answer": (
                        action_metadata[
                            "raw_model_answer"
                        ]
                    ),
                    "all_raw_answers": (
                        action_metadata[
                            "all_raw_answers"
                        ]
                    ),
                    "parsed_action": (
                        parsed_action
                    ),
                    "parsed_action_name": (
                        action_name
                    ),
                    "legal_action_names": (
                        action_metadata[
                            "legal_action_names"
                        ]
                    ),
                    "shield_used": (
                        action_metadata[
                            "shield_used"
                        ]
                    ),
                    "shield_reprompts": (
                        action_metadata[
                            "shield_reprompts"
                        ]
                    ),
                    "parse_failure": (
                        False
                    ),
                    "executed": (
                        True
                    ),
                    "robot_before": (
                        robot_before
                    ),
                    "robot_after": (
                        robot_after
                    ),
                    "target_grid_cell": (
                        target_cell
                    ),
                    "expected_waypoint": {
                        "x": (
                            expected_x
                        ),
                        "y": (
                            expected_y
                        ),
                    },
                    "physical_position_error_m": (
                        position_error
                    ),
                    "grid_state_after": (
                        next_state
                    ),
                    "reward": (
                        reward
                    ),
                    "terminated": (
                        terminated
                    ),
                    "truncated": (
                        truncated
                    ),
                    "info": (
                        info
                    ),
                }
            )

            if terminated:
                reached_goal = True

                print()
                print("=" * 80)
                print(
                    "GOAL REACHED"
                )
                print("=" * 80)

                break

        # ============================================================
        # FINAL STATE
        # ============================================================

        final_grid_state = (
            env.get_state()
        )

        final_robot_state = (
            adapter.get_state()
        )

        result = {
            "run_type": (
                "gazebo_farama_replay"
            ),
            "env_name": (
                env_name
            ),
            "seed": (
                int(seed)
            ),
            "policy_type": (
                policy_type
            ),
            "mode": (
                mode
            ),
            "max_steps": (
                max_steps
            ),
            "max_reprompts": (
                2
            ),
            "reached_goal": (
                reached_goal
            ),
            "retry_budget_exhausted": (
                retry_budget_exhausted
            ),
            "num_steps": (
                len(
                    step_logs
                )
            ),
            "initial_grid_state": (
                initial_grid_state
            ),
            "final_grid_state": (
                final_grid_state
            ),
            "initial_robot_state": (
                initial_robot_state
            ),
            "final_robot_state": (
                final_robot_state
            ),
            "cell_size_metres": (
                CELL_SIZE_METRES
            ),
            "execution_mode": (
                "absolute_grid_waypoints"
            ),
            "physical_walls_enabled": (
                True
            ),
            "physical_wall_model": (
                physical_world_metadata
            ),
            "physical_walls_cleaned_after_run": (
                cleanup_walls
            ),
            "step_logs": (
                step_logs
            ),
        }

        return result

    finally:
        # Delete the physical map before destroying the ROS node.
        if cleanup_walls:
            physical_world.cleanup()

        env.close()

        adapter.close()


def build_argument_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Replay an exact Farama MiniGrid "
            "benchmark state using a TurtleBot3 "
            "execution backend in Gazebo."
        )
    )

    parser.add_argument(
        "--environment",
        default=(
            "MiniGrid-SimpleCrossingS9N1-v0"
        ),
        choices=list(
            MAX_STEPS_BY_ENV
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--policy",
        choices=[
            "legality_shield",
            "reprompt_control",
        ],
        default=(
            "legality_shield"
        ),
    )

    parser.add_argument(
        "--mode",
        choices=[
            "allocentric",
            "egocentric",
        ],
        default=(
            "allocentric"
        ),
    )

    parser.add_argument(
        "--model",
        default=(
            "openai/gpt-4o-mini"
        ),
    )

    parser.add_argument(
        "--provider",
        default=(
            "OpenAI"
        ),
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--output",
        default=(
            "gazebo_farama_replay.json"
        ),
    )

    parser.add_argument(
        "--keep-walls",
        action="store_true",
        help=(
            "Leave the physical MiniGrid "
            "wall model in Gazebo after the "
            "episode for visual inspection."
        ),
    )

    return parser


def main():
    args = (
        build_argument_parser()
        .parse_args()
    )

    if args.max_steps is None:
        max_steps = (
            MAX_STEPS_BY_ENV[
                args.environment
            ]
        )

    else:
        max_steps = (
            args.max_steps
        )

    print()
    print(
        "Creating OpenRouter policy..."
    )

    policy_fn = (
        make_openai_policy_fn(
            model=(
                args.model
            ),
            temperature=(
                0.0
            ),
            provider=(
                args.provider
            ),
            max_output_tokens=(
                16
            ),
        )
    )

    result = run_episode(
        env_name=(
            args.environment
        ),
        seed=(
            args.seed
        ),
        policy_type=(
            args.policy
        ),
        mode=(
            args.mode
        ),
        policy_fn=(
            policy_fn
        ),
        max_steps=(
            max_steps
        ),
        cleanup_walls=(
            not args.keep_walls
        ),
    )

    result["model"] = (
        args.model
    )

    result["provider"] = (
        args.provider
    )

    result["temperature"] = (
        0.0
    )

    result[
        "intervention_exhaustion_behavior"
    ] = (
        "parse_failure; no illegal "
        "action is physically executed"
    )

    save_result(
        result,
        args.output,
    )

    print()
    print("=" * 80)
    print(
        "REPLAY COMPLETE"
    )
    print("=" * 80)

    print(
        "Environment:",
        result[
            "env_name"
        ],
    )

    print(
        "Seed:",
        result[
            "seed"
        ],
    )

    print(
        "Reached goal:",
        result[
            "reached_goal"
        ],
    )

    print(
        "Steps:",
        result[
            "num_steps"
        ],
    )

    print(
        "Final logical cell:",
        result[
            "final_grid_state"
        ][
            "agent"
        ],
    )

    print(
        "Saved:",
        args.output,
    )


if __name__ == "__main__":
    main()