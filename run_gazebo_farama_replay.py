import argparse
import json
import math
import time
from pathlib import Path

from llm_policy import make_openai_policy_fn

from minigrid_wrapper import (
    MiniGridCardinalWrapper,
    ACTION_NAMES,
    ACTION_TO_DELTA,
)

# Reuse the already-validated Farama experiment logic.
# The existing Farama runner itself is not modified.
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
    Match the TurtleBot's physical heading to the initial MiniGrid facing.

    After each executed cardinal action, the robot is left facing the same
    cardinal direction as MiniGridCardinalWrapper.step_cardinal().
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
    Convert a MiniGrid cell into its absolute Gazebo waypoint.

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


def run_episode(
    env_name,
    seed,
    policy_type,
    mode,
    policy_fn,
    max_steps,
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
    # GAZEBO PHYSICAL EXECUTION BACKEND
    # ================================================================

    adapter = GazeboAdapter()

    waypoint_executor = (
        GazeboWaypointExecutor(
            adapter
        )
    )

    step_logs = []

    reached_goal = False
    retry_budget_exhausted = False

    try:
        # ------------------------------------------------------------
        # Align physical orientation with the exact MiniGrid state.
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

        # The current Gazebo pose becomes the physical location of the
        # MiniGrid starting cell.
        anchor_x = (
            initial_robot_state["x"]
        )

        anchor_y = (
            initial_robot_state["y"]
        )

        anchor_cell = tuple(
            initial_grid_state[
                "agent"
            ]
        )

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
            "Obstacles:",
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
            # Exact prompt from the existing Farama wrapper.
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
                        ACTION_NAMES[action]
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
            # Retry-budget exhaustion.
            # No illegal action is physically executed.
            # --------------------------------------------------------

            if parsed_action is None:
                retry_budget_exhausted = True

                step_logs.append(
                    {
                        "step": step,
                        "grid_state_before": (
                            state_before
                        ),
                        "grid_text": (
                            grid_text
                        ),
                        "prompt": prompt,
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
            # Safety assertion.
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
            # DETERMINE THE NEXT LOGICAL CELL BEFORE PHYSICAL MOTION
            # ========================================================

            current_row, current_col = (
                state_before["agent"]
            )

            delta_row, delta_col = (
                ACTION_TO_DELTA[
                    parsed_action
                ]
            )

            target_cell = (
                current_row
                + delta_row,
                current_col
                + delta_col,
            )

            # Convert the target MiniGrid cell into an absolute Gazebo
            # waypoint anchored to the episode's physical start pose.
            expected_x, expected_y = (
                expected_world_position(
                    anchor_x=anchor_x,
                    anchor_y=anchor_y,
                    anchor_cell=(
                        anchor_cell
                    ),
                    target_cell=(
                        target_cell
                    ),
                )
            )

            # ========================================================
            # PHYSICAL EXECUTION TO ABSOLUTE CELL CENTRE
            # ========================================================

            robot_before = (
                adapter.get_state()
            )

            robot_after = (
                waypoint_executor.execute_to_waypoint(
                    action=action_name,
                    target_x=expected_x,
                    target_y=expected_y,
                )
            )

            # ========================================================
            # UPDATE THE EXACT FARAMA LOGICAL STATE
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

            # The logical simulator should have moved into exactly the
            # cell whose waypoint we just physically executed.
            if (
                tuple(
                    next_state["agent"]
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
            # PHYSICAL / LOGICAL ALIGNMENT CHECK
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
            # LOG STEP
            # ========================================================

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
                    "parse_failure": False,
                    "executed": True,
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

        return {
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
                len(step_logs)
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
            "step_logs": (
                step_logs
            ),
        }

    finally:
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
            model=args.model,
            temperature=0.0,
            provider=(
                args.provider
            ),
            max_output_tokens=16,
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