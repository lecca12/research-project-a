import argparse
import json
import time
from pathlib import Path

import rclpy

from llm_policy import make_openai_policy_fn

from gazebo_turtlebot.gazebo_adapter import GazeboAdapter
from gazebo_turtlebot.gazebo_grid_env import (
    GazeboGridEnv,
    load_layout,
)

from run_gazebo_pilot import (
    SHORT_TEMPERATURE,
    SHORT_MAX_OUTPUT_TOKENS,
    MAX_REPROMPTS,
    EARLY_STOP_REPEATS,
    check_legal_action_ordering,
    reset_simulation,
    run_episode,
    save_trace,
)


DEFAULT_SEEDS = [0, 1, 2, 3, 4]

ARMS = [
    "legality_shield",
    "reprompt_control",
]

MODES = [
    "allocentric",
    "egocentric",
]


def save_json(data, path):
    path = Path(path)

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            indent=2,
        )


def condition_key(result):
    return (
        int(result["seed"]),
        result["policy_type"],
        result["mode"],
    )


def wait_for_reset_odometry(
    adapter,
    timeout_seconds=10.0,
    position_tolerance=0.05,
    yaw_tolerance=0.10,
    confirmations_required=3,
):
    """
    After /reset_simulation, actively process /odom messages until
    the persistent GridActionController has observed the robot back
    at the Gazebo origin.

    This prevents a new episode from being initialised using stale
    odometry from the previous episode.
    """

    deadline = (
        time.monotonic()
        + timeout_seconds
    )

    confirmations = 0
    last_pose = None

    print(
        "Waiting for reset odometry..."
    )

    while (
        time.monotonic()
        < deadline
    ):
        # The persistent adapter/controller does not automatically
        # process callbacks while the supervisor's reset function
        # is sleeping. Spin it here so fresh /odom is consumed.
        rclpy.spin_once(
            adapter.controller,
            timeout_sec=0.1,
        )

        pose = adapter.get_state()
        last_pose = pose

        at_origin = (
            abs(pose["x"])
            <= position_tolerance
            and abs(pose["y"])
            <= position_tolerance
            and abs(pose["yaw"])
            <= yaw_tolerance
        )

        if at_origin:
            confirmations += 1

            if (
                confirmations
                >= confirmations_required
            ):
                print(
                    "Reset odometry confirmed: "
                    f"x={pose['x']:.3f}, "
                    f"y={pose['y']:.3f}, "
                    f"yaw="
                    f"{pose['yaw_degrees']:.1f} deg"
                )
                return

        else:
            confirmations = 0

    if last_pose is None:
        raise RuntimeError(
            "Timed out waiting for "
            "odometry after Gazebo reset."
        )

    raise RuntimeError(
        "Timed out waiting for reset "
        "odometry. "
        f"Last pose: "
        f"x={last_pose['x']:.3f}, "
        f"y={last_pose['y']:.3f}, "
        f"yaw="
        f"{last_pose['yaw_degrees']:.1f} deg"
    )


def summarise(results):
    summary = {}

    for arm in ARMS:
        for mode in MODES:
            matching = [
                result
                for result in results
                if (
                    result[
                        "policy_type"
                    ]
                    == arm
                    and result[
                        "mode"
                    ]
                    == mode
                )
            ]

            n = len(matching)

            if n == 0:
                continue

            successes = sum(
                bool(
                    result[
                        "reached_goal"
                    ]
                )
                for result
                in matching
            )

            early_stops = sum(
                bool(
                    result.get(
                        "early_stopped",
                        False,
                    )
                )
                for result
                in matching
            )

            retry_exhaustions = 0
            total_steps = 0
            total_reprompts = 0
            executed_steps = 0
            correct_steps = 0

            max_step_errors = []

            for result in matching:
                total_steps += int(
                    result[
                        "num_steps"
                    ]
                )

                max_error = (
                    result.get(
                        "max_step_error_metres"
                    )
                )

                if (
                    max_error
                    is not None
                ):
                    max_step_errors.append(
                        float(
                            max_error
                        )
                    )

                logs = result.get(
                    "step_logs",
                    [],
                )

                # In the supervisor runner, exhausted retries
                # end as a parse failure / no executable action.
                if (
                    logs
                    and logs[-1].get(
                        "parse_failure",
                        False,
                    )
                ):
                    retry_exhaustions += 1

                for log in logs:
                    total_reprompts += int(
                        log.get(
                            "shield_reprompts",
                            0,
                        )
                    )

                    if not log.get(
                        "parse_failure",
                        False,
                    ):
                        executed_steps += 1

                        if log.get(
                            "is_correct",
                            False,
                        ):
                            correct_steps += 1

            if (
                executed_steps
                > 0
            ):
                step_accuracy = (
                    correct_steps
                    / executed_steps
                )
            else:
                step_accuracy = 0.0

            summary[
                f"{arm}__{mode}"
            ] = {
                "arm": arm,
                "mode": mode,
                "episodes": n,
                "successes": (
                    successes
                ),
                "success_rate": (
                    successes / n
                ),
                "retry_budget_exhaustions": (
                    retry_exhaustions
                ),
                "retry_budget_exhaustion_rate": (
                    retry_exhaustions
                    / n
                ),
                "early_stops": (
                    early_stops
                ),
                "average_steps": (
                    total_steps / n
                ),
                "total_reprompts": (
                    total_reprompts
                ),
                "executed_steps": (
                    executed_steps
                ),
                "correct_steps": (
                    correct_steps
                ),
                "executed_step_accuracy": (
                    step_accuracy
                ),
                "max_physical_step_error_metres": (
                    max(
                        max_step_errors
                    )
                    if max_step_errors
                    else None
                ),
            }

    return summary


def print_summary(summary):
    print()
    print("=" * 80)
    print(
        "GAZEBO TRANSFER SUMMARY"
    )
    print("=" * 80)

    for arm in ARMS:
        for mode in MODES:
            key = (
                f"{arm}__{mode}"
            )

            if (
                key
                not in summary
            ):
                continue

            group = (
                summary[key]
            )

            print()
            print(
                f"{arm} | {mode}"
            )

            print(
                "Episodes:",
                group[
                    "episodes"
                ],
            )

            print(
                "Success:",
                (
                    f"{group['successes']}/"
                    f"{group['episodes']} "
                    f"("
                    f"{100 * group['success_rate']:.1f}%"
                    f")"
                ),
            )

            print(
                "Retry exhaustion:",
                (
                    f"{group['retry_budget_exhaustions']}/"
                    f"{group['episodes']} "
                    f"("
                    f"{100 * group['retry_budget_exhaustion_rate']:.1f}%"
                    f")"
                ),
            )

            print(
                "Average steps:",
                (
                    f"{group['average_steps']:.2f}"
                ),
            )

            print(
                "Executed step accuracy:",
                (
                    f"{100 * group['executed_step_accuracy']:.1f}% "
                    f"("
                    f"{group['correct_steps']}/"
                    f"{group['executed_steps']}"
                    f")"
                ),
            )

            print(
                "Total reprompts:",
                group[
                    "total_reprompts"
                ],
            )

            max_error = group[
                "max_physical_step_error_metres"
            ]

            if (
                max_error
                is not None
            ):
                print(
                    "Max physical "
                    "step error:",
                    (
                        f"{max_error:.4f} m"
                    ),
                )
            else:
                print(
                    "Max physical "
                    "step error: N/A"
                )


def layout_path_for_seed(
    seed,
):
    return (
        "layout_"
        "simplecrossing_s9n1_"
        f"seed{seed}.json"
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Supervisor-aligned "
            "Gazebo transfer batch: "
            "legality shield versus "
            "reprompt control."
        )
    )

    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=(
            DEFAULT_SEEDS
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
        default="OpenAI",
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=81,
    )

    parser.add_argument(
        "--cell-size",
        type=float,
        default=0.50,
    )

    parser.add_argument(
        "--output",
        default=(
            "gazebo_supervisor_"
            "batch_5seed.json"
        ),
    )

    parser.add_argument(
        "--trace-dir",
        default=(
            "gazebo_supervisor_"
            "batch_5seed_traces"
        ),
    )

    args = (
        parser.parse_args()
    )

    # ----------------------------------------------------------
    # Load and validate all frozen layouts before running Gazebo.
    # ----------------------------------------------------------

    layouts = {}

    for seed in args.seeds:
        path = Path(
            layout_path_for_seed(
                seed
            )
        )

        if not path.exists():
            raise FileNotFoundError(
                "Missing layout for "
                f"seed {seed}: {path}"
            )

        layout = load_layout(
            path
        )

        if (
            int(
                layout["seed"]
            )
            != seed
        ):
            raise RuntimeError(
                f"Layout {path} says "
                f"seed {layout['seed']}, "
                f"expected {seed}."
            )

        if (
            layout["env_name"]
            != (
                "MiniGrid-"
                "SimpleCrossingS9N1-v0"
            )
        ):
            raise RuntimeError(
                "Unexpected "
                "environment in "
                f"{path}: "
                f"{layout['env_name']}"
            )

        # Supervisor startup guard.
        # Refuses to run if legal action
        # ordering differs from benchmark.
        check_legal_action_ordering(
            layout,
            args.cell_size,
        )

        layouts[
            seed
        ] = layout

    # ----------------------------------------------------------
    # LLM policy.
    # Same model interface/settings as benchmark.
    # ----------------------------------------------------------

    policy_fn = (
        make_openai_policy_fn(
            model=(
                args.model
            ),
            temperature=(
                SHORT_TEMPERATURE
            ),
            max_output_tokens=(
                SHORT_MAX_OUTPUT_TOKENS
            ),
            provider=(
                args.provider
            ),
        )
    )

    output_path = Path(
        args.output
    )

    trace_dir = Path(
        args.trace_dir
    )

    trace_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ----------------------------------------------------------
    # Explicit experiment metadata.
    # ----------------------------------------------------------

    metadata = {
        "run_type": (
            "gazebo_transfer_batch"
        ),
        "environment": (
            "MiniGrid-"
            "SimpleCrossingS9N1-v0"
        ),
        "seeds": list(
            args.seeds
        ),
        "num_seeds": len(
            args.seeds
        ),
        "model": (
            args.model
        ),
        "provider_pin": (
            args.provider
        ),
        "arms": list(
            ARMS
        ),
        "modes": list(
            MODES
        ),
        "max_steps": (
            args.max_steps
        ),
        "short_temperature": (
            SHORT_TEMPERATURE
        ),
        "short_max_output_tokens": (
            SHORT_MAX_OUTPUT_TOKENS
        ),
        "max_reprompts": (
            MAX_REPROMPTS
        ),
        "early_stop_repeats": (
            EARLY_STOP_REPEATS
        ),
        "cell_size_metres": (
            args.cell_size
        ),
        "layout_files": {
            str(seed): (
                layout_path_for_seed(
                    seed
                )
            )
            for seed
            in args.seeds
        },
        "execution_backend": (
            "TurtleBot3 Gazebo"
        ),
        "comparison_scope": (
            "legality_shield_vs_"
            "reprompt_control"
        ),
        "reset_odometry_sync": {
            "enabled": True,
            "position_tolerance_metres": (
                0.05
            ),
            "yaw_tolerance_radians": (
                0.10
            ),
            "confirmations_required": (
                3
            ),
            "timeout_seconds": (
                10.0
            ),
        },
    }

    # ----------------------------------------------------------
    # Resume support.
    #
    # IMPORTANT:
    # Do not delete the existing JSON after a crash.
    # Completed conditions are loaded and skipped.
    # ----------------------------------------------------------

    results = []

    if output_path.exists():
        try:
            existing = (
                json.loads(
                    output_path.read_text(
                        encoding="utf-8"
                    )
                )
            )

            if isinstance(
                existing,
                dict,
            ):
                results = (
                    existing.get(
                        "results",
                        [],
                    )
                )

                print(
                    "Resuming with "
                    f"{len(results)} "
                    "completed episodes."
                )

        except Exception as exc:
            print(
                "Warning: could not "
                "load existing output "
                f"for resume: {exc}"
            )

            results = []

    completed = {
        condition_key(
            result
        )
        for result
        in results
    }

    total_conditions = (
        len(args.seeds)
        * len(ARMS)
        * len(MODES)
    )

    # One persistent ROS adapter for the entire batch.
    adapter = (
        GazeboAdapter()
    )

    run_index = len(
        results
    )

    try:
        for seed in args.seeds:
            layout = (
                layouts[seed]
            )

            # Alternate policy order by seed so one intervention
            # is not always executed first.
            if (
                seed % 2
                == 0
            ):
                arm_order = [
                    "legality_shield",
                    "reprompt_control",
                ]
            else:
                arm_order = [
                    "reprompt_control",
                    "legality_shield",
                ]

            for mode in MODES:
                for arm in arm_order:
                    key = (
                        seed,
                        arm,
                        mode,
                    )

                    if (
                        key
                        in completed
                    ):
                        print(
                            "Skipping "
                            "completed:",
                            key,
                        )
                        continue

                    run_index += 1

                    print()
                    print(
                        "#" * 80
                    )
                    print(
                        f"RUN {run_index}/"
                        f"{total_conditions}"
                    )
                    print(
                        f"Seed: {seed}"
                    )
                    print(
                        f"Arm: {arm}"
                    )
                    print(
                        f"Mode: {mode}"
                    )
                    print(
                        "#" * 80
                    )

                    # ------------------------------------------
                    # Reset Gazebo.
                    # ------------------------------------------

                    reset_simulation()

                    # ------------------------------------------
                    # CRITICAL FIX:
                    #
                    # reset_simulation() resets Gazebo's robot
                    # pose, but the persistent ROS controller
                    # may still contain the previous episode's
                    # cached odometry.
                    #
                    # Actively spin /odom until the controller
                    # has observed the reset pose.
                    # ------------------------------------------

                    wait_for_reset_odometry(
                        adapter
                    )

                    # ------------------------------------------
                    # Create a fresh logical environment for
                    # this frozen MiniGrid layout.
                    # ------------------------------------------

                    env = GazeboGridEnv(
                        adapter=(
                            adapter
                        ),
                        layout=(
                            layout
                        ),
                        cell_size=(
                            args.cell_size
                        ),
                    )

                    env.initialise_from_current_pose()

                    # ------------------------------------------
                    # Run supervisor episode implementation.
                    # ------------------------------------------

                    result = (
                        run_episode(
                            env=env,
                            mode=mode,
                            policy_type=(
                                arm
                            ),
                            policy_fn=(
                                policy_fn
                            ),
                            max_steps=(
                                args.max_steps
                            ),
                            verbose=True,
                        )
                    )

                    # Add batch-level metadata directly
                    # to each episode as well.
                    result[
                        "episode"
                    ] = 0

                    result[
                        "model"
                    ] = (
                        args.model
                    )

                    result[
                        "provider_pin"
                    ] = (
                        args.provider
                    )

                    result[
                        "max_steps"
                    ] = (
                        args.max_steps
                    )

                    results.append(
                        result
                    )

                    completed.add(
                        key
                    )

                    # ------------------------------------------
                    # Save trace.
                    # ------------------------------------------

                    save_trace(
                        result,
                        trace_dir,
                    )

                    # ------------------------------------------
                    # Save the entire batch after EVERY episode.
                    # ------------------------------------------

                    summary = (
                        summarise(
                            results
                        )
                    )

                    save_json(
                        {
                            "metadata": (
                                metadata
                            ),
                            "results": (
                                results
                            ),
                            "summary": (
                                summary
                            ),
                        },
                        output_path,
                    )

                    print()
                    print(
                        "Completed:",
                        (
                            f"seed={seed}, "
                            f"arm={arm}, "
                            f"mode={mode}, "
                            f"success="
                            f"{result['reached_goal']}, "
                            f"steps="
                            f"{result['num_steps']}"
                        ),
                    )

    finally:
        adapter.close()

    # ----------------------------------------------------------
    # Final summary and save.
    # ----------------------------------------------------------

    summary = (
        summarise(
            results
        )
    )

    save_json(
        {
            "metadata": (
                metadata
            ),
            "results": (
                results
            ),
            "summary": (
                summary
            ),
        },
        output_path,
    )

    print_summary(
        summary
    )

    print()
    print(
        "Saved:",
        output_path,
    )

    print(
        "Traces:",
        trace_dir,
    )


if __name__ == "__main__":
    main()