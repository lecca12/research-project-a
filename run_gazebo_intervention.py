import argparse
import json
from pathlib import Path

from llm_policy import make_openai_policy_fn

from gazebo_turtlebot.gazebo_turtlebot.gazebo_adapter import GazeboAdapter
from gazebo_turtlebot.gazebo_turtlebot.gazebo_grid_world import GazeboGridWorld
from gazebo_turtlebot.gazebo_turtlebot.gazebo_policy import (
    choose_action,
    get_prompt,
)


DEFAULT_OBSTACLES = [
    (3, 2),
    (2, 2),
]


def make_json_safe(obj):
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


def run_episode(
    policy_type,
    mode,
    policy_fn,
    max_steps=20,
    rows=5,
    cols=5,
    cell_size=0.50,
    start_cell=(4, 0),
    goal_cell=(0, 4),
    obstacle_cells=None,
    verbose=True,
):
    adapter = GazeboAdapter()

    world = GazeboGridWorld(
        adapter=adapter,
        rows=rows,
        cols=cols,
        cell_size=cell_size,
        start_cell=start_cell,
        goal_cell=goal_cell,
        obstacle_cells=(
            obstacle_cells
            if obstacle_cells is not None
            else DEFAULT_OBSTACLES
        ),
    )

    step_logs = []

    reached_goal = False
    exhausted_retry_budget = False

    try:
        world.initialise_from_current_pose()

        initial_state = (
            world.get_state()
        )

        print()
        print("=" * 80)
        print("GAZEBO INTERVENTION EPISODE")
        print("=" * 80)

        print(
            f"Policy: {policy_type}"
        )

        print(
            f"Mode: {mode}"
        )

        print(
            f"Start cell: "
            f"{initial_state['agent']}"
        )

        print(
            f"Goal cell: "
            f"{world.goal_cell}"
        )

        print(
            f"Obstacles: "
            f"{sorted(world.obstacle_cells)}"
        )

        print(
            f"Initial facing: "
            f"{initial_state['facing']}"
        )

        print("=" * 80)

        for step in range(
            max_steps
        ):
            state_before = (
                world.get_state()
            )

            if world.reached_goal():
                reached_goal = True
                break

            prompt = get_prompt(
                world,
                mode,
            )

            legal_cardinal_actions = (
                world.get_legal_actions()
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
                    "Agent:",
                    state_before["agent"],
                )

                print(
                    "Facing:",
                    state_before["facing"],
                )

                print(
                    "Goal:",
                    state_before["goal"],
                )

                print(
                    "Legal cardinal actions:",
                    legal_cardinal_actions,
                )

            (
                parsed_action,
                action_metadata,
            ) = choose_action(
                env=world,
                mode=mode,
                policy_type=(
                    policy_type
                ),
                prompt=prompt,
                policy_fn=policy_fn,
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

                print(
                    "Final parsed action:",
                    parsed_action,
                )

            parse_failure = (
                parsed_action is None
            )

            if parse_failure:
                exhausted_retry_budget = True

                step_logs.append(
                    {
                        "step": step,
                        "policy_type": (
                            policy_type
                        ),
                        "mode": mode,
                        "agent_pos": (
                            state_before[
                                "agent"
                            ]
                        ),
                        "goal_pos": (
                            state_before[
                                "goal"
                            ]
                        ),
                        "facing": (
                            state_before[
                                "facing"
                            ]
                        ),
                        "pose_before": (
                            state_before[
                                "pose"
                            ]
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
                        "parsed_action": None,
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
                        "blocked": False,
                        "blocked_type": None,
                        "state_after": None,
                    }
                )

                print()
                print(
                    "Retry budget exhausted. "
                    "Episode ends as parse failure."
                )

                break

            execution_result = (
                world.execute_action(
                    parsed_action
                )
            )

            state_after = (
                world.get_state()
            )

            if verbose:
                print(
                    "Execution result:",
                    execution_result,
                )

                print(
                    "New cell:",
                    state_after[
                        "agent"
                    ],
                )

                print(
                    "New facing:",
                    state_after[
                        "facing"
                    ],
                )

            step_logs.append(
                {
                    "step": step,
                    "policy_type": (
                        policy_type
                    ),
                    "mode": mode,
                    "agent_pos": (
                        state_before[
                            "agent"
                        ]
                    ),
                    "goal_pos": (
                        state_before[
                            "goal"
                        ]
                    ),
                    "facing": (
                        state_before[
                            "facing"
                        ]
                    ),
                    "pose_before": (
                        state_before[
                            "pose"
                        ]
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
                    "parsed_action": (
                        parsed_action
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
                    "executed": (
                        execution_result[
                            "executed"
                        ]
                    ),
                    "blocked": (
                        execution_result[
                            "blocked"
                        ]
                    ),
                    "blocked_type": (
                        execution_result[
                            "blocked_type"
                        ]
                    ),
                    "execution_result": (
                        execution_result
                    ),
                    "state_after": (
                        state_after
                    ),
                }
            )

            if world.reached_goal():
                reached_goal = True

                print()
                print(
                    "=" * 80
                )

                print(
                    "GOAL REACHED"
                )

                print(
                    "=" * 80
                )

                break

        final_state = (
            world.get_state()
        )

        result = {
            "environment_type": (
                "gazebo_turtlebot3"
            ),
            "policy_type": (
                policy_type
            ),
            "mode": mode,
            "rows": rows,
            "cols": cols,
            "cell_size": (
                cell_size
            ),
            "start_cell": (
                start_cell
            ),
            "goal_cell": (
                goal_cell
            ),
            "obstacle_cells": (
                sorted(
                    world.obstacle_cells
                )
            ),
            "max_steps": (
                max_steps
            ),
            "num_steps": (
                len(step_logs)
            ),
            "reached_goal": (
                reached_goal
            ),
            "retry_budget_exhausted": (
                exhausted_retry_budget
            ),
            "initial_state": (
                initial_state
            ),
            "final_state": (
                final_state
            ),
            "step_logs": (
                step_logs
            ),
        }

        return result

    finally:
        adapter.close()


def build_argument_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Run a TurtleBot3 Gazebo "
            "legality-shield or matched-reprompt "
            "control episode."
        )
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
        help=(
            "Intervention arm to run."
        ),
    )

    parser.add_argument(
        "--mode",
        choices=[
            "allocentric",
            "egocentric",
        ],
        default="allocentric",
        help=(
            "Action framing."
        ),
    )

    parser.add_argument(
        "--model",
        default=(
            "openai/gpt-4o-mini"
        ),
        help=(
            "OpenRouter model ID."
        ),
    )

    parser.add_argument(
        "--provider",
        default="OpenAI",
        help=(
            "Optional OpenRouter provider pin."
        ),
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=20,
        help=(
            "Maximum physical grid actions "
            "for the episode."
        ),
    )

    parser.add_argument(
        "--output",
        default=(
            "gazebo_intervention_result.json"
        ),
        help=(
            "JSON output filename."
        ),
    )

    return parser


def main():
    args = (
        build_argument_parser()
        .parse_args()
    )

    print()
    print(
        "Creating OpenRouter policy..."
    )

    policy_fn = (
        make_openai_policy_fn(
            model=args.model,
            temperature=0.0,
            provider=args.provider,
            max_output_tokens=16,
        )
    )

    result = run_episode(
        policy_type=args.policy,
        mode=args.mode,
        policy_fn=policy_fn,
        max_steps=(
            args.max_steps
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

    result["max_reprompts"] = (
        2
    )

    result[
        "intervention_exhaustion_behavior"
    ] = (
        "parse_failure; no illegal "
        "action is executed"
    )

    save_result(
        result,
        args.output,
    )

    print()
    print("=" * 80)
    print("EPISODE COMPLETE")
    print("=" * 80)

    print(
        "Reached goal:",
        result[
            "reached_goal"
        ],
    )

    print(
        "Retry budget exhausted:",
        result[
            "retry_budget_exhausted"
        ],
    )

    print(
        "Steps:",
        result[
            "num_steps"
        ],
    )

    print(
        "Final cell:",
        result[
            "final_state"
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