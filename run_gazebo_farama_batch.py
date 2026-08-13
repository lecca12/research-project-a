import argparse
import json
import time
from pathlib import Path

from llm_policy import make_openai_policy_fn

from run_gazebo_farama_replay import (
    MAX_STEPS_BY_ENV,
    run_episode,
)


DEFAULT_POLICIES = [
    "legality_shield",
    "reprompt_control",
]

DEFAULT_MODES = [
    "allocentric",
    "egocentric",
]


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


def save_json(
    data,
    path,
):
    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = (
        path.with_suffix(".tmp")
    )

    with temporary_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            make_json_safe(data),
            file,
            indent=2,
        )

    temporary_path.replace(
        path
    )


def make_episode_filename(
    env_name,
    seed,
    policy,
    mode,
):
    safe_env = (
        env_name
        .replace(
            "MiniGrid-",
            "",
        )
        .replace(
            "-v0",
            "",
        )
        .replace(
            "/",
            "_",
        )
    )

    return (
        f"{safe_env}"
        f"_seed{seed}"
        f"_{policy}"
        f"_{mode}.json"
    )


def build_summary(
    episodes,
):
    summary = {}

    for episode in episodes:
        if episode.get(
            "status"
        ) != "complete":
            continue

        policy = (
            episode[
                "policy_type"
            ]
        )

        mode = (
            episode[
                "mode"
            ]
        )

        key = (
            f"{policy}__{mode}"
        )

        if key not in summary:
            summary[key] = {
                "policy_type": (
                    policy
                ),
                "mode": (
                    mode
                ),
                "episodes": 0,
                "successes": 0,
                "retry_budget_exhaustions": 0,
                "total_steps": 0,
                "total_reprompts": 0,
                "episodes_using_reprompt": 0,
            }

        group = (
            summary[key]
        )

        group[
            "episodes"
        ] += 1

        if episode[
            "reached_goal"
        ]:
            group[
                "successes"
            ] += 1

        if episode[
            "retry_budget_exhausted"
        ]:
            group[
                "retry_budget_exhaustions"
            ] += 1

        group[
            "total_steps"
        ] += episode[
            "num_steps"
        ]

        episode_reprompts = 0

        for step in episode[
            "step_logs"
        ]:
            episode_reprompts += (
                step.get(
                    "shield_reprompts",
                    0,
                )
            )

        group[
            "total_reprompts"
        ] += (
            episode_reprompts
        )

        if episode_reprompts > 0:
            group[
                "episodes_using_reprompt"
            ] += 1

    for group in summary.values():
        n = group[
            "episodes"
        ]

        if n > 0:
            group[
                "success_rate"
            ] = (
                group[
                    "successes"
                ]
                / n
            )

            group[
                "retry_budget_exhaustion_rate"
            ] = (
                group[
                    "retry_budget_exhaustions"
                ]
                / n
            )

            group[
                "average_steps"
            ] = (
                group[
                    "total_steps"
                ]
                / n
            )

            group[
                "average_reprompts"
            ] = (
                group[
                    "total_reprompts"
                ]
                / n
            )

    return summary


def print_summary(
    summary,
):
    print()
    print("=" * 80)
    print(
        "BATCH SUMMARY"
    )
    print("=" * 80)

    for key in sorted(
        summary
    ):
        group = (
            summary[key]
        )

        print()
        print(
            f"{group['policy_type']} "
            f"| {group['mode']}"
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
            f"{group['average_steps']:.2f}",
        )

        print(
            "Total reprompts:",
            group[
                "total_reprompts"
            ],
        )


def run_batch(
    environments,
    seeds,
    policies,
    modes,
    policy_fn,
    max_steps_override,
    output_dir,
    batch_output,
    pause_between_runs,
):
    output_dir = Path(
        output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    batch_result = {
        "run_type": (
            "gazebo_farama_batch"
        ),
        "environments": (
            environments
        ),
        "seeds": (
            seeds
        ),
        "policies": (
            policies
        ),
        "modes": (
            modes
        ),
        "episodes": [],
        "summary": {},
    }

    total_runs = (
        len(environments)
        * len(seeds)
        * len(policies)
        * len(modes)
    )

    run_number = 0

    for env_name in environments:
        for seed in seeds:
            for mode in modes:

                # Alternate which policy goes first across seeds.
                # This avoids always giving one intervention the
                # first physical run of a matched pair.
                if seed % 2 == 0:
                    ordered_policies = (
                        policies
                    )

                else:
                    ordered_policies = (
                        list(
                            reversed(
                                policies
                            )
                        )
                    )

                for policy in ordered_policies:
                    run_number += 1

                    print()
                    print("#" * 80)

                    print(
                        f"BATCH RUN "
                        f"{run_number}/"
                        f"{total_runs}"
                    )

                    print(
                        f"Environment: "
                        f"{env_name}"
                    )

                    print(
                        f"Seed: {seed}"
                    )

                    print(
                        f"Mode: {mode}"
                    )

                    print(
                        f"Policy: {policy}"
                    )

                    print("#" * 80)

                    if (
                        max_steps_override
                        is None
                    ):
                        max_steps = (
                            MAX_STEPS_BY_ENV[
                                env_name
                            ]
                        )

                    else:
                        max_steps = (
                            max_steps_override
                        )

                    episode_filename = (
                        make_episode_filename(
                            env_name=(
                                env_name
                            ),
                            seed=(
                                seed
                            ),
                            policy=(
                                policy
                            ),
                            mode=(
                                mode
                            ),
                        )
                    )

                    episode_path = (
                        output_dir
                        / episode_filename
                    )

                    try:
                        episode = (
                            run_episode(
                                env_name=(
                                    env_name
                                ),
                                seed=(
                                    seed
                                ),
                                policy_type=(
                                    policy
                                ),
                                mode=(
                                    mode
                                ),
                                policy_fn=(
                                    policy_fn
                                ),
                                max_steps=(
                                    max_steps
                                ),
                                cleanup_walls=(
                                    True
                                ),
                                verbose=(
                                    True
                                ),
                            )
                        )

                        episode[
                            "status"
                        ] = (
                            "complete"
                        )

                        save_json(
                            episode,
                            episode_path,
                        )

                        print()
                        print(
                            "Episode saved:",
                            episode_path,
                        )

                    except Exception as exc:
                        print()
                        print(
                            "EPISODE ERROR:"
                        )

                        print(
                            str(exc)
                        )

                        episode = {
                            "status": (
                                "error"
                            ),
                            "env_name": (
                                env_name
                            ),
                            "seed": (
                                seed
                            ),
                            "policy_type": (
                                policy
                            ),
                            "mode": (
                                mode
                            ),
                            "error": (
                                repr(
                                    exc
                                )
                            ),
                        }

                    batch_result[
                        "episodes"
                    ].append(
                        episode
                    )

                    batch_result[
                        "summary"
                    ] = (
                        build_summary(
                            batch_result[
                                "episodes"
                            ]
                        )
                    )

                    # Save after every episode so a later interruption
                    # does not destroy completed results.
                    save_json(
                        batch_result,
                        batch_output,
                    )

                    if (
                        run_number
                        < total_runs
                    ):
                        print()
                        print(
                            "Waiting before "
                            "next episode..."
                        )

                        time.sleep(
                            pause_between_runs
                        )

    return batch_result


def build_argument_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Run matched Gazebo/Farama "
            "legality-shield vs reprompt-control "
            "episodes across multiple seeds."
        )
    )

    parser.add_argument(
        "--environments",
        nargs="+",
        default=[
            "MiniGrid-SimpleCrossingS9N1-v0",
        ],
        choices=list(
            MAX_STEPS_BY_ENV
        ),
    )

    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[
            0,
            1,
            2,
        ],
    )

    parser.add_argument(
        "--policies",
        nargs="+",
        default=(
            DEFAULT_POLICIES
        ),
        choices=[
            "legality_shield",
            "reprompt_control",
        ],
    )

    parser.add_argument(
        "--modes",
        nargs="+",
        default=(
            DEFAULT_MODES
        ),
        choices=[
            "allocentric",
            "egocentric",
        ],
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
        "--output-dir",
        default=(
            "gazebo_batch_results"
        ),
    )

    parser.add_argument(
        "--batch-output",
        default=(
            "gazebo_batch_results.json"
        ),
    )

    parser.add_argument(
        "--pause-between-runs",
        type=float,
        default=1.0,
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

    print()
    print("=" * 80)
    print(
        "GAZEBO BATCH EXPERIMENT"
    )
    print("=" * 80)

    print(
        "Environments:",
        args.environments,
    )

    print(
        "Seeds:",
        args.seeds,
    )

    print(
        "Policies:",
        args.policies,
    )

    print(
        "Modes:",
        args.modes,
    )

    total_runs = (
        len(
            args.environments
        )
        * len(
            args.seeds
        )
        * len(
            args.policies
        )
        * len(
            args.modes
        )
    )

    print(
        "Total episodes:",
        total_runs,
    )

    print("=" * 80)

    result = run_batch(
        environments=(
            args.environments
        ),
        seeds=(
            args.seeds
        ),
        policies=(
            args.policies
        ),
        modes=(
            args.modes
        ),
        policy_fn=(
            policy_fn
        ),
        max_steps_override=(
            args.max_steps
        ),
        output_dir=(
            args.output_dir
        ),
        batch_output=(
            args.batch_output
        ),
        pause_between_runs=(
            args.pause_between_runs
        ),
    )

    print_summary(
        result[
            "summary"
        ]
    )

    print()
    print(
        "Batch result saved:",
        args.batch_output,
    )


if __name__ == "__main__":
    main()