import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


EVENT_ORDER = [
    "directional_or_detour_error",
    "obstacle_blindness",
    "boundary_error",
    "parse_failure",
    "no_first_error",
]

LABEL_MAP = {
    "directional_or_detour_error": "Directional",
    "obstacle_blindness": "Obstacle blindness",
    "boundary_error": "Boundary",
    "parse_failure": "Parse failure",
    "no_first_error": "No first error",
}

POLICY_ORDER = [
    "baseline",
    "reprompt_control",
    "legality_shield",
    "reasoning",
]

POLICY_LABEL = {
    "baseline": "Baseline",
    "reprompt_control": "Retry Control",
    "legality_shield": "Shield",
    "reasoning": "Reasoning",
}

ENV_ORDER = [
    "MiniGrid-SimpleCrossingS9N1-v0",
    "MiniGrid-SimpleCrossingS9N2-v0",
    "MiniGrid-SimpleCrossingS9N3-v0",
    "MiniGrid-FourRooms-v0",
]

ENV_LABEL = {
    "MiniGrid-SimpleCrossingS9N1-v0": "SimpleCrossing N1",
    "MiniGrid-SimpleCrossingS9N2-v0": "SimpleCrossing N2",
    "MiniGrid-SimpleCrossingS9N3-v0": "SimpleCrossing N3",
    "MiniGrid-FourRooms-v0": "FourRooms",
}


def load_results(paths):
    """Load one merged result file or concatenate several chunk files."""
    if isinstance(paths, (str, Path)):
        paths = [paths]

    combined = []

    for path in paths:
        with open(path, "r", encoding="utf-8") as file:
            payload = json.load(file)

        if not isinstance(payload, list):
            raise ValueError(
                f"Expected a JSON list of episodes in {path}, "
                f"got {type(payload).__name__}."
            )

        combined.extend(payload)

    return combined


def safe_div(a, b):
    return a / b if b else 0.0


def classify_error(step):
    if (
        step.get("parse_failure", False)
        or not step.get("is_valid_format", True)
    ):
        return "parse_failure"

    if step.get("hit_obstacle", False):
        return "obstacle_blindness"

    if step.get("hit_wall", False):
        return "boundary_error"

    if not step.get("is_correct", False):
        return "directional_or_detour_error"

    return "correct"


def get_first_event(step_logs):
    for step in step_logs:
        event = classify_error(step)

        if event != "correct":
            return event

    return "no_first_error"


def summarize(results):
    summary = defaultdict(
        lambda: {
            "episodes": 0,
            "successes": 0,
            "first_events": defaultdict(int),
        }
    )

    for episode in results:
        key = (
            episode["env_name"],
            episode.get("policy_type", "baseline"),
            episode["mode"],
        )

        summary[key]["episodes"] += 1

        if episode.get("reached_goal", False):
            summary[key]["successes"] += 1

        event = get_first_event(
            episode.get("step_logs", [])
        )

        summary[key]["first_events"][event] += 1

    return summary


def get_envs(summary):
    present_envs = {
        env_name
        for env_name, _, _ in summary.keys()
    }

    ordered = [
        env_name
        for env_name in ENV_ORDER
        if env_name in present_envs
    ]

    additional = sorted(
        present_envs.difference(ENV_ORDER)
    )

    return ordered + additional


def get_policies(summary, env_name):
    policies = {
        policy
        for env, policy, _ in summary.keys()
        if env == env_name
    }

    ordered = [
        policy
        for policy in POLICY_ORDER
        if policy in policies
    ]

    additional = sorted(
        policies.difference(POLICY_ORDER)
    )

    return ordered + additional


def get_policy_label(policy):
    return POLICY_LABEL.get(
        policy,
        policy.replace("_", " ").title(),
    )


def plot_success_by_env(summary, output_dir):
    for env_name in get_envs(summary):
        policies = get_policies(
            summary,
            env_name,
        )

        categories = []

        for policy in policies:
            label = get_policy_label(policy)

            categories.append(
                (
                    policy,
                    "allocentric",
                    f"{label}\nAllocentric",
                )
            )

            categories.append(
                (
                    policy,
                    "egocentric",
                    f"{label}\nEgocentric",
                )
            )

        labels = [
            label
            for _, _, label in categories
        ]

        values = []

        for policy, mode, _ in categories:
            data = summary.get(
                (
                    env_name,
                    policy,
                    mode,
                )
            )

            if data is None:
                values.append(0.0)
                continue

            values.append(
                100
                * safe_div(
                    data["successes"],
                    data["episodes"],
                )
            )

        figure_width = max(
            10,
            len(categories) * 1.25,
        )

        fig, ax = plt.subplots(
            figsize=(figure_width, 5)
        )

        ax.bar(
            labels,
            values,
        )

        ax.set_title(
            f"{ENV_LABEL.get(env_name, env_name)} "
            "Success Rates"
        )

        ax.set_ylabel(
            "Success (%)"
        )

        ax.set_ylim(
            0,
            100,
        )

        ax.grid(
            axis="y",
            alpha=0.3,
        )

        for index, value in enumerate(values):
            ax.text(
                index,
                min(value + 2, 98),
                f"{value:.1f}%",
                ha="center",
                va="bottom",
            )

        fig.tight_layout()

        safe_name = env_name.replace(
            "-",
            "_",
        )

        fig.savefig(
            output_dir
            / f"{safe_name}_success_rates.png",
            dpi=300,
        )

        plt.close(fig)


def plot_first_events_by_env(
    summary,
    output_dir,
):
    for env_name in get_envs(summary):
        policies = get_policies(
            summary,
            env_name,
        )

        categories = []

        for policy in policies:
            label = get_policy_label(policy)

            categories.append(
                (
                    policy,
                    "allocentric",
                    f"{label}\nAllocentric",
                )
            )

            categories.append(
                (
                    policy,
                    "egocentric",
                    f"{label}\nEgocentric",
                )
            )

        labels = [
            label
            for _, _, label in categories
        ]

        x = list(
            range(len(labels))
        )

        bottoms = [
            0.0
            for _ in labels
        ]

        figure_width = max(
            11,
            len(categories) * 1.3,
        )

        fig, ax = plt.subplots(
            figsize=(figure_width, 5)
        )

        for event in EVENT_ORDER:
            values = []

            for policy, mode, _ in categories:
                data = summary.get(
                    (
                        env_name,
                        policy,
                        mode,
                    )
                )

                if data is None:
                    values.append(0.0)
                    continue

                count = data[
                    "first_events"
                ].get(
                    event,
                    0,
                )

                values.append(
                    100
                    * safe_div(
                        count,
                        data["episodes"],
                    )
                )

            if all(
                value == 0
                for value in values
            ):
                continue

            ax.bar(
                x,
                values,
                bottom=bottoms,
                label=LABEL_MAP[event],
            )

            bottoms = [
                bottom + value
                for bottom, value
                in zip(bottoms, values)
            ]

        ax.set_title(
            f"{ENV_LABEL.get(env_name, env_name)} "
            "First-Event Outcomes"
        )

        ax.set_ylabel(
            "Percentage of Episodes (%)"
        )

        ax.set_xticks(x)
        ax.set_xticklabels(labels)

        ax.set_ylim(
            0,
            100,
        )

        ax.legend(
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
        )

        ax.grid(
            axis="y",
            alpha=0.3,
        )

        fig.tight_layout()

        safe_name = env_name.replace(
            "-",
            "_",
        )

        fig.savefig(
            output_dir
            / f"{safe_name}_first_events.png",
            dpi=300,
            bbox_inches="tight",
        )

        plt.close(fig)


def plot_success_sweep(
    summary,
    output_dir,
):
    rows = []

    for env_name in get_envs(summary):
        for policy in get_policies(
            summary,
            env_name,
        ):
            for mode in [
                "allocentric",
                "egocentric",
            ]:
                data = summary.get(
                    (
                        env_name,
                        policy,
                        mode,
                    )
                )

                if data is None:
                    continue

                mode_label = (
                    "Alloc."
                    if mode == "allocentric"
                    else "Ego."
                )

                rows.append(
                    {
                        "label": (
                            f"{ENV_LABEL.get(env_name, env_name)}\n"
                            f"{get_policy_label(policy)}\n"
                            f"{mode_label}"
                        ),
                        "success": (
                            100
                            * safe_div(
                                data["successes"],
                                data["episodes"],
                            )
                        ),
                    }
                )

    labels = [
        row["label"]
        for row in rows
    ]

    values = [
        row["success"]
        for row in rows
    ]

    figure_width = max(
        14,
        len(rows) * 0.7,
    )

    fig, ax = plt.subplots(
        figsize=(figure_width, 6)
    )

    ax.bar(
        labels,
        values,
    )

    ax.set_title(
        "Farama MiniGrid Success Rates "
        "Across Environments"
    )

    ax.set_ylabel(
        "Success (%)"
    )

    ax.set_ylim(
        0,
        100,
    )

    ax.grid(
        axis="y",
        alpha=0.3,
    )

    for index, value in enumerate(values):
        ax.text(
            index,
            min(value + 2, 98),
            f"{value:.0f}%",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.tick_params(
        axis="x",
        labelrotation=45,
    )

    fig.tight_layout()

    fig.savefig(
        output_dir
        / "farama_success_sweep.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Plot one merged Farama result file "
            "or multiple chunk result files."
        )
    )

    parser.add_argument(
        "inputs",
        nargs="*",
        default=[
            "farama_minigrid_results_final.json"
        ],
        help=(
            "Merged JSON file, or multiple chunk "
            "JSON files."
        ),
    )

    parser.add_argument(
        "--output-dir",
        default="plots_farama",
        help="Directory in which plots are saved.",
    )

    args = parser.parse_args()

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = load_results(
        args.inputs
    )

    summary = summarize(
        results
    )

    plot_success_by_env(
        summary,
        output_dir,
    )

    plot_first_events_by_env(
        summary,
        output_dir,
    )

    plot_success_sweep(
        summary,
        output_dir,
    )

    print(
        "Saved plots to:",
        output_dir,
    )

    print(
        "- one success plot per environment"
    )

    print(
        "- one first-event plot per environment"
    )

    print(
        "- farama_success_sweep.png"
    )


if __name__ == "__main__":
    main()