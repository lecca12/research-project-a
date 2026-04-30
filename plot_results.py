import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


MODES = ["allocentric", "egocentric"]


def load_results(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_div(numerator, denominator):
    if denominator == 0:
        return 0.0
    return numerator / denominator


LABEL_MAP = {
    "directional_or_detour_error": "Directional",
    "obstacle_blindness": "Obstacle blindness",
    "boundary_error": "Boundary",
    "parse_failure": "Parse failure",
    "no_first_error": "No first error",
}


EVENT_ORDER = [
    "directional_or_detour_error",
    "obstacle_blindness",
    "boundary_error",
    "parse_failure",
    "no_first_error",
]


def classify_error(step):
    """
    Classify a single timestep.

    The order matters:
    - parse failures are output-format failures
    - obstacle and wall collisions are constraint violations
    - valid but non-optimal actions are directional/detour errors
    - otherwise the action is correct
    """
    if step.get("parse_failure", False) or not step.get("is_valid_format", True):
        return "parse_failure"

    if step.get("hit_obstacle", False):
        return "obstacle_blindness"

    if step.get("hit_wall", False):
        return "boundary_error"

    if not step.get("is_correct", False):
        return "directional_or_detour_error"

    return "correct"


def get_first_event(step_logs):
    """
    Return the first non-correct event in the episode.

    If the episode reaches the goal without any non-correct action,
    return 'no_first_error'. This matches the thesis terminology and
    ensures each episode contributes exactly one outcome category.
    """
    for step in step_logs:
        event = classify_error(step)
        if event != "correct":
            return event

    return "no_first_error"


def analyze(results):
    summary = defaultdict(lambda: {
        "episodes": 0,
        "successes": 0,
        "first_events": defaultdict(int),
    })

    for episode in results:
        key = (
            episode.get("grid_size"),
            episode.get("obstacle_density"),
            episode.get("mode"),
        )

        summary[key]["episodes"] += 1

        if episode.get("reached_goal", False):
            summary[key]["successes"] += 1

        first_event = get_first_event(episode.get("step_logs", []))
        summary[key]["first_events"][first_event] += 1

    return summary


def get_conditions(summary):
    return sorted(
        {(grid, density) for grid, density, _ in summary.keys()},
        key=lambda x: (int(x[0]), float(x[1])),
    )


def get_event_types(summary):
    event_types = set()
    for data in summary.values():
        event_types.update(data["first_events"].keys())

    return [event for event in EVENT_ORDER if event in event_types]


def condition_label(grid_size, density):
    return f"{grid_size}x{grid_size}\n{int(float(density) * 100)}%"


def plot_success_rate(summary, output_dir):
    conditions = get_conditions(summary)

    labels = [condition_label(grid, density) for grid, density in conditions]
    x = list(range(len(conditions)))
    width = 0.35

    allocentric_rates = []
    egocentric_rates = []

    for grid, density in conditions:
        for mode, target in [
            ("allocentric", allocentric_rates),
            ("egocentric", egocentric_rates),
        ]:
            data = summary.get((grid, density, mode), None)
            if data is None:
                target.append(0.0)
            else:
                target.append(100 * safe_div(data["successes"], data["episodes"]))

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.bar([i - width / 2 for i in x], allocentric_rates, width, label="Allocentric")
    ax.bar([i + width / 2 for i in x], egocentric_rates, width, label="Egocentric")

    ax.set_title("Success Rate by Condition")
    ax.set_ylabel("Success (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 100)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "success_rate.png", dpi=300)
    plt.close()


def plot_first_event(summary, output_dir):
    conditions = get_conditions(summary)
    event_types = get_event_types(summary)

    labels = []
    for grid, density in conditions:
        for mode in MODES:
            labels.append(f"{condition_label(grid, density)}\n{mode}")

    x = list(range(len(labels)))
    bottoms = [0.0 for _ in labels]

    fig, ax = plt.subplots(figsize=(12, 6))

    for event_type in event_types:
        values = []

        for grid, density in conditions:
            for mode in MODES:
                data = summary.get((grid, density, mode), None)

                if data is None:
                    values.append(0.0)
                    continue

                count = data["first_events"].get(event_type, 0)
                values.append(100 * safe_div(count, data["episodes"]))

        ax.bar(
            x,
            values,
            bottom=bottoms,
            label=LABEL_MAP.get(event_type, event_type),
        )

        bottoms = [b + v for b, v in zip(bottoms, values)]

    ax.set_title("First Episode Outcome by Condition")
    ax.set_ylabel("Percentage of Episodes (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylim(0, 100)

    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()

    # Preferred filename used in thesis.
    fig.savefig(output_dir / "first_event.png", dpi=300, bbox_inches="tight")

    # Backward-compatible filename in case older LaTeX/PPT references remain.
    fig.savefig(output_dir / "first_error.png", dpi=300, bbox_inches="tight")

    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("json_path", default="main_results.json", nargs="?")
    parser.add_argument("--output-dir", default="plots")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    results = load_results(args.json_path)
    summary = analyze(results)

    plot_success_rate(summary, output_dir)
    plot_first_event(summary, output_dir)

    print("Saved plots to:", output_dir)
    print("- success_rate.png")
    print("- first_event.png")
    print("- first_error.png")


if __name__ == "__main__":
    main()