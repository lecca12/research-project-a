import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


MODES = ["allocentric", "egocentric"]

LABEL_MAP = {
    "directional_or_detour_error": "Directional",
    "obstacle_blindness": "Obstacle blindness",
    "boundary_error": "Boundary",
    "parse_failure": "Parse failure",
    "no_first_error": "No first error",
}

ERROR_ORDER = [
    "directional_or_detour_error",
    "obstacle_blindness",
    "boundary_error",
    "parse_failure",
]

EVENT_ORDER = [
    "directional_or_detour_error",
    "obstacle_blindness",
    "boundary_error",
    "parse_failure",
    "no_first_error",
]


def load_results(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_div(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def classify_error(step):
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
    for step in step_logs:
        event = classify_error(step)
        if event != "correct":
            return event
    return "no_first_error"


def analyze(results):
    by_condition = defaultdict(lambda: {
        "episodes": 0,
        "successes": 0,
        "first_events": defaultdict(int),
    })

    by_mode = defaultdict(lambda: {
        "episodes": 0,
        "successes": 0,
        "step_events": defaultdict(int),
        "step_total": 0,
        "first_events": defaultdict(int),
    })

    for episode in results:
        grid = episode.get("grid_size")
        density = episode.get("obstacle_density")
        mode = episode.get("mode")
        step_logs = episode.get("step_logs", [])

        condition_key = (grid, density, mode)

        by_condition[condition_key]["episodes"] += 1
        by_mode[mode]["episodes"] += 1

        if episode.get("reached_goal", False):
            by_condition[condition_key]["successes"] += 1
            by_mode[mode]["successes"] += 1

        first_event = get_first_event(step_logs)
        by_condition[condition_key]["first_events"][first_event] += 1
        by_mode[mode]["first_events"][first_event] += 1

        for step in step_logs:
            event = classify_error(step)
            if event != "correct":
                by_mode[mode]["step_events"][event] += 1
                by_mode[mode]["step_total"] += 1

    return by_condition, by_mode


def get_conditions(summary):
    return sorted(
        {(grid, density) for grid, density, _ in summary.keys()},
        key=lambda x: (int(x[0]), float(x[1])),
    )


def condition_label(grid_size, density):
    return f"{grid_size}x{grid_size}\n{int(float(density) * 100)}%"


def plot_success_rate(by_condition, output_dir):
    conditions = get_conditions(by_condition)
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
            data = by_condition.get((grid, density, mode))
            target.append(100 * safe_div(data["successes"], data["episodes"]) if data else 0.0)

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


def plot_first_event_by_condition(by_condition, output_dir):
    conditions = get_conditions(by_condition)

    labels = []
    for grid, density in conditions:
        for mode in MODES:
            labels.append(f"{condition_label(grid, density)}\n{mode}")

    x = list(range(len(labels)))
    bottoms = [0.0 for _ in labels]

    fig, ax = plt.subplots(figsize=(12, 6))

    for event_type in EVENT_ORDER:
        values = []

        for grid, density in conditions:
            for mode in MODES:
                data = by_condition.get((grid, density, mode))
                if data is None:
                    values.append(0.0)
                else:
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
    fig.savefig(output_dir / "first_event.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "first_error.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_step_level_errors_by_mode(by_mode, output_dir):
    labels = [LABEL_MAP[e] for e in ERROR_ORDER]
    x = list(range(len(labels)))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))

    for offset, mode in [(-width / 2, "allocentric"), (width / 2, "egocentric")]:
        data = by_mode[mode]
        values = [
            100 * safe_div(data["step_events"].get(event, 0), data["step_total"])
            for event in ERROR_ORDER
        ]
        ax.bar([i + offset for i in x], values, width, label=mode.capitalize())

    ax.set_title("Old Measurement: Step-Level Error Breakdown")
    ax.set_ylabel("Percentage of Non-Correct Steps (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylim(0, 100)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "step_level_errors_by_mode.png", dpi=300)
    plt.close()


def plot_first_event_by_mode(by_mode, output_dir):
    labels = [LABEL_MAP[e] for e in EVENT_ORDER]
    x = list(range(len(labels)))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))

    for offset, mode in [(-width / 2, "allocentric"), (width / 2, "egocentric")]:
        data = by_mode[mode]
        values = [
            100 * safe_div(data["first_events"].get(event, 0), data["episodes"])
            for event in EVENT_ORDER
        ]
        ax.bar([i + offset for i in x], values, width, label=mode.capitalize())

    ax.set_title("New Measurement: First-Event Outcome by Mode")
    ax.set_ylabel("Percentage of Episodes (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylim(0, 100)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "first_event_by_mode.png", dpi=300)
    plt.close()


def plot_before_after_obstacle_boundary(by_mode, output_dir):
    categories = [
        ("allocentric", "obstacle_blindness", "Allocentric\nobstacle"),
        ("allocentric", "boundary_error", "Allocentric\nboundary"),
        ("egocentric", "obstacle_blindness", "Egocentric\nobstacle"),
        ("egocentric", "boundary_error", "Egocentric\nboundary"),
    ]

    labels = [label for _, _, label in categories]
    x = list(range(len(labels)))
    width = 0.35

    step_values = []
    first_values = []

    for mode, event, _ in categories:
        data = by_mode[mode]
        step_values.append(
            100 * safe_div(data["step_events"].get(event, 0), data["step_total"])
        )
        first_values.append(
            100 * safe_div(data["first_events"].get(event, 0), data["episodes"])
        )

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.bar([i - width / 2 for i in x], step_values, width, label="Step-level")
    ax.bar([i + width / 2 for i in x], first_values, width, label="First-event")

    ax.set_title("Before vs After De-Duplication")
    ax.set_ylabel("Percentage (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 100)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "before_after_obstacle_boundary.png", dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("json_path", default="main_results.json", nargs="?")
    parser.add_argument("--output-dir", default="plots")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    results = load_results(args.json_path)
    by_condition, by_mode = analyze(results)

    plot_success_rate(by_condition, output_dir)
    plot_first_event_by_condition(by_condition, output_dir)
    plot_step_level_errors_by_mode(by_mode, output_dir)
    plot_first_event_by_mode(by_mode, output_dir)
    plot_before_after_obstacle_boundary(by_mode, output_dir)

    print("Saved plots to:", output_dir)
    print("- success_rate.png")
    print("- first_event.png")
    print("- first_error.png")
    print("- step_level_errors_by_mode.png")
    print("- first_event_by_mode.png")
    print("- before_after_obstacle_boundary.png")


if __name__ == "__main__":
    main()