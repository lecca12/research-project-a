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


def load_results(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_div(a, b):
    return a / b if b else 0.0


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


def summarize(results):
    summary = defaultdict(lambda: {
        "episodes": 0,
        "successes": 0,
        "first_events": defaultdict(int),
    })

    for ep in results:
        key = (
            ep.get("policy_type", "baseline"),
            ep["mode"],
        )

        summary[key]["episodes"] += 1

        if ep.get("reached_goal", False):
            summary[key]["successes"] += 1

        event = get_first_event(ep.get("step_logs", []))
        summary[key]["first_events"][event] += 1

    return summary


def plot_success(summary, title, output_path):
    categories = [
        ("baseline", "allocentric", "Baseline\nAllocentric"),
        ("baseline", "egocentric", "Baseline\nEgocentric"),
        ("legality_shield", "allocentric", "Shield\nAllocentric"),
        ("legality_shield", "egocentric", "Shield\nEgocentric"),
    ]

    labels = [label for _, _, label in categories]
    values = []

    for policy, mode, _ in categories:
        data = summary.get((policy, mode))
        if data is None:
            values.append(0.0)
        else:
            values.append(100 * safe_div(data["successes"], data["episodes"]))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(labels, values)

    ax.set_title(title)
    ax.set_ylabel("Success (%)")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.3)

    for i, value in enumerate(values):
        ax.text(i, value + 2, f"{value:.1f}%", ha="center", va="bottom")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close()


def plot_first_events(summary, title, output_path):
    categories = [
        ("baseline", "allocentric", "Baseline\nAllocentric"),
        ("baseline", "egocentric", "Baseline\nEgocentric"),
        ("legality_shield", "allocentric", "Shield\nAllocentric"),
        ("legality_shield", "egocentric", "Shield\nEgocentric"),
    ]

    labels = [label for _, _, label in categories]
    x = list(range(len(labels)))
    bottoms = [0.0 for _ in labels]

    fig, ax = plt.subplots(figsize=(9, 5))

    for event in EVENT_ORDER:
        values = []

        for policy, mode, _ in categories:
            data = summary.get((policy, mode))

            if data is None:
                values.append(0.0)
                continue

            count = data["first_events"].get(event, 0)
            values.append(100 * safe_div(count, data["episodes"]))

        if all(v == 0 for v in values):
            continue

        ax.bar(
            x,
            values,
            bottom=bottoms,
            label=LABEL_MAP[event],
        )

        bottoms = [b + v for b, v in zip(bottoms, values)]

    ax.set_title(title)
    ax.set_ylabel("Percentage of Episodes (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 100)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    output_dir = Path("plots_legality")
    output_dir.mkdir(exist_ok=True)

    custom_results = load_results("custom_minigrid_results.json")
    simple_results = load_results("minigrid_results.json")

    custom_summary = summarize(custom_results)
    simple_summary = summarize(simple_results)

    plot_success(
        custom_summary,
        "Custom MiniGrid Success Rates",
        output_dir / "custom_minigrid_success_rates.png",
    )

    plot_first_events(
        custom_summary,
        "Custom MiniGrid First-Event Outcomes",
        output_dir / "custom_minigrid_first_events.png",
    )

    plot_success(
        simple_summary,
        "SimpleCrossing Success Rates",
        output_dir / "simplecrossing_success_rates.png",
    )

    plot_first_events(
        simple_summary,
        "SimpleCrossing First-Event Outcomes",
        output_dir / "simplecrossing_first_events.png",
    )

    print("Saved plots to:", output_dir)
    print("- custom_minigrid_success_rates.png")
    print("- custom_minigrid_first_events.png")
    print("- simplecrossing_success_rates.png")
    print("- simplecrossing_first_events.png")


if __name__ == "__main__":
    main()