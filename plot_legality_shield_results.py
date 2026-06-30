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

POLICY_ORDER = ["baseline", "legality_shield", "reasoning"]
POLICY_LABEL = {
    "baseline": "Baseline",
    "legality_shield": "Shield",
    "reasoning": "Reasoning",
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
        key = (ep.get("policy_type", "baseline"), ep["mode"])
        summary[key]["episodes"] += 1

        if ep.get("reached_goal", False):
            summary[key]["successes"] += 1

        event = get_first_event(ep.get("step_logs", []))
        summary[key]["first_events"][event] += 1

    return summary


def available_policies(summary):
    policies = {policy for policy, _ in summary.keys()}
    return [p for p in POLICY_ORDER if p in policies]


def plot_success(summary, title, output_path):
    policies = available_policies(summary)
    categories = []

    for policy in policies:
        categories.append((policy, "allocentric", f"{POLICY_LABEL[policy]}\nAllocentric"))
        categories.append((policy, "egocentric", f"{POLICY_LABEL[policy]}\nEgocentric"))

    labels = [label for _, _, label in categories]
    values = []

    for policy, mode, _ in categories:
        data = summary.get((policy, mode))
        values.append(100 * safe_div(data["successes"], data["episodes"]) if data else 0.0)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(labels, values)

    ax.set_title(title)
    ax.set_ylabel("Success (%)")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.3)

    for i, value in enumerate(values):
        ax.text(i, min(value + 2, 98), f"{value:.1f}%", ha="center", va="bottom")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close()


def plot_first_events(summary, title, output_path):
    policies = available_policies(summary)
    categories = []

    for policy in policies:
        categories.append((policy, "allocentric", f"{POLICY_LABEL[policy]}\nAllocentric"))
        categories.append((policy, "egocentric", f"{POLICY_LABEL[policy]}\nEgocentric"))

    labels = [label for _, _, label in categories]
    x = list(range(len(labels)))
    bottoms = [0.0 for _ in labels]

    fig, ax = plt.subplots(figsize=(11, 5))

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
    custom_summary = summarize(custom_results)

    plot_success(
        custom_summary,
        "Custom MiniGrid Success Rates",
        output_dir / "custom_minigrid_success_rates_with_reasoning.png",
    )

    plot_first_events(
        custom_summary,
        "Custom MiniGrid First-Event Outcomes",
        output_dir / "custom_minigrid_first_events_with_reasoning.png",
    )

    try:
        simple_results = load_results("minigrid_results.json")
        simple_summary = summarize(simple_results)

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
    except FileNotFoundError:
        print("minigrid_results.json not found; skipped SimpleCrossing plots.")

    print("Saved plots to:", output_dir)
    print("- custom_minigrid_success_rates_with_reasoning.png")
    print("- custom_minigrid_first_events_with_reasoning.png")
    print("- simplecrossing_success_rates.png, if minigrid_results.json exists")
    print("- simplecrossing_first_events.png, if minigrid_results.json exists")


if __name__ == "__main__":
    main()