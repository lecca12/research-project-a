
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


def classify_error(step):
    """
    First-error taxonomy.

    We use first error per episode to avoid repeated stuck-state errors
    dominating the results.
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


def get_first_error(step_logs):
    for step in step_logs:
        error = classify_error(step)
        if error != "correct":
            return error
    return None


def analyze(results):
    summary = defaultdict(lambda: {
        "episodes": 0,
        "successes": 0,
        "first_errors": defaultdict(int),
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

        first_error = get_first_error(episode.get("step_logs", []))
        if first_error is not None:
            summary[key]["first_errors"][first_error] += 1

    return summary


def get_conditions(summary):
    return sorted(
        {(grid, density) for grid, density, _ in summary.keys()},
        key=lambda x: (int(x[0]), float(x[1])),
    )


def get_error_types(summary):
    error_types = set()
    for data in summary.values():
        error_types.update(data["first_errors"].keys())

    preferred_order = [
        "directional_or_detour_error",
        "obstacle_blindness",
        "boundary_error",
        "parse_failure",
    ]

    ordered = [e for e in preferred_order if e in error_types]
    ordered += sorted(error_types - set(ordered))
    return ordered


def condition_label(grid_size, density):
    return f"{grid_size}x{grid_size}\n{int(float(density) * 100)}% obs"


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

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.bar([i - width / 2 for i in x], allocentric_rates, width, label="Allocentric")
    ax.bar([i + width / 2 for i in x], egocentric_rates, width, label="Egocentric")

    for i, value in enumerate(allocentric_rates):
        ax.text(i - width / 2, value, f"{value:.0f}%", ha="center", va="bottom", fontsize=8)

    for i, value in enumerate(egocentric_rates):
        ax.text(i + width / 2, value, f"{value:.0f}%", ha="center", va="bottom", fontsize=8)

    ax.set_title("Episode success rate by condition")
    ax.set_ylabel("Success rate (%)")
    ax.set_xlabel("Condition")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 105)
    ax.legend(title="Prompt framing")
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    output_path = output_dir / "success_rate_by_condition.png"
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    return output_path


def plot_first_error_by_condition(summary, output_dir):
    conditions = get_conditions(summary)
    error_types = get_error_types(summary)

    labels = []
    for grid, density in conditions:
        for mode in MODES:
            labels.append(f"{condition_label(grid, density)}\n{mode}")

    x = list(range(len(labels)))
    bottoms = [0 for _ in labels]

    fig, ax = plt.subplots(figsize=(12, 6))

    for error_type in error_types:
        values = []

        for grid, density in conditions:
            for mode in MODES:
                data = summary.get((grid, density, mode), None)

                if data is None:
                    values.append(0.0)
                    continue

                count = data["first_errors"].get(error_type, 0)
                values.append(100 * safe_div(count, data["episodes"]))

        ax.bar(x, values, bottom=bottoms, label=error_type)
        bottoms = [b + v for b, v in zip(bottoms, values)]

    ax.set_title("First error per episode by condition")
    ax.set_ylabel("Episodes with first error type (%)")
    ax.set_xlabel("Condition and prompt framing")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylim(0, 105)
    ax.legend(title="First error type")
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    output_path = output_dir / "first_error_by_condition.png"
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    return output_path


def write_summary_text(summary, output_dir):
    conditions = get_conditions(summary)
    lines = []

    lines.append("RESULT SUMMARY")
    lines.append("=" * 80)
    lines.append("")
    lines.append("Success rate by condition:")
    lines.append("")

    for grid, density in conditions:
        lines.append(f"{grid}x{grid}, obstacle density={density}")

        for mode in MODES:
            data = summary.get((grid, density, mode), None)
            if data is None:
                lines.append(f"  {mode}: no data")
                continue

            success_rate = 100 * safe_div(data["successes"], data["episodes"])
            lines.append(
                f"  {mode}: {data['successes']}/{data['episodes']} "
                f"successful episodes ({success_rate:.1f}%)"
            )

        lines.append("")

    lines.append("")
    lines.append("First error per episode:")
    lines.append("")

    for grid, density in conditions:
        lines.append(f"{grid}x{grid}, obstacle density={density}")

        for mode in MODES:
            data = summary.get((grid, density, mode), None)
            if data is None:
                lines.append(f"  {mode}: no data")
                continue

            lines.append(f"  {mode}:")
            if not data["first_errors"]:
                lines.append("    no first errors")
            else:
                for error_type, count in sorted(data["first_errors"].items()):
                    pct = 100 * safe_div(count, data["episodes"])
                    lines.append(f"    {error_type}: {count}/{data['episodes']} ({pct:.1f}%)")

        lines.append("")

    output_path = output_dir / "results_summary.txt"
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Plot simplified GridWorld experiment results.")
    parser.add_argument(
        "json_path",
        nargs="?",
        default="main_results.json",
        help="Path to results JSON file.",
    )
    parser.add_argument(
        "--output-dir",
        default="plots",
        help="Folder to save plots.",
    )
    args = parser.parse_args()

    json_path = Path(args.json_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = load_results(json_path)
    summary = analyze(results)

    success_plot = plot_success_rate(summary, output_dir)
    error_plot = plot_first_error_by_condition(summary, output_dir)
    summary_text = write_summary_text(summary, output_dir)

    print("Saved:")
    print(f"- {success_plot}")
    print(f"- {error_plot}")
    print(f"- {summary_text}")


if __name__ == "__main__":
    main()
