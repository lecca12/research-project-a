import json
from collections import defaultdict


def load_results(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


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


def analyze(results):
    summary = defaultdict(lambda: {
        "episodes": 0,
        "success": 0,
        "first_errors": defaultdict(int),
    })

    for ep in results:
        key = (
            ep["grid_size"],
            ep["obstacle_density"],
            ep["mode"],
        )

        s = summary[key]
        s["episodes"] += 1

        if ep["reached_goal"]:
            s["success"] += 1

        # first error only
        for step in ep["step_logs"]:
            error = classify_error(step)
            if error != "correct":
                s["first_errors"][error] += 1
                break

    return summary


def print_results(summary):
    print("\nRESULTS")
    print("=" * 80)

    for key in sorted(summary.keys()):
        grid, density, mode = key
        s = summary[key]

        success_rate = s["success"] / s["episodes"]

        print(f"\n{grid}x{grid}, density={density}, {mode}")
        print(f"Success: {s['success']}/{s['episodes']} ({success_rate:.2f})")

        print("First error breakdown:")
        total_errors = sum(s["first_errors"].values())

        for err, count in s["first_errors"].items():
            pct = count / total_errors if total_errors > 0 else 0
            print(f"  {err}: {count} ({pct:.2f})")


def main():
    results = load_results("main_results.json")
    summary = analyze(results)
    print_results(summary)


if __name__ == "__main__":
    main()