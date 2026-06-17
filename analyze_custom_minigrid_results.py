import json
from collections import defaultdict


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
    mode_summary = defaultdict(lambda: {
        "episodes": 0,
        "successes": 0,
        "step_events": defaultdict(int),
        "step_total": 0,
        "first_events": defaultdict(int),
    })

    condition_summary = defaultdict(lambda: {
        "episodes": 0,
        "successes": 0,
        "first_events": defaultdict(int),
    })

    for episode in results:
        mode = episode["mode"]

        condition_key = (
            episode["grid_size"],
            episode["obstacle_density"],
            episode["mode"],
        )

        mode_data = mode_summary[mode]
        cond_data = condition_summary[condition_key]

        mode_data["episodes"] += 1
        cond_data["episodes"] += 1

        if episode.get("reached_goal", False):
            mode_data["successes"] += 1
            cond_data["successes"] += 1

        step_logs = episode.get("step_logs", [])

        for step in step_logs:
            event = classify_error(step)

            if event != "correct":
                mode_data["step_events"][event] += 1
                mode_data["step_total"] += 1

        first_event = get_first_event(step_logs)

        mode_data["first_events"][first_event] += 1
        cond_data["first_events"][first_event] += 1

    return mode_summary, condition_summary


def print_breakdown(title, events, denominator):
    print(title)

    for event in EVENT_ORDER:
        count = events.get(event, 0)

        if count == 0:
            continue

        pct = 100 * safe_div(count, denominator)

        print(
            f"  {LABEL_MAP[event]}: "
            f"{count} ({pct:.1f}%)"
        )


def print_mode_summary(mode_summary):
    print("\nMODE-LEVEL SUMMARY")
    print("=" * 80)

    for mode in ["allocentric", "egocentric"]:
        data = mode_summary[mode]

        episodes = data["episodes"]
        successes = data["successes"]

        print(f"\n{mode.upper()}")
        print("-" * 80)
        print(f"Episodes: {episodes}")
        print(
            f"Success: {successes}/{episodes} "
            f"({100 * safe_div(successes, episodes):.1f}%)"
        )

        print()

        print_breakdown(
            "Step-level non-correct events:",
            data["step_events"],
            data["step_total"],
        )

        print()

        print_breakdown(
            "First-event-per-episode outcomes:",
            data["first_events"],
            episodes,
        )


def print_condition_summary(condition_summary):
    print("\nCONDITION-LEVEL SUMMARY")
    print("=" * 80)

    sorted_keys = sorted(condition_summary.keys())

    for grid_size, density, mode in sorted_keys:
        data = condition_summary[(grid_size, density, mode)]

        episodes = data["episodes"]
        successes = data["successes"]

        print(
            f"\n{grid_size}x{grid_size}, "
            f"density={density}, "
            f"mode={mode}"
        )
        print("-" * 80)

        print(
            f"Success: {successes}/{episodes} "
            f"({100 * safe_div(successes, episodes):.1f}%)"
        )

        print("First-event-per-episode breakdown:")

        for event in EVENT_ORDER:
            count = data["first_events"].get(event, 0)

            if count == 0:
                continue

            pct = 100 * safe_div(count, episodes)

            print(
                f"  {LABEL_MAP[event]}: "
                f"{count} ({pct:.1f}%)"
            )


def main():
    results = load_results("custom_minigrid_results.json")

    mode_summary, condition_summary = analyze(results)

    print_mode_summary(mode_summary)
    print_condition_summary(condition_summary)


if __name__ == "__main__":
    main()