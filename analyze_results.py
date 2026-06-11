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
    if denominator == 0:
        return 0.0
    return numerator / denominator


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
    """
    First-event-per-episode metric:
    return the first non-correct action type, or 'no_first_error'
    if the episode contains no non-correct action.
    """
    for step in step_logs:
        event = classify_error(step)
        if event != "correct":
            return event

    return "no_first_error"


def analyze(results):
    by_condition = defaultdict(lambda: {
        "episodes": 0,
        "successes": 0,
        "step_events": defaultdict(int),
        "step_total": 0,
        "first_events": defaultdict(int),
    })

    by_mode = defaultdict(lambda: {
        "episodes": 0,
        "successes": 0,
        "step_events": defaultdict(int),
        "step_total": 0,
        "first_events": defaultdict(int),
    })

    for ep in results:
        grid_size = ep["grid_size"]
        obstacle_density = ep["obstacle_density"]
        mode = ep["mode"]

        condition_key = (grid_size, obstacle_density, mode)

        for summary in [by_condition[condition_key], by_mode[mode]]:
            summary["episodes"] += 1

            if ep.get("reached_goal", False):
                summary["successes"] += 1

            step_logs = ep.get("step_logs", [])

            # Old measurement: every step-level event.
            for step in step_logs:
                event = classify_error(step)
                if event != "correct":
                    summary["step_events"][event] += 1
                    summary["step_total"] += 1

            # New measurement: first-event-per-episode.
            first_event = get_first_event(step_logs)
            summary["first_events"][first_event] += 1

    return by_condition, by_mode


def print_event_breakdown(events, denominator, indent="  "):
    for event in EVENT_ORDER:
        count = events.get(event, 0)
        if count == 0:
            continue

        pct = 100 * safe_div(count, denominator)
        label = LABEL_MAP.get(event, event)
        print(f"{indent}{label}: {count} ({pct:.1f}%)")


def print_mode_summary(by_mode):
    print("\nMODE-LEVEL SUMMARY")
    print("=" * 80)

    for mode in ["allocentric", "egocentric"]:
        s = by_mode[mode]
        episodes = s["episodes"]
        successes = s["successes"]
        success_rate = 100 * safe_div(successes, episodes)

        print(f"\n{mode.upper()}")
        print("-" * 80)
        print(f"Episodes: {episodes}")
        print(f"Success: {successes}/{episodes} ({success_rate:.1f}%)")

        print("\nOld measurement: step-level errors")
        print_event_breakdown(
            events=s["step_events"],
            denominator=s["step_total"],
            indent="  ",
        )

        print("\nNew measurement: first-event-per-episode")
        print_event_breakdown(
            events=s["first_events"],
            denominator=episodes,
            indent="  ",
        )

        obstacle_step = 100 * safe_div(
            s["step_events"].get("obstacle_blindness", 0),
            s["step_total"],
        )
        obstacle_first = 100 * safe_div(
            s["first_events"].get("obstacle_blindness", 0),
            episodes,
        )

        boundary_step = 100 * safe_div(
            s["step_events"].get("boundary_error", 0),
            s["step_total"],
        )
        boundary_first = 100 * safe_div(
            s["first_events"].get("boundary_error", 0),
            episodes,
        )

        print("\nBefore vs after de-duplication")
        print(f"  Obstacle blindness: {obstacle_step:.1f}% step-level -> {obstacle_first:.1f}% first-event")
        print(f"  Boundary error: {boundary_step:.1f}% step-level -> {boundary_first:.1f}% first-event")


def print_condition_summary(by_condition):
    print("\nCONDITION-LEVEL SUMMARY")
    print("=" * 80)

    for key in sorted(by_condition.keys()):
        grid_size, obstacle_density, mode = key
        s = by_condition[key]

        episodes = s["episodes"]
        successes = s["successes"]
        success_rate = 100 * safe_div(successes, episodes)

        print(f"\n{grid_size}x{grid_size}, density={obstacle_density}, mode={mode}")
        print("-" * 80)
        print(f"Success: {successes}/{episodes} ({success_rate:.1f}%)")

        print("First-event-per-episode breakdown:")
        print_event_breakdown(
            events=s["first_events"],
            denominator=episodes,
            indent="  ",
        )


def main():
    results = load_results("main_results.json")
    by_condition, by_mode = analyze(results)

    print_mode_summary(by_mode)
    print_condition_summary(by_condition)


if __name__ == "__main__":
    main()