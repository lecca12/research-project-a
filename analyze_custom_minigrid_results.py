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
    summary = defaultdict(lambda: {
        "episodes": 0,
        "successes": 0,
        "early_stopped": 0,
        "shield_used_steps": 0,
        "shield_reprompts": 0,
        "total_steps": 0,
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
        policy_type = episode.get("policy_type", "baseline")
        mode = episode["mode"]

        key = (policy_type, mode)
        condition_key = (
            policy_type,
            episode["grid_size"],
            episode["obstacle_density"],
            mode,
        )

        s = summary[key]
        c = condition_summary[condition_key]

        s["episodes"] += 1
        c["episodes"] += 1

        if episode.get("reached_goal", False):
            s["successes"] += 1
            c["successes"] += 1

        if episode.get("early_stopped", False):
            s["early_stopped"] += 1

        step_logs = episode.get("step_logs", [])
        s["total_steps"] += len(step_logs)

        for step in step_logs:
            if step.get("shield_used", False):
                s["shield_used_steps"] += 1

            s["shield_reprompts"] += int(step.get("shield_reprompts", 0))

            event = classify_error(step)
            if event != "correct":
                s["step_events"][event] += 1
                s["step_total"] += 1

        first_event = get_first_event(step_logs)
        s["first_events"][first_event] += 1
        c["first_events"][first_event] += 1

    return summary, condition_summary


def print_breakdown(title, events, denominator):
    print(title)

    for event in EVENT_ORDER:
        count = events.get(event, 0)
        if count == 0:
            continue

        pct = 100 * safe_div(count, denominator)
        print(f"  {LABEL_MAP[event]}: {count} ({pct:.1f}%)")


def print_summary(summary):
    print("\nPOLICY x MODE SUMMARY")
    print("=" * 80)

    for policy_type, mode in sorted(summary.keys()):
        s = summary[(policy_type, mode)]

        episodes = s["episodes"]
        successes = s["successes"]

        print(f"\nPOLICY={policy_type}, MODE={mode}")
        print("-" * 80)
        print(f"Episodes: {episodes}")
        print(f"Success: {successes}/{episodes} ({100 * safe_div(successes, episodes):.1f}%)")
        print(f"Early stopped: {s['early_stopped']}/{episodes} ({100 * safe_div(s['early_stopped'], episodes):.1f}%)")
        print(f"Average steps: {safe_div(s['total_steps'], episodes):.1f}")

        if "shield" in policy_type:
            print(f"Shield-used steps: {s['shield_used_steps']}")
            print(f"Total shield reprompts: {s['shield_reprompts']}")

        print()
        print_breakdown(
            "Step-level non-correct events:",
            s["step_events"],
            s["step_total"],
        )

        print()
        print_breakdown(
            "First-event-per-episode outcomes:",
            s["first_events"],
            episodes,
        )


def print_condition_summary(condition_summary):
    print("\nCONDITION-LEVEL SUMMARY")
    print("=" * 80)

    for policy_type, grid_size, density, mode in sorted(condition_summary.keys()):
        data = condition_summary[(policy_type, grid_size, density, mode)]

        episodes = data["episodes"]
        successes = data["successes"]

        print(f"\nPolicy={policy_type}, {grid_size}x{grid_size}, density={density}, mode={mode}")
        print("-" * 80)
        print(f"Success: {successes}/{episodes} ({100 * safe_div(successes, episodes):.1f}%)")
        print("First-event-per-episode breakdown:")

        for event in EVENT_ORDER:
            count = data["first_events"].get(event, 0)
            if count == 0:
                continue

            pct = 100 * safe_div(count, episodes)
            print(f"  {LABEL_MAP[event]}: {count} ({pct:.1f}%)")


def main():
    results = load_results("custom_minigrid_results.json")

    summary, condition_summary = analyze(results)

    print_summary(summary)
    print_condition_summary(condition_summary)


if __name__ == "__main__":
    main()