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
        "step_events": defaultdict(int),
        "step_total": 0,
        "first_events": defaultdict(int),
        "first_raw_answers": defaultdict(int),
    })

    for episode in results:
        mode = episode["mode"]
        s = summary[mode]

        s["episodes"] += 1

        if episode.get("reached_goal", False):
            s["successes"] += 1

        step_logs = episode.get("step_logs", [])

        for step in step_logs:
            event = classify_error(step)
            if event != "correct":
                s["step_events"][event] += 1
                s["step_total"] += 1

        first_event = get_first_event(step_logs)
        s["first_events"][first_event] += 1

        if step_logs:
            first_bad = None
            for step in step_logs:
                if classify_error(step) != "correct":
                    first_bad = step
                    break

            if first_bad is not None:
                raw = first_bad.get("raw_model_answer", "")
                parsed = first_bad.get("parsed_action_name", first_bad.get("parsed_action"))
                optimal = tuple(first_bad.get("optimal_action_names", []))
                key = f"raw={raw}, parsed={parsed}, optimal={optimal}"
                s["first_raw_answers"][key] += 1

    return summary


def print_breakdown(title, events, denominator):
    print(title)
    for event in EVENT_ORDER:
        count = events.get(event, 0)
        if count == 0:
            continue

        pct = 100 * safe_div(count, denominator)
        label = LABEL_MAP.get(event, event)
        print(f"  {label}: {count} ({pct:.1f}%)")


def main():
    results = load_results("minigrid_results.json")
    summary = analyze(results)

    print("\nMINIGRID RESULTS SUMMARY")
    print("=" * 80)

    for mode in ["allocentric", "egocentric"]:
        s = summary[mode]

        episodes = s["episodes"]
        successes = s["successes"]
        success_rate = 100 * safe_div(successes, episodes)

        print("\n" + mode.upper())
        print("-" * 80)
        print(f"Episodes: {episodes}")
        print(f"Success: {successes}/{episodes} ({success_rate:.1f}%)")

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

        print("\nMost common first bad actions:")
        for key, count in sorted(
            s["first_raw_answers"].items(),
            key=lambda item: item[1],
            reverse=True,
        )[:10]:
            print(f"  {count}x {key}")


if __name__ == "__main__":
    main()