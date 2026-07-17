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


def analyze(results):
    summary = defaultdict(lambda: {
        "episodes": 0,
        "successes": 0,
        "early_stopped": 0,
        "total_steps": 0,
        "shield_used_steps": 0,
        "shield_reprompts": 0,
        "step_events": defaultdict(int),
        "step_total": 0,
        "first_events": defaultdict(int),
    })

    for episode in results:
        key = (
            episode["env_name"],
            episode.get("policy_type", "baseline"),
            episode["mode"],
        )

        s = summary[key]
        s["episodes"] += 1
        s["total_steps"] += episode.get("num_steps", 0)

        if episode.get("reached_goal", False):
            s["successes"] += 1

        if episode.get("early_stopped", False):
            s["early_stopped"] += 1

        step_logs = episode.get("step_logs", [])

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

    return summary


def print_breakdown(title, events, denominator):
    print(title)
    for event in EVENT_ORDER:
        count = events.get(event, 0)
        if count == 0:
            continue
        pct = 100 * safe_div(count, denominator)
        print(f"  {LABEL_MAP[event]}: {count} ({pct:.1f}%)")


def main():
    # results = load_results("farama_minigrid_results.json")
    results = load_results("farama_minigrid_results_final.json")
    summary = analyze(results)

    print("\nFARAMA MINIGRID RESULTS SUMMARY")
    print("=" * 80)

    for env_name, policy_type, mode in sorted(summary.keys()):
        s = summary[(env_name, policy_type, mode)]
        episodes = s["episodes"]
        successes = s["successes"]

        print(f"\nENV={env_name}")
        print(f"POLICY={policy_type}, MODE={mode}")
        print("-" * 80)
        print(f"Episodes: {episodes}")
        print(f"Success: {successes}/{episodes} ({100 * safe_div(successes, episodes):.1f}%)")
        print(f"Early stopped: {s['early_stopped']}/{episodes} ({100 * safe_div(s['early_stopped'], episodes):.1f}%)")
        print(f"Average steps: {safe_div(s['total_steps'], episodes):.1f}")

        if "shield" in policy_type:
            print(f"Shield-used steps: {s['shield_used_steps']}")
            print(f"Total shield reprompts: {s['shield_reprompts']}")

        print()
        print_breakdown("Step-level non-correct events:", s["step_events"], s["step_total"])

        print()
        print_breakdown("First-event-per-episode outcomes:", s["first_events"], episodes)


if __name__ == "__main__":
    main()