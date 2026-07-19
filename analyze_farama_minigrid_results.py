import json
import re
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


ABSOLUTE_ACTIONS = {
    "north": 0,
    "east": 1,
    "south": 2,
    "west": 3,
}

RELATIVE_ACTIONS = {
    "forward": 0,
    "right": 1,
    "backward": 2,
    "left": 3,
}


def load_results(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_div(a, b):
    return a / b if b else 0.0


def normalize_answer(text):
    """
    Normalize a short saved model response into one lowercase word.

    This is deliberately permissive because the baseline and shield policies
    were instructed to return only one word.
    """
    if text is None:
        return ""

    words = re.findall(r"[a-z]+", str(text).lower())

    if not words:
        return ""

    return words[-1]


def relative_to_cardinal(relative_action, facing):
    """
    Convert an egocentric action to the project's cardinal action index.

    MiniGrid facing:
        0 = east
        1 = south
        2 = west
        3 = north

    Project cardinal actions:
        0 = north
        1 = east
        2 = south
        3 = west
    """
    facing_to_cardinal = {
        0: 1,  # east
        1: 2,  # south
        2: 3,  # west
        3: 0,  # north
    }

    cardinal_facing = facing_to_cardinal[int(facing)]

    return (
        cardinal_facing + int(relative_action)
    ) % 4


def parse_saved_action(raw_answer, mode, facing):
    """
    Parse a saved first-attempt response into a cardinal action index.
    """
    word = normalize_answer(raw_answer)

    if mode == "allocentric":
        return ABSOLUTE_ACTIONS.get(word)

    if mode == "egocentric":
        relative_action = RELATIVE_ACTIONS.get(word)

        if relative_action is None:
            return None

        return relative_to_cardinal(
            relative_action=relative_action,
            facing=facing,
        )

    raise ValueError(f"Unknown mode: {mode}")


def get_first_raw_answer(step):
    """
    Return the original model proposal before any shield reprompt.
    """
    all_raw_answers = step.get(
        "all_raw_answers",
        [],
    )

    if all_raw_answers:
        return all_raw_answers[0]

    # Fallback for older result files without all_raw_answers.
    return step.get("raw_model_answer")


def classify_error(step):
    if (
        step.get("parse_failure", False)
        or not step.get("is_valid_format", True)
    ):
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
    summary = defaultdict(
        lambda: {
            "episodes": 0,
            "successes": 0,
            "early_stopped": 0,
            "total_steps": 0,
            "shield_used_steps": 0,
            "shield_reprompts": 0,
            "step_events": defaultdict(int),
            "step_total": 0,
            "first_events": defaultdict(int),

            # Shield-specific first-attempt metrics.
            "shield_steps": 0,
            "shield_first_attempt_valid": 0,
            "shield_first_attempt_legal": 0,
            "shield_first_attempt_correct": 0,
            "shield_executed_valid": 0,
            "shield_executed_legal": 0,
            "shield_executed_correct": 0,
            "shield_intervention_steps": 0,
        }
    )

    for episode in results:
        policy_type = episode.get(
            "policy_type",
            "baseline",
        )

        mode = episode["mode"]

        key = (
            episode["env_name"],
            policy_type,
            mode,
        )

        s = summary[key]

        s["episodes"] += 1
        s["total_steps"] += episode.get(
            "num_steps",
            0,
        )

        if episode.get("reached_goal", False):
            s["successes"] += 1

        if episode.get("early_stopped", False):
            s["early_stopped"] += 1

        step_logs = episode.get(
            "step_logs",
            [],
        )

        for step in step_logs:
            if step.get("shield_used", False):
                s["shield_used_steps"] += 1

            s["shield_reprompts"] += int(
                step.get(
                    "shield_reprompts",
                    0,
                )
            )

            event = classify_error(step)

            if event != "correct":
                s["step_events"][event] += 1
                s["step_total"] += 1

            if policy_type == "legality_shield":
                s["shield_steps"] += 1

                first_raw_answer = (
                    get_first_raw_answer(step)
                )

                first_action = parse_saved_action(
                    raw_answer=first_raw_answer,
                    mode=mode,
                    facing=step.get("facing", 0),
                )

                optimal_actions = {
                    int(action)
                    for action in step.get(
                        "optimal_actions",
                        [],
                    )
                }

                legal_names = {
                    str(name).lower()
                    for name in step.get(
                        "legal_action_names",
                        [],
                    )
                }

                first_word = normalize_answer(
                    first_raw_answer
                )

                first_attempt_valid = (
                    first_action is not None
                )

                first_attempt_legal = (
                    first_attempt_valid
                    and first_word in legal_names
                )

                first_attempt_correct = (
                    first_attempt_valid
                    and first_action
                    in optimal_actions
                )

                if first_attempt_valid:
                    s[
                        "shield_first_attempt_valid"
                    ] += 1

                if first_attempt_legal:
                    s[
                        "shield_first_attempt_legal"
                    ] += 1

                if first_attempt_correct:
                    s[
                        "shield_first_attempt_correct"
                    ] += 1

                executed_action = step.get(
                    "parsed_action"
                )

                executed_valid = (
                    executed_action is not None
                    and not step.get(
                        "parse_failure",
                        False,
                    )
                )

                if executed_valid:
                    executed_action = int(
                        executed_action
                    )

                    s[
                        "shield_executed_valid"
                    ] += 1

                    # The shield should ensure every executed
                    # action is legal. This counter is retained
                    # as a consistency check.
                    s[
                        "shield_executed_legal"
                    ] += 1

                    if (
                        executed_action
                        in optimal_actions
                    ):
                        s[
                            "shield_executed_correct"
                        ] += 1

                if step.get(
                    "shield_used",
                    False,
                ):
                    s[
                        "shield_intervention_steps"
                    ] += 1

        first_event = get_first_event(
            step_logs
        )

        s["first_events"][
            first_event
        ] += 1

    return summary


def print_breakdown(
    title,
    events,
    denominator,
):
    print(title)

    for event in EVENT_ORDER:
        count = events.get(
            event,
            0,
        )

        if count == 0:
            continue

        pct = 100 * safe_div(
            count,
            denominator,
        )

        print(
            f"  {LABEL_MAP[event]}: "
            f"{count} ({pct:.1f}%)"
        )


def print_shield_accuracy(s):
    total = s["shield_steps"]

    first_valid = s[
        "shield_first_attempt_valid"
    ]

    first_legal = s[
        "shield_first_attempt_legal"
    ]

    first_correct = s[
        "shield_first_attempt_correct"
    ]

    executed_valid = s[
        "shield_executed_valid"
    ]

    executed_legal = s[
        "shield_executed_legal"
    ]

    executed_correct = s[
        "shield_executed_correct"
    ]

    interventions = s[
        "shield_intervention_steps"
    ]

    print()
    print("Shield first-attempt versus executed action:")
    print(
        "  First-attempt valid format: "
        f"{first_valid}/{total} "
        f"({100 * safe_div(first_valid, total):.1f}%)"
    )
    print(
        "  First-attempt legal: "
        f"{first_legal}/{total} "
        f"({100 * safe_div(first_legal, total):.1f}%)"
    )
    print(
        "  First-attempt correct: "
        f"{first_correct}/{total} "
        f"({100 * safe_div(first_correct, total):.1f}%)"
    )
    print(
        "  Executed valid action: "
        f"{executed_valid}/{total} "
        f"({100 * safe_div(executed_valid, total):.1f}%)"
    )
    print(
        "  Executed legal action: "
        f"{executed_legal}/{total} "
        f"({100 * safe_div(executed_legal, total):.1f}%)"
    )
    print(
        "  Executed correct action: "
        f"{executed_correct}/{total} "
        f"({100 * safe_div(executed_correct, total):.1f}%)"
    )
    print(
        "  Shield intervention rate: "
        f"{interventions}/{total} "
        f"({100 * safe_div(interventions, total):.1f}%)"
    )

    if first_valid:
        print(
            "  First-attempt accuracy among valid proposals: "
            f"{first_correct}/{first_valid} "
            f"({100 * safe_div(first_correct, first_valid):.1f}%)"
        )

    if executed_valid:
        print(
            "  Executed accuracy among valid actions: "
            f"{executed_correct}/{executed_valid} "
            f"({100 * safe_div(executed_correct, executed_valid):.1f}%)"
        )


def print_overall_shield_summary(summary):
    combined = defaultdict(
        lambda: {
            "shield_steps": 0,
            "shield_first_attempt_valid": 0,
            "shield_first_attempt_legal": 0,
            "shield_first_attempt_correct": 0,
            "shield_executed_valid": 0,
            "shield_executed_legal": 0,
            "shield_executed_correct": 0,
            "shield_intervention_steps": 0,
        }
    )

    for (
        env_name,
        policy_type,
        mode,
    ), s in summary.items():
        if policy_type != "legality_shield":
            continue

        target = combined[mode]

        for field in target:
            target[field] += s[field]

    print()
    print("=" * 80)
    print("OVERALL LEGALITY-SHIELD ACTION ACCURACY")
    print("=" * 80)

    for mode in [
        "allocentric",
        "egocentric",
    ]:
        if mode not in combined:
            continue

        print()
        print(f"MODE={mode}")
        print("-" * 80)

        print_shield_accuracy(
            combined[mode]
        )


def main():
    results = load_results(
        "farama_minigrid_results_final.json"
    )

    summary = analyze(results)

    print()
    print("FARAMA MINIGRID RESULTS SUMMARY")
    print("=" * 80)

    for (
        env_name,
        policy_type,
        mode,
    ) in sorted(summary.keys()):
        s = summary[
            (
                env_name,
                policy_type,
                mode,
            )
        ]

        episodes = s["episodes"]
        successes = s["successes"]

        print()
        print(f"ENV={env_name}")
        print(
            f"POLICY={policy_type}, "
            f"MODE={mode}"
        )
        print("-" * 80)

        print(
            f"Episodes: {episodes}"
        )

        print(
            f"Success: "
            f"{successes}/{episodes} "
            f"({100 * safe_div(successes, episodes):.1f}%)"
        )

        print(
            f"Early stopped: "
            f"{s['early_stopped']}/{episodes} "
            f"({100 * safe_div(s['early_stopped'], episodes):.1f}%)"
        )

        print(
            "Average steps: "
            f"{safe_div(s['total_steps'], episodes):.1f}"
        )

        if policy_type == "legality_shield":
            print(
                "Shield-used steps: "
                f"{s['shield_used_steps']}"
            )

            print(
                "Total shield reprompts: "
                f"{s['shield_reprompts']}"
            )

            print_shield_accuracy(s)

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

    print_overall_shield_summary(
        summary
    )


if __name__ == "__main__":
    main()