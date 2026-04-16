import argparse
import json
from pathlib import Path


FACING_TO_NAME = {
    0: "north",
    1: "east",
    2: "south",
    3: "west",
}


def load_results(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        return [data]
    return data


def classify_step_error(step):
    if step.get("parse_failure", False) or not step.get("is_valid_format", True):
        return "parse_failure"
    if step.get("hit_obstacle", False):
        return "obstacle_blindness"
    if step.get("hit_wall", False):
        return "boundary_error"
    if not step.get("is_correct", False):
        return "directional_or_detour_error"
    return "correct"


def show_step(episode_index, episode, step):
    print("\n" + "=" * 100)
    print(
        f"Episode {episode_index} | Mode={episode.get('mode')} | Seed={episode.get('seed')} | "
        f"Step={step.get('step')} | Error={classify_step_error(step)}"
    )
    print(f"Agent={step.get('agent_pos')} Goal={step.get('goal_pos')} Facing={step.get('facing_name', FACING_TO_NAME.get(step.get('facing', 0), 'unknown'))}")
    print(step.get("grid_text_before", "[no grid text logged]"))
    print()
    print("Raw model answer:", repr(step.get("raw_model_answer")))
    print("Parsed action:", step.get("parsed_action_name"))
    print("Correct answer text:", step.get("correct_answer_text"))
    print("Optimal absolute actions:", step.get("optimal_action_names", []))
    print("Valid format:", step.get("is_valid_format"))
    print("Is correct:", step.get("is_correct"))
    print("Hit wall:", step.get("hit_wall"))
    print("Hit obstacle:", step.get("hit_obstacle"))
    print("=" * 100)


def main():
    parser = argparse.ArgumentParser(description="Replay steps from a results JSON file.")
    parser.add_argument("results_path", help="Path to results JSON, e.g. main_results.json")
    parser.add_argument("--mode", choices=["allocentric", "egocentric"], default=None)
    parser.add_argument(
        "--error-type",
        choices=["parse_failure", "obstacle_blindness", "boundary_error", "directional_or_detour_error", "correct"],
        default=None,
    )
    parser.add_argument("--episode-index", type=int, default=None)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--show-correct", action="store_true")
    args = parser.parse_args()

    results = load_results(Path(args.results_path))
    shown = 0

    for episode_index, episode in enumerate(results):
        if args.episode_index is not None and episode_index != args.episode_index:
            continue
        if args.mode is not None and episode.get("mode") != args.mode:
            continue

        for step in episode.get("step_logs", []):
            error_type = classify_step_error(step)

            if not args.show_correct and error_type == "correct":
                continue
            if args.error_type is not None and error_type != args.error_type:
                continue

            show_step(episode_index, episode, step)
            shown += 1

            if shown >= args.limit:
                print(f"\nStopped after showing {shown} steps.")
                return

    print(f"\nFinished. Displayed {shown} matching steps.")


if __name__ == "__main__":
    main()
