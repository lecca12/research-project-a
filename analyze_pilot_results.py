import json
import csv
from pathlib import Path
from collections import defaultdict


def safe_div(numerator, denominator):
    if denominator == 0:
        return 0.0
    return numerator / denominator


def load_results(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        return [data]
    if isinstance(data, list):
        return data
    raise ValueError("pilot_results.json must contain a list of episode results or a single result dict.")


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


def init_mode_summary():
    return {
        "episodes": 0,
        "successful_episodes": 0,
        "total_episode_steps": 0,
        "steps": 0,
        "correct_steps": 0,
        "parse_failures": 0,
        "obstacle_hits": 0,
        "wall_hits": 0,
        "terminated_steps": 0,
        "truncated_steps": 0,
        "error_type_counts": defaultdict(int),
    }


def analyze_results(results):
    summaries = defaultdict(init_mode_summary)
    incorrect_rows = []
    episode_rows = []

    for episode_idx, episode in enumerate(results):
        mode = episode.get("mode", "unknown")
        summary = summaries[mode]
        summary["episodes"] += 1

        reached_goal = bool(episode.get("reached_goal", False))
        if reached_goal:
            summary["successful_episodes"] += 1

        num_steps = int(episode.get("num_steps", 0))
        summary["total_episode_steps"] += num_steps

        step_logs = episode.get("step_logs", [])
        seed = episode.get("seed", None)

        episode_rows.append({
            "episode_index": episode_idx,
            "mode": mode,
            "seed": seed,
            "num_steps": num_steps,
            "reached_goal": reached_goal,
        })

        for step in step_logs:
            summary["steps"] += 1

            is_correct = bool(step.get("is_correct", False))
            if is_correct:
                summary["correct_steps"] += 1

            parse_failure = bool(step.get("parse_failure", False)) or not bool(step.get("is_valid_format", True))
            hit_obstacle = bool(step.get("hit_obstacle", False))
            hit_wall = bool(step.get("hit_wall", False))
            terminated = bool(step.get("terminated", False))
            truncated = bool(step.get("truncated", False))

            if parse_failure:
                summary["parse_failures"] += 1
            if hit_obstacle:
                summary["obstacle_hits"] += 1
            if hit_wall:
                summary["wall_hits"] += 1
            if terminated:
                summary["terminated_steps"] += 1
            if truncated:
                summary["truncated_steps"] += 1

            error_type = classify_step_error(step)
            summary["error_type_counts"][error_type] += 1

            if error_type != "correct":
                incorrect_rows.append({
                    "episode_index": episode_idx,
                    "mode": mode,
                    "seed": seed,
                    "step": step.get("step"),
                    "agent_pos": step.get("agent_pos"),
                    "goal_pos": step.get("goal_pos"),
                    "facing": step.get("facing"),
                    "raw_model_answer": step.get("raw_model_answer"),
                    "parsed_action": step.get("parsed_action"),
                    "correct_answer_text": step.get("correct_answer_text"),
                    "optimal_actions": step.get("optimal_actions"),
                    "optimal_action_names": step.get("optimal_action_names"),
                    "is_valid_format": step.get("is_valid_format"),
                    "is_correct": step.get("is_correct"),
                    "hit_wall": step.get("hit_wall"),
                    "hit_obstacle": step.get("hit_obstacle"),
                    "parse_failure": step.get("parse_failure"),
                    "error_type": error_type,
                    "prompt": step.get("prompt"),
                })

    return summaries, episode_rows, incorrect_rows


def print_summary_table(summaries):
    print("\nPILOT SUMMARY")
    print("=" * 100)
    header = (
        f"{'Mode':<14}"
        f"{'Episodes':>10}"
        f"{'Success%':>12}"
        f"{'AvgSteps':>12}"
        f"{'StepAcc%':>12}"
        f"{'ParseFail':>12}"
        f"{'ObsHits':>10}"
        f"{'WallHits':>10}"
    )
    print(header)
    print("-" * 100)

    for mode in sorted(summaries.keys()):
        s = summaries[mode]
        success_rate = 100 * safe_div(s["successful_episodes"], s["episodes"])
        avg_steps = safe_div(s["total_episode_steps"], s["episodes"])
        step_acc = 100 * safe_div(s["correct_steps"], s["steps"])

        row = (
            f"{mode:<14}"
            f"{s['episodes']:>10}"
            f"{success_rate:>12.1f}"
            f"{avg_steps:>12.2f}"
            f"{step_acc:>12.1f}"
            f"{s['parse_failures']:>12}"
            f"{s['obstacle_hits']:>10}"
            f"{s['wall_hits']:>10}"
        )
        print(row)
    print("=" * 100)


def print_error_breakdown(summaries):
    print("\nERROR BREAKDOWN")
    print("=" * 100)
    for mode in sorted(summaries.keys()):
        s = summaries[mode]
        print(f"\nMode: {mode}")
        total_steps = s["steps"]
        for error_type, count in sorted(s["error_type_counts"].items()):
            pct = 100 * safe_div(count, total_steps)
            print(f"  {error_type:<28} {count:>6} ({pct:>5.1f}%)")
    print("=" * 100)


def save_csv(path, rows, fieldnames):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(base_dir, summaries, episode_rows, incorrect_rows):
    summary_rows = []
    error_rows = []

    for mode in sorted(summaries.keys()):
        s = summaries[mode]
        summary_rows.append({
            "mode": mode,
            "episodes": s["episodes"],
            "successful_episodes": s["successful_episodes"],
            "success_rate": safe_div(s["successful_episodes"], s["episodes"]),
            "avg_steps": safe_div(s["total_episode_steps"], s["episodes"]),
            "steps": s["steps"],
            "correct_steps": s["correct_steps"],
            "step_accuracy": safe_div(s["correct_steps"], s["steps"]),
            "parse_failures": s["parse_failures"],
            "obstacle_hits": s["obstacle_hits"],
            "wall_hits": s["wall_hits"],
            "terminated_steps": s["terminated_steps"],
            "truncated_steps": s["truncated_steps"],
        })

        for error_type, count in sorted(s["error_type_counts"].items()):
            error_rows.append({
                "mode": mode,
                "error_type": error_type,
                "count": count,
                "percent_of_steps": safe_div(count, s["steps"]),
            })

    save_csv(
        base_dir / "pilot_episode_summary.csv",
        episode_rows,
        ["episode_index", "mode", "seed", "num_steps", "reached_goal"],
    )

    save_csv(
        base_dir / "pilot_mode_summary.csv",
        summary_rows,
        [
            "mode",
            "episodes",
            "successful_episodes",
            "success_rate",
            "avg_steps",
            "steps",
            "correct_steps",
            "step_accuracy",
            "parse_failures",
            "obstacle_hits",
            "wall_hits",
            "terminated_steps",
            "truncated_steps",
        ],
    )

    save_csv(
        base_dir / "pilot_error_breakdown.csv",
        error_rows,
        ["mode", "error_type", "count", "percent_of_steps"],
    )

    if incorrect_rows:
        save_csv(
            base_dir / "pilot_incorrect_steps.csv",
            incorrect_rows,
            [
                "episode_index",
                "mode",
                "seed",
                "step",
                "agent_pos",
                "goal_pos",
                "facing",
                "raw_model_answer",
                "parsed_action",
                "correct_answer_text",
                "optimal_actions",
                "optimal_action_names",
                "is_valid_format",
                "is_correct",
                "hit_wall",
                "hit_obstacle",
                "parse_failure",
                "error_type",
                "prompt",
            ],
        )


def main():
    input_path = Path("pilot_results.json")
    if not input_path.exists():
        raise FileNotFoundError(
            "Could not find pilot_results.json in the current folder. "
            "Place this script in the same folder as pilot_results.json and run it there."
        )

    results = load_results(input_path)
    summaries, episode_rows, incorrect_rows = analyze_results(results)

    print_summary_table(summaries)
    print_error_breakdown(summaries)

    write_outputs(Path("."), summaries, episode_rows, incorrect_rows)

    print("\nSaved files:")
    print("- pilot_episode_summary.csv")
    print("- pilot_mode_summary.csv")
    print("- pilot_error_breakdown.csv")
    if incorrect_rows:
        print("- pilot_incorrect_steps.csv")

    print("\nDone.")


if __name__ == "__main__":
    main()
