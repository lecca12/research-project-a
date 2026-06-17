import json
from pathlib import Path
from collections import defaultdict

from custom_minigrid_wrapper import (
    CustomMiniGridWrapper,
    generate_custom_minigrid_states,
)
from experiment_utils import normalize_answer
from llm_policy import make_openai_policy_fn


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


def make_json_safe(obj):
    if hasattr(obj, "item"):
        return obj.item()
    if isinstance(obj, tuple):
        return [make_json_safe(x) for x in obj]
    if isinstance(obj, list):
        return [make_json_safe(x) for x in obj]
    if isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}
    return obj


def parse_action(text, mode, env):
    word = normalize_answer(text)

    if mode == "allocentric":
        return ABSOLUTE_ACTIONS.get(word)

    if mode == "egocentric":
        rel_action = RELATIVE_ACTIONS.get(word)
        if rel_action is None:
            return None
        return env.relative_to_cardinal(rel_action)

    raise ValueError(f"Unknown mode: {mode}")


def get_prompt(env, mode):
    if mode == "allocentric":
        return env.make_allocentric_description()
    if mode == "egocentric":
        return env.make_egocentric_description()
    raise ValueError(f"Unknown mode: {mode}")


def classify_error(parse_failure, hit_wall, hit_obstacle, is_correct):
    if parse_failure:
        return "parse_failure"
    if hit_obstacle:
        return "obstacle_blindness"
    if hit_wall:
        return "boundary_error"
    if not is_correct:
        return "directional_or_detour_error"
    return "correct"


def should_early_stop(repeat_counts, state_before, parsed_action, threshold):
    if threshold is None:
        return False, None

    key = (
        tuple(state_before["agent"]),
        int(state_before["facing"]),
        int(parsed_action) if parsed_action is not None else None,
    )

    repeat_counts[key] += 1

    if repeat_counts[key] >= threshold:
        return True, key

    return False, key


def print_generated_states(fixed_states):
    print("\nGENERATED STATES")
    print("-" * 80)

    for i, state in enumerate(fixed_states):
        preview_env = CustomMiniGridWrapper(
            size=state["size"],
            obstacle_density=state["obstacle_density"],
            max_steps=state["max_steps"],
        )
        preview_env.reset(state=state)

        print(f"\nState {i + 1} (seed={state['seed']})")
        print(preview_env.render_text())
        print("Agent:", preview_env.get_state()["agent"])
        print("Goal:", preview_env.get_state()["goal"])
        print("Facing:", preview_env.get_state()["facing_name"])
        print("Obstacles:", preview_env.get_state()["obstacle_cells"])
        print("Optimal allocentric:", preview_env.get_optimal_action_names())
        print("Optimal egocentric:", preview_env.get_optimal_relative_action_names())


def run_episode(
    fixed_state,
    mode,
    policy_fn,
    max_steps,
    early_stop_repeats=3,
    verbose=False,
):
    env = CustomMiniGridWrapper(
        size=fixed_state["size"],
        obstacle_density=fixed_state["obstacle_density"],
        max_steps=max_steps,
    )
    env.reset(state=fixed_state)

    step_logs = []
    reached_goal = False
    early_stopped = False
    repeat_counts = defaultdict(int)

    for step in range(max_steps):
        prompt = get_prompt(env, mode)
        optimal_actions = env.get_optimal_actions()
        optimal_action_names = env.get_optimal_action_names()
        optimal_relative_action_names = env.get_optimal_relative_action_names()
        grid_text = env.render_text()
        state_before = env.get_state()

        raw_answer = policy_fn(prompt)
        parsed_action = parse_action(raw_answer, mode, env)

        parse_failure = parsed_action is None
        is_correct = (not parse_failure) and parsed_action in optimal_actions

        if parse_failure:
            step_logs.append({
                "step": step,
                "agent_pos": state_before["agent"],
                "goal_pos": state_before["goal"],
                "facing": state_before["facing"],
                "facing_name": state_before["facing_name"],
                "raw_model_answer": raw_answer,
                "parsed_action": None,
                "optimal_actions": sorted(optimal_actions),
                "optimal_action_names": optimal_action_names,
                "optimal_relative_action_names": optimal_relative_action_names,
                "is_valid_format": False,
                "is_correct": False,
                "parse_failure": True,
                "error_type": "parse_failure",
                "reward": None,
                "hit_wall": False,
                "hit_obstacle": False,
                "terminated": False,
                "truncated": False,
                "early_stopped": False,
                "grid_text": grid_text,
                "prompt": prompt,
            })
            break

        next_state, reward, terminated, truncated, info = env.step_cardinal(parsed_action)

        error_type = classify_error(
            parse_failure=False,
            hit_wall=info["hit_wall"],
            hit_obstacle=info["hit_obstacle"],
            is_correct=is_correct,
        )

        stop_now, repeat_key = should_early_stop(
            repeat_counts=repeat_counts,
            state_before=state_before,
            parsed_action=parsed_action,
            threshold=early_stop_repeats,
        )

        if stop_now and not terminated:
            early_stopped = True
            truncated = True

        step_logs.append({
            "step": step,
            "agent_pos": state_before["agent"],
            "goal_pos": state_before["goal"],
            "facing": state_before["facing"],
            "facing_name": state_before["facing_name"],
            "raw_model_answer": raw_answer,
            "parsed_action": parsed_action,
            "parsed_action_name": info["action_name"],
            "optimal_actions": sorted(optimal_actions),
            "optimal_action_names": optimal_action_names,
            "optimal_relative_action_names": optimal_relative_action_names,
            "is_valid_format": True,
            "is_correct": is_correct,
            "parse_failure": False,
            "error_type": error_type,
            "reward": reward,
            "hit_wall": info["hit_wall"],
            "hit_obstacle": info["hit_obstacle"],
            "blocked_type": info["blocked_type"],
            "terminated": terminated,
            "truncated": truncated,
            "early_stopped": early_stopped,
            "early_stop_repeat_key": repeat_key,
            "next_agent_pos": next_state["agent"],
            "next_facing": next_state["facing"],
            "next_facing_name": next_state["facing_name"],
            "grid_text": grid_text,
            "prompt": prompt,
        })

        if verbose:
            print("\n" + "=" * 80)
            print(f"Mode={mode}, step={step}")
            print(grid_text)
            print("Raw answer:", raw_answer)
            print("Parsed:", parsed_action, info["action_name"])
            print("Optimal allocentric:", optimal_action_names)
            print("Optimal egocentric:", optimal_relative_action_names)
            print("Correct:", is_correct)
            print("Error:", error_type)
            print("Early stop:", stop_now)

        if terminated:
            reached_goal = True
            break

        if stop_now:
            break

    final_state = env.get_state()

    return {
        "mode": mode,
        "seed": fixed_state.get("seed"),
        "grid_size": fixed_state["size"],
        "obstacle_density": fixed_state["obstacle_density"],
        "num_steps": len(step_logs),
        "reached_goal": reached_goal,
        "early_stopped": early_stopped,
        "final_state": final_state["agent"],
        "final_facing": final_state["facing"],
        "final_facing_name": final_state["facing_name"],
        "max_steps": max_steps,
        "early_stop_repeats": early_stop_repeats,
        "step_logs": step_logs,
    }


def main():
    conditions = [
        {"size": 5, "obstacle_density": 0.10},
        {"size": 5, "obstacle_density": 0.30},
        {"size": 8, "obstacle_density": 0.10},
        {"size": 8, "obstacle_density": 0.30},
    ]

    num_states = 10
    start_seed = 0
    modes = ["allocentric", "egocentric"]

    # Diagnostic speed-up. Set to None for final no-shortcut runs.
    early_stop_repeats = 3

    # Set True first to inspect maps before spending API calls.
    preview_only = False
    print_generated_grids = True

    model = "gpt-4o-mini"
    temperature = 0.0
    max_output_tokens = 16

    output_path = Path("custom_minigrid_results.json")
    metadata_path = Path("custom_minigrid_results_metadata.json")

    all_results = []

    print("\nRUNNING CUSTOM MINIGRID EXPERIMENT")
    print("=" * 80)
    print("Conditions:", conditions)
    print("States per condition:", num_states)
    print("Modes:", modes)
    print("Early stop repeats:", early_stop_repeats)
    print("Preview only:", preview_only)
    print("Print generated grids:", print_generated_grids)

    if not preview_only:
        policy_fn = make_openai_policy_fn(
            model=model,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )
    else:
        policy_fn = None

    for condition in conditions:
        size = condition["size"]
        density = condition["obstacle_density"]
        max_steps = size ** 2

        print("\n" + "=" * 80)
        print(f"Condition: {size}x{size}, density={density}, max_steps={max_steps}")
        print("=" * 80)

        fixed_states = generate_custom_minigrid_states(
            size=size,
            obstacle_density=density,
            num_states=num_states,
            start_seed=start_seed,
        )

        if print_generated_grids:
            print_generated_states(fixed_states)

        if preview_only:
            continue

        for state_index, state in enumerate(fixed_states):
            seed = state["seed"]
            print(f"\nState {state_index + 1}/{num_states}, seed={seed}")

            for mode in modes:
                print(f"  Running mode: {mode}")

                result = run_episode(
                    fixed_state=state,
                    mode=mode,
                    policy_fn=policy_fn,
                    max_steps=max_steps,
                    early_stop_repeats=early_stop_repeats,
                    verbose=False,
                )

                result["model"] = model
                result["temperature"] = temperature
                result["max_output_tokens"] = max_output_tokens

                all_results.append(result)

                print(
                    f"    reached_goal={result['reached_goal']}, "
                    f"early_stopped={result['early_stopped']}, "
                    f"steps={result['num_steps']}, "
                    f"final_state={result['final_state']}"
                )

    if preview_only:
        print("\nPreview only enabled. No API calls were made and no results were saved.")
        return

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(make_json_safe(all_results), f, indent=2)

    metadata = {
        "conditions": conditions,
        "num_states": num_states,
        "start_seed": start_seed,
        "modes": modes,
        "early_stop_repeats": early_stop_repeats,
        "preview_only": preview_only,
        "print_generated_grids": print_generated_grids,
        "max_steps_rule": "grid_size_squared",
        "model": model,
        "temperature": temperature,
        "max_output_tokens": max_output_tokens,
        "output_file": str(output_path),
    }

    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(make_json_safe(metadata), f, indent=2)

    print("\nSaved:")
    print("-", output_path)
    print("-", metadata_path)


if __name__ == "__main__":
    main()