import json
from pathlib import Path

from experiment_utils import normalize_answer
from llm_policy import make_openai_policy_fn
from minigrid_wrapper import MiniGridCardinalWrapper


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


def parse_minigrid_action(text, mode, env):
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


def run_episode(env_name, seed, mode, policy_fn, max_steps, verbose=False):
    env = MiniGridCardinalWrapper(env_name=env_name, seed=seed)
    env.reset(seed=seed)

    step_logs = []
    reached_goal = False

    for step in range(max_steps):
        prompt = get_prompt(env, mode)
        optimal_actions = env.get_optimal_actions()
        optimal_action_names = env.get_optimal_action_names()
        optimal_relative_action_names = env.get_optimal_relative_action_names()
        grid_text = env.render_text()
        state_before = env.get_state()

        raw_answer = policy_fn(prompt)
        parsed_action = parse_minigrid_action(raw_answer, mode, env)

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
            "next_agent_pos": next_state["agent"],
            "next_facing": next_state["facing"],
            "next_facing_name": next_state["facing_name"],
            "grid_text": grid_text,
            "prompt": prompt,
        })

        if verbose:
            print("\n" + "=" * 80)
            print(f"Seed={seed}, mode={mode}, step={step}")
            print(grid_text)
            print("Raw answer:", raw_answer)
            print("Parsed action:", parsed_action, info["action_name"])
            print("Optimal allocentric actions:", optimal_action_names)
            print("Optimal egocentric actions:", optimal_relative_action_names)
            print("Correct:", is_correct)
            print("Reward:", reward)
            print("Info:", info)

        if terminated:
            reached_goal = True
            break

    final_state = env.get_state()
    env.close()

    return {
        "env_name": env_name,
        "mode": mode,
        "seed": seed,
        "num_steps": len(step_logs),
        "reached_goal": reached_goal,
        "final_state": final_state["agent"],
        "final_facing": final_state["facing"],
        "final_facing_name": final_state["facing_name"],
        "max_steps": max_steps,
        "step_logs": step_logs,
    }


def main():
    env_name = "MiniGrid-SimpleCrossingS9N1-v0"
    seeds = list(range(10))
    modes = ["allocentric", "egocentric"]

    # SimpleCrossingS9N1 is 9x9, so this matches the GridWorld rule.
    max_steps = 9 ** 2

    model = "gpt-4o-mini"
    temperature = 0.0
    max_output_tokens = 16
    output_path = Path("minigrid_results.json")
    metadata_path = Path("minigrid_results_metadata.json")

    policy_fn = make_openai_policy_fn(
        model=model,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )

    all_results = []

    print("\nRUNNING MINIGRID EXPERIMENT")
    print("=" * 80)
    print("Environment:", env_name)
    print("Seeds:", seeds)
    print("Modes:", modes)
    print("Max steps:", max_steps)

    for seed in seeds:
        print("\n" + "-" * 80)
        print(f"Seed {seed}")

        for mode in modes:
            print(f"Running mode: {mode}")

            result = run_episode(
                env_name=env_name,
                seed=seed,
                mode=mode,
                policy_fn=policy_fn,
                max_steps=max_steps,
                verbose=False,
            )

            result["model"] = model
            result["temperature"] = temperature
            result["max_output_tokens"] = max_output_tokens

            all_results.append(result)

            print(
                f"  reached_goal={result['reached_goal']}, "
                f"steps={result['num_steps']}, "
                f"final_state={result['final_state']}"
            )

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    metadata = {
        "env_name": env_name,
        "seeds": seeds,
        "num_seeds": len(seeds),
        "modes": modes,
        "max_steps": max_steps,
        "max_steps_rule": "grid_size_squared",
        "model": model,
        "temperature": temperature,
        "max_output_tokens": max_output_tokens,
        "output_file": str(output_path),
    }

    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("\nSaved:")
    print("-", output_path)
    print("-", metadata_path)


if __name__ == "__main__":
    main()