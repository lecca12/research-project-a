import json
import re
from pathlib import Path
from collections import defaultdict

from experiment_utils import normalize_answer
from llm_policy import make_openai_policy_fn
from minigrid_wrapper import (
    MiniGridCardinalWrapper,
    ACTION_TO_DELTA,
    ACTION_NAMES,
    RELATIVE_ACTION_NAMES,
)


ABSOLUTE_ACTIONS = {"north": 0, "east": 1, "south": 2, "west": 3}
RELATIVE_ACTIONS = {"forward": 0, "right": 1, "backward": 2, "left": 3}


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


def extract_final_action_word(text, mode):
    if text is None:
        return None

    valid_words = ABSOLUTE_ACTIONS if mode == "allocentric" else RELATIVE_ACTIONS
    lowered = text.strip().lower()

    patterns = [
        r"final action\s*:\s*([a-z]+)",
        r"final answer\s*:\s*([a-z]+)",
        r"answer\s*:\s*([a-z]+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, lowered)
        if match:
            word = normalize_answer(match.group(1))
            if word in valid_words:
                return word

    words = re.findall(r"[a-z]+", lowered)
    for word in reversed(words):
        if word in valid_words:
            return word

    return None


def parse_reasoning_action(text, mode, env):
    word = extract_final_action_word(text, mode)

    if word is None:
        return None

    if mode == "allocentric":
        return ABSOLUTE_ACTIONS.get(word)

    rel_action = RELATIVE_ACTIONS.get(word)
    if rel_action is None:
        return None

    return env.relative_to_cardinal(rel_action)


def get_prompt(env, mode):
    if mode == "allocentric":
        return env.make_allocentric_description()
    if mode == "egocentric":
        return env.make_egocentric_description()
    raise ValueError(f"Unknown mode: {mode}")


def make_reasoning_prompt(base_prompt, mode):
    if mode == "allocentric":
        action_list = ["north", "east", "south", "west"]
        final_options = "north, east, south, or west"
    else:
        action_list = ["forward", "right", "backward", "left"]
        final_options = "forward, right, backward, or left"

    candidate_lines = "\n".join(
        f"- {action}: resulting cell = ?, legal/illegal = ?, shortest-path progress = ?"
        for action in action_list
    )

    return f"""{base_prompt}

Now use candidate-evaluation reasoning before choosing.

Evaluate every candidate action explicitly:

{candidate_lines}

For each action:
1. Work out the resulting cell.
2. State whether the action is legal or illegal.
3. State whether it appears to move along a shortest valid path toward the goal.

Then choose the best legal action.

After the reasoning, give your final answer on a new line using exactly this format:
Final action: <one of {final_options}>
"""


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


def get_legal_cardinal_actions(env):
    state = env.get_state()
    row, col = state["agent"]

    legal = set()
    for action, (dr, dc) in ACTION_TO_DELTA.items():
        blocked, _ = env.is_blocked(row + dr, col + dc)
        if not blocked:
            legal.add(action)

    return legal


def cardinal_to_relative(action, facing):
    # MiniGrid facing: 0=east, 1=south, 2=west, 3=north
    # Project cardinal: 0=north, 1=east, 2=south, 3=west
    facing_to_cardinal = {0: 1, 1: 2, 2: 3, 3: 0}
    cardinal_facing = facing_to_cardinal[facing]
    return (action - cardinal_facing) % 4


def legal_action_names(env, mode):
    legal = get_legal_cardinal_actions(env)

    if mode == "allocentric":
        return [ACTION_NAMES[a] for a in sorted(legal)]

    facing = env.get_state()["facing"]
    names = []
    for action in sorted(legal):
        rel = cardinal_to_relative(action, facing)
        names.append(RELATIVE_ACTION_NAMES[rel])

    return names


def make_legality_reprompt(original_prompt, raw_answer, legal_names):
    legal_text = ", ".join(legal_names)

    return f"""{original_prompt}

Your previous answer was: {raw_answer}

That action is illegal from the current state because it would move into a wall, obstacle, or outside the grid.

Choose a legal action instead.

Legal actions available now:
{legal_text}

Answer with one word only from: {legal_text}
"""


def choose_action(
    env,
    mode,
    policy_type,
    prompt,
    short_policy_fn,
    reasoning_policy_fn,
    max_reprompts=2,
):
    if policy_type == "baseline":
        raw_answer = short_policy_fn(prompt)
        parsed_action = parse_action(raw_answer, mode, env)

        return parsed_action, {
            "policy_type": policy_type,
            "raw_model_answer": raw_answer,
            "all_raw_answers": [raw_answer],
            "shield_used": False,
            "shield_reprompts": 0,
            "reasoning_used": False,
            "reasoning_prompt": None,
            "legal_action_names": legal_action_names(env, mode),
        }

    if policy_type == "reasoning":
        reasoning_prompt = make_reasoning_prompt(prompt, mode)
        raw_answer = reasoning_policy_fn(reasoning_prompt)
        parsed_action = parse_reasoning_action(raw_answer, mode, env)

        return parsed_action, {
            "policy_type": policy_type,
            "raw_model_answer": raw_answer,
            "all_raw_answers": [raw_answer],
            "shield_used": False,
            "shield_reprompts": 0,
            "reasoning_used": True,
            "reasoning_prompt": reasoning_prompt,
            "legal_action_names": legal_action_names(env, mode),
        }

    if policy_type == "legality_shield":
        legal_actions = get_legal_cardinal_actions(env)
        legal_names = legal_action_names(env, mode)

        all_raw_answers = []
        current_prompt = prompt

        for attempt in range(max_reprompts + 1):
            raw_answer = short_policy_fn(current_prompt)
            all_raw_answers.append(raw_answer)

            parsed_action = parse_action(raw_answer, mode, env)

            if parsed_action is not None and parsed_action in legal_actions:
                return parsed_action, {
                    "policy_type": policy_type,
                    "raw_model_answer": raw_answer,
                    "all_raw_answers": all_raw_answers,
                    "shield_used": attempt > 0,
                    "shield_reprompts": attempt,
                    "reasoning_used": False,
                    "reasoning_prompt": None,
                    "legal_action_names": legal_names,
                }

            current_prompt = make_legality_reprompt(
                original_prompt=prompt,
                raw_answer=raw_answer,
                legal_names=legal_names,
            )

        return parsed_action, {
            "policy_type": policy_type,
            "raw_model_answer": all_raw_answers[-1] if all_raw_answers else None,
            "all_raw_answers": all_raw_answers,
            "shield_used": True,
            "shield_reprompts": max_reprompts,
            "reasoning_used": False,
            "reasoning_prompt": None,
            "legal_action_names": legal_names,
        }

    raise ValueError(f"Unknown policy_type: {policy_type}")


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


def save_episode_trace(result, output_dir="farama_minigrid_traces"):
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    safe_env_name = result["env_name"].replace("/", "_").replace("-", "_")
    filename = (
        f"{safe_env_name}_"
        f"{result['policy_type']}_"
        f"{result['mode']}_"
        f"seed{result['seed']}.txt"
    )

    path = output_path / filename

    with path.open("w", encoding="utf-8") as f:
        f.write(f"Environment: {result['env_name']}\n")
        f.write(f"Policy: {result['policy_type']}\n")
        f.write(f"Mode: {result['mode']}\n")
        f.write(f"Seed: {result['seed']}\n")
        f.write(f"Reached goal: {result['reached_goal']}\n")
        f.write(f"Early stopped: {result['early_stopped']}\n")
        f.write(f"Steps: {result['num_steps']}\n")
        f.write("=" * 80 + "\n\n")

        for step in result["step_logs"]:
            f.write(f"STEP {step['step']}\n")
            f.write("-" * 80 + "\n")
            f.write(step["grid_text"] + "\n\n")
            f.write(f"Raw answer: {step.get('raw_model_answer')}\n")
            f.write(f"All raw answers: {step.get('all_raw_answers')}\n")
            f.write(f"Parsed action: {step.get('parsed_action_name')}\n")
            f.write(f"Parsed action index: {step.get('parsed_action')}\n")
            f.write(f"Optimal actions: {step.get('optimal_action_names')}\n")
            f.write(f"Optimal relative actions: {step.get('optimal_relative_action_names')}\n")
            f.write(f"Legal actions: {step.get('legal_action_names')}\n")
            f.write(f"Correct: {step.get('is_correct')}\n")
            f.write(f"Hit wall/boundary: {step.get('hit_wall')}\n")
            f.write(f"Hit obstacle/interior wall: {step.get('hit_obstacle')}\n")
            f.write(f"Blocked type: {step.get('blocked_type')}\n")
            f.write(f"Error type: {step.get('error_type')}\n")
            f.write(f"Shield used: {step.get('shield_used')}\n")
            f.write(f"Shield reprompts: {step.get('shield_reprompts')}\n")
            f.write(f"Reasoning used: {step.get('reasoning_used')}\n")
            f.write(f"Terminated: {step.get('terminated')}\n")
            f.write(f"Truncated: {step.get('truncated')}\n")
            f.write(f"Early stopped: {step.get('early_stopped')}\n")
            f.write(f"Early stop repeat key: {step.get('early_stop_repeat_key')}\n")
            f.write("\n" + "=" * 80 + "\n\n")


def run_episode(
    env_name,
    seed,
    mode,
    policy_type,
    short_policy_fn,
    reasoning_policy_fn,
    max_steps,
    early_stop_repeats=3,
    verbose=False,
):
    env = MiniGridCardinalWrapper(env_name=env_name, seed=seed)
    env.reset(seed=seed)

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

        parsed_action, action_meta = choose_action(
            env=env,
            mode=mode,
            policy_type=policy_type,
            prompt=prompt,
            short_policy_fn=short_policy_fn,
            reasoning_policy_fn=reasoning_policy_fn,
        )

        raw_answer = action_meta["raw_model_answer"]
        parse_failure = parsed_action is None
        is_correct = (not parse_failure) and parsed_action in optimal_actions

        if parse_failure:
            step_logs.append({
                "step": step,
                "policy_type": policy_type,
                "agent_pos": state_before["agent"],
                "goal_pos": state_before["goal"],
                "facing": state_before["facing"],
                "facing_name": state_before["facing_name"],
                "raw_model_answer": raw_answer,
                "all_raw_answers": action_meta["all_raw_answers"],
                "parsed_action": None,
                "optimal_actions": sorted(optimal_actions),
                "optimal_action_names": optimal_action_names,
                "optimal_relative_action_names": optimal_relative_action_names,
                "legal_action_names": action_meta["legal_action_names"],
                "shield_used": action_meta["shield_used"],
                "shield_reprompts": action_meta["shield_reprompts"],
                "reasoning_used": action_meta["reasoning_used"],
                "reasoning_prompt": action_meta["reasoning_prompt"],
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
                "early_stop_repeat_key": None,
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
            "policy_type": policy_type,
            "agent_pos": state_before["agent"],
            "goal_pos": state_before["goal"],
            "facing": state_before["facing"],
            "facing_name": state_before["facing_name"],
            "raw_model_answer": raw_answer,
            "all_raw_answers": action_meta["all_raw_answers"],
            "parsed_action": parsed_action,
            "parsed_action_name": info["action_name"],
            "optimal_actions": sorted(optimal_actions),
            "optimal_action_names": optimal_action_names,
            "optimal_relative_action_names": optimal_relative_action_names,
            "legal_action_names": action_meta["legal_action_names"],
            "shield_used": action_meta["shield_used"],
            "shield_reprompts": action_meta["shield_reprompts"],
            "reasoning_used": action_meta["reasoning_used"],
            "reasoning_prompt": action_meta["reasoning_prompt"],
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
            print(f"Env={env_name}, policy={policy_type}, seed={seed}, mode={mode}, step={step}")
            print(grid_text)
            print("Raw answer:", raw_answer)
            print("Parsed action:", parsed_action, info["action_name"])
            print("Legal actions:", action_meta["legal_action_names"])
            print("Optimal allocentric:", optimal_action_names)
            print("Optimal egocentric:", optimal_relative_action_names)
            print("Correct:", is_correct)
            print("Error:", error_type)
            print("Reward:", reward)
            print("Info:", info)
            print("Early stop:", stop_now)

        if terminated:
            reached_goal = True
            break

        if stop_now:
            break

    final_state = env.get_state()
    env.close()

    return {
        "env_name": env_name,
        "policy_type": policy_type,
        "mode": mode,
        "seed": seed,
        "num_steps": len(step_logs),
        "reached_goal": reached_goal,
        "early_stopped": early_stopped,
        "early_stop_repeats": early_stop_repeats,
        "final_state": final_state["agent"],
        "final_facing": final_state["facing"],
        "final_facing_name": final_state["facing_name"],
        "max_steps": max_steps,
        "step_logs": step_logs,
    }


def main():
    env_names = [
        "MiniGrid-SimpleCrossingS9N1-v0",
        "MiniGrid-SimpleCrossingS9N2-v0",
        "MiniGrid-SimpleCrossingS9N3-v0",
        "MiniGrid-FourRooms-v0",
    ]

    seeds = list(range(10))
    modes = ["allocentric", "egocentric"]

    # Use this for a quick meeting run:
    policy_types = ["baseline"]

    # Later, switch to:
    # policy_types = ["baseline", "legality_shield", "reasoning"]

    max_steps_by_env = {
        "MiniGrid-SimpleCrossingS9N1-v0": 9 ** 2,
        "MiniGrid-SimpleCrossingS9N2-v0": 9 ** 2,
        "MiniGrid-SimpleCrossingS9N3-v0": 9 ** 2,
        "MiniGrid-FourRooms-v0": 19 ** 2,
    }

    early_stop_repeats = 3
    save_traces = True

    model = "gpt-4o-mini"
    temperature = 0.0

    output_path = Path("farama_minigrid_results.json")
    metadata_path = Path("farama_minigrid_results_metadata.json")

    short_policy_fn = make_openai_policy_fn(
        model=model,
        temperature=temperature,
        max_output_tokens=16,
    )

    reasoning_policy_fn = make_openai_policy_fn(
        model=model,
        temperature=temperature,
        max_output_tokens=256,
    )

    all_results = []

    print("\nRUNNING OFFICIAL FARAMA MINIGRID EXPERIMENT")
    print("=" * 80)
    print("Environments:", env_names)
    print("Seeds:", seeds)
    print("Modes:", modes)
    print("Policy types:", policy_types)
    print("Early stop repeats:", early_stop_repeats)
    print("Save traces:", save_traces)

    for env_name in env_names:
        max_steps = max_steps_by_env[env_name]

        print("\n" + "=" * 80)
        print(f"ENVIRONMENT: {env_name}, max_steps={max_steps}")
        print("=" * 80)

        for seed in seeds:
            print("\n" + "-" * 80)
            print(f"Seed {seed}")

            for policy_type in policy_types:
                for mode in modes:
                    print(f"Running policy={policy_type}, mode={mode}")

                    result = run_episode(
                        env_name=env_name,
                        seed=seed,
                        mode=mode,
                        policy_type=policy_type,
                        short_policy_fn=short_policy_fn,
                        reasoning_policy_fn=reasoning_policy_fn,
                        max_steps=max_steps,
                        early_stop_repeats=early_stop_repeats,
                        verbose=False,
                    )

                    result["model"] = model
                    result["temperature"] = temperature
                    result["short_max_output_tokens"] = 16
                    result["reasoning_max_output_tokens"] = 256

                    all_results.append(result)

                    if save_traces:
                        save_episode_trace(result)

                    print(
                        f"  reached_goal={result['reached_goal']}, "
                        f"early_stopped={result['early_stopped']}, "
                        f"steps={result['num_steps']}, "
                        f"final_state={result['final_state']}"
                    )

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(make_json_safe(all_results), f, indent=2)

    metadata = {
        "env_names": env_names,
        "seeds": seeds,
        "num_seeds": len(seeds),
        "modes": modes,
        "policy_types": policy_types,
        "max_steps_by_env": max_steps_by_env,
        "early_stop_repeats": early_stop_repeats,
        "save_traces": save_traces,
        "max_steps_rule": "environment_specific_grid_size_squared",
        "model": model,
        "temperature": temperature,
        "short_max_output_tokens": 16,
        "reasoning_max_output_tokens": 256,
        "output_file": str(output_path),
    }

    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(make_json_safe(metadata), f, indent=2)

    print("\nSaved:")
    print("-", output_path)
    print("-", metadata_path)
    if save_traces:
        print("- farama_minigrid_traces/")


if __name__ == "__main__":
    main()