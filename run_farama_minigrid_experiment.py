import json
import re
from pathlib import Path

from experiment_utils import normalize_answer
from llm_policy import make_openai_policy_fn
from minigrid_wrapper import (
    MiniGridCardinalWrapper,
    ACTION_TO_DELTA,
    ACTION_NAMES,
    RELATIVE_ACTION_NAMES,
)


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


REASONING_SYSTEM_INSTRUCTIONS = (
    "You are a precise navigation assistant for a grid world experiment. "
    "Follow the requested candidate-by-candidate reasoning process carefully. "
    "After completing the reasoning, finish with exactly one final line in "
    "the form 'Answer: <action>', where <action> is one allowed action word."
)


def make_json_safe(obj):
    if hasattr(obj, "item"):
        return obj.item()

    if isinstance(obj, tuple):
        return [make_json_safe(item) for item in obj]

    if isinstance(obj, list):
        return [make_json_safe(item) for item in obj]

    if isinstance(obj, dict):
        return {
            key: make_json_safe(value)
            for key, value in obj.items()
        }

    return obj


def save_results_incrementally(results, output_path):
    """
    Rewrite the current complete result list after every finished episode.

    This prevents a late API failure from losing all completed episodes.
    """
    temporary_path = output_path.with_suffix(".tmp")

    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(
            make_json_safe(results),
            file,
            indent=2,
        )

    temporary_path.replace(output_path)


def parse_action(text, mode, env):
    """
    Parse baseline and legality-shield responses.

    These policies are instructed to return one action word only.
    """
    word = normalize_answer(text)

    if mode == "allocentric":
        return ABSOLUTE_ACTIONS.get(word)

    if mode == "egocentric":
        relative_action = RELATIVE_ACTIONS.get(word)

        if relative_action is None:
            return None

        return env.relative_to_cardinal(relative_action)

    raise ValueError(f"Unknown mode: {mode}")


def extract_final_action_word(text, mode):
    """
    Extract the final action from a reasoning response.

    Important parser guard:
    - If an Answer/Final Answer/Final Action marker exists, only the word
      directly following that marker is considered.
    - If that marked word is invalid, return None.
    - Only use whole-response fallback scanning when no marker exists.
    """
    if text is None:
        return None

    if mode == "allocentric":
        valid_words = set(ABSOLUTE_ACTIONS)
    elif mode == "egocentric":
        valid_words = set(RELATIVE_ACTIONS)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    lowered = text.strip().lower()

    marker_pattern = re.compile(
        r"(?:answer|final\s+answer|final\s+action)\s*:\s*([a-z]+)",
        flags=re.IGNORECASE,
    )

    marker_match = marker_pattern.search(lowered)

    if marker_match is not None:
        marked_word = normalize_answer(marker_match.group(1))

        if marked_word in valid_words:
            return marked_word

        # Example: "Answer: none" must be a parse failure.
        # Do not scan earlier reasoning text for another action.
        return None

    # Fallback is only allowed when there is no answer marker at all.
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

    relative_action = RELATIVE_ACTIONS.get(word)

    if relative_action is None:
        return None

    return env.relative_to_cardinal(relative_action)


def get_prompt(env, mode):
    if mode == "allocentric":
        return env.make_allocentric_description()

    if mode == "egocentric":
        return env.make_egocentric_description()

    raise ValueError(f"Unknown mode: {mode}")


def make_reasoning_prompt(base_prompt, mode):
    if mode == "allocentric":
        action_list = [
            "north",
            "east",
            "south",
            "west",
        ]
        final_options = "north, east, south, or west"

    elif mode == "egocentric":
        action_list = [
            "forward",
            "right",
            "backward",
            "left",
        ]
        final_options = "forward, right, backward, or left"

    else:
        raise ValueError(f"Unknown mode: {mode}")

    candidate_lines = "\n".join(
        (
            f"- {action}: resulting cell = ?, "
            f"legal/illegal = ?, shortest-path progress = ?"
        )
        for action in action_list
    )

    return f"""{base_prompt}

Use candidate-evaluation reasoning before choosing.

Evaluate every candidate action explicitly:

{candidate_lines}

For each candidate action:
1. Work out the resulting cell.
2. State whether the action is legal or illegal.
3. State whether it appears to preserve progress along a shortest valid path
   toward the goal.

Then choose the best legal action.

After the reasoning, give the final answer on a new line using exactly this
format:

Answer: <one of {final_options}>
"""


def classify_error(
    parse_failure,
    hit_wall,
    hit_obstacle,
    is_correct,
):
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

    legal_actions = set()

    for action, (delta_row, delta_col) in ACTION_TO_DELTA.items():
        target_row = row + delta_row
        target_col = col + delta_col

        blocked, _ = env.is_blocked(
            target_row,
            target_col,
        )

        if not blocked:
            legal_actions.add(action)

    return legal_actions


def cardinal_to_relative(action, facing):
    """
    Convert project cardinal action index to an egocentric action index.

    MiniGrid facing:
        0=east, 1=south, 2=west, 3=north

    Project cardinal:
        0=north, 1=east, 2=south, 3=west
    """
    facing_to_cardinal = {
        0: 1,
        1: 2,
        2: 3,
        3: 0,
    }

    cardinal_facing = facing_to_cardinal[int(facing)]
    return (action - cardinal_facing) % 4


def legal_action_names(env, mode):
    legal_actions = get_legal_cardinal_actions(env)

    if mode == "allocentric":
        return [
            ACTION_NAMES[action]
            for action in sorted(legal_actions)
        ]

    if mode == "egocentric":
        facing = env.get_state()["facing"]
        relative_names = []

        for action in sorted(legal_actions):
            relative_action = cardinal_to_relative(
                action,
                facing,
            )
            relative_names.append(
                RELATIVE_ACTION_NAMES[relative_action]
            )

        return relative_names

    raise ValueError(f"Unknown mode: {mode}")


def make_legality_reprompt(
    original_prompt,
    raw_answer,
    legal_names,
):
    legal_text = ", ".join(legal_names)

    return f"""{original_prompt}

Your previous answer was: {raw_answer}

That proposed action is illegal from the current state because it would move
into an obstacle, the outer boundary, or outside the grid.

The illegal action was not executed.

Choose again using only one of the currently legal actions:

{legal_text}

Answer with one word only from:
{legal_text}
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
        parsed_action = parse_action(
            raw_answer,
            mode,
            env,
        )

        return parsed_action, {
            "policy_type": policy_type,
            "raw_model_answer": raw_answer,
            "all_raw_answers": [raw_answer],
            "shield_used": False,
            "shield_reprompts": 0,
            "reasoning_used": False,
            "reasoning_prompt": None,
            "legal_action_names": legal_action_names(
                env,
                mode,
            ),
        }

    if policy_type == "reasoning":
        reasoning_prompt = make_reasoning_prompt(
            prompt,
            mode,
        )

        raw_answer = reasoning_policy_fn(
            reasoning_prompt
        )

        parsed_action = parse_reasoning_action(
            raw_answer,
            mode,
            env,
        )

        return parsed_action, {
            "policy_type": policy_type,
            "raw_model_answer": raw_answer,
            "all_raw_answers": [raw_answer],
            "shield_used": False,
            "shield_reprompts": 0,
            "reasoning_used": True,
            "reasoning_prompt": reasoning_prompt,
            "legal_action_names": legal_action_names(
                env,
                mode,
            ),
        }

    if policy_type == "legality_shield":
        legal_actions = get_legal_cardinal_actions(env)
        legal_names = legal_action_names(
            env,
            mode,
        )

        all_raw_answers = []
        current_prompt = prompt
        last_parsed_action = None

        for attempt in range(max_reprompts + 1):
            raw_answer = short_policy_fn(
                current_prompt
            )
            all_raw_answers.append(raw_answer)

            parsed_action = parse_action(
                raw_answer,
                mode,
                env,
            )
            last_parsed_action = parsed_action

            if (
                parsed_action is not None
                and parsed_action in legal_actions
            ):
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

        return last_parsed_action, {
            "policy_type": policy_type,
            "raw_model_answer": (
                all_raw_answers[-1]
                if all_raw_answers
                else None
            ),
            "all_raw_answers": all_raw_answers,
            "shield_used": True,
            "shield_reprompts": max_reprompts,
            "reasoning_used": False,
            "reasoning_prompt": None,
            "legal_action_names": legal_names,
        }

    raise ValueError(
        f"Unknown policy_type: {policy_type}"
    )


def update_consecutive_repeat_state(
    previous_key,
    previous_streak,
    current_key,
    threshold,
):
    """
    Update the consecutive-repeat streak.

    A repeat only counts when the current state-action key is identical to the
    immediately preceding key.

    Examples:
        A, A, A -> streak reaches 3 and stops
        A, B, A -> streak is reset; does not stop
    """
    if threshold is None:
        return False, current_key, 1

    if current_key == previous_key:
        current_streak = previous_streak + 1
    else:
        current_streak = 1

    should_stop = current_streak >= threshold

    return (
        should_stop,
        current_key,
        current_streak,
    )


def save_episode_trace(
    result,
    output_dir="farama_minigrid_traces_s9n1_sanity",
):
    output_directory = Path(output_dir)
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    safe_environment_name = (
        result["env_name"]
        .replace("/", "_")
        .replace("-", "_")
    )

    filename = (
        f"{safe_environment_name}_"
        f"{result['policy_type']}_"
        f"{result['mode']}_"
        f"seed{result['seed']}.txt"
    )

    output_path = output_directory / filename

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        file.write(
            f"Environment: {result['env_name']}\n"
        )
        file.write(
            f"Policy: {result['policy_type']}\n"
        )
        file.write(
            f"Mode: {result['mode']}\n"
        )
        file.write(
            f"Seed: {result['seed']}\n"
        )
        file.write(
            f"Reached goal: {result['reached_goal']}\n"
        )
        file.write(
            f"Early stopped: {result['early_stopped']}\n"
        )
        file.write(
            f"Steps: {result['num_steps']}\n"
        )
        file.write("=" * 80 + "\n\n")

        for step_log in result["step_logs"]:
            file.write(
                f"STEP {step_log['step']}\n"
            )
            file.write("-" * 80 + "\n")
            file.write(
                step_log["grid_text"] + "\n\n"
            )
            file.write(
                "Raw answer: "
                f"{step_log.get('raw_model_answer')}\n"
            )
            file.write(
                "All raw answers: "
                f"{step_log.get('all_raw_answers')}\n"
            )
            file.write(
                "Parsed action: "
                f"{step_log.get('parsed_action_name')}\n"
            )
            file.write(
                "Parsed action index: "
                f"{step_log.get('parsed_action')}\n"
            )
            file.write(
                "Optimal actions: "
                f"{step_log.get('optimal_action_names')}\n"
            )
            file.write(
                "Optimal relative actions: "
                f"{step_log.get('optimal_relative_action_names')}\n"
            )
            file.write(
                "Legal actions: "
                f"{step_log.get('legal_action_names')}\n"
            )
            file.write(
                f"Correct: {step_log.get('is_correct')}\n"
            )
            file.write(
                f"Hit boundary: {step_log.get('hit_wall')}\n"
            )
            file.write(
                "Hit obstacle/interior wall: "
                f"{step_log.get('hit_obstacle')}\n"
            )
            file.write(
                "Blocked type: "
                f"{step_log.get('blocked_type')}\n"
            )
            file.write(
                "Error type: "
                f"{step_log.get('error_type')}\n"
            )
            file.write(
                "Shield used: "
                f"{step_log.get('shield_used')}\n"
            )
            file.write(
                "Shield reprompts: "
                f"{step_log.get('shield_reprompts')}\n"
            )
            file.write(
                "Reasoning used: "
                f"{step_log.get('reasoning_used')}\n"
            )
            file.write(
                "Consecutive repeat streak: "
                f"{step_log.get('consecutive_repeat_streak')}\n"
            )
            file.write(
                "Early stop key: "
                f"{step_log.get('early_stop_repeat_key')}\n"
            )
            file.write(
                "Terminated: "
                f"{step_log.get('terminated')}\n"
            )
            file.write(
                "Truncated: "
                f"{step_log.get('truncated')}\n"
            )
            file.write(
                "Early stopped: "
                f"{step_log.get('early_stopped')}\n"
            )
            file.write(
                "\n" + "=" * 80 + "\n\n"
            )


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
    env = MiniGridCardinalWrapper(
        env_name=env_name,
        seed=seed,
    )
    env.reset(seed=seed)

    step_logs = []
    reached_goal = False
    early_stopped = False

    previous_repeat_key = None
    consecutive_repeat_streak = 0

    for step in range(max_steps):
        prompt = get_prompt(env, mode)

        optimal_actions = env.get_optimal_actions()
        optimal_action_names = (
            env.get_optimal_action_names()
        )
        optimal_relative_action_names = (
            env.get_optimal_relative_action_names()
        )

        grid_text = env.render_text()
        state_before = env.get_state()

        parsed_action, action_metadata = choose_action(
            env=env,
            mode=mode,
            policy_type=policy_type,
            prompt=prompt,
            short_policy_fn=short_policy_fn,
            reasoning_policy_fn=reasoning_policy_fn,
        )

        raw_answer = action_metadata[
            "raw_model_answer"
        ]

        parse_failure = parsed_action is None
        is_correct = (
            not parse_failure
            and parsed_action in optimal_actions
        )

        if parse_failure:
            step_logs.append({
                "step": step,
                "policy_type": policy_type,
                "agent_pos": state_before["agent"],
                "goal_pos": state_before["goal"],
                "facing": state_before["facing"],
                "facing_name": state_before[
                    "facing_name"
                ],
                "raw_model_answer": raw_answer,
                "all_raw_answers": action_metadata[
                    "all_raw_answers"
                ],
                "parsed_action": None,
                "parsed_action_name": None,
                "optimal_actions": sorted(
                    optimal_actions
                ),
                "optimal_action_names": (
                    optimal_action_names
                ),
                "optimal_relative_action_names": (
                    optimal_relative_action_names
                ),
                "legal_action_names": action_metadata[
                    "legal_action_names"
                ],
                "shield_used": action_metadata[
                    "shield_used"
                ],
                "shield_reprompts": action_metadata[
                    "shield_reprompts"
                ],
                "reasoning_used": action_metadata[
                    "reasoning_used"
                ],
                "reasoning_prompt": action_metadata[
                    "reasoning_prompt"
                ],
                "is_valid_format": False,
                "is_correct": False,
                "parse_failure": True,
                "error_type": "parse_failure",
                "reward": None,
                "hit_wall": False,
                "hit_obstacle": False,
                "blocked_type": None,
                "terminated": False,
                "truncated": False,
                "early_stopped": False,
                "early_stop_repeat_key": None,
                "consecutive_repeat_streak": 0,
                "grid_text": grid_text,
                "prompt": prompt,
            })
            break

        (
            next_state,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step_cardinal(parsed_action)

        error_type = classify_error(
            parse_failure=False,
            hit_wall=info["hit_wall"],
            hit_obstacle=info["hit_obstacle"],
            is_correct=is_correct,
        )

        repeat_key = (
            tuple(state_before["agent"]),
            int(state_before["facing"]),
            int(parsed_action),
        )

        (
            stop_now,
            previous_repeat_key,
            consecutive_repeat_streak,
        ) = update_consecutive_repeat_state(
            previous_key=previous_repeat_key,
            previous_streak=consecutive_repeat_streak,
            current_key=repeat_key,
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
            "facing_name": state_before[
                "facing_name"
            ],
            "raw_model_answer": raw_answer,
            "all_raw_answers": action_metadata[
                "all_raw_answers"
            ],
            "parsed_action": parsed_action,
            "parsed_action_name": info[
                "action_name"
            ],
            "optimal_actions": sorted(
                optimal_actions
            ),
            "optimal_action_names": (
                optimal_action_names
            ),
            "optimal_relative_action_names": (
                optimal_relative_action_names
            ),
            "legal_action_names": action_metadata[
                "legal_action_names"
            ],
            "shield_used": action_metadata[
                "shield_used"
            ],
            "shield_reprompts": action_metadata[
                "shield_reprompts"
            ],
            "reasoning_used": action_metadata[
                "reasoning_used"
            ],
            "reasoning_prompt": action_metadata[
                "reasoning_prompt"
            ],
            "is_valid_format": True,
            "is_correct": is_correct,
            "parse_failure": False,
            "error_type": error_type,
            "reward": reward,
            "hit_wall": info["hit_wall"],
            "hit_obstacle": info[
                "hit_obstacle"
            ],
            "blocked_type": info[
                "blocked_type"
            ],
            "terminated": terminated,
            "truncated": truncated,
            "early_stopped": early_stopped,
            "early_stop_repeat_key": repeat_key,
            "consecutive_repeat_streak": (
                consecutive_repeat_streak
            ),
            "next_agent_pos": next_state[
                "agent"
            ],
            "next_facing": next_state[
                "facing"
            ],
            "next_facing_name": next_state[
                "facing_name"
            ],
            "grid_text": grid_text,
            "prompt": prompt,
        })

        if verbose:
            print("\n" + "=" * 80)
            print(
                f"Environment={env_name}, "
                f"policy={policy_type}, "
                f"seed={seed}, mode={mode}, "
                f"step={step}"
            )
            print(grid_text)
            print("Raw answer:", raw_answer)
            print(
                "Parsed action:",
                parsed_action,
                info["action_name"],
            )
            print(
                "Legal actions:",
                action_metadata[
                    "legal_action_names"
                ],
            )
            print(
                "Optimal allocentric:",
                optimal_action_names,
            )
            print(
                "Optimal egocentric:",
                optimal_relative_action_names,
            )
            print("Correct:", is_correct)
            print("Error:", error_type)
            print(
                "Consecutive repeat streak:",
                consecutive_repeat_streak,
            )
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
        "seed": int(seed),
        "num_steps": len(step_logs),
        "reached_goal": reached_goal,
        "early_stopped": early_stopped,
        "early_stop_repeats": early_stop_repeats,
        "final_state": final_state["agent"],
        "final_facing": final_state["facing"],
        "final_facing_name": final_state[
            "facing_name"
        ],
        "max_steps": max_steps,
        "step_logs": step_logs,
    }


def main():
    # Supervisor-requested sanity run:
    # rerun only SimpleCrossing S9N1 with 10 paired seeds.
    env_names = [
        "MiniGrid-SimpleCrossingS9N1-v0",
    ]

    seeds = list(range(10))

    modes = [
        "allocentric",
        "egocentric",
    ]

    policy_types = [
        "baseline",
        "legality_shield",
        "reasoning",
    ]

    max_steps_by_env = {
        "MiniGrid-SimpleCrossingS9N1-v0": 9 ** 2,
    }

    early_stop_repeats = 3
    save_traces = True

    model = "gpt-4o-mini"

    # GPT-4o-mini supports deterministic temperature zero in all arms.
    short_temperature = 0.0
    reasoning_temperature = 0.0

    short_max_output_tokens = 16
    reasoning_max_output_tokens = 1500

    # Use new filenames so the previous verified run is preserved.
    output_path = Path(
        "farama_minigrid_results_s9n1_sanity.json"
    )
    metadata_path = Path(
        "farama_minigrid_results_s9n1_sanity_metadata.json"
    )

    short_policy_fn = make_openai_policy_fn(
        model=model,
        temperature=short_temperature,
        max_output_tokens=short_max_output_tokens,
    )

    reasoning_policy_fn = make_openai_policy_fn(
        model=model,
        temperature=reasoning_temperature,
        system_instructions=(
            REASONING_SYSTEM_INSTRUCTIONS
        ),
        max_output_tokens=(
            reasoning_max_output_tokens
        ),
    )

    all_results = []

    # Start a fresh sanity-run results file immediately.
    save_results_incrementally(
        all_results,
        output_path,
    )

    metadata = {
        "run_type": "s9n1_prompt_cleanup_sanity_check",
        "env_names": env_names,
        "seeds": seeds,
        "num_seeds": len(seeds),
        "modes": modes,
        "policy_types": policy_types,
        "max_steps_by_env": max_steps_by_env,
        "early_stop_rule": (
            "three consecutive identical "
            "(agent_position, facing, parsed_action) keys"
        ),
        "early_stop_repeats": early_stop_repeats,
        "save_traces": save_traces,
        "model": model,
        "short_temperature": short_temperature,
        "reasoning_temperature": (
            reasoning_temperature
        ),
        "short_max_output_tokens": (
            short_max_output_tokens
        ),
        "reasoning_max_output_tokens": (
            reasoning_max_output_tokens
        ),
        "output_file": str(output_path),
    }

    with metadata_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            make_json_safe(metadata),
            file,
            indent=2,
        )

    print(
        "\nRUNNING OFFICIAL FARAMA MINIGRID "
        "S9N1 SANITY EXPERIMENT"
    )
    print("=" * 80)
    print("Environments:", env_names)
    print("Seeds:", seeds)
    print("Modes:", modes)
    print("Policy types:", policy_types)
    print(
        "Early stop repeats:",
        early_stop_repeats,
    )
    print("Save traces:", save_traces)
    print(
        "Short temperature:",
        short_temperature,
    )
    print(
        "Reasoning temperature:",
        reasoning_temperature,
    )
    print(
        "Short max output tokens:",
        short_max_output_tokens,
    )
    print(
        "Reasoning max output tokens:",
        reasoning_max_output_tokens,
    )

    for env_name in env_names:
        max_steps = max_steps_by_env[
            env_name
        ]

        print("\n" + "=" * 80)
        print(
            f"ENVIRONMENT: {env_name}, "
            f"max_steps={max_steps}"
        )
        print("=" * 80)

        for seed in seeds:
            print("\n" + "-" * 80)
            print(f"Seed {seed}")

            for policy_type in policy_types:
                for mode in modes:
                    print(
                        f"Running policy={policy_type}, "
                        f"mode={mode}"
                    )

                    result = run_episode(
                        env_name=env_name,
                        seed=seed,
                        mode=mode,
                        policy_type=policy_type,
                        short_policy_fn=short_policy_fn,
                        reasoning_policy_fn=(
                            reasoning_policy_fn
                        ),
                        max_steps=max_steps,
                        early_stop_repeats=(
                            early_stop_repeats
                        ),
                        verbose=False,
                    )

                    result["model"] = model
                    result[
                        "short_temperature"
                    ] = short_temperature
                    result[
                        "reasoning_temperature"
                    ] = reasoning_temperature
                    result[
                        "short_max_output_tokens"
                    ] = short_max_output_tokens
                    result[
                        "reasoning_max_output_tokens"
                    ] = reasoning_max_output_tokens

                    all_results.append(result)

                    # Save immediately after every completed episode.
                    save_results_incrementally(
                        all_results,
                        output_path,
                    )

                    if save_traces:
                        save_episode_trace(result)

                    print(
                        "  "
                        f"reached_goal="
                        f"{result['reached_goal']}, "
                        f"early_stopped="
                        f"{result['early_stopped']}, "
                        f"steps="
                        f"{result['num_steps']}, "
                        f"final_state="
                        f"{result['final_state']}"
                    )

    print("\nSaved:")
    print("-", output_path)
    print("-", metadata_path)

    if save_traces:
        print(
            "- "
            "farama_minigrid_traces_s9n1_sanity/"
        )


if __name__ == "__main__":
    main()