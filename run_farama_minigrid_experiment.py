import argparse
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
    Save all completed episode results after every episode.

    The temporary file is replaced atomically so a late API failure does not
    discard the episodes that have already completed.
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
    Parse one-word baseline and legality-shield responses.
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

    The answer marker must begin a line, but common Markdown formatting is
    accepted. The parser uses the first valid action word after the marker on
    the same line. It never scans the rest of the response as a fallback.

    Accepted examples include:
        Answer: west
        **Answer:** west
        **Answer: west**
        - Answer: west
        Answer: go west now, not north
    """
    if text is None:
        return None

    if mode == "allocentric":
        valid_words = set(ABSOLUTE_ACTIONS)
    elif mode == "egocentric":
        valid_words = set(RELATIVE_ACTIONS)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    marker_pattern = re.compile(
        r"""
        ^[ \t]*
        (?:[-*+][ \t]+|\#{1,6}[ \t]+)?
        (?:\*\*|__)?
        (?:final[ \t]+answer|final[ \t]+action|answer)
        [ \t]*
        (?:\*\*|__)?
        [ \t]*:[ \t]*
        (?:\*\*|__)?
        (?P<answer_text>[^\r\n]*)
        $ 
        """,
        flags=re.IGNORECASE | re.MULTILINE | re.VERBOSE,
    )

    matches = list(marker_pattern.finditer(text))
    if not matches:
        return None

    answer_text = matches[-1].group("answer_text")
    words = re.findall(r"[a-z]+", answer_text.lower())

    for word in words:
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

After the reasoning, give the final answer on a new line using exactly this format:

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
    Convert a cardinal action into a relative action index.

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
        names = []

        for action in sorted(legal_actions):
            relative_action = cardinal_to_relative(
                action,
                facing,
            )

            names.append(
                RELATIVE_ACTION_NAMES[relative_action]
            )

        return names

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


def make_reprompt_control_prompt(
    original_prompt,
    raw_answer,
):
    """
    Retry prompt for the matched-reprompt control arm.

    This deliberately withholds the legal-action list so the only difference
    from the legality shield is the hint content.
    """
    return f"""{original_prompt}

Your previous answer was: {raw_answer}

That proposed action is illegal from the current state because it would move
into an obstacle, the outer boundary, or outside the grid.

The illegal action was not executed.

Choose another action.

Answer with one word only using the action choices from the original prompt.
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

    if policy_type in {"legality_shield", "reprompt_control"}:
        legal_actions = get_legal_cardinal_actions(env)
        legal_names = legal_action_names(
            env,
            mode,
        )

        all_raw_answers = []
        current_prompt = prompt

        for attempt in range(max_reprompts + 1):
            raw_answer = short_policy_fn(current_prompt)
            all_raw_answers.append(raw_answer)

            parsed_action = parse_action(
                raw_answer,
                mode,
                env,
            )

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

            if policy_type == "legality_shield":
                current_prompt = make_legality_reprompt(
                    original_prompt=prompt,
                    raw_answer=raw_answer,
                    legal_names=legal_names,
                )
            else:
                current_prompt = make_reprompt_control_prompt(
                    original_prompt=prompt,
                    raw_answer=raw_answer,
                )

        # Neither intervention arm ever executes an illegal action.
        # Exhausting the shared retry budget becomes a parse failure.
        return None, {
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
    Stop only after consecutive repeats of the same state-action key.

    A, A, A -> stop
    A, B, A -> do not stop
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
    output_dir="farama_minigrid_traces_final",
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
                f"Terminated: {step_log.get('terminated')}\n"
            )
            file.write(
                f"Truncated: {step_log.get('truncated')}\n"
            )
            file.write(
                f"Early stopped: {step_log.get('early_stopped')}\n"
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
                "facing_name": state_before["facing_name"],
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
            "facing_name": state_before["facing_name"],
            "raw_model_answer": raw_answer,
            "all_raw_answers": action_metadata[
                "all_raw_answers"
            ],
            "parsed_action": parsed_action,
            "parsed_action_name": info["action_name"],
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
            "hit_obstacle": info["hit_obstacle"],
            "blocked_type": info["blocked_type"],
            "terminated": terminated,
            "truncated": truncated,
            "early_stopped": early_stopped,
            "early_stop_repeat_key": repeat_key,
            "consecutive_repeat_streak": (
                consecutive_repeat_streak
            ),
            "next_agent_pos": next_state["agent"],
            "next_facing": next_state["facing"],
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
                f"seed={seed}, mode={mode}, step={step}"
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
                action_metadata["legal_action_names"],
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
        "final_facing_name": final_state["facing_name"],
        "max_steps": max_steps,
        "step_logs": step_logs,
    }


def test_reasoning_parser():
    valid_cases = [
        ("Answer: west", "west"),
        ("**Answer:** west", "west"),
        ("**Answer: west**", "west"),
        ("- Answer: west", "west"),
        ("### Answer: west", "west"),
        ("Final answer: west", "west"),
        ("Answer: go west now, not north", "west"),
    ]

    for text, expected in valid_cases:
        parsed = extract_final_action_word(
            text,
            "allocentric",
        )
        assert parsed == expected, (
            f"Expected {expected}, got {parsed}: {text!r}"
        )

    invalid_cases = [
        "Answer:",
        "Answer: none",
        "I considered west first.",
        "The best move is west.",
        "Final answer:",
    ]

    for text in invalid_cases:
        parsed = extract_final_action_word(
            text,
            "allocentric",
        )
        assert parsed is None, (
            f"Expected None, got {parsed}: {text!r}"
        )

    print("Reasoning parser tests passed.")


def parse_csv_or_space_values(values):
    parsed = []
    for value in values:
        parsed.extend(
            item.strip()
            for item in value.split(",")
            if item.strip()
        )
    return parsed


def parse_seed_spec(values):
    """Parse seed lists such as 0 1 2, 0,1,2, or ranges such as 0-9."""
    seeds = []
    for token in parse_csv_or_space_values(values):
        range_match = re.fullmatch(r"(-?\d+)-(-?\d+)", token)
        if range_match:
            start = int(range_match.group(1))
            end = int(range_match.group(2))
            step = 1 if end >= start else -1
            seeds.extend(range(start, end + step, step))
        else:
            seeds.append(int(token))

    # Preserve requested order while removing duplicates.
    return list(dict.fromkeys(seeds))


def build_argument_parser():
    parser = argparse.ArgumentParser(
        description="Run split Farama MiniGrid benchmark chunks."
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        default=["0-9"],
        help="Seed range/list, e.g. 0-9, 0,2,4, or 0 1 2.",
    )
    parser.add_argument(
        "--arms",
        nargs="+",
        default=[
            "baseline",
            "legality_shield",
            "reprompt_control",
            "reasoning",
        ],
        help="Arms separated by spaces or commas.",
    )
    parser.add_argument(
        "--environments",
        nargs="+",
        default=[
            "MiniGrid-SimpleCrossingS9N1-v0",
            "MiniGrid-SimpleCrossingS9N2-v0",
            "MiniGrid-SimpleCrossingS9N3-v0",
            "MiniGrid-FourRooms-v0",
        ],
        help="Environment IDs separated by spaces or commas.",
    )
    parser.add_argument(
        "--model",
        default="openai/gpt-4o-mini",
        help=(
            "OpenRouter model ID, for example openai/gpt-4.1-mini or "
            "anthropic/claude-haiku-4.5. Recorded verbatim in the run "
            "metadata."
        ),
    )

    parser.add_argument(
        "--provider",
        default=None,
        help=(
            "Optional OpenRouter upstream provider to pin, for example "
            "OpenAI or Anthropic. When set, all calls hit that one upstream. "
            "Recorded in the run metadata."
        ),
    )

    parser.add_argument(
        "--output",
        default="farama_minigrid_results_final.json",
        help="Unique JSON output filename for this chunk.",
    )
    return parser


def main():
    args = build_argument_parser().parse_args()

    env_names = parse_csv_or_space_values(args.environments)
    seeds = parse_seed_spec(args.seeds)

    modes = [
        "allocentric",
        "egocentric",
    ]

    policy_types = parse_csv_or_space_values(args.arms)

    valid_policy_types = {
        "baseline",
        "legality_shield",
        "reprompt_control",
        "reasoning",
    }
    unknown_policy_types = set(policy_types) - valid_policy_types
    if unknown_policy_types:
        raise ValueError(
            "Unknown arms: "
            + ", ".join(sorted(unknown_policy_types))
        )

    max_steps_by_env = {
        "MiniGrid-SimpleCrossingS9N1-v0": 9 ** 2,
        "MiniGrid-SimpleCrossingS9N2-v0": 9 ** 2,
        "MiniGrid-SimpleCrossingS9N3-v0": 9 ** 2,
        "MiniGrid-FourRooms-v0": 19 ** 2,
    }

    early_stop_repeats = 3
    save_traces = True

    model = args.model
    provider_pin = args.provider

    short_temperature = 0.0
    reasoning_temperature = 0.0

    short_max_output_tokens = 16
    reasoning_max_output_tokens = 1500

    output_path = Path(args.output)
    metadata_path = output_path.with_name(
        f"{output_path.stem}_metadata.json"
    )
    trace_directory = f"{output_path.stem}_traces"

    short_policy_fn = make_openai_policy_fn(
        model=model,
        temperature=short_temperature,
        max_output_tokens=short_max_output_tokens,
        provider=provider_pin,
    )

    reasoning_policy_fn = make_openai_policy_fn(
        model=model,
        temperature=reasoning_temperature,
        system_instructions=(
            REASONING_SYSTEM_INSTRUCTIONS
        ),
        max_output_tokens=reasoning_max_output_tokens,
        provider=provider_pin,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    all_results = []

    save_results_incrementally(
        all_results,
        output_path,
    )

    metadata = {
        "run_type": "split_full_farama_benchmark",
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
        "trace_directory": trace_directory,
        "model": model,
        "provider_pin": provider_pin,
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
        "intervention_arms": {
            "legality_shield": (
                "two-reprompt budget; reprompt lists legal actions"
            ),
            "reprompt_control": (
                "two-reprompt budget; reprompt does not list legal actions"
            ),
        },
        "max_reprompts": 2,
        "intervention_exhaustion_behavior": (
            "parse_failure; no illegal action is executed"
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
        "\nRUNNING FULL CORRECTED "
        "FARAMA MINIGRID BENCHMARK"
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
        max_steps = max_steps_by_env[env_name]

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

                    save_results_incrementally(
                        all_results,
                        output_path,
                    )

                    if save_traces:
                        save_episode_trace(
                            result,
                            output_dir=trace_directory,
                        )

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
            f"- {trace_directory}/"
        )


if __name__ == "__main__":
    test_reasoning_parser()
    main()