from prompt_utils import make_allocentric_prompt, make_egocentric_prompt

ABSOLUTE_TEXT_TO_ACTION = {"north": 0, "east": 1, "south": 2, "west": 3}
RELATIVE_TEXT_TO_ACTION = {"forward": 0, "right": 1, "backward": 2, "left": 3}


def normalize_answer(text):
    if text is None:
        return None

    text = text.strip().lower()
    for ch in [".", ",", "!", "?", ":", ";", '"', "'", "`", "(", ")"]:
        text = text.replace(ch, "")

    text = text.splitlines()[0].strip()
    parts = text.split()
    if not parts:
        return None
    return parts[0]


def parse_allocentric_answer(text):
    answer = normalize_answer(text)
    return ABSOLUTE_TEXT_TO_ACTION.get(answer)


def parse_egocentric_answer(text):
    answer = normalize_answer(text)
    return RELATIVE_TEXT_TO_ACTION.get(answer)


def relative_to_absolute_action(agent_facing, relative_action):
    if relative_action is None:
        return None
    return (agent_facing + relative_action) % 4


def get_prompt_for_mode(obs, mode):
    if mode == "allocentric":
        return make_allocentric_prompt(obs)
    if mode == "egocentric":
        return make_egocentric_prompt(obs)
    raise ValueError(f"Unknown mode: {mode}")


def get_correct_answers_for_mode(env, mode):
    if mode == "allocentric":
        return env.get_optimal_action_names()
    if mode == "egocentric":
        return env.get_optimal_relative_action_names()
    raise ValueError(f"Unknown mode: {mode}")


def parse_model_answer_to_absolute_action(obs, mode, model_text):
    if mode == "allocentric":
        return parse_allocentric_answer(model_text)
    if mode == "egocentric":
        relative_action = parse_egocentric_answer(model_text)
        if relative_action is None:
            return None
        return relative_to_absolute_action(obs["facing"], relative_action)
    raise ValueError(f"Unknown mode: {mode}")


def evaluate_single_decision(env, obs, mode, model_text):
    prompt = get_prompt_for_mode(obs, mode)
    correct_answer_texts = get_correct_answers_for_mode(env, mode)
    parsed_action = parse_model_answer_to_absolute_action(obs, mode, model_text)
    optimal_actions = env.get_optimal_actions()

    is_valid_format = parsed_action is not None
    is_correct = bool(is_valid_format and parsed_action in optimal_actions)

    return {
        "prompt": prompt,
        "correct_answer_texts": correct_answer_texts,
        "parsed_action": parsed_action,
        "optimal_actions": sorted(optimal_actions),
        "is_valid_format": is_valid_format,
        "is_correct": is_correct,
        "raw_model_answer": model_text,
    }


def generate_fixed_states(size, obstacle_density, num_states, max_steps=50, ensure_path=True, start_seed=0):
    from gridworld import SimpleGridWorld

    states = []
    for episode_seed in range(start_seed, start_seed + num_states):
        env = SimpleGridWorld(
            size=size,
            obstacle_density=obstacle_density,
            max_steps=max_steps,
            ensure_path=ensure_path,
        )
        env.reset(seed=episode_seed)
        states.append(env.export_state())
    return states


def run_episode(env, mode, policy_fn, max_steps=None, verbose=False, seed=None, fixed_state=None):
    """
    Run one full episode.
    policy_fn(prompt) -> model output string
    """
    if fixed_state is not None:
        obs, info = env.reset(options={"state": fixed_state})
    else:
        obs, info = env.reset(seed=seed)

    done = False
    step_logs = []
    step_count = 0

    if max_steps is None:
        max_steps = env.max_steps

    while not done and step_count < max_steps:
        prompt = get_prompt_for_mode(obs, mode)
        correct_answer_texts = get_correct_answers_for_mode(env, mode)
        optimal_actions = env.get_optimal_actions()

        model_text = policy_fn(prompt)
        chosen_action = parse_model_answer_to_absolute_action(obs, mode, model_text)

        valid_format = chosen_action is not None
        parse_failure = not valid_format
        is_correct = bool(valid_format and chosen_action in optimal_actions)

        if verbose:
            print("=" * 60)
            print("STEP", step_count)
            print(prompt)
            print("Model answer:", model_text)
            print("Correct answers:", correct_answer_texts)

        if chosen_action is None:
            step_logs.append(
                {
                    "step": step_count,
                    "seed": seed if fixed_state is None else fixed_state.get("seed"),
                    "agent_pos": obs["agent"].tolist(),
                    "goal_pos": obs["goal"].tolist(),
                    "facing": int(obs["facing"]),
                    "prompt_mode": mode,
                    "prompt": prompt,
                    "raw_model_answer": model_text,
                    "parsed_action": None,
                    "correct_answer_texts": correct_answer_texts,
                    "optimal_actions": sorted(optimal_actions),
                    "is_valid_format": False,
                    "parse_failure": True,
                    "is_correct": False,
                    "reward": None,
                    "terminated": False,
                    "truncated": False,
                    "hit_wall": None,
                    "hit_obstacle": None,
                    "manhattan_distance": int(info["manhattan_distance"]),
                }
            )
            break

        next_obs, reward, terminated, truncated, step_info = env.step(chosen_action)

        step_logs.append(
            {
                "step": step_count,
                "seed": seed if fixed_state is None else fixed_state.get("seed"),
                "agent_pos": obs["agent"].tolist(),
                "goal_pos": obs["goal"].tolist(),
                "facing": int(obs["facing"]),
                "prompt_mode": mode,
                "prompt": prompt,
                "raw_model_answer": model_text,
                "parsed_action": chosen_action,
                "correct_answer_texts": correct_answer_texts,
                "optimal_actions": sorted(optimal_actions),
                "is_valid_format": True,
                "parse_failure": parse_failure,
                "is_correct": is_correct,
                "reward": reward,
                "terminated": terminated,
                "truncated": truncated,
                "hit_wall": bool(step_info["hit_wall"]),
                "hit_obstacle": bool(step_info["hit_obstacle"]),
                "manhattan_distance": int(info["manhattan_distance"]),
            }
        )

        obs = next_obs
        info = step_info
        done = bool(terminated or truncated)
        step_count += 1

    reached_goal = bool(step_logs and step_logs[-1]["terminated"])
    final_state = env.export_state() if hasattr(env, "export_state") else None

    return {
        "mode": mode,
        "seed": seed if fixed_state is None else fixed_state.get("seed"),
        "num_steps": len(step_logs),
        "reached_goal": reached_goal,
        "step_logs": step_logs,
        "final_state": final_state,
    }