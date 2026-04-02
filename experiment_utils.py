from prompt_utils import make_allocentric_prompt, make_egocentric_prompt


ABSOLUTE_TEXT_TO_ACTION = {
    "north": 0,
    "east": 1,
    "south": 2,
    "west": 3,
}

RELATIVE_TEXT_TO_ACTION = {
    "forward": 0,
    "right": 1,
    "backward": 2,
    "left": 3,
}


def normalize_answer(text):
    """
    Clean model output into a lowercase first-token answer.
    """
    if text is None:
        return None

    text = text.strip().lower()

    for ch in [".", ",", "!", "?", ":", ";", '"', "'", "`"]:
        text = text.replace(ch, "")

    lines = text.splitlines()
    if not lines:
        return None

    text = lines[0].strip()
    parts = text.split()
    if not parts:
        return None

    return parts[0]


def parse_allocentric_answer(text):
    answer = normalize_answer(text)
    if answer in ABSOLUTE_TEXT_TO_ACTION:
        return ABSOLUTE_TEXT_TO_ACTION[answer]
    return None


def parse_egocentric_answer(text):
    answer = normalize_answer(text)
    if answer in RELATIVE_TEXT_TO_ACTION:
        return RELATIVE_TEXT_TO_ACTION[answer]
    return None


def relative_to_absolute_action(agent_facing, relative_action):
    """
    Facing:
        0=north, 1=east, 2=south, 3=west
    Relative:
        0=forward, 1=right, 2=backward, 3=left
    """
    if relative_action is None:
        return None
    return (agent_facing + relative_action) % 4


def get_prompt_for_mode(obs, mode):
    if mode == "allocentric":
        return make_allocentric_prompt(obs)
    if mode == "egocentric":
        return make_egocentric_prompt(obs)
    raise ValueError(f"Unknown mode: {mode}")


def get_correct_answer_for_mode(env, mode):
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
    correct_answer_text = get_correct_answer_for_mode(env, mode)

    parsed_action = parse_model_answer_to_absolute_action(obs, mode, model_text)
    optimal_actions = env.get_optimal_actions()

    is_valid_format = parsed_action is not None
    is_correct = parsed_action in optimal_actions if is_valid_format else False

    return {
        "prompt": prompt,
        "correct_answer_text": correct_answer_text,
        "parsed_action": parsed_action,
        "optimal_absolute_actions": sorted(optimal_actions),
        "is_valid_format": is_valid_format,
        "is_correct": is_correct,
        "raw_model_answer": model_text,
    }


def _serialize_state(obs):
    return {
        "agent": obs["agent"].tolist(),
        "goal": obs["goal"].tolist(),
        "obstacles": obs["obstacles"].tolist(),
        "facing": int(obs["facing"]),
    }


def generate_fixed_states(num_states, size, obstacle_density, max_steps=50, ensure_path=True, seed_start=0):
    """
    Pre-generate fixed states so different prompt modes can be compared on the
    exact same layouts.
    """
    from gridworld import SimpleGridWorld

    states = []
    for i in range(num_states):
        seed = seed_start + i
        env = SimpleGridWorld(
            size=size,
            obstacle_density=obstacle_density,
            max_steps=max_steps,
            ensure_path=ensure_path,
        )
        obs, _ = env.reset(seed=seed)
        state = _serialize_state(obs)
        state["seed"] = seed
        states.append(state)

    return states


def run_episode(env, mode, policy_fn, max_steps=None, verbose=False, seed=None, fixed_state=None):
    """
    Run one full episode.

    policy_fn(prompt) -> model output string

    If fixed_state is provided, the environment is reset to that exact state so
    allocentric and egocentric conditions can be compared on the same map.
    """
    if fixed_state is not None:
        obs, info = env.reset(seed=seed, options={"state": fixed_state})
    else:
        obs, info = env.reset(seed=seed)

    done = False
    step_logs = []
    step_count = 0

    if max_steps is None:
        max_steps = env.max_steps

    while not done and step_count < max_steps:
        prompt = get_prompt_for_mode(obs, mode)
        correct_answer_text = get_correct_answer_for_mode(env, mode)
        optimal_actions = env.get_optimal_actions()

        model_text = policy_fn(prompt)
        chosen_action = parse_model_answer_to_absolute_action(obs, mode, model_text)

        valid_format = chosen_action is not None
        is_correct = valid_format and (chosen_action in optimal_actions)

        if verbose:
            print("=" * 60)
            print("STEP", step_count)
            print(prompt)
            print("Model answer:", model_text)
            print("Correct answers:", correct_answer_text)
            print("Optimal absolute actions:", sorted(optimal_actions))

        if chosen_action is None:
            step_logs.append({
                "step": step_count,
                "seed": fixed_state.get("seed", seed) if fixed_state is not None else seed,
                "agent_pos": obs["agent"].tolist(),
                "goal_pos": obs["goal"].tolist(),
                "obstacles": obs["obstacles"].tolist(),
                "facing": int(obs["facing"]),
                "prompt_mode": mode,
                "prompt": prompt,
                "raw_model_answer": model_text,
                "parsed_action": None,
                "correct_answer_text": correct_answer_text,
                "optimal_absolute_actions": sorted(optimal_actions),
                "is_valid_format": False,
                "is_correct": False,
                "parse_failure": True,
                "reward": None,
                "terminated": False,
                "truncated": False,
                "hit_wall": None,
                "hit_obstacle": None,
                "manhattan_distance": info.get("manhattan_distance"),
                "shortest_path_length": info.get("shortest_path_length"),
            })
            break

        next_obs, reward, terminated, truncated, step_info = env.step(chosen_action)

        step_logs.append({
            "step": step_count,
            "seed": fixed_state.get("seed", seed) if fixed_state is not None else seed,
            "agent_pos": obs["agent"].tolist(),
            "goal_pos": obs["goal"].tolist(),
            "obstacles": obs["obstacles"].tolist(),
            "facing": int(obs["facing"]),
            "prompt_mode": mode,
            "prompt": prompt,
            "raw_model_answer": model_text,
            "parsed_action": chosen_action,
            "correct_answer_text": correct_answer_text,
            "optimal_absolute_actions": sorted(optimal_actions),
            "is_valid_format": True,
            "is_correct": is_correct,
            "parse_failure": False,
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
            "hit_wall": step_info["hit_wall"],
            "hit_obstacle": step_info["hit_obstacle"],
            "manhattan_distance": info.get("manhattan_distance"),
            "shortest_path_length": info.get("shortest_path_length"),
        })

        obs = next_obs
        info = step_info
        done = terminated or truncated
        step_count += 1

    reached_goal = False
    if step_logs:
        reached_goal = bool(step_logs[-1]["terminated"])

    return {
        "mode": mode,
        "seed": fixed_state.get("seed", seed) if fixed_state is not None else seed,
        "fixed_state_used": fixed_state is not None,
        "num_steps": len(step_logs),
        "reached_goal": reached_goal,
        "step_logs": step_logs,
    }
