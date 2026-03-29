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

    for ch in [".", ",", "!", "?", ":", ";", "\"", "'", "`"]:
        text = text.replace(ch, "")

    text = text.splitlines()[0].strip()

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
    elif mode == "egocentric":
        return make_egocentric_prompt(obs)
    else:
        raise ValueError(f"Unknown mode: {mode}")


def get_correct_answer_for_mode(env, mode):
    if mode == "allocentric":
        return env.get_optimal_action_name()
    elif mode == "egocentric":
        return env.get_optimal_relative_action_name()
    else:
        raise ValueError(f"Unknown mode: {mode}")


def parse_model_answer_to_absolute_action(obs, mode, model_text):
    if mode == "allocentric":
        return parse_allocentric_answer(model_text)

    elif mode == "egocentric":
        relative_action = parse_egocentric_answer(model_text)
        if relative_action is None:
            return None
        return relative_to_absolute_action(obs["facing"], relative_action)

    else:
        raise ValueError(f"Unknown mode: {mode}")


def evaluate_single_decision(env, obs, mode, model_text):
    prompt = get_prompt_for_mode(obs, mode)
    correct_answer_text = get_correct_answer_for_mode(env, mode)

    parsed_action = parse_model_answer_to_absolute_action(obs, mode, model_text)
    correct_absolute_action = env.get_optimal_action()

    is_valid_format = parsed_action is not None
    is_correct = parsed_action == correct_absolute_action if is_valid_format else False

    return {
        "prompt": prompt,
        "correct_answer_text": correct_answer_text,
        "parsed_action": parsed_action,
        "correct_absolute_action": correct_absolute_action,
        "is_valid_format": is_valid_format,
        "is_correct": is_correct,
        "raw_model_answer": model_text,
    }


def run_episode(env, mode, policy_fn, max_steps=None, verbose=False):
    """
    Run one full episode.
    policy_fn(prompt) -> model output string
    """
    obs, info = env.reset()
    done = False
    step_logs = []
    step_count = 0

    if max_steps is None:
        max_steps = env.max_steps

    while not done and step_count < max_steps:
        prompt = get_prompt_for_mode(obs, mode)
        correct_answer_text = get_correct_answer_for_mode(env, mode)
        correct_absolute_action = env.get_optimal_action()

        model_text = policy_fn(prompt)
        chosen_action = parse_model_answer_to_absolute_action(obs, mode, model_text)

        valid_format = chosen_action is not None
        is_correct = valid_format and (chosen_action == correct_absolute_action)

        if verbose:
            print("=" * 60)
            print("STEP", step_count)
            print(prompt)
            print("Model answer:", model_text)
            print("Correct answer:", correct_answer_text)

        if chosen_action is None:
            step_logs.append({
                "step": step_count,
                "agent_pos": obs["agent"].copy(),
                "goal_pos": obs["goal"].copy(),
                "facing": int(obs["facing"]),
                "prompt_mode": mode,
                "prompt": prompt,
                "raw_model_answer": model_text,
                "parsed_action": None,
                "correct_answer_text": correct_answer_text,
                "correct_absolute_action": correct_absolute_action,
                "is_valid_format": False,
                "is_correct": False,
                "reward": None,
                "terminated": False,
                "truncated": True,
                "hit_wall": None,
                "hit_obstacle": None,
            })
            break

        next_obs, reward, terminated, truncated, step_info = env.step(chosen_action)

        step_logs.append({
            "step": step_count,
            "agent_pos": obs["agent"].copy(),
            "goal_pos": obs["goal"].copy(),
            "facing": int(obs["facing"]),
            "prompt_mode": mode,
            "prompt": prompt,
            "raw_model_answer": model_text,
            "parsed_action": chosen_action,
            "correct_answer_text": correct_answer_text,
            "correct_absolute_action": correct_absolute_action,
            "is_valid_format": True,
            "is_correct": is_correct,
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
            "hit_wall": step_info["hit_wall"],
            "hit_obstacle": step_info["hit_obstacle"],
        })

        obs = next_obs
        info = step_info
        done = terminated or truncated
        step_count += 1

    reached_goal = False
    if step_logs:
        last = step_logs[-1]
        reached_goal = bool(last["terminated"])

    return {
        "mode": mode,
        "num_steps": len(step_logs),
        "reached_goal": reached_goal,
        "step_logs": step_logs,
    }
