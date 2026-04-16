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


def action_name(env, action):
    if action is None:
        return None
    return env.action_to_name[action]


def action_names(env, actions):
    return [env.action_to_name[a] for a in sorted(actions)]


def run_episode(env, mode, policy_fn, seed=None, max_steps=None, verbose=False):
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
        grid_text_before = env.render_text()

        model_text = policy_fn(prompt)
        chosen_action = parse_model_answer_to_absolute_action(obs, mode, model_text)

        valid_format = chosen_action is not None
        is_correct = valid_format and (chosen_action in optimal_actions)

        if verbose:
            print("=" * 60)
            print("STEP", step_count)
            print(grid_text_before)
            print()
            print(prompt)
            print("Model answer:", model_text)
            print("Chosen absolute action:", action_name(env, chosen_action))
            print("Optimal absolute actions:", action_names(env, optimal_actions))
            print("Correct answer text:", correct_answer_text)

        if chosen_action is None:
            step_logs.append({
                "step": step_count,
                "seed": seed,
                "grid_size": env.size,
                "agent_pos": obs["agent"].tolist(),
                "goal_pos": obs["goal"].tolist(),
                "facing": int(obs["facing"]),
                "facing_name": env.facing_to_name[int(obs["facing"])],
                "obstacles": obs["obstacles"].tolist(),
                "prompt_mode": mode,
                "prompt": prompt,
                "grid_text_before": grid_text_before,
                "grid_text_after": grid_text_before,
                "raw_model_answer": model_text,
                "parsed_action": None,
                "parsed_action_name": None,
                "correct_answer_text": correct_answer_text,
                "optimal_actions": sorted(list(optimal_actions)),
                "optimal_action_names": action_names(env, optimal_actions),
                "is_valid_format": False,
                "is_correct": False,
                "parse_failure": True,
                "reward": None,
                "terminated": False,
                "truncated": False,
                "hit_wall": None,
                "hit_obstacle": None,
            })
            break

        next_obs, reward, terminated, truncated, step_info = env.step(chosen_action)
        grid_text_after = env.render_text()

        step_logs.append({
            "step": step_count,
            "seed": seed,
            "grid_size": env.size,
            "agent_pos": obs["agent"].tolist(),
            "goal_pos": obs["goal"].tolist(),
            "facing": int(obs["facing"]),
            "facing_name": env.facing_to_name[int(obs["facing"])],
            "obstacles": obs["obstacles"].tolist(),
            "prompt_mode": mode,
            "prompt": prompt,
            "grid_text_before": grid_text_before,
            "grid_text_after": grid_text_after,
            "raw_model_answer": model_text,
            "parsed_action": chosen_action,
            "parsed_action_name": action_name(env, chosen_action),
            "correct_answer_text": correct_answer_text,
            "optimal_actions": sorted(list(optimal_actions)),
            "optimal_action_names": action_names(env, optimal_actions),
            "is_valid_format": True,
            "is_correct": is_correct,
            "parse_failure": False,
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
            "hit_wall": step_info["hit_wall"],
            "hit_obstacle": step_info["hit_obstacle"],
        })

        if verbose:
            print("Reward:", reward)
            print("Hit wall:", step_info["hit_wall"])
            print("Hit obstacle:", step_info["hit_obstacle"])
            print("Terminated:", terminated)
            print("Truncated:", truncated)
            print()

        obs = next_obs
        info = step_info
        done = terminated or truncated
        step_count += 1

    reached_goal = False
    if step_logs:
        reached_goal = bool(step_logs[-1]["terminated"])

    return {
        "mode": mode,
        "seed": seed,
        "num_steps": len(step_logs),
        "reached_goal": reached_goal,
        "step_logs": step_logs,
    }
