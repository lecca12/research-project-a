def normalize_answer(text):
    if text is None:
        return None

    text = text.strip().lower()

    for ch in [".", ",", "!", "?", ":", ";", "\"", "'", "`", "(", ")"]:
        text = text.replace(ch, "")

    text = text.splitlines()[0].strip()

    parts = text.split()
    if not parts:
        return None

    return parts[0]


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


def parse_allocentric(text):
    word = normalize_answer(text)
    return ABSOLUTE_ACTIONS.get(word)


def parse_egocentric(text):
    word = normalize_answer(text)
    return RELATIVE_ACTIONS.get(word)


def relative_to_absolute(facing, rel_action):
    if rel_action is None:
        return None
    return (facing + rel_action) % 4


def get_prompt(obs, mode):
    from prompt_utils import make_allocentric_prompt, make_egocentric_prompt

    if mode == "allocentric":
        return make_allocentric_prompt(obs)
    elif mode == "egocentric":
        return make_egocentric_prompt(obs)
    else:
        raise ValueError("Unknown mode")


def parse_action(obs, mode, text):
    if mode == "allocentric":
        return parse_allocentric(text)

    if mode == "egocentric":
        rel = parse_egocentric(text)
        if rel is None:
            return None
        return relative_to_absolute(obs["facing"], rel)

    raise ValueError("Unknown mode")


def generate_fixed_states(size, obstacle_density, num_states, max_steps, start_seed=0):
    from gridworld import SimpleGridWorld

    states = []

    for seed in range(start_seed, start_seed + num_states):
        env = SimpleGridWorld(
            size=size,
            obstacle_density=obstacle_density,
            max_steps=max_steps,
        )
        env.reset(seed=seed)
        states.append(env.export_state())

    return states


def run_episode(env, mode, policy_fn, fixed_state=None, verbose=False):
    if fixed_state is not None:
        obs, info = env.reset(options={"state": fixed_state})
        seed = fixed_state.get("seed")
    else:
        obs, info = env.reset()
        seed = None

    done = False
    step_logs = []
    step = 0

    while not done:
        prompt = get_prompt(obs, mode)
        optimal_actions = env.get_optimal_actions()
        grid_text = env.render_text()

        model_text = policy_fn(prompt)
        action = parse_action(obs, mode, model_text)

        valid = action is not None
        correct = valid and action in optimal_actions

        if action is None:
            step_logs.append({
                "step": step,
                "agent_pos": obs["agent"].tolist(),
                "goal_pos": obs["goal"].tolist(),
                "facing": int(obs["facing"]),
                "raw_model_answer": model_text,
                "parsed_action": None,
                "optimal_actions": list(optimal_actions),
                "is_valid_format": False,
                "is_correct": False,
                "parse_failure": True,
                "reward": None,
                "grid_text": grid_text,
            })
            break

        next_obs, reward, terminated, truncated, info = env.step(action)

        step_logs.append({
            "step": step,
            "agent_pos": obs["agent"].tolist(),
            "goal_pos": obs["goal"].tolist(),
            "facing": int(obs["facing"]),
            "raw_model_answer": model_text,
            "parsed_action": action,
            "optimal_actions": list(optimal_actions),
            "is_valid_format": True,
            "is_correct": correct,
            "parse_failure": False,
            "reward": reward,
            "hit_wall": info["hit_wall"],
            "hit_obstacle": info["hit_obstacle"],
            "terminated": terminated,
            "truncated": truncated,
            "grid_text": grid_text,
        })

        if verbose:
            print("=" * 60)
            print(grid_text)
            print(prompt)
            print("LLM:", model_text)
            print("Parsed:", action)
            print("Correct:", correct)

        obs = next_obs
        done = terminated or truncated
        step += 1

    return {
        "mode": mode,
        "seed": seed,
        "num_steps": len(step_logs),
        "reached_goal": bool(step_logs and step_logs[-1].get("terminated")),
        "final_state": obs["agent"].tolist(),
        "max_steps": env.max_steps,
        "prompt": prompt,
        "step_logs": step_logs,
    }