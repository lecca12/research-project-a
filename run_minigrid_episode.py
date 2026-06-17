from minigrid_wrapper import MiniGridCardinalWrapper
from llm_policy import make_openai_policy_fn
from experiment_utils import normalize_answer


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


def run_minigrid_episode(mode, seed=0, max_steps=64, verbose=True):
    env = MiniGridCardinalWrapper(
        env_name="MiniGrid-SimpleCrossingS9N1-v0",
        seed=seed,
    )
    env.reset(seed=seed)

    policy_fn = make_openai_policy_fn(
        model="gpt-4o-mini",
        temperature=0.0,
        max_output_tokens=16,
    )

    step_logs = []
    reached_goal = False

    for step in range(max_steps):
        prompt = get_prompt(env, mode)
        optimal_actions = env.get_optimal_actions()
        optimal_names = env.get_optimal_action_names()

        raw_answer = policy_fn(prompt)
        parsed_action = parse_minigrid_action(raw_answer, mode, env)

        parse_failure = parsed_action is None
        is_correct = (not parse_failure) and parsed_action in optimal_actions

        if parse_failure:
            step_logs.append({
                "step": step,
                "raw_answer": raw_answer,
                "parsed_action": None,
                "optimal_actions": sorted(optimal_actions),
                "optimal_action_names": optimal_names,
                "parse_failure": True,
                "is_correct": False,
            })
            break

        state, reward, terminated, truncated, info = env.step_cardinal(parsed_action)

        log = {
            "step": step,
            "mode": mode,
            "raw_answer": raw_answer,
            "parsed_action": parsed_action,
            "optimal_actions": sorted(optimal_actions),
            "optimal_action_names": optimal_names,
            "parse_failure": False,
            "is_correct": is_correct,
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
            "hit_wall": info["hit_wall"],
            "hit_obstacle": info["hit_obstacle"],
            "agent": state["agent"],
            "goal": state["goal"],
            "facing": state["facing_name"],
            "grid_text": env.render_text(),
        }

        step_logs.append(log)

        if verbose:
            print("\n" + "=" * 80)
            print(f"Step {step}")
            print(env.render_text())
            print("Mode:", mode)
            print("Raw answer:", raw_answer)
            print("Parsed action:", parsed_action)
            print("Optimal actions:", optimal_names)
            print("Correct:", is_correct)
            print("Reward:", reward)
            print("Info:", info)

        if terminated:
            reached_goal = True
            break

    env.close()

    return {
        "mode": mode,
        "seed": seed,
        "reached_goal": reached_goal,
        "num_steps": len(step_logs),
        "step_logs": step_logs,
    }


def main():
    for mode in ["allocentric", "egocentric"]:
        print("\n" + "#" * 80)
        print(f"RUNNING MINIGRID EPISODE: {mode.upper()}")
        print("#" * 80)

        result = run_minigrid_episode(
            mode=mode,
            seed=0,
            max_steps=64,
            verbose=True,
        )

        print("\nRESULT")
        print("=" * 80)
        print("Mode:", result["mode"])
        print("Reached goal:", result["reached_goal"])
        print("Steps:", result["num_steps"])


if __name__ == "__main__":
    main()