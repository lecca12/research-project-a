"""
Small pilot runner for manual sanity checks with GPT-4o-mini.

Runs the same fixed grid states in both allocentric and egocentric modes,
so you can compare the model on identical maps.
"""

import json

from experiment_utils import generate_fixed_states, run_episode
from gridworld import SimpleGridWorld
from llm_policy import make_openai_policy_fn


def main():
    policy_fn = make_openai_policy_fn(model="gpt-4o-mini")

    fixed_states = generate_fixed_states(
        size=5,
        obstacle_density=0.10,
        num_states=3,
        max_steps=20,
        ensure_path=True,
        start_seed=0,
    )

    all_results = []
    for state in fixed_states:
        for mode in ["allocentric", "egocentric"]:
            env = SimpleGridWorld(
                size=5,
                obstacle_density=0.10,
                max_steps=20,
                ensure_path=True,
            )
            result = run_episode(
                env=env,
                mode=mode,
                policy_fn=policy_fn,
                fixed_state=state,
                verbose=True,
            )
            all_results.append(result)

    with open("pilot_results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print("Saved pilot results to pilot_results.json")


if __name__ == "__main__":
    main()
