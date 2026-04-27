import json

from gridworld import SimpleGridWorld
from experiment_utils import generate_fixed_states, run_episode
from llm_policy import make_openai_policy_fn


def main():
    # small pilot setup
    size = 5
    obstacle_density = 0.10
    num_states = 3
    max_steps = 20  # keep short for debugging

    policy_fn = make_openai_policy_fn(model="gpt-4o-mini")

    # generate fixed states so both modes use identical maps
    fixed_states = generate_fixed_states(
        size=size,
        obstacle_density=obstacle_density,
        num_states=num_states,
        max_steps=max_steps,
        start_seed=0,
    )

    results = []

    for state in fixed_states:
        for mode in ["allocentric", "egocentric"]:
            env = SimpleGridWorld(
                size=size,
                obstacle_density=obstacle_density,
                max_steps=max_steps,
            )

            episode_result = run_episode(
                env=env,
                mode=mode,
                policy_fn=policy_fn,
                fixed_state=state,
                verbose=True,  # important for debugging
            )

            # add metadata
            episode_result["grid_size"] = size
            episode_result["obstacle_density"] = obstacle_density
            episode_result["model"] = "gpt-4o-mini"

            results.append(episode_result)

    # save results
    with open("pilot_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\nSaved pilot results to pilot_results.json")


if __name__ == "__main__":
    main()