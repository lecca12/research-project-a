import json
from pathlib import Path

from gridworld import SimpleGridWorld
from experiment_utils import generate_fixed_states, run_episode
from llm_policy import make_openai_policy_fn


def make_experiment_conditions():
    return [
        {"grid_size": 5, "obstacle_density": 0.10},
        {"grid_size": 5, "obstacle_density": 0.30},
        {"grid_size": 8, "obstacle_density": 0.10},
        {"grid_size": 8, "obstacle_density": 0.30},
    ]


def main():
    num_states = 10
    start_seed = 0

    model = "gpt-4o-mini"
    temperature = 0.0
    max_output_tokens = 16
    verbose = False

    policy_fn = make_openai_policy_fn(
        model=model,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )

    all_results = []
    conditions = make_experiment_conditions()

    for condition in conditions:
        grid_size = condition["grid_size"]
        obstacle_density = condition["obstacle_density"]
        max_steps = grid_size ** 2

        print("\n" + "=" * 80)
        print(
            f"Running condition: grid={grid_size}x{grid_size}, "
            f"density={obstacle_density}, states={num_states}, max_steps={max_steps}"
        )
        print("=" * 80)

        fixed_states = generate_fixed_states(
            size=grid_size,
            obstacle_density=obstacle_density,
            num_states=num_states,
            max_steps=max_steps,
            start_seed=start_seed,
        )

        for state_index, state in enumerate(fixed_states):
            seed = state.get("seed")
            print(f"State {state_index + 1}/{num_states}, seed={seed}")

            for mode in ["allocentric", "egocentric"]:
                env = SimpleGridWorld(
                    size=grid_size,
                    obstacle_density=obstacle_density,
                    max_steps=max_steps,
                )

                result = run_episode(
                    env=env,
                    mode=mode,
                    policy_fn=policy_fn,
                    fixed_state=state,
                    verbose=verbose,
                )

                result["grid_size"] = grid_size
                result["obstacle_density"] = obstacle_density
                result["max_steps"] = max_steps
                result["model"] = model
                result["temperature"] = temperature
                result["max_output_tokens"] = max_output_tokens

                all_results.append(result)

    with open("main_results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    metadata = {
        "num_states": num_states,
        "start_seed": start_seed,
        "max_steps_rule": "grid_size_squared",
        "model": model,
        "temperature": temperature,
        "max_output_tokens": max_output_tokens,
        "conditions": conditions,
        "modes": ["allocentric", "egocentric"],
        "output_file": "main_results.json",
    }

    with open("main_results_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("\nSaved:")
    print("- main_results.json")
    print("- main_results_metadata.json")


if __name__ == "__main__":
    main()