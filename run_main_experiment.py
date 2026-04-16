import json
from pathlib import Path

from gridworld import SimpleGridWorld
from experiment_utils import run_episode
from llm_policy import make_openai_policy_fn


def make_experiment_conditions():
    return [
        {"grid_size": 5, "obstacle_density": 0.10, "mode": "allocentric"},
        {"grid_size": 5, "obstacle_density": 0.10, "mode": "egocentric"},
        {"grid_size": 5, "obstacle_density": 0.30, "mode": "allocentric"},
        {"grid_size": 5, "obstacle_density": 0.30, "mode": "egocentric"},
        {"grid_size": 8, "obstacle_density": 0.10, "mode": "allocentric"},
        {"grid_size": 8, "obstacle_density": 0.10, "mode": "egocentric"},
        {"grid_size": 8, "obstacle_density": 0.30, "mode": "allocentric"},
        {"grid_size": 8, "obstacle_density": 0.30, "mode": "egocentric"},
    ]


def run_condition(
    *,
    grid_size,
    obstacle_density,
    mode,
    seeds,
    model="gpt-4o-mini",
    temperature=0.0,
    max_output_tokens=16,
    verbose=False,
):
    env = SimpleGridWorld(
        size=grid_size,
        obstacle_density=obstacle_density,
        ensure_path=True,
    )

    policy_fn = make_openai_policy_fn(
        model=model,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )

    results = []

    print("\n" + "=" * 100)
    print(
        f"Running condition: size={grid_size}, density={obstacle_density}, "
        f"mode={mode}, episodes={len(seeds)}"
    )
    print("=" * 100)

    for idx, seed in enumerate(seeds, start=1):
        print(f"[{idx}/{len(seeds)}] seed={seed}")

        episode_result = run_episode(
            env=env,
            mode=mode,
            policy_fn=policy_fn,
            seed=seed,
            verbose=verbose,
        )

        episode_result["grid_size"] = grid_size
        episode_result["obstacle_density"] = obstacle_density
        episode_result["model"] = model
        episode_result["temperature"] = temperature
        episode_result["max_output_tokens"] = max_output_tokens

        results.append(episode_result)

    return results


def main():
    # Adjust this first for your next run.
    num_seeds = 10
    seeds = list(range(num_seeds))

    model = "gpt-4o-mini"
    temperature = 0.0
    max_output_tokens = 16
    verbose = False

    conditions = make_experiment_conditions()
    all_results = []

    for condition in conditions:
        condition_results = run_condition(
            grid_size=condition["grid_size"],
            obstacle_density=condition["obstacle_density"],
            mode=condition["mode"],
            seeds=seeds,
            model=model,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            verbose=verbose,
        )
        all_results.extend(condition_results)

    output_path = Path("main_results.json")
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    metadata_path = Path("main_results_metadata.json")
    metadata = {
        "num_seeds": num_seeds,
        "seeds": seeds,
        "model": model,
        "temperature": temperature,
        "max_output_tokens": max_output_tokens,
        "conditions": conditions,
        "output_file": str(output_path),
    }
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("\nSaved files:")
    print(f"- {output_path.resolve()}")
    print(f"- {metadata_path.resolve()}")
    print("\nDone.")


if __name__ == "__main__":
    main()
