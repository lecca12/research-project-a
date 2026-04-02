from gridworld import SimpleGridWorld
from experiment_utils import run_episode, generate_fixed_states


def always_north_policy(prompt):
    return "north"


def always_forward_policy(prompt):
    return "forward"


fixed_states = generate_fixed_states(num_states=1, size=5, obstacle_density=0.2, seed_start=123)
state = fixed_states[0]

print("=== Allocentric test (fixed state) ===")
env = SimpleGridWorld(size=5, obstacle_density=0.2, ensure_path=True)
result = run_episode(
    env=env,
    mode="allocentric",
    policy_fn=always_north_policy,
    verbose=True,
    fixed_state=state,
)

print("\nEpisode summary:")
print("Reached goal:", result["reached_goal"])
print("Steps:", result["num_steps"])
print("Seed:", result["seed"])

print("\n=== Egocentric test (same fixed state) ===")
env = SimpleGridWorld(size=5, obstacle_density=0.2, ensure_path=True)
result = run_episode(
    env=env,
    mode="egocentric",
    policy_fn=always_forward_policy,
    verbose=True,
    fixed_state=state,
)

print("\nEpisode summary:")
print("Reached goal:", result["reached_goal"])
print("Steps:", result["num_steps"])
print("Seed:", result["seed"])
