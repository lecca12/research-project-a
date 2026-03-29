from gridworld import SimpleGridWorld
from experiment_utils import run_episode


def always_north_policy(prompt):
    return "north"


def always_forward_policy(prompt):
    return "forward"


print("=== Allocentric test ===")
env = SimpleGridWorld(size=5, obstacle_density=0.2, ensure_path=True)
result = run_episode(
    env=env,
    mode="allocentric",
    policy_fn=always_north_policy,
    verbose=True,
)

print("\nEpisode summary:")
print("Reached goal:", result["reached_goal"])
print("Steps:", result["num_steps"])

print("\n=== Egocentric test ===")
env = SimpleGridWorld(size=5, obstacle_density=0.2, ensure_path=True)
result = run_episode(
    env=env,
    mode="egocentric",
    policy_fn=always_forward_policy,
    verbose=True,
)

print("\nEpisode summary:")
print("Reached goal:", result["reached_goal"])
print("Steps:", result["num_steps"])
