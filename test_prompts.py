from gridworld import SimpleGridWorld
from prompt_utils import make_allocentric_prompt, make_egocentric_prompt

env = SimpleGridWorld(size=5, obstacle_density=0.2, ensure_path=True)

obs, info = env.reset()

env.render()

print("Optimal absolute action:", env.get_optimal_action_name())
print("Optimal relative action:", env.get_optimal_relative_action_name())
print()

print("=== Allocentric Prompt ===")
print(make_allocentric_prompt(obs))
print()

print("=== Egocentric Prompt ===")
print(make_egocentric_prompt(obs))
print()
