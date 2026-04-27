def make_allocentric_prompt(obs):
    agent = tuple(obs["agent"])
    goal = tuple(obs["goal"])
    obstacles = obs["obstacles"]

    obstacle_cells = []
    size = obstacles.shape[0]
    for r in range(size):
        for c in range(size):
            if obstacles[r, c] == 1:
                obstacle_cells.append((r, c))

    prompt = f"""You are navigating a grid world.

Grid size: {size} x {size}
Agent position: {agent}
Goal position: {goal}
Obstacle cells: {', '.join(map(str, obstacle_cells)) if obstacle_cells else 'None'}

Coordinate system:
- Row 0 is the top (north), and row numbers increase downward (south).
- Column 0 is the left (west), and column numbers increase to the right (east).

Choose exactly one action from:
north, east, south, west

Rules:
- Do not move into an obstacle.
- Do not move outside the grid.
- Choose the best next move on a shortest valid path to the goal.

Answer with one word only:
north, east, south, or west
"""
    return prompt


def make_egocentric_prompt(obs):
    agent = tuple(obs["agent"])
    goal = tuple(obs["goal"])
    obstacles = obs["obstacles"]
    facing = obs["facing"]

    size = obstacles.shape[0]

    facing_names = ["north", "east", "south", "west"]
    facing_name = facing_names[facing]

    obstacle_cells = []
    for r in range(size):
        for c in range(size):
            if obstacles[r, c] == 1:
                obstacle_cells.append((r, c))

    prompt = f"""You are navigating a grid world.

Grid size: {size} x {size}
Agent position: {agent}
Goal position: {goal}
Obstacle cells: {', '.join(map(str, obstacle_cells)) if obstacle_cells else 'None'}

The agent is currently facing {facing_name}.

Relative actions:
- forward: move in the direction you are facing
- right: turn right relative to your current facing and move
- backward: turn around and move
- left: turn left relative to your current facing and move

Choose exactly one action from:
forward, right, backward, left

Rules:
- Do not move into an obstacle.
- Do not move outside the grid.
- Choose the best next move on a shortest valid path to the goal.

Answer with one word only:
forward, right, backward, or left
"""
    return prompt