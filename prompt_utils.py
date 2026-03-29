def obstacle_list_from_grid(obstacles):
    coords = []
    rows, cols = obstacles.shape
    for r in range(rows):
        for c in range(cols):
            if obstacles[r, c] == 1:
                coords.append((r, c))
    return coords


def make_allocentric_prompt(obs):
    """
    Allocentric prompt:
    Uses absolute directions: north, east, south, west
    """
    agent_r, agent_c = obs["agent"]
    goal_r, goal_c = obs["goal"]
    obstacles = obs["obstacles"]

    obstacle_coords = obstacle_list_from_grid(obstacles)
    obstacle_text = ", ".join([f"({r},{c})" for r, c in obstacle_coords])
    if not obstacle_text:
        obstacle_text = "none"

    prompt = f"""
You are navigating a grid world.

Grid size: {obstacles.shape[0]} x {obstacles.shape[1]}
Agent position: ({agent_r},{agent_c})
Goal position: ({goal_r},{goal_c})
Obstacle cells: {obstacle_text}

Choose exactly one action from:
north, east, south, west

Rules:
- Do not move into an obstacle.
- Do not move outside the grid.
- Choose the best next move to get closer to the goal.

Answer with one word only:
north, east, south, or west
""".strip()

    return prompt


def make_egocentric_prompt(obs):
    """
    Egocentric prompt:
    Uses relative directions: forward, right, backward, left
    """
    agent_r, agent_c = obs["agent"]
    goal_r, goal_c = obs["goal"]
    obstacles = obs["obstacles"]
    facing = obs["facing"]

    facing_names = {
        0: "north",
        1: "east",
        2: "south",
        3: "west",
    }

    obstacle_coords = obstacle_list_from_grid(obstacles)
    obstacle_text = ", ".join([f"({r},{c})" for r, c in obstacle_coords])
    if not obstacle_text:
        obstacle_text = "none"

    prompt = f"""
You are navigating a grid world.

Grid size: {obstacles.shape[0]} x {obstacles.shape[1]}
Your current position is ({agent_r},{agent_c})
The goal is at ({goal_r},{goal_c})
You are currently facing {facing_names[facing]}
Obstacle cells: {obstacle_text}

Choose exactly one action from:
forward, right, backward, left

Rules:
- Do not move into an obstacle.
- Do not move outside the grid.
- Choose the best next move to get closer to the goal.

Answer with one word only:
forward, right, backward, or left
""".strip()

    return prompt
