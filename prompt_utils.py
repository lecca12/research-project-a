def obstacle_list_from_grid(obstacles):
    coords = []
    rows, cols = obstacles.shape
    for r in range(rows):
        for c in range(cols):
            if obstacles[r, c] == 1:
                coords.append((r, c))
    return coords


def _shared_grid_description(obs):
    agent_r, agent_c = obs["agent"]
    goal_r, goal_c = obs["goal"]
    obstacles = obs["obstacles"]

    obstacle_coords = obstacle_list_from_grid(obstacles)
    obstacle_text = ", ".join([f"({r},{c})" for r, c in obstacle_coords])
    if not obstacle_text:
        obstacle_text = "none"

    coordinate_text = "\n".join(
        [
            "Coordinate system:",
            "- Row 0 is the top (north), and row numbers increase downward (south).",
            "- Column 0 is the left (west), and column numbers increase to the right (east).",
        ]
    )

    return {
        "grid_size": f"{obstacles.shape[0]} x {obstacles.shape[1]}",
        "agent": (int(agent_r), int(agent_c)),
        "goal": (int(goal_r), int(goal_c)),
        "obstacles_text": obstacle_text,
        "coordinate_text": coordinate_text,
    }


def make_allocentric_prompt(obs):
    shared = _shared_grid_description(obs)
    agent_r, agent_c = shared["agent"]
    goal_r, goal_c = shared["goal"]

    prompt = f"""
You are navigating a grid world.

Grid size: {shared['grid_size']}
Agent position: ({agent_r},{agent_c})
Goal position: ({goal_r},{goal_c})
Obstacle cells: {shared['obstacles_text']}

{shared['coordinate_text']}

Choose exactly one action from:
north, east, south, west

Rules:
- Do not move into an obstacle.
- Do not move outside the grid.
- Choose the best next move that is part of a shortest valid path to the goal.

Answer with one word only:
north, east, south, or west
""".strip()

    return prompt


def make_egocentric_prompt(obs):
    shared = _shared_grid_description(obs)
    agent_r, agent_c = shared["agent"]
    goal_r, goal_c = shared["goal"]
    facing = obs["facing"]

    facing_names = {0: "north", 1: "east", 2: "south", 3: "west"}

    prompt = f"""
You are navigating a grid world.

Grid size: {shared['grid_size']}
Your current position is ({agent_r},{agent_c})
The goal is at ({goal_r},{goal_c})
You are currently facing {facing_names[facing]}
Obstacle cells: {shared['obstacles_text']}

{shared['coordinate_text']}

Relative action meanings:
- forward = keep moving in the direction you are currently facing
- right = turn/move 90 degrees to your right
- backward = move in the opposite direction
- left = turn/move 90 degrees to your left

Choose exactly one action from:
forward, right, backward, left

Rules:
- Do not move into an obstacle.
- Do not move outside the grid.
- Choose the best next move that is part of a shortest valid path to the goal.

Answer with one word only:
forward, right, backward, or left
""".strip()

    return prompt
