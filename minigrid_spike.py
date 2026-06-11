import gymnasium as gym
import minigrid


DIRECTION_NAMES = {
    0: "east",
    1: "south",
    2: "west",
    3: "north",
}


def describe_state(env):
    grid = env.unwrapped.grid
    width = grid.width
    height = grid.height

    agent_pos = tuple(env.unwrapped.agent_pos)
    agent_dir = env.unwrapped.agent_dir
    goal_pos = None
    walls = []

    for y in range(height):
        for x in range(width):
            cell = grid.get(x, y)
            if cell is None:
                continue

            if cell.type == "goal":
                goal_pos = (y, x)
            elif cell.type == "wall":
                walls.append((y, x))

    agent_rc = (agent_pos[1], agent_pos[0])
    facing = DIRECTION_NAMES[agent_dir]

    print("\nFULL GRID")
    print("=" * 60)

    for y in range(height):
        row = []
        for x in range(width):
            if (x, y) == agent_pos:
                row.append("A")
                continue

            cell = grid.get(x, y)
            if cell is None:
                row.append(".")
            elif cell.type == "wall":
                row.append("X")
            elif cell.type == "goal":
                row.append("G")
            else:
                row.append("?")
        print(" ".join(row))

    print("\nALLOCENTRIC DESCRIPTION")
    print("=" * 60)
    print(f"Grid size: {height} x {width}")
    print(f"Agent position: {agent_rc}")
    print(f"Goal position: {goal_pos}")
    print(f"Wall cells: {walls}")
    print("Allowed actions: north, east, south, west")

    print("\nEGOCENTRIC DESCRIPTION")
    print("=" * 60)
    print(f"Grid size: {height} x {width}")
    print(f"Agent position: {agent_rc}")
    print(f"Goal position: {goal_pos}")
    print(f"Wall cells: {walls}")
    print(f"The agent is currently facing {facing}.")
    print("Allowed actions: forward, right, backward, left")


def main():
    env = gym.make("MiniGrid-Empty-8x8-v0", render_mode="rgb_array")
    obs, info = env.reset(seed=0)

    describe_state(env)

    env.close()


if __name__ == "__main__":
    main()