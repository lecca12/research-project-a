from gazebo_turtlebot.gazebo_adapter import GazeboAdapter
from gazebo_turtlebot.gazebo_grid_world import GazeboGridWorld


def main():
    adapter = GazeboAdapter()

    world = GazeboGridWorld(
        adapter=adapter,
        rows=5,
        cols=5,
        start_cell=(4, 0),
        goal_cell=(0, 4),
        obstacle_cells=[],
    )

    try:
        world.initialise_from_current_pose()

        print()
        print("Initial state:")
        print(world.get_state())

        actions = [
            "east",
            "north",
            "east",
            "north",
        ]

        for action in actions:
            print()
            print(
                f"Legal actions: "
                f"{world.get_legal_actions()}"
            )

            print(
                f"Executing: {action}"
            )

            result = world.execute_action(
                action
            )

            print(
                "Result:",
                result,
            )

            print(
                "Discrete state:",
                world.get_state(),
            )

    finally:
        adapter.close()


if __name__ == "__main__":
    main()
