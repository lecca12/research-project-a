from gazebo_turtlebot.gazebo_adapter import GazeboAdapter


def main():
    adapter = GazeboAdapter()

    try:
        actions = [
            "east",
            "north",
            "west",
            "south",
        ]

        print()
        print("Starting Gazebo adapter test")
        print("=" * 50)

        for action in actions:
            print()
            print(f"Executing: {action}")

            state = adapter.execute(action)

            print(
                f"Pose after {action}: "
                f"x={state['x']:.3f}, "
                f"y={state['y']:.3f}, "
                f"yaw={state['yaw_degrees']:.1f} deg"
            )

        print()
        print("Test complete.")

    finally:
        adapter.close()


if __name__ == "__main__":
    main()