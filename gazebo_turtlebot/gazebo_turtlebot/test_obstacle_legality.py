from gazebo_turtlebot.gazebo_adapter import GazeboAdapter
from gazebo_turtlebot.gazebo_grid_world import GazeboGridWorld


def print_state(world, label):
    state = world.get_state()

    print()
    print(label)
    print("-" * 50)
    print("Agent cell:", state["agent"])
    print("Goal cell:", state["goal"])
    print("Facing:", state["facing"])
    print(
        "Pose:",
        f"x={state['pose']['x']:.3f}, "
        f"y={state['pose']['y']:.3f}, "
        f"yaw={state['pose']['yaw_degrees']:.1f} deg",
    )
    print("Legal actions:", world.get_legal_actions())


def main():
    adapter = GazeboAdapter()

    # Simple test layout:
    #
    # row 0: . . . . G
    # row 1: . . . . .
    # row 2: . . # . .
    # row 3: . . # . .
    # row 4: A . . . .
    #
    # Start at (4, 0).
    #
    # We move:
    # east  -> (4, 1)
    # north -> (3, 1)
    #
    # Then east would attempt to enter obstacle (3, 2)
    # and MUST be blocked without moving the robot.

    world = GazeboGridWorld(
        adapter=adapter,
        rows=5,
        cols=5,
        start_cell=(4, 0),
        goal_cell=(0, 4),
        obstacle_cells=[
            (3, 2),
            (2, 2),
        ],
    )

    try:
        world.initialise_from_current_pose()

        print_state(
            world,
            "INITIAL STATE",
        )

        print()
        print("=" * 70)
        print("1. Executing legal action: east")
        print("=" * 70)

        result = world.execute_action(
            "east"
        )

        print("Result:", result)

        print_state(
            world,
            "AFTER EAST",
        )

        print()
        print("=" * 70)
        print("2. Executing legal action: north")
        print("=" * 70)

        result = world.execute_action(
            "north"
        )

        print("Result:", result)

        print_state(
            world,
            "AFTER NORTH",
        )

        before_blocked = world.get_state()

        print()
        print("=" * 70)
        print(
            "3. Attempting ILLEGAL action: east "
            "into obstacle at (3, 2)"
        )
        print("=" * 70)

        result = world.execute_action(
            "east"
        )

        after_blocked = world.get_state()

        print()
        print("Blocked result:")
        print(result)

        print_state(
            world,
            "AFTER BLOCKED EAST",
        )

        print()
        print("=" * 70)
        print("VALIDATION")
        print("=" * 70)

        assert result["executed"] is False
        assert result["blocked"] is True
        assert result["blocked_type"] == "obstacle"

        assert before_blocked["agent"] == (
            3,
            1,
        )

        assert after_blocked["agent"] == (
            3,
            1,
        )

        print(
            "PASS: obstacle action was blocked "
            "before physical execution."
        )

        print(
            "PASS: TurtleBot remained in cell (3, 1)."
        )

        print()
        print("=" * 70)
        print(
            "4. Attempting boundary action: west "
            "after returning to start-side column"
        )
        print("=" * 70)

        # South back to (4,1), then west to (4,0)
        world.execute_action(
            "south"
        )

        world.execute_action(
            "west"
        )

        before_boundary = world.get_state()

        result = world.execute_action(
            "west"
        )

        after_boundary = world.get_state()

        print()
        print("Boundary result:")
        print(result)

        assert result["executed"] is False
        assert result["blocked"] is True
        assert result["blocked_type"] == "boundary"

        assert before_boundary["agent"] == (
            4,
            0,
        )

        assert after_boundary["agent"] == (
            4,
            0,
        )

        print(
            "PASS: boundary action was blocked "
            "before physical execution."
        )

        print()
        print("All legality tests passed.")

    finally:
        adapter.close()


if __name__ == "__main__":
    main()