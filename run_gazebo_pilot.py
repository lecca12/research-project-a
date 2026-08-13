"""
Run the legality-shield arm on the TurtleBot3 in Gazebo.

Copy this file to the repository root, next to llm_policy.py.

Usage:

    source /opt/ros/humble/setup.bash
    export PYTHONPATH="$PYTHONPATH:$HOME/research-project-a:$HOME/research-project-a/gazebo_turtlebot"
    export OPENROUTER_API_KEY="..."
    python3 run_gazebo_pilot.py --episodes 1

The model interface is the same one the text experiment uses. The prompts, the
parser, the shield and the two-reprompt budget all come from existing code. The
only thing that changes is that a step drives the robot to the next cell instead
of updating a text grid.
"""

import argparse
import json
import subprocess
import time
from pathlib import Path

from gazebo_turtlebot.gazebo_adapter import GazeboAdapter
from gazebo_turtlebot.gazebo_grid_env import GazeboGridEnv, load_layout
from gazebo_turtlebot import gazebo_policy
from llm_policy import make_openai_policy_fn


# Matches run_farama_minigrid_experiment.py. These values are part of the
# interface being transferred, so do not change them for the Gazebo run.
SHORT_TEMPERATURE = 0.0
SHORT_MAX_OUTPUT_TOKENS = 16
MAX_REPROMPTS = 2
EARLY_STOP_REPEATS = 3


def classify_error(parse_failure, hit_wall, hit_obstacle, is_correct):
    """
    Copied from run_farama_minigrid_experiment.py.

    It is copied rather than imported because importing that module pulls in
    gymnasium and minigrid, which are not installed on the robot machine.
    """
    if parse_failure:
        return "parse_failure"

    if hit_obstacle:
        return "obstacle_blindness"

    if hit_wall:
        return "boundary_error"

    if not is_correct:
        return "directional_or_detour_error"

    return "correct"


def update_consecutive_repeat_state(
    previous_key,
    previous_streak,
    current_key,
    threshold,
):
    """
    Copied from run_farama_minigrid_experiment.py.

    Stop only after consecutive repeats of the same state-action key.
    """
    if threshold is None:
        return False, current_key, 1

    if current_key == previous_key:
        current_streak = previous_streak + 1
    else:
        current_streak = 1

    return (
        current_streak >= threshold,
        current_key,
        current_streak,
    )


class _StaticAdapter:
    """
    Stand-in adapter used by the startup check. Touches no hardware.
    """

    def get_state(self):
        return {"x": 0.0, "y": 0.0, "yaw": 0.0, "yaw_degrees": 0.0}


def check_legal_action_ordering(layout, cell_size):
    """
    Fail early if the shield would list legal actions in the wrong order.

    The text experiment sorts legal actions by cardinal action number, giving
    north, east, south, west. Sorting the names as strings instead gives
    east, north, south, west, which changes the text of the shield reprompt.
    See the spec for the two-line fix in gazebo_policy.py.
    """
    # Checked on a throwaway environment, never on the live one. Moving the
    # live robot's cell around to probe it would leave the episode starting
    # from the wrong cell if anything raised part way through.
    probe = GazeboGridEnv(adapter=_StaticAdapter(), layout=layout, cell_size=cell_size)

    for row in range(probe.rows):
        for col in range(probe.cols):
            if probe.is_blocked(row, col)[0]:
                continue

            probe.cell = (row, col)

            from_policy = gazebo_policy.legal_action_names(probe, "allocentric")
            expected = probe.get_legal_action_names_in_project_order()

            if from_policy != expected:
                raise RuntimeError(
                    f"At cell {probe.cell}, gazebo_policy.legal_action_names "
                    f"returns {from_policy}, but the text experiment produces "
                    f"{expected}. The shield reprompt would not match. Apply "
                    "the ordering fix in gazebo_policy.py described in the "
                    "spec before running."
                )


def reset_simulation():
    """
    Return the robot to its spawn pose between episodes.
    """
    # TODO: confirm this service name on your machine with `ros2 service list`.
    # Gazebo Classic normally exposes /reset_simulation via gazebo_ros.
    subprocess.run(
        [
            "ros2",
            "service",
            "call",
            "/reset_simulation",
            "std_srvs/srv/Empty",
            "{}",
        ],
        check=True,
    )

    # Give the odometry a moment to republish from zero.
    time.sleep(2.0)


def run_episode(env, mode, policy_type, policy_fn, max_steps, verbose=True):
    """
    One episode. Mirrors run_episode in run_farama_minigrid_experiment.py.
    """
    step_logs = []
    reached_goal = False
    early_stopped = False

    previous_repeat_key = None
    consecutive_repeat_streak = 0

    for step in range(max_steps):
        prompt = gazebo_policy.get_prompt(env, mode)

        optimal_action_names = env.get_optimal_action_names()
        optimal_relative_action_names = env.get_optimal_relative_action_names()

        grid_text = env.render_text()
        state_before = env.get_state()

        parsed_action, action_metadata = gazebo_policy.choose_action(
            env=env,
            mode=mode,
            policy_type=policy_type,
            prompt=prompt,
            policy_fn=policy_fn,
            max_reprompts=MAX_REPROMPTS,
        )

        parse_failure = parsed_action is None
        is_correct = not parse_failure and parsed_action in optimal_action_names

        log = {
            "step": step,
            "policy_type": policy_type,
            "mode": mode,
            "agent_pos": list(state_before["agent"]),
            "goal_pos": list(state_before["goal"]),
            "facing_name": state_before["facing_name"],
            "raw_model_answer": action_metadata["raw_model_answer"],
            "all_raw_answers": action_metadata["all_raw_answers"],
            "parsed_action_name": parsed_action,
            "optimal_action_names": optimal_action_names,
            "optimal_relative_action_names": optimal_relative_action_names,
            "legal_action_names": action_metadata["legal_action_names"],
            "shield_used": action_metadata["shield_used"],
            "shield_reprompts": action_metadata["shield_reprompts"],
            "is_correct": is_correct,
            "parse_failure": parse_failure,
            "grid_text": grid_text,
            "prompt": prompt,
        }

        if parse_failure:
            # The shield exhausted its retries. No illegal action is executed
            # and the episode ends, exactly as in the text experiment.
            log["error_type"] = "parse_failure"
            log["hit_wall"] = False
            log["hit_obstacle"] = False
            step_logs.append(log)
            break

        action_index = {
            "north": 0,
            "east": 1,
            "south": 2,
            "west": 3,
        }[parsed_action]

        _, reward, terminated, truncated, info = env.step_cardinal(action_index)

        log["error_type"] = classify_error(
            parse_failure=False,
            hit_wall=info["hit_wall"],
            hit_obstacle=info["hit_obstacle"],
            is_correct=is_correct,
        )
        log["hit_wall"] = info["hit_wall"]
        log["hit_obstacle"] = info["hit_obstacle"]
        log["blocked_type"] = info["blocked_type"]
        log["reward"] = reward
        log["step_error_metres"] = info["execution"].get("step_error_metres")
        log["cumulative_offset_metres"] = info["execution"].get(
            "cumulative_offset_metres"
        )
        log["next_agent_pos"] = list(env.cell)

        repeat_key = (
            tuple(state_before["agent"]),
            state_before["facing_name"],
            parsed_action,
        )

        (
            stop_now,
            previous_repeat_key,
            consecutive_repeat_streak,
        ) = update_consecutive_repeat_state(
            previous_key=previous_repeat_key,
            previous_streak=consecutive_repeat_streak,
            current_key=repeat_key,
            threshold=EARLY_STOP_REPEATS,
        )

        log["consecutive_repeat_streak"] = consecutive_repeat_streak
        step_logs.append(log)

        if verbose:
            print(grid_text)
            print(
                f"step {step}: answer={log['raw_model_answer']!r} "
                f"action={parsed_action} optimal={optimal_action_names} "
                f"correct={is_correct} reprompts={log['shield_reprompts']}"
            )

        if terminated:
            reached_goal = True
            break

        if stop_now:
            early_stopped = True
            break

    return {
        "env_name": env.layout["env_name"],
        "seed": env.layout["seed"],
        "policy_type": policy_type,
        "mode": mode,
        "num_steps": len(step_logs),
        "reached_goal": reached_goal,
        "early_stopped": early_stopped,
        "optimal_path_length": env.layout["optimal_path_length"],
        "final_cell": list(env.cell),
        "max_step_error_metres": round(env.max_step_error_metres, 4),
        "step_logs": step_logs,
    }


def save_trace(result, output_dir):
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)

    name = (
        f"gazebo_{result['policy_type']}_{result['mode']}_"
        f"seed{result['seed']}_episode{result['episode']}.txt"
    )

    with (directory / name).open("w", encoding="utf-8") as file:
        file.write(f"Environment: {result['env_name']}\n")
        file.write(f"Policy: {result['policy_type']}\n")
        file.write(f"Mode: {result['mode']}\n")
        file.write(f"Reached goal: {result['reached_goal']}\n")
        file.write(f"Steps: {result['num_steps']}\n")
        file.write(f"Optimal: {result['optimal_path_length']}\n")
        file.write("=" * 80 + "\n\n")

        for log in result["step_logs"]:
            file.write(f"STEP {log['step']}\n")
            file.write("-" * 80 + "\n")
            file.write(log["grid_text"] + "\n\n")
            file.write(f"Raw answer: {log.get('raw_model_answer')}\n")
            file.write(f"All raw answers: {log.get('all_raw_answers')}\n")
            file.write(f"Parsed action: {log.get('parsed_action_name')}\n")
            file.write(f"Optimal actions: {log.get('optimal_action_names')}\n")
            file.write(f"Legal actions: {log.get('legal_action_names')}\n")
            file.write(f"Correct: {log.get('is_correct')}\n")
            file.write(f"Error type: {log.get('error_type')}\n")
            file.write(f"Shield reprompts: {log.get('shield_reprompts')}\n")
            file.write(f"Step error (m): {log.get('step_error_metres')}\n")
            file.write("\n" + "=" * 80 + "\n\n")


def build_argument_parser():
    parser = argparse.ArgumentParser(
        description="Run the legality-shield arm on the TurtleBot3 in Gazebo."
    )
    parser.add_argument(
        "--layout",
        default="layout_simplecrossing_s9n1_seed0.json",
        help="Frozen MiniGrid layout to mirror.",
    )
    parser.add_argument(
        "--mode",
        default="allocentric",
        choices=["allocentric", "egocentric"],
    )
    parser.add_argument(
        "--arm",
        default="legality_shield",
        choices=["legality_shield", "reprompt_control"],
    )
    parser.add_argument(
        "--model",
        default="openai/gpt-4o-mini",
        help="OpenRouter model ID. Use the same one as the text run.",
    )
    parser.add_argument("--provider", default=None)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=81)
    parser.add_argument("--cell-size", type=float, default=0.50)
    parser.add_argument("--output", default="gazebo_pilot_results.json")
    return parser


def main():
    args = build_argument_parser().parse_args()

    layout = load_layout(args.layout)

    policy_fn = make_openai_policy_fn(
        model=args.model,
        temperature=SHORT_TEMPERATURE,
        max_output_tokens=SHORT_MAX_OUTPUT_TOKENS,
        provider=args.provider,
    )

    output_path = Path(args.output)
    trace_directory = f"{output_path.stem}_traces"

    metadata = {
        "run_type": "gazebo_transfer_pilot",
        "env_name": layout["env_name"],
        "seed": layout["seed"],
        "mode": args.mode,
        "arm": args.arm,
        "model": args.model,
        "provider_pin": args.provider,
        "short_temperature": SHORT_TEMPERATURE,
        "short_max_output_tokens": SHORT_MAX_OUTPUT_TOKENS,
        "max_reprompts": MAX_REPROMPTS,
        "early_stop_repeats": EARLY_STOP_REPEATS,
        "cell_size_metres": args.cell_size,
        "episodes": args.episodes,
        "optimal_path_length": layout["optimal_path_length"],
    }

    with Path(f"{output_path.stem}_metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    adapter = GazeboAdapter()
    results = []

    try:
        for episode in range(args.episodes):
            if episode > 0:
                reset_simulation()

            env = GazeboGridEnv(
                adapter=adapter,
                layout=layout,
                cell_size=args.cell_size,
            )
            env.initialise_from_current_pose()

            check_legal_action_ordering(layout, args.cell_size)

            print("=" * 80)
            print(f"EPISODE {episode}: {args.arm}, {args.mode}, {args.model}")
            print("=" * 80)

            result = run_episode(
                env=env,
                mode=args.mode,
                policy_type=args.arm,
                policy_fn=policy_fn,
                max_steps=args.max_steps,
            )

            result["episode"] = episode
            result["model"] = args.model
            results.append(result)

            # Written after every episode so a later failure cannot discard
            # an episode that already finished.
            with output_path.open("w", encoding="utf-8") as file:
                json.dump(results, file, indent=2)

            save_trace(result, trace_directory)

            print(
                f"episode {episode}: reached_goal={result['reached_goal']} "
                f"steps={result['num_steps']} "
                f"optimal={result['optimal_path_length']} "
                f"max_step_error={result['max_step_error_metres']} m"
            )

    finally:
        adapter.close()

    print(f"\nSaved {output_path} and {trace_directory}/")


if __name__ == "__main__":
    main()
