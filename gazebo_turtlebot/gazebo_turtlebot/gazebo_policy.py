from experiment_utils import normalize_answer


ABSOLUTE_ACTIONS = {
    "north": "north",
    "east": "east",
    "south": "south",
    "west": "west",
}

RELATIVE_ACTIONS = {
    "forward": 0,
    "right": 1,
    "backward": 2,
    "left": 3,
}


def parse_action(text, mode, env):
    """
    Parse one-word model responses for the Gazebo experiment.

    Allocentric responses become cardinal action strings.
    Egocentric responses are converted to cardinal actions using the
    robot's current facing direction.
    """
    word = normalize_answer(text)

    if mode == "allocentric":
        return ABSOLUTE_ACTIONS.get(word)

    if mode == "egocentric":
        relative_action = RELATIVE_ACTIONS.get(word)

        if relative_action is None:
            return None

        return env.relative_to_cardinal(
            relative_action
        )

    raise ValueError(
        f"Unknown mode: {mode}"
    )


def get_prompt(env, mode):
    if mode == "allocentric":
        return (
            env.make_allocentric_description()
        )

    if mode == "egocentric":
        return (
            env.make_egocentric_description()
        )

    raise ValueError(
        f"Unknown mode: {mode}"
    )


def get_legal_cardinal_actions(env):
    """
    Return the legal cardinal actions at the robot's current grid cell.
    """
    return set(
        env.get_legal_actions()
    )


CARDINAL_ORDER = [
    "north",
    "east",
    "south",
    "west",
]


def legal_action_names(env, mode):
    legal_cardinal = (
        get_legal_cardinal_actions(env)
    )

    ordered = [
        name
        for name in CARDINAL_ORDER
        if name in legal_cardinal
    ]

    if mode == "allocentric":
        return ordered

    if mode == "egocentric":
        return [
            env.cardinal_to_relative_name(name)
            for name in ordered
        ]

    raise ValueError(
        f"Unknown mode: {mode}"
    )


def make_legality_reprompt(
    original_prompt,
    raw_answer,
    legal_names,
):
    """
    Exact legality-shield retry structure used in the Farama experiment.

    The legal-action set is explicitly revealed.
    """
    legal_text = ", ".join(
        legal_names
    )

    return f"""{original_prompt}

Your previous answer was: {raw_answer}

That proposed action is illegal from the current state because it would move
into an obstacle, the outer boundary, or outside the grid.

The illegal action was not executed.

Choose again using only one of the currently legal actions:

{legal_text}

Answer with one word only from:
{legal_text}
"""


def make_reprompt_control_prompt(
    original_prompt,
    raw_answer,
):
    """
    Matched retry control.

    Same retry budget and illegal-action blocking as the legality shield,
    but the legal-action list is deliberately withheld.
    """
    return f"""{original_prompt}

Your previous answer was: {raw_answer}

That proposed action is illegal from the current state because it would move
into an obstacle, the outer boundary, or outside the grid.

The illegal action was not executed.

Choose another action.

Answer with one word only using the action choices from the original prompt.
"""


def choose_action(
    env,
    mode,
    policy_type,
    prompt,
    policy_fn,
    max_reprompts=2,
):
    """
    Gazebo intervention logic matched to the validated Farama implementation.

    Supported arms:
        legality_shield
        reprompt_control

    Both arms:
        - block illegal actions before physical execution
        - allow at most two reprompts
        - return None on retry-budget exhaustion
        - never execute an illegal action

    The only difference is whether the reprompt explicitly lists legal actions.
    """

    if policy_type not in {
        "legality_shield",
        "reprompt_control",
    }:
        raise ValueError(
            "Gazebo intervention runner currently supports only "
            "'legality_shield' and 'reprompt_control'."
        )

    legal_actions = (
        get_legal_cardinal_actions(env)
    )

    legal_names = legal_action_names(
        env,
        mode,
    )

    all_raw_answers = []
    current_prompt = prompt

    for attempt in range(
        max_reprompts + 1
    ):
        raw_answer = policy_fn(
            current_prompt
        )

        all_raw_answers.append(
            raw_answer
        )

        parsed_action = parse_action(
            raw_answer,
            mode,
            env,
        )

        if (
            parsed_action is not None
            and parsed_action
            in legal_actions
        ):
            return parsed_action, {
                "policy_type": policy_type,
                "raw_model_answer": raw_answer,
                "all_raw_answers": (
                    all_raw_answers
                ),
                "shield_used": (
                    attempt > 0
                ),
                "shield_reprompts": (
                    attempt
                ),
                "legal_action_names": (
                    legal_names
                ),
            }

        if (
            policy_type
            == "legality_shield"
        ):
            current_prompt = (
                make_legality_reprompt(
                    original_prompt=prompt,
                    raw_answer=raw_answer,
                    legal_names=legal_names,
                )
            )

        else:
            current_prompt = (
                make_reprompt_control_prompt(
                    original_prompt=prompt,
                    raw_answer=raw_answer,
                )
            )

    return None, {
        "policy_type": policy_type,
        "raw_model_answer": (
            all_raw_answers[-1]
            if all_raw_answers
            else None
        ),
        "all_raw_answers": (
            all_raw_answers
        ),
        "shield_used": True,
        "shield_reprompts": (
            max_reprompts
        ),
        "legal_action_names": (
            legal_names
        ),
    }