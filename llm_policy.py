"""
OpenAI-backed policy function for GridWorld experiments.

Usage:
    from llm_policy import make_openai_policy_fn
    policy_fn = make_openai_policy_fn(model="gpt-4o-mini")
    answer = policy_fn(prompt)
"""

import os
from typing import Callable

from openai import OpenAI


DEFAULT_SYSTEM_INSTRUCTIONS = (
    "You are a precise navigation assistant for a grid world experiment. "
    "Return exactly one allowed action word and nothing else. "
    "Do not explain your reasoning."
)


class OpenAIPolicyError(RuntimeError):
    pass


def make_openai_policy_fn(
    model: str = "gpt-4o-mini",
    api_key: str | None = None,
    system_instructions: str = DEFAULT_SYSTEM_INSTRUCTIONS,
    max_output_tokens: int = 16,
) -> Callable[[str], str]:
    """
    Build a policy_fn(prompt) that calls the OpenAI Responses API.
    """
    resolved_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not resolved_key:
        raise OpenAIPolicyError(
            "OPENAI_API_KEY is not set. Export your API key before running the policy."
        )

    client = OpenAI(api_key=resolved_key)

    def policy_fn(prompt: str) -> str:
        try:
            response = client.responses.create(
                model=model,
                instructions=system_instructions,
                input=prompt,
                max_output_tokens=max_output_tokens,
            )
        except Exception as exc:
            raise OpenAIPolicyError(f"OpenAI API call failed: {exc}") from exc

        text = getattr(response, "output_text", None)
        if text is None:
            raise OpenAIPolicyError("Responses API returned no text output.")
        return text.strip()

    return policy_fn