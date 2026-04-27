from openai import OpenAI
import time
import os
from typing import Callable


DEFAULT_SYSTEM_INSTRUCTIONS = (
    "You are a precise navigation assistant for a grid world experiment. "
    "Return exactly one allowed action word and nothing else. "
    "Do not explain your reasoning."
)


class OpenAIPolicyError(RuntimeError):
    pass


def make_openai_policy_fn(
    model: str = "gpt-4o-mini",
    temperature: float = 0.0,
    api_key: str | None = None,
    system_instructions: str = DEFAULT_SYSTEM_INSTRUCTIONS,
    max_output_tokens: int = 16,
) -> Callable[[str], str]:
    """
    Returns a function: policy_fn(prompt) -> model response text
    """

    # Resolve API key
    resolved_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not resolved_key:
        raise OpenAIPolicyError(
            "OPENAI_API_KEY is not set. Export your API key before running."
        )

    client = OpenAI(api_key=resolved_key)

    def policy_fn(prompt: str) -> str:
        try:
            # Avoid hitting rate limits
            time.sleep(0.3)

            response = client.responses.create(
                model=model,
                instructions=system_instructions,
                input=prompt,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
            )

        except Exception as exc:
            raise OpenAIPolicyError(f"OpenAI API call failed: {exc}") from exc

        text = getattr(response, "output_text", None)
        if text is None:
            raise OpenAIPolicyError("No text output returned from API.")

        return text.strip()

    return policy_fn