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
    temperature: float | None = 0.0,
    api_key: str | None = None,
    system_instructions: str = DEFAULT_SYSTEM_INSTRUCTIONS,
    max_output_tokens: int = 16,
) -> Callable[[str], str]:
    """
    Returns a function: policy_fn(prompt) -> model response text.

    If temperature is None, the temperature parameter is omitted from the API call.
    This is useful for reasoning/newer models that reject or ignore custom temperature.
    """

    resolved_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not resolved_key:
        raise OpenAIPolicyError(
            "OPENAI_API_KEY is not set. Export your API key before running."
        )

    client = OpenAI(api_key=resolved_key)

    def policy_fn(prompt: str) -> str:
        try:
            time.sleep(0.3)

            request_kwargs = {
                "model": model,
                "instructions": system_instructions,
                "input": prompt,
                "max_output_tokens": max_output_tokens,
            }

            if temperature is not None:
                request_kwargs["temperature"] = temperature

            response = client.responses.create(**request_kwargs)

        except Exception as exc:
            raise OpenAIPolicyError(f"OpenAI API call failed: {exc}") from exc

        text = getattr(response, "output_text", None)
        if text is None:
            raise OpenAIPolicyError("No text output returned from API.")

        return text.strip()

    return policy_fn