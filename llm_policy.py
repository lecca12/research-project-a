import os
import time
from typing import Callable

from openai import OpenAI


DEFAULT_SYSTEM_INSTRUCTIONS = (
    "You are a precise navigation assistant for a grid world experiment. "
    "Return exactly one allowed action word and nothing else. "
    "Do not explain your reasoning."
)


class OpenAIPolicyError(RuntimeError):
    """Raised when an OpenAI policy request cannot be completed."""


def make_openai_policy_fn(
    model: str = "gpt-4o-mini",
    temperature: float | None = 0.0,
    api_key: str | None = None,
    system_instructions: str = DEFAULT_SYSTEM_INSTRUCTIONS,
    max_output_tokens: int = 16,
    max_retries: int = 3,
    retry_base_delay: float = 1.0,
    request_delay: float = 0.3,
) -> Callable[[str], str]:
    """
    Return a function with the interface:

        policy_fn(prompt) -> response text

    Parameters
    ----------
    model:
        OpenAI model name.

    temperature:
        Sampling temperature. When None, the parameter is omitted from the
        request. This supports models that reject custom temperature values.

    api_key:
        Optional explicit API key. When omitted, OPENAI_API_KEY is read from
        the environment.

    system_instructions:
        Instructions supplied separately from the user prompt.

    max_output_tokens:
        Maximum output-token allowance for each request.

    max_retries:
        Number of additional attempts after the initial request fails.

    retry_base_delay:
        Initial retry delay in seconds. Later retries use exponential backoff.

    request_delay:
        Small delay before each API request to reduce rapid request bursts.
    """
    resolved_key = api_key or os.environ.get("OPENAI_API_KEY")

    if not resolved_key:
        raise OpenAIPolicyError(
            "OPENAI_API_KEY is not set. "
            "Set it in the current shell before running the experiment."
        )

    if max_retries < 0:
        raise ValueError("max_retries must be zero or greater.")

    if retry_base_delay < 0:
        raise ValueError("retry_base_delay must be zero or greater.")

    if request_delay < 0:
        raise ValueError("request_delay must be zero or greater.")

    client = OpenAI(api_key=resolved_key)

    def policy_fn(prompt: str) -> str:
        if not isinstance(prompt, str) or not prompt.strip():
            raise OpenAIPolicyError(
                "The policy prompt must be a non-empty string."
            )

        request_kwargs = {
            "model": model,
            "instructions": system_instructions,
            "input": prompt,
            "max_output_tokens": max_output_tokens,
        }

        # Some models reject or ignore temperature. Omitting the field when
        # temperature is None lets each runner choose the appropriate setup.
        if temperature is not None:
            request_kwargs["temperature"] = temperature

        total_attempts = max_retries + 1
        last_exception = None

        for attempt_index in range(total_attempts):
            try:
                if request_delay > 0:
                    time.sleep(request_delay)

                response = client.responses.create(**request_kwargs)

                text = getattr(response, "output_text", None)

                if text is None:
                    raise OpenAIPolicyError(
                        "The API response did not contain output_text."
                    )

                cleaned_text = text.strip()

                if not cleaned_text:
                    raise OpenAIPolicyError(
                        "The API returned an empty text response."
                    )

                return cleaned_text

            except Exception as exc:
                last_exception = exc

                is_final_attempt = attempt_index == total_attempts - 1

                if is_final_attempt:
                    break

                retry_delay = retry_base_delay * (2 ** attempt_index)

                print(
                    "OpenAI request failed "
                    f"(attempt {attempt_index + 1}/{total_attempts}): "
                    f"{exc}"
                )
                print(
                    f"Retrying after {retry_delay:.1f} seconds..."
                )

                if retry_delay > 0:
                    time.sleep(retry_delay)

        raise OpenAIPolicyError(
            f"OpenAI API call failed after {total_attempts} attempts: "
            f"{last_exception}"
        ) from last_exception

    return policy_fn