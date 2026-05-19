"""Thin wrappers over Anthropic and OpenAI SDKs. Returns dicts with text + token counts."""
import os
from typing import Any


def call_anthropic(client: Any, model: str, system: str, user: str,
                   temperature: float, max_tokens: int) -> dict:
    resp = client.messages.create(
        model=model,
        system=system,
        messages=[{"role": "user", "content": user}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    text = "".join(block.text for block in resp.content if hasattr(block, "text"))
    return {
        "text": text,
        "input_tokens": resp.usage.input_tokens,
        "output_tokens": resp.usage.output_tokens,
    }


def call_openai(client: Any, model: str, system: str, user: str,
                temperature: float, max_tokens: int, seed: int | None = None) -> dict:
    kwargs = dict(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    if seed is not None:
        kwargs["seed"] = seed
    resp = client.chat.completions.create(**kwargs)
    return {
        "text": resp.choices[0].message.content,
        "input_tokens": resp.usage.prompt_tokens,
        "output_tokens": resp.usage.completion_tokens,
    }


def get_anthropic_client():
    from anthropic import Anthropic
    return Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])


def get_openai_client():
    from openai import OpenAI
    return OpenAI(api_key=os.environ["OPENAI_API_KEY"])
