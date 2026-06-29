"""Answerers: the FIXED frontier model that answers from selected frames.

Kept identical across conditions so the only thing that varies is the selector.
`EchoAnswerer` needs no API key and estimates token counts, so the full pipeline
(selection -> scoring -> aggregation) can be smoke-tested offline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional

from .media import Frame, frame_to_base64


@dataclass
class AnswerResult:
    text: str
    input_tokens: int
    output_tokens: int
    raw: dict = field(default_factory=dict)


class Answerer:
    name = "base"

    def answer(self, question: str, frames: List[Frame]) -> AnswerResult:
        raise NotImplementedError


def _estimate_tokens(text: str, n_frames: int, tokens_per_image: int) -> int:
    return max(1, len(text) // 4) + n_frames * tokens_per_image


class EchoAnswerer(Answerer):
    """Offline dry-run. Stub answer (so accuracy will be ~0) but realistic token
    bookkeeping, for validating plumbing without spending API calls."""

    name = "echo"

    def __init__(self, tokens_per_image: int = 512, max_side: int = 768):
        self.tokens_per_image = tokens_per_image
        self.max_side = max_side  # unused; kept for signature parity

    def answer(self, question: str, frames: List[Frame]) -> AnswerResult:
        text = f"[echo] would answer the question using {len(frames)} frame(s)."
        return AnswerResult(
            text=text,
            input_tokens=_estimate_tokens(question, len(frames), self.tokens_per_image),
            output_tokens=8,
        )


class AnthropicAnswerer(Answerer):
    name = "anthropic"

    def __init__(self, model: str = "claude-sonnet-4-5", max_tokens: int = 512, max_side: int = 768):
        import anthropic

        self.client = anthropic.Anthropic()
        self.model = model
        self.max_tokens = max_tokens
        self.max_side = max_side

    def answer(self, question: str, frames: List[Frame]) -> AnswerResult:
        content = [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": frame_to_base64(f, max_side=self.max_side),
                },
            }
            for f in frames
        ]
        content.append({"type": "text", "text": question})
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": content}],
        )
        text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
        return AnswerResult(
            text=text,
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
            raw={"model": self.model},
        )


class OpenAIAnswerer(Answerer):
    name = "openai"

    def __init__(self, model: str = "gpt-4o", max_tokens: int = 512, max_side: int = 768):
        from openai import OpenAI

        self.client = OpenAI()
        self.model = model
        self.max_tokens = max_tokens
        self.max_side = max_side

    def answer(self, question: str, frames: List[Frame]) -> AnswerResult:
        content: list = [{"type": "text", "text": question}]
        for f in frames:
            b64 = frame_to_base64(f, max_side=self.max_side)
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
        resp = self.client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": content}],
        )
        return AnswerResult(
            text=resp.choices[0].message.content or "",
            input_tokens=resp.usage.prompt_tokens,
            output_tokens=resp.usage.completion_tokens,
            raw={"model": self.model},
        )


def build_answerer(provider: str, model: Optional[str], max_side: int = 768) -> Answerer:
    if provider == "echo":
        return EchoAnswerer(max_side=max_side)
    if provider == "anthropic":
        return AnthropicAnswerer(model=model or "claude-sonnet-4-5", max_side=max_side)
    if provider == "openai":
        return OpenAIAnswerer(model=model or "gpt-4o", max_side=max_side)
    raise ValueError(f"unknown answerer provider: {provider!r}")


def make_text_judge(provider: str, model: Optional[str]) -> Callable[[str], str]:
    """Build a text-only LLM judge callable: prompt -> short verdict string."""
    if provider == "anthropic":
        import anthropic

        client = anthropic.Anthropic()
        mdl = model or "claude-sonnet-4-5"

        def fn(prompt: str) -> str:
            r = client.messages.create(
                model=mdl, max_tokens=8, messages=[{"role": "user", "content": prompt}]
            )
            return "".join(b.text for b in r.content if getattr(b, "type", None) == "text")

        return fn

    if provider == "openai":
        from openai import OpenAI

        client = OpenAI()
        mdl = model or "gpt-4o"

        def fn(prompt: str) -> str:
            r = client.chat.completions.create(
                model=mdl, max_tokens=8, messages=[{"role": "user", "content": prompt}]
            )
            return r.choices[0].message.content or ""

        return fn

    # echo / no-provider: judging is unavailable, fall back to "no" so exact_match drives accuracy
    def fn(prompt: str) -> str:
        return "no"

    return fn
