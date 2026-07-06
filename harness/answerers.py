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

    def __init__(self, model: str = "gpt-4o", max_tokens: int = 512, max_side: int = 768,
                 detail: Optional[str] = None, max_retries: int = 6,
                 reasoning_effort: str = "low"):
        from openai import OpenAI

        self.client = OpenAI()
        self.model = model
        # gpt-5* are reasoning models: they reject `max_tokens` (need `max_completion_tokens`)
        # and spend part of the budget on hidden reasoning tokens -> give a bigger ceiling.
        self.is_reasoning = "gpt-5" in model
        self.max_tokens = 2048 if (self.is_reasoning and max_tokens < 2048) else max_tokens
        self.reasoning_effort = reasoning_effort
        self.max_side = max_side
        # detail=low ~85 tok/frame (full-dump); high = fine-grained (top-k); None -> provider default
        self.detail = detail
        self.max_retries = max_retries

    def _image_part(self, f: Frame) -> dict:
        b64 = frame_to_base64(f, max_side=self.max_side)
        img = {"url": f"data:image/jpeg;base64,{b64}"}
        if self.detail:
            img["detail"] = self.detail
        return {"type": "image_url", "image_url": img}

    def _create_with_backoff(self, messages: list):
        import random
        import time as _time

        from openai import APIConnectionError, APIStatusError, RateLimitError

        if self.is_reasoning:
            kw = {"max_completion_tokens": self.max_tokens, "reasoning_effort": self.reasoning_effort}
        else:
            kw = {"max_tokens": self.max_tokens}

        last_exc = None
        for attempt in range(self.max_retries):
            try:
                return self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    **kw,
                )
            except RateLimitError as e:  # 429
                last_exc = e
            except APIConnectionError as e:  # transient network
                last_exc = e
            except APIStatusError as e:  # retry only 5xx; 4xx (e.g. 400 too-many-images) is fatal
                if e.status_code < 500:
                    raise
                last_exc = e
            sleep = min(60.0, (2 ** attempt) + random.random())
            print(f"  [openai retry {attempt+1}/{self.max_retries} after {type(last_exc).__name__}] sleeping {sleep:.1f}s")
            _time.sleep(sleep)
        raise last_exc  # exhausted retries

    def answer(self, question: str, frames: List[Frame]) -> AnswerResult:
        content: list = [{"type": "text", "text": question}]
        content.extend(self._image_part(f) for f in frames)
        resp = self._create_with_backoff([{"role": "user", "content": content}])
        return AnswerResult(
            text=resp.choices[0].message.content or "",
            input_tokens=resp.usage.prompt_tokens,
            output_tokens=resp.usage.completion_tokens,
            raw={"model": self.model},
        )


def build_answerer(provider: str, model: Optional[str], max_side: int = 768,
                   detail: Optional[str] = None) -> Answerer:
    if provider == "echo":
        return EchoAnswerer(max_side=max_side)
    if provider == "anthropic":
        return AnthropicAnswerer(model=model or "claude-sonnet-4-5", max_side=max_side)
    if provider == "openai":
        return OpenAIAnswerer(model=model or "gpt-4o", max_side=max_side, detail=detail)
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
        # judge should be cheap + deterministic; never a reasoning model (8-tok budget would be
        # eaten by hidden reasoning -> empty verdict). Force a small non-reasoning default.
        mdl = model or "gpt-4.1"
        if "gpt-5" in mdl:
            kw = {"max_completion_tokens": 512, "reasoning_effort": "low"}
        else:
            kw = {"max_tokens": 8}

        def fn(prompt: str) -> str:
            r = client.chat.completions.create(
                model=mdl, messages=[{"role": "user", "content": prompt}], **kw
            )
            return r.choices[0].message.content or ""

        return fn

    # echo / no-provider: judging is unavailable, fall back to "no" so exact_match drives accuracy
    def fn(prompt: str) -> str:
        return "no"

    return fn
