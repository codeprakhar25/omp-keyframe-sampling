#!/usr/bin/env python3
"""Unit tests for GPT harness frac-resize / pick normalize (no API, no video)."""
from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image

_SLM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_SLM))

from scripts.gpt_mini_lvbench import (  # noqa: E402
    apply_frac_resize,
    apply_max_side,
    normalize_gold,
    normalize_pick_entries,
    picks_mean_frac,
    prompt_has_options,
    format_question,
    item_option_lines,
    question_text_for_item,
    _MIN_PIXELS_FLOOR,
)


def test_normalize_secs():
    e = normalize_pick_entries([17.0, 76.0, 90.0])
    assert e == [(17.0, 1.0), (76.0, 1.0), (90.0, 1.0)]


def test_normalize_resprop():
    e = normalize_pick_entries([[17.0, 0.5], [76.0, 1.0], [90.0, 0.25]])
    assert e[0] == (17.0, 0.5)
    assert e[1] == (76.0, 1.0)
    assert abs(e[2][1] - 0.25) < 1e-9


def test_normalize_rejects_bad_frac():
    try:
        normalize_pick_entries([[1.0, 0.0]])
        assert False, "should reject frac=0"
    except ValueError:
        pass


def test_frac_shrinks_area():
    # Large canvas so 0.5*pixels stays above MIN_PIXELS_FLOOR (200704).
    img = Image.new("RGB", (1200, 800), color=(10, 20, 30))
    out = apply_frac_resize(img, 0.5)
    assert out.size[0] * out.size[1] < img.size[0] * img.size[1]
    ratio = (out.size[0] * out.size[1]) / (img.size[0] * img.size[1])
    assert 0.45 < ratio < 0.55, ratio


def test_frac_floor_binds_on_small_cap():
    # 768x432 capped frame: 0.5*area < floor → pinned near floor (~0.60 of cap).
    img = Image.new("RGB", (768, 432))
    out = apply_frac_resize(img, 0.5)
    ratio = (out.size[0] * out.size[1]) / (img.size[0] * img.size[1])
    assert out.size[0] * out.size[1] >= _MIN_PIXELS_FLOOR * 0.85
    assert 0.55 < ratio < 0.70, ratio


def test_frac_noop_near_one():
    img = Image.new("RGB", (768, 432))
    out = apply_frac_resize(img, 0.9995)
    assert out.size == img.size


def test_max_side_then_frac():
    img = Image.new("RGB", (1920, 1080))
    capped = apply_max_side(img, 768)
    assert max(capped.size) <= 768
    half = apply_frac_resize(capped, 0.53)
    assert half.size[0] * half.size[1] < capped.size[0] * capped.size[1]


def test_floor_pins_tiny():
    img = Image.new("RGB", (100, 100))  # 10k < floor
    # frac would want even smaller; floor keeps >= min when starting above floor
    big = Image.new("RGB", (800, 800))
    out = apply_frac_resize(big, 0.01)
    assert out.size[0] * out.size[1] >= _MIN_PIXELS_FLOOR * 0.9  # approx after round


def test_gold():
    assert normalize_gold("C") == "C"
    assert normalize_gold("C) foo") == "C"
    assert normalize_gold("E. bar") == "E"


def test_mean_frac():
    e = normalize_pick_entries([[1, 1.0], [2, 0.5], [3, 0.5]])
    assert abs(picks_mean_frac(e) - 2 / 3) < 1e-9


def test_lvbench_prompt_options_gate():
    # Real LVBench shape: bare stem + options only in full question.
    item = {
        "question_stem": "What year appears in the opening caption of the video?",
        "question": "What year appears in the opening caption of the video?\n(A) 1636\n(B) 1366\n(C) 1363\n(D) 1633",
    }
    q = question_text_for_item(item)
    assert "(A)" in q
    opts = item_option_lines(item, bench="lvbench")
    text = format_question(q, "lvbench", options=opts or None)
    assert prompt_has_options(text), text[-200:]
    assert "(A)" in text and "(D)" in text


def test_lvb_paren_and_dot():
    assert prompt_has_options("Q\nA. a\nB. b\nC. c")
    assert prompt_has_options("Q\nA) a\nB) b")
    assert prompt_has_options("Q\n(A) a\n(B) b\n(C) c")
    assert not prompt_has_options("Q with no choices")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
    print("ALL_OK")
