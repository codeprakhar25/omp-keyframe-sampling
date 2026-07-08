"""Unit tests for the Stage-1 retriever core (harness/union_retrieval.py).

Pure-Python, deterministic, NO torch import -- runs with zero deps beyond the stdlib.
Two ways to run:
    python3 -m pytest tests/test_union.py
    python3 tests/test_union.py            # no pytest needed
"""

from __future__ import annotations

import os
import sys

# make `import harness...` work regardless of cwd / invocation style (pytest vs bare python3)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harness.union_retrieval import any_hit, build_union_indices, full_recall  # noqa: E402


def _flat(n: int, value: float = 0.0) -> list[float]:
    return [value] * n


def _times(n: int) -> list[float]:
    return [float(i) for i in range(n)]


# --- 1. lone sharp spike -------------------------------------------------------------------

def test_lone_spike_is_recovered():
    n = 20
    scores = _flat(n)
    scores[10] = 10.0
    times = _times(n)
    union = build_union_indices(scores, times, budget=100, gmin=10.0, pad=2.0, k_std=1.0)
    assert 10 in union
    assert len(union) > 0


# --- 2. multi-needle: two peaks far apart both survive -------------------------------------

def test_two_distant_peaks_both_survive():
    n = 200
    scores = _flat(n)
    scores[20] = 10.0
    scores[120] = 10.0  # 100s away >> gmin -> NMS must not suppress either
    times = _times(n)
    union = build_union_indices(scores, times, budget=100, gmin=10.0, pad=2.0, k_std=1.0)
    assert 20 in union
    assert 120 in union


# --- 3. NMS gap: near-duplicate peak doesn't get its own peak-slot --------------------------

def test_close_peak_does_not_steal_a_peak_slot():
    n = 100
    scores = _flat(n)
    scores[50] = 10.0
    scores[53] = 8.0  # 3s away, < default gmin=10 -> should NOT be accepted as its own peak
    times = _times(n)

    # narrow pad: if 53 got its own peak-slot, its OWN window (53±1) would surface it
    # regardless of what the peak at 50 does. It must NOT appear here.
    tight = build_union_indices(scores, times, budget=100, gmin=10.0, pad=1.0, k_std=1.0)
    assert 53 not in tight
    assert 50 in tight

    # wide pad: 53 can still show up, but only as a rider inside peak-50's window, not
    # because it earned a second peak-slot.
    wide = build_union_indices(scores, times, budget=100, gmin=10.0, pad=4.0, k_std=1.0)
    assert 53 in wide
    assert 50 in wide


# --- 4. budget bound ------------------------------------------------------------------------

def test_union_never_exceeds_budget():
    n = 400
    scores = _flat(n)
    for i in range(0, n, 20):  # 20 strong, well-separated peaks
        scores[i] = 10.0
    times = _times(n)
    union = build_union_indices(scores, times, budget=5, gmin=10.0, pad=2.0, k_std=1.0)
    assert len(union) <= 5
    assert len(union) > 0

    union_big = build_union_indices(scores, times, budget=100, gmin=10.0, pad=2.0, k_std=1.0)
    assert len(union_big) <= 100


# --- 5. fallback: no peak clears the noise floor --------------------------------------------

def test_fallback_when_no_peak_exceeds_tau():
    n = 50
    scores = _flat(n, 1.0)  # perfectly flat -> std=0, tau=mean -> nothing is STRICTLY > tau
    times = _times(n)
    union = build_union_indices(scores, times, budget=10, gmin=10.0, pad=2.0, k_std=1.0)
    assert len(union) == 10
    assert union == sorted(union)  # non-empty, well-formed
    assert union  # non-empty


# --- 6. any-hit / full-recall helpers --------------------------------------------------------

def test_any_hit_and_full_recall():
    gold_spans = [(10.0, 12.0), (50.0, 52.0)]  # disjoint, multi-needle

    # covers only the first span
    u1 = [10.5, 30.0]
    assert any_hit(u1, gold_spans, tol=0.5) is True
    assert full_recall(u1, gold_spans, tol=0.5) is False

    # covers both spans
    u2 = [10.5, 51.0]
    assert any_hit(u2, gold_spans, tol=0.5) is True
    assert full_recall(u2, gold_spans, tol=0.5) is True

    # covers neither
    u3 = [0.0, 100.0]
    assert any_hit(u3, gold_spans, tol=0.5) is False
    assert full_recall(u3, gold_spans, tol=0.5) is False

    # tolerance boundary: just inside vs just outside the ±0.5s pad
    assert any_hit([9.6], [(10.0, 12.0)], tol=0.5) is True
    assert any_hit([9.4], [(10.0, 12.0)], tol=0.5) is False

    # no gold spans -> both trivially False, never crash
    assert any_hit([1.0], None, tol=0.5) is False
    assert full_recall([1.0], [], tol=0.5) is False


if __name__ == "__main__":
    import inspect

    fails = 0
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and inspect.isfunction(f)]
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except AssertionError as e:
            fails += 1
            print(f"FAIL {name}: {e}")
    print(f"\n{len(tests) - fails}/{len(tests)} passed")
    sys.exit(1 if fails else 0)
