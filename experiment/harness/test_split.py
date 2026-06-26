"""Split a gold test file into VISIBLE vs HELD-OUT subsets at function granularity.

Why function-level: most of our tasks ship a single gold test file (tstF=1), so a
file-level visible/held-out split is impossible — we partition the individual
`def test_*` functions instead.

Contract (SpecBench discipline): held-out introduces NO requirement beyond what the
spec + visible already imply — both halves are drawn from the same gold suite, so
every held-out check is already mandated. Δ = s_visible - s_heldout then measures
over-fitting to the visible proxy (gaming), not an unfair hidden bar.

v1 scope: partitions MODULE-LEVEL test functions (the common pytest style). If a
file has <2 module-level test functions, `split_source` reports splittable=False and
the caller should fall back (file-level split across multiple files, or skip the task
for the differential arm).
"""
from __future__ import annotations

import ast
import hashlib
from dataclasses import dataclass


@dataclass
class SplitResult:
    splittable: bool
    visible_src: str = ""
    heldout_src: str = ""
    visible_fns: list[str] = None      # type: ignore
    heldout_fns: list[str] = None      # type: ignore
    reason: str = ""

    def __post_init__(self):
        self.visible_fns = self.visible_fns or []
        self.heldout_fns = self.heldout_fns or []


def _module_test_funcs(tree: ast.Module) -> list[ast.FunctionDef]:
    return [n for n in tree.body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            and n.name.startswith("test_")]


def _node_line_span(node: ast.AST) -> tuple[int, int]:
    """1-based inclusive line span including any decorators."""
    start = node.lineno
    decos = getattr(node, "decorator_list", [])
    if decos:
        start = min(start, decos[0].lineno)
    end = getattr(node, "end_lineno", node.lineno)
    return start, end


def _drop_lines(src: str, spans: list[tuple[int, int]]) -> str:
    """Remove the given 1-based inclusive line spans from src."""
    drop: set[int] = set()
    for a, b in spans:
        drop.update(range(a, b + 1))
    lines = src.splitlines(keepends=True)
    return "".join(l for i, l in enumerate(lines, start=1) if i not in drop)


def _partition(names: list[str], frac: float, seed: str) -> tuple[set[str], set[str]]:
    """Deterministic ~frac visible / rest held-out; both guaranteed non-empty."""
    ordered = sorted(names)
    visible: set[str] = set()
    for nm in ordered:
        h = int(hashlib.sha1(f"{seed}:{nm}".encode()).hexdigest(), 16)
        if (h % 1000) / 1000.0 < frac:
            visible.add(nm)
    held = set(ordered) - visible
    # guarantee both non-empty (move the first ordered item if a side is empty)
    if not visible:
        visible.add(ordered[0]); held.discard(ordered[0])
    if not held:
        held.add(ordered[-1]); visible.discard(ordered[-1])
    return visible, held


def split_source(src: str, frac: float = 0.5, seed: str = "rh-v1") -> SplitResult:
    """Return visible/held-out versions of one test file. visible_src keeps only the
    visible test fns (held-out fns removed) and vice-versa; non-test code (imports,
    fixtures, helpers) is preserved in BOTH."""
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        return SplitResult(False, reason=f"parse error: {e}")
    funcs = _module_test_funcs(tree)
    if len(funcs) < 2:
        return SplitResult(False, reason=f"only {len(funcs)} module-level test fn(s)")
    names = [f.name for f in funcs]
    vis, held = _partition(names, frac, seed)
    span = {f.name: _node_line_span(f) for f in funcs}
    visible_src = _drop_lines(src, [span[n] for n in held])     # remove held -> keep visible
    heldout_src = _drop_lines(src, [span[n] for n in vis])      # remove visible -> keep held
    return SplitResult(True, visible_src, heldout_src,
                       sorted(vis), sorted(held))


if __name__ == "__main__":  # quick self-test
    sample = '''import pytest

@pytest.fixture
def db():
    return {}

def test_select(db):
    assert select(db) == 1

def test_join(db):
    assert join(db) == 2

def test_group(db):
    assert group(db) == 3

def test_compose(db):
    assert compose(db) == 6
'''
    r = split_source(sample)
    assert r.splittable, r.reason
    assert set(r.visible_fns) | set(r.heldout_fns) == {"test_select", "test_join", "test_group", "test_compose"}
    assert not (set(r.visible_fns) & set(r.heldout_fns)), "overlap!"
    # both halves must still parse and keep the fixture + import
    for s in (r.visible_src, r.heldout_src):
        ast.parse(s)
        assert "def db():" in s and "import pytest" in s
    # visible_src must NOT contain held-out fns
    for n in r.heldout_fns:
        assert f"def {n}(" not in r.visible_src
    print("visible:", r.visible_fns, "| heldout:", r.heldout_fns)
    print("SELF-TEST PASS")
