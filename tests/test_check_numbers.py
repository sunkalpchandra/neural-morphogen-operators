"""Verify the audit gate can actually fail.

`make check-numbers` is the only thing standing between a stale number and the
manuscript, and it reports a pass count that reads as reassurance. One of its
checks was dead for the entire period after the paper was split into
`preamble.tex` + `sections/`: the stub parser searched only the section files,
so the set of stubbed macros was always empty and the check compared nothing
against nothing. It still reported itself as passing.

These tests inject the defect each check exists to catch and assert it is
caught. A check that cannot fail is not a check.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load():
    """Import check_numbers as a module without executing main()."""
    spec = importlib.util.spec_from_file_location(
        "check_numbers", ROOT / "scripts" / "check_numbers.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["check_numbers"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def cn():
    return _load()


def test_the_stub_parser_finds_the_real_stub_block(cn):
    """Task 44. The regression that motivated this file.

    `stubbed` must be populated from preamble.tex. If it is empty the provenance
    check degenerates to a no-op that still reports success.
    """
    rep = cn.Report()
    cn.check_scope(rep)
    pre = (cn.PAPER / "preamble.tex").read_text()
    assert r"\@for\@nmocmd" in pre, "stub block is not in preamble.tex any more"
    # Parse it the way check_scope does and assert it is non-trivial.
    import re
    lines = pre.split("\n")
    lo = next(i for i, l in enumerate(lines) if r"\@for\@nmocmd" in l)
    hi = next(i for i, l in enumerate(lines[lo:], lo) if r"\makeatother" in l)
    blk = "\n".join(lines[lo:hi + 1])
    stubbed = set(re.findall(r"[A-Za-z]+", blk.split(":=")[-1].split(r"}\do")[0]))
    assert len(stubbed) > 100, f"only {len(stubbed)} stubs parsed; parser is broken"
    assert "NMOPearson" in stubbed and "MSSpecimens" in stubbed


def test_registry_macros_are_either_stubbed_or_generated(cn):
    """Every macro the registry knows about must render as a number or a visible
    ??; a macro that is neither is an undefined control sequence at build time."""
    import re
    defined = set(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}",
                             (cn.PAPER / "numbers.tex").read_text()))
    pre = (cn.PAPER / "preamble.tex").read_text()
    lines = pre.split("\n")
    lo = next(i for i, l in enumerate(lines) if r"\@for\@nmocmd" in l)
    hi = next(i for i, l in enumerate(lines[lo:], lo) if r"\makeatother" in l)
    blk = "\n".join(lines[lo:hi + 1])
    stubbed = set(re.findall(r"[A-Za-z]+", blk.split(":=")[-1].split(r"}\do")[0]))
    orphans = sorted(set(cn.PROVENANCE) - stubbed - defined)
    assert not orphans, f"registry macros with no definition and no stub: {orphans}"


def test_literal_sweep_catches_a_hard_coded_number(cn, tmp_path, monkeypatch):
    """Task 43. The check that stops a number being typed straight into prose.
    Inject one and assert it is reported."""
    sec = cn.PAPER / "sections"
    victim = sec / "results.tex"
    if not victim.exists():
        pytest.skip("results.tex not present")
    original = victim.read_text()
    try:
        victim.write_text(original + "\nThe model reaches 0.7391 Pearson r.\n")
        rep = cn.Report()
        cn.check_literals(rep)
        hits = [r for r in rep.rows if "0.7391" in str(r)]
        assert hits, "a bare 0.7391 in the results prose was not flagged"
    finally:
        victim.write_text(original)


def test_gate_passes_on_the_committed_tree(cn):
    """The inverse of the above: with nothing injected, the checks agree.
    If this fails, the repository is in a state the paper should not be built
    from, and the failure message names which check."""
    rep = cn.Report()
    cn.check_scope(rep)
    cn.check_literals(rep)
    cn.check_spelling(rep)
    # rep.rows holds plain tuples (claim, where, artifact, value, verdict, ok).
    # getattr(row, "ok", True) on a tuple is always True, which is how the first
    # draft of this assertion managed to pass unconditionally.
    # Report.add() is only called on a violation, so an empty rows list is the
    # passing state here -- unlike the literal-injection test above, which
    # asserts a row appears.
    bad = [r for r in rep.rows if not r[-1]]
    assert not bad, f"{len(bad)} failing: {bad[:3]}"
    assert not rep.errors, rep.errors[:3]
