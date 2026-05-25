"""Compact unified-diff style text by collapsing long unchanged context.

Stdlib-only. Deterministic.

Levels:
  off         identity
  lite        collapse unchanged runs >=20 lines, keep 3 context each side
  standard    collapse unchanged runs >=10 lines, keep 2 context each side
  aggressive  collapse unchanged runs >=5 lines, keep 1 context each side;
              additionally drop adjacent ``-X``/``+X`` no-op revert pairs

Only files that look like a unified diff (contain at least one ``@@`` hunk
header) are touched. ``+``/``-`` lines and hunk headers themselves are never
modified (other than the aggressive revert-pair drop).
"""

from __future__ import annotations

import re

from .base import EngineResult, make_result, validate_level

NAME = "diff_compactor"

# Real unified-diff hunk header: ``@@ -1,4 +1,4 @@``. Requiring this (rather
# than a bare ``@@`` substring) avoids treating prose containing ``@@``
# (emails, ``HEAD@@{1}``, etc.) as a diff.
_HUNK_HEADER_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+\d+(?:,\d+)? @@", re.MULTILINE)


def _level_config(level: str) -> tuple[int, int, bool]:
    """Return (threshold, keep_each_side, drop_revert_pairs)."""
    if level == "lite":
        return 20, 3, False
    if level == "standard":
        return 10, 2, False
    return 5, 1, True  # aggressive


def _is_context(line: str) -> bool:
    """Unified diff context lines begin with a single space."""
    return line.startswith(" ")


def _collapse_context(
    lines: list[str], threshold: int, keep: int
) -> tuple[list[str], int]:
    out: list[str] = []
    collapsed = 0
    i = 0
    n = len(lines)
    while i < n:
        if not _is_context(lines[i]):
            out.append(lines[i])
            i += 1
            continue
        j = i
        while j < n and _is_context(lines[j]):
            j += 1
        run = j - i
        if run >= threshold and run > 2 * keep:
            omitted = run - 2 * keep
            marker = f"[... {omitted} unchanged lines ...]"
            out.extend(lines[i : i + keep])
            out.append(marker)
            out.extend(lines[j - keep : j])
            collapsed += omitted
        else:
            out.extend(lines[i:j])
        i = j
    return out, collapsed


def _drop_revert_pairs(lines: list[str]) -> tuple[list[str], int]:
    """Drop adjacent ``-X``/``+X`` pairs that net to no change.

    File headers ``--- a/foo`` / ``+++ b/foo`` are explicitly skipped so we
    never confuse them for change markers.
    """
    out: list[str] = []
    removed = 0
    i = 0
    n = len(lines)
    while i < n:
        cur = lines[i]
        nxt = lines[i + 1] if i + 1 < n else None
        if (
            nxt is not None
            and cur.startswith("-")
            and not cur.startswith("---")
            and nxt.startswith("+")
            and not nxt.startswith("+++")
            and cur[1:] == nxt[1:]
        ):
            removed += 1
            i += 2
            continue
        out.append(cur)
        i += 1
    return out, removed


def compress(text: str, *, level: str = "standard") -> EngineResult:
    validate_level(level)
    if level == "off" or not text:
        return EngineResult(
            text=text, original_chars=len(text), compressed_chars=len(text)
        )
    if not _HUNK_HEADER_RE.search(text):
        return EngineResult(
            text=text, original_chars=len(text), compressed_chars=len(text)
        )

    threshold, keep, do_reverts = _level_config(level)
    lines = text.split("\n")

    new_lines, collapsed_context = _collapse_context(lines, threshold, keep)

    reverts = 0
    if do_reverts:
        new_lines, reverts = _drop_revert_pairs(new_lines)

    out_text = "\n".join(new_lines)
    parts: list[str] = []
    if collapsed_context:
        parts.append(f"collapsed {collapsed_context} context lines")
    if reverts:
        parts.append(f"dropped {reverts} revert pairs")
    return make_result(text, out_text, "; ".join(parts))
