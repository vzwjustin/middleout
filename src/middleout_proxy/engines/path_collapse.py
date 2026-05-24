"""Replace repeated long absolute paths with short ``<P1>`` aliases.

Stdlib-only. Deterministic — aliases are numbered by first-occurrence order.

Levels:
  off         identity
  lite        only alias paths >=80 chars that appear >=5 times
  standard    paths >=60 chars, >=3 occurrences
  aggressive  paths >=40 chars, >=2 occurrences

The original first occurrence of each aliased path is kept verbatim; only
later occurrences are rewritten to the alias. A legend line is prepended::

    Path aliases: <P1>=/abs/path/one; <P2>=/abs/path/two

An alias is only emitted when the legend overhead is strictly smaller than
the savings from rewriting later occurrences, so the engine cannot make the
text longer.
"""

from __future__ import annotations

import re

from .base import EngineResult, make_result, validate_level

NAME = "path_collapse"

# Per spec: optional drive letter prefix, then >= 2 ``/segment`` segments.
# A segment may contain word chars, dots, hyphens and spaces.
_PATH_RE = re.compile(r"(?:[A-Za-z]:)?(?:/[\w.\- ]+){2,}")


def _level_config(level: str) -> tuple[int, int]:
    """Return (min_path_chars, min_occurrences)."""
    if level == "lite":
        return 80, 5
    if level == "standard":
        return 60, 3
    return 40, 2  # aggressive


def compress(text: str, *, level: str = "standard") -> EngineResult:
    validate_level(level)
    if level == "off" or not text:
        return EngineResult(
            text=text, original_chars=len(text), compressed_chars=len(text)
        )

    min_len, min_occs = _level_config(level)

    groups: dict[str, list[tuple[int, int]]] = {}
    for m in _PATH_RE.finditer(text):
        path = m.group(0)
        if len(path) < min_len:
            continue
        groups.setdefault(path, []).append(m.span())

    candidates = [(p, spans) for p, spans in groups.items() if len(spans) >= min_occs]
    if not candidates:
        return EngineResult(
            text=text, original_chars=len(text), compressed_chars=len(text)
        )

    # Deterministic: sort by first occurrence position.
    candidates.sort(key=lambda kv: kv[1][0][0])

    # Decide which aliases pay for themselves. Use the eventual alias label
    # length so the estimate matches what will land in the output.
    selected: list[tuple[str, str, list[tuple[int, int]]]] = []
    for path, spans in candidates:
        alias = f"<P{len(selected) + 1}>"
        rewrites = len(spans) - 1  # only later occurrences are replaced
        savings = rewrites * (len(path) - len(alias))
        # Legend entry roughly: "<P1>=path; " or trailing "<P1>=path".
        legend_entry = len(alias) + 1 + len(path) + 2
        if savings > legend_entry:
            selected.append((alias, path, spans))
    if not selected:
        return EngineResult(
            text=text, original_chars=len(text), compressed_chars=len(text)
        )

    # Build replacement ops; replace later occurrences only.
    ops: list[tuple[int, int, str]] = []
    for alias, _path, spans in selected:
        for start, end in spans[1:]:
            ops.append((start, end, alias))

    # Apply right-to-left so earlier offsets remain valid.
    ops.sort(key=lambda x: x[0], reverse=True)
    out = text
    for start, end, alias in ops:
        out = out[:start] + alias + out[end:]

    legend_body = "; ".join(f"{a}={p}" for a, p, _ in selected)
    legend = f"Path aliases: {legend_body}\n"
    out_text = legend + out

    note = f"{len(selected)} path aliases"
    return make_result(text, out_text, note)
