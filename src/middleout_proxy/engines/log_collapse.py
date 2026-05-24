"""Collapse repeated log lines.

Stdlib-only. Deterministic.

Levels:
  off         identity
  lite        collapse runs of >=10 byte-identical lines
  standard    collapse runs of >=5 lines that agree after stripping leading
              timestamps (ISO 8601, ``HH:MM:SS``, ``[YYYY-MM-DD HH:MM:SS]``)
  aggressive  standard + numeric tokens normalized to ``#`` before comparing

A collapsed run is rendered as::

    <first line>
    [... N identical lines collapsed ...]
    <last line>

with the first and last of the run preserved verbatim. A collapse is only
emitted when it actually saves characters versus the original block.
"""

from __future__ import annotations

import re

from .base import EngineResult, make_result, validate_level

NAME = "log_collapse"

# Leading timestamps we know how to strip. Order matters: try the most
# specific (longest) pattern first.
_TS_PATTERNS = (
    # [YYYY-MM-DD HH:MM:SS] or [YYYY-MM-DD HH:MM:SS.ms]
    re.compile(r"^\s*\[\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?\]\s*"),
    # ISO 8601: 2023-01-15T10:30:45(.ms)?(Z|+HH:MM)?
    re.compile(
        r"^\s*\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}"
        r"(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?\s*"
    ),
    # bare HH:MM:SS or HH:MM:SS.ms at line start
    re.compile(r"^\s*\d{2}:\d{2}:\d{2}(?:\.\d+)?\s*"),
)

_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")


def _strip_timestamps(line: str) -> str:
    for pat in _TS_PATTERNS:
        new = pat.sub("", line, count=1)
        if new != line:
            return new
    return line


def _normalize(line: str, level: str) -> str:
    if level == "lite":
        return line
    s = _strip_timestamps(line)
    if level == "aggressive":
        s = _NUMBER_RE.sub("#", s)
    return s


def _threshold(level: str) -> int:
    if level == "lite":
        return 10
    return 5  # standard + aggressive


def compress(text: str, *, level: str = "standard") -> EngineResult:
    validate_level(level)
    if level == "off" or not text:
        return EngineResult(
            text=text, original_chars=len(text), compressed_chars=len(text)
        )

    threshold = _threshold(level)
    lines = text.split("\n")
    keys = [_normalize(line, level) for line in lines]

    out: list[str] = []
    collapsed_lines = 0
    i = 0
    n = len(lines)
    while i < n:
        j = i + 1
        while j < n and keys[j] == keys[i]:
            j += 1
        run = j - i
        if run >= threshold:
            omitted = run - 2
            marker = f"[... {omitted} identical lines collapsed ...]"
            proposed = "\n".join((lines[i], marker, lines[j - 1]))
            original = "\n".join(lines[i:j])
            if len(proposed) < len(original):
                out.append(lines[i])
                out.append(marker)
                out.append(lines[j - 1])
                collapsed_lines += omitted
                i = j
                continue
        out.append(lines[i])
        i += 1

    out_text = "\n".join(out)
    note = f"collapsed {collapsed_lines} lines" if collapsed_lines else ""
    return make_result(text, out_text, note)
