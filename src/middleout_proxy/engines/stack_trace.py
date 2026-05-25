"""Collapse repeated stack frames in Python / JS / Java / Rust tracebacks.

Stdlib-only and deterministic.

Levels:
  off         identity
  lite        collapse runs of >=5 identical frames
  standard    collapse runs of >=3 identical frames; also runs of >=3 frames
              sharing file+function but with different line numbers
  aggressive  lower thresholds to >=2; additionally truncate any contiguous
              block of frame-or-marker lines longer than 5 down to
              head 2 + marker + tail 1

A "frame" is a single source line that matches one of the per-language
regexes below. Python's frame line is often followed by an indented context
line; we glue that into the frame's identity so that adjacent repeating
``recurse``-style frames collapse correctly.
"""

from __future__ import annotations

import re

from .base import EngineResult, make_result, validate_level

NAME = "stack_trace"

# Python: '  File "foo.py", line 10, in funcname'
_PY_FRAME_RE = re.compile(
    r'^\s*File "(?P<file>[^"]+)", line (?P<line>\d+), in (?P<func>\S+)\s*$'
)

# Java: '\tat com.foo.Bar.method(Bar.java:42)'
_JAVA_FRAME_RE = re.compile(
    r"^\s*at\s+(?P<func>[\w$.<>]+)\((?P<file>[\w./$]+):(?P<line>\d+)\)\s*$"
)

# JavaScript / Node (V8): 'at funcName (file.js:10:5)'
_JS_FRAME_RE = re.compile(
    r"^\s*at\s+(?P<func>[\w$.<>]+)\s+\((?P<file>[^():\s]+):(?P<line>\d+)(?::\d+)?\)\s*$"
)
# JavaScript anonymous frame: 'at file.js:10:5'
_JS_BARE_RE = re.compile(
    r"^\s*at\s+(?P<file>[^():\s]+):(?P<line>\d+)(?::\d+)?\s*$"
)

# Rust backtrace frame: '   12: my_crate::my_fn'
# Require at least one ``::`` so plain numbered prose lists (``1: alpha``,
# ``2: beta``) don't get mistaken for stack frames at aggressive level.
_RUST_FRAME_RE = re.compile(
    r"^\s*\d+:\s+(?P<func>[\w<>$]+(?:::[\w<>$]+)+)\s*$"
)

# A marker we emit; recognized so the aggressive pass can include them when
# scanning contiguous frame blocks to truncate.
_MARKER_RE = re.compile(r"^\s*\[\.\.\..*frames?.*\.\.\.\]\s*$")


def _parse_frame(line: str) -> tuple[str, str, str] | None:
    """Return (file, func, line_no) for any supported language, else None."""
    m = _PY_FRAME_RE.match(line)
    if m:
        return (m.group("file"), m.group("func"), m.group("line"))
    m = _JAVA_FRAME_RE.match(line)
    if m:
        return (m.group("file"), m.group("func"), m.group("line"))
    m = _JS_FRAME_RE.match(line)
    if m:
        return (m.group("file"), m.group("func"), m.group("line"))
    m = _JS_BARE_RE.match(line)
    if m:
        return (m.group("file"), "<anon>", m.group("line"))
    m = _RUST_FRAME_RE.match(line)
    if m:
        return ("<rust>", m.group("func"), "0")
    return None


def _is_context_line(line: str, frame_line: str) -> bool:
    """Heuristic: Python's frame is followed by a context line indented further."""
    if not line.strip():
        return False
    if _parse_frame(line) is not None:
        return False
    return _indent(line) > _indent(frame_line)


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip())


def _scan_units(lines: list[str]) -> list[tuple[str, tuple, tuple[int, int]]]:
    """Walk lines, emit (kind, identity, (start, end_exclusive)).

    A 'frame' unit may consume one or two source lines: the frame line itself,
    plus an immediately following context line if present. ``identity`` for
    a frame is ((file, func, line_no), context_text_or_None) so that adjacent
    identical (frame, context) pairs hash equal.

    Previously-emitted collapse markers are recognised as their own ``marker``
    unit so the aggressive truncate pass can group them with surrounding
    frame units.
    """
    units: list[tuple[str, tuple, tuple[int, int]]] = []
    i = 0
    n = len(lines)
    while i < n:
        if _MARKER_RE.match(lines[i]):
            units.append(("marker", (lines[i],), (i, i + 1)))
            i += 1
            continue
        parsed = _parse_frame(lines[i])
        if parsed is None:
            units.append(("other", (lines[i],), (i, i + 1)))
            i += 1
            continue
        consumed = 1
        ctx: str | None = None
        if i + 1 < n and _is_context_line(lines[i + 1], lines[i]):
            ctx = lines[i + 1]
            consumed = 2
        units.append(("frame", (parsed, ctx), (i, i + consumed)))
        i += consumed
    return units


def _level_config(level: str) -> tuple[int, int | None, bool]:
    """Return (identical_threshold, similar_threshold_or_None, truncate_blocks)."""
    if level == "lite":
        return 5, None, False
    if level == "standard":
        return 3, 3, False
    # aggressive
    return 2, 2, True


def _collapse_runs(
    lines: list[str], ident_thresh: int, similar_thresh: int | None
) -> tuple[list[str], int]:
    units = _scan_units(lines)
    out: list[str] = []
    collapsed = 0
    i = 0
    while i < len(units):
        kind, identity, (start, end) = units[i]
        if kind != "frame":
            out.extend(lines[start:end])
            i += 1
            continue

        # Run of identical (frame+context) units.
        j = i + 1
        while (
            j < len(units)
            and units[j][0] == "frame"
            and units[j][1] == identity
        ):
            j += 1
        run = j - i
        if run >= ident_thresh:
            out.append(f"[... {run} identical frames collapsed ...]")
            collapsed += run - 1
            i = j
            continue

        # Run of same (file, func) but different line.
        if similar_thresh is not None:
            file, func, _ = identity[0]
            j = i + 1
            while j < len(units) and units[j][0] == "frame":
                p_file, p_func, _ = units[j][1][0]
                if (p_file, p_func) != (file, func):
                    break
                j += 1
            run = j - i
            if run >= similar_thresh:
                out.append(
                    f"[... {run} similar frames in {func}() at {file} collapsed ...]"
                )
                collapsed += run - 1
                i = j
                continue

        # Default: emit the original lines for this unit.
        out.extend(lines[start:end])
        i += 1
    return out, collapsed


def _truncate_trace_blocks(lines: list[str]) -> tuple[list[str], int]:
    """Aggressive: any contiguous block of ``frame``/``marker`` units longer
    than 5 is reduced to head 2 + omission marker + tail 1.

    We scan in *units* (a frame may consume two source lines: header + context)
    so the truncated output keeps each visible frame and its context line
    together.
    """
    units = _scan_units(lines)
    out: list[str] = []
    truncated = 0
    i = 0
    n = len(units)
    while i < n:
        kind, _ident, (start, end) = units[i]
        if kind == "other":
            out.extend(lines[start:end])
            i += 1
            continue
        j = i
        while j < n and units[j][0] in ("frame", "marker"):
            j += 1
        block = units[i:j]
        if len(block) > 5:
            omitted = len(block) - 3
            for u in block[:2]:
                out.extend(lines[u[2][0] : u[2][1]])
            out.append(f"[... {omitted} frames omitted; truncated trace ...]")
            for u in block[-1:]:
                out.extend(lines[u[2][0] : u[2][1]])
            truncated += omitted
        else:
            for u in block:
                out.extend(lines[u[2][0] : u[2][1]])
        i = j
    return out, truncated


def compress(text: str, *, level: str = "standard") -> EngineResult:
    validate_level(level)
    if level == "off" or not text:
        return EngineResult(
            text=text, original_chars=len(text), compressed_chars=len(text)
        )

    ident_thresh, similar_thresh, truncate = _level_config(level)

    lines = text.split("\n")
    new_lines, collapsed = _collapse_runs(lines, ident_thresh, similar_thresh)
    truncated = 0
    if truncate:
        new_lines, truncated = _truncate_trace_blocks(new_lines)

    out_text = "\n".join(new_lines)

    parts = []
    if collapsed:
        parts.append(f"collapsed {collapsed} frames")
    if truncated:
        parts.append(f"truncated {truncated} frames")
    note = "; ".join(parts)

    return make_result(text, out_text, note)
