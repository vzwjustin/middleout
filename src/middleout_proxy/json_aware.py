"""JSON-aware compression: minify JSON blocks, collapse whitespace in code/prose.

Pure-stdlib. Deterministic. Operates safely:
  - JSON minification is only applied when a block parses cleanly.
  - Whitespace normalization preserves indentation inside whitespace-sensitive
    fenced code blocks (python, yaml, makefile).
  - Aggressive level strips JSONC comments (// and /* */) and trailing commas,
    but refuses when the comment marker appears inside a JSON string literal.

Levels:
  - safe       : JSON minify only when block parses.
  - standard   : + collapse blank lines / trim trailing whitespace in prose and
                 in non-whitespace-sensitive fenced blocks.
  - aggressive : + strip JSONC comments and trailing commas before reparsing;
                 refuses when comment markers appear inside string literals.

Public API:
  compress(text: str, level: str) -> tuple[str, dict]
      Returns (new_text, stats={chars_in, chars_out, blocks_found}).
"""

from __future__ import annotations

import json
import re

_LEVELS = ("safe", "standard", "aggressive")

_FENCE_SPLIT_RE = re.compile(r"```")

_WHITESPACE_SAFE_LANGS = frozenset(
    {"python", "py", "yaml", "yml", "makefile", "make", "haml", "coffee", "coffeescript",
     "fsharp", "f#", "sass"}
)

_HEADER_LANG_RE = re.compile(r"^([A-Za-z0-9_+\-]*)\s*(.*)$", re.DOTALL)


def _parse_json_strict(text: str) -> object | None:
    """Try to parse text as JSON. Return parsed object on success, None otherwise."""
    stripped = text.strip()
    if not stripped:
        return None
    if stripped[0] not in "{[":
        return None
    try:
        return json.loads(stripped)
    except (ValueError, RecursionError):
        return None


def _minify_json(parsed: object) -> str:
    """Produce a deterministic minified JSON string."""
    return json.dumps(parsed, ensure_ascii=False, separators=(",", ":"), sort_keys=False)


def _has_comment_marker_in_string(text: str) -> bool:
    """Detect if // or /* appears inside a JSON string literal."""
    in_string = False
    escape = False
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            elif ch == "/" and i + 1 < n and text[i + 1] in "/*":
                return True
            i += 1
            continue
        if ch == '"':
            in_string = True
        i += 1
    return False


_JSONC_LINE_COMMENT_RE = re.compile(r"//[^\n]*")
_JSONC_BLOCK_COMMENT_RE = re.compile(r"/\*.*?\*/", re.DOTALL)
_TRAILING_COMMA_RE = re.compile(r",(\s*[}\]])")


def _strip_jsonc(text: str) -> str | None:
    """Strip JSONC comments and trailing commas. Returns None if unsafe."""
    if _has_comment_marker_in_string(text):
        return None
    stripped = _JSONC_BLOCK_COMMENT_RE.sub("", text)
    stripped = _JSONC_LINE_COMMENT_RE.sub("", stripped)
    stripped = _TRAILING_COMMA_RE.sub(r"\1", stripped)
    return stripped


def _try_minify_block(text: str, level: str) -> tuple[str, bool]:
    """Attempt to minify a block of text as JSON. Returns (new_text, did_minify)."""
    parsed = _parse_json_strict(text)
    if parsed is not None:
        return _minify_json(parsed), True
    if level == "aggressive":
        stripped = _strip_jsonc(text)
        if stripped is None:
            return text, False
        parsed = _parse_json_strict(stripped)
        if parsed is not None:
            return _minify_json(parsed), True
    return text, False


def _collapse_prose_whitespace(text: str) -> str:
    """Trim trailing whitespace on lines and collapse 3+ blank lines to 2."""
    lines = text.split("\n")
    trimmed = [ln.rstrip() for ln in lines]
    out: list[str] = []
    blank_run = 0
    for ln in trimmed:
        if ln == "":
            blank_run += 1
            if blank_run <= 2:
                out.append(ln)
        else:
            blank_run = 0
            out.append(ln)
    return "\n".join(out)


def _process_fenced_segment(segment: str, level: str) -> tuple[str, int]:
    """Process the content of a single fenced segment.

    Returns (new_segment, blocks_minified). The segment includes the language
    header on its first line.
    """
    header_end = segment.find("\n")
    if header_end == -1:
        return segment, 0
    header = segment[:header_end]
    body = segment[header_end + 1:]
    lang_match = _HEADER_LANG_RE.match(header)
    lang = (lang_match.group(1) if lang_match else "").lower()

    blocks_minified = 0
    new_body = body

    # Only attempt JSON minification when the fence is explicitly a JSON-ish
    # language, or when there is no language tag at all. A ``python``/``js``
    # fence whose body happens to also parse as JSON is still source code in
    # that language and must keep its formatting.
    json_eligible = lang in ("json", "jsonc", "json5") or lang == ""
    if json_eligible and (
        _parse_json_strict(body) is not None
        or (level == "aggressive" and body.lstrip().startswith(("{", "[")))
    ):
        minified, did = _try_minify_block(body, level)
        if did:
            new_body = minified
            blocks_minified += 1

    if level in ("standard", "aggressive") and lang not in _WHITESPACE_SAFE_LANGS:
        if blocks_minified == 0:
            new_body = _collapse_prose_whitespace(new_body)

    # Preserve a trailing newline before the closing fence so it stays at
    # column 0 — markdown requires the closing ``` to start its own line.
    if body.endswith("\n") and not new_body.endswith("\n"):
        new_body = new_body + "\n"
    return f"{header}\n{new_body}", blocks_minified


def _process_prose_segment(segment: str, level: str) -> tuple[str, int]:
    """Try to JSON-minify the prose segment if it looks like JSON.

    Always returns prose with whitespace collapsed at standard+.
    """
    blocks_minified = 0
    candidate = segment.strip()
    new_segment = segment
    if candidate and candidate[0] in "{[":
        minified, did = _try_minify_block(segment, level)
        if did:
            leading = segment[: len(segment) - len(segment.lstrip())]
            trailing = segment[len(segment.rstrip()):]
            new_segment = f"{leading}{minified}{trailing}"
            blocks_minified += 1
    if level in ("standard", "aggressive") and blocks_minified == 0:
        new_segment = _collapse_prose_whitespace(new_segment)
    return new_segment, blocks_minified


def compress(text: str, level: str = "safe") -> tuple[str, dict]:
    """JSON-aware compression. Returns (new_text, stats)."""
    if level not in _LEVELS:
        raise ValueError(f"json_aware level must be one of {_LEVELS}, got {level!r}")
    chars_in = len(text)
    if not text:
        return text, {"chars_in": 0, "chars_out": 0, "blocks_found": 0}

    parts = _FENCE_SPLIT_RE.split(text)
    rebuilt: list[str] = []
    blocks_found = 0
    for i, segment in enumerate(parts):
        if i % 2 == 1:
            new_segment, hits = _process_fenced_segment(segment, level)
        else:
            new_segment, hits = _process_prose_segment(segment, level)
        rebuilt.append(new_segment)
        blocks_found += hits

    out = "```".join(rebuilt)
    return out, {
        "chars_in": chars_in,
        "chars_out": len(out),
        "blocks_found": blocks_found,
    }
