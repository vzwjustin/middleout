"""Strip comments inside fenced code blocks.

Stdlib-only. Deterministic. Only touches content between matched ``` fences;
prose outside fences is preserved byte-for-byte. Language is inferred from
the fence info string (``\u200b```python`` etc.); unknown fences are left
alone.

Levels:
  off         identity
  lite        full-line ``# ...`` / ``// ...`` comments only
  standard    lite + trailing line comments + single-line ``/* ... */``
  aggressive  standard + multi-line ``/* ... */`` + Python docstrings
              (triple-quoted strings immediately following ``def`` or
              ``class`` headers)

Safety contract:
  * Shebang lines (``#!``) are NEVER removed.
  * A comment marker preceded by an odd number of unescaped quote chars on
    the same line is treated as being inside a string literal and skipped.
  * ``//`` immediately preceded by ``:`` is treated as part of a URL.
"""

from __future__ import annotations

import re

from .base import EngineResult, make_result, validate_level

NAME = "comment_strip"

# A code fence line: optional leading whitespace then ``` plus an info string.
_FENCE_RE = re.compile(r"^(?P<indent>\s*)```(?P<info>\S*)\s*$")

# Multi-line C-style block comment: spans newlines, non-greedy.
_C_BLOCK_RE = re.compile(r"/\*.*?\*/", re.DOTALL)

# Python def/class header (allow async def).
_PY_DEF_RE = re.compile(r"^\s*(?:async\s+def|def|class)\s")


def _python_style() -> dict:
    return {"line": ("#",), "block": None, "docstring": True}


def _c_style() -> dict:
    return {"line": ("//",), "block": ("/*", "*/"), "docstring": False}


_LANG_STYLES: dict[str, dict] = {
    # Python + shell-likes (hash comments)
    "python": _python_style(),
    "py": _python_style(),
    "ruby": {"line": ("#",), "block": None, "docstring": False},
    "rb": {"line": ("#",), "block": None, "docstring": False},
    "yaml": {"line": ("#",), "block": None, "docstring": False},
    "yml": {"line": ("#",), "block": None, "docstring": False},
    "sh": {"line": ("#",), "block": None, "docstring": False},
    "bash": {"line": ("#",), "block": None, "docstring": False},
    "zsh": {"line": ("#",), "block": None, "docstring": False},
    "shell": {"line": ("#",), "block": None, "docstring": False},
    "perl": {"line": ("#",), "block": None, "docstring": False},
    "r": {"line": ("#",), "block": None, "docstring": False},
    "toml": {"line": ("#",), "block": None, "docstring": False},
    "ini": {"line": ("#", ";"), "block": None, "docstring": False},
    # C-likes (slash comments + block comments)
    "javascript": _c_style(),
    "js": _c_style(),
    "typescript": _c_style(),
    "ts": _c_style(),
    "tsx": _c_style(),
    "jsx": _c_style(),
    "java": _c_style(),
    "c": _c_style(),
    "cpp": _c_style(),
    "c++": _c_style(),
    "h": _c_style(),
    "hpp": _c_style(),
    "cs": _c_style(),
    "csharp": _c_style(),
    "go": _c_style(),
    "rust": _c_style(),
    "rs": _c_style(),
    "swift": _c_style(),
    "kotlin": _c_style(),
    "kt": _c_style(),
    "scala": _c_style(),
    "dart": _c_style(),
    "php": {"line": ("//", "#"), "block": ("/*", "*/"), "docstring": False},
}


def _detect_style(info: str) -> dict | None:
    info = info.strip().lower()
    if not info:
        return None
    # Some info strings carry attributes after a space or comma; take first token.
    info = re.split(r"[\s,{]", info, maxsplit=1)[0]
    return _LANG_STYLES.get(info)


def _find_comment_pos(line: str, marker: str) -> int | None:
    """Return first index of ``marker`` not inside a string literal, or None.

    Heuristic per spec: count unescaped single/double/backtick quote chars
    before the candidate position. If any class has an odd count, the marker
    is inside an open string.
    """
    start = 0
    while True:
        pos = line.find(marker, start)
        if pos == -1:
            return None
        before = line[:pos]
        # Count unescaped quotes for each delimiter class.
        in_string = False
        for q in ('"', "'", "`"):
            quotes = before.count(q) - before.count("\\" + q)
            if quotes % 2 == 1:
                in_string = True
                break
        if in_string:
            start = pos + len(marker)
            continue
        if marker == "//" and pos > 0 and line[pos - 1] == ":":
            start = pos + len(marker)
            continue
        return pos


def _remove_inline_block(line: str, open_tok: str, close_tok: str) -> str:
    """Remove ``/* ... */`` runs that open and close on the same line, ignoring
    matches that begin inside a string literal.
    """
    out: list[str] = []
    i = 0
    n = len(line)
    while i < n:
        if line.startswith(open_tok, i):
            # Skip if the open token is inside a string literal.
            before = line[:i]
            in_string = False
            for q in ('"', "'", "`"):
                quotes = before.count(q) - before.count("\\" + q)
                if quotes % 2 == 1:
                    in_string = True
                    break
            if in_string:
                out.append(line[i])
                i += 1
                continue
            end = line.find(close_tok, i + len(open_tok))
            if end == -1:
                out.append(line[i])
                i += 1
                continue
            i = end + len(close_tok)
            continue
        out.append(line[i])
        i += 1
    return "".join(out)


def _strip_trailing(line: str, style: dict) -> str:
    block = style.get("block")
    if block:
        line = _remove_inline_block(line, block[0], block[1])
    best_pos: int | None = None
    for marker in style["line"]:
        pos = _find_comment_pos(line, marker)
        if pos is not None and (best_pos is None or pos < best_pos):
            best_pos = pos
    if best_pos is None:
        return line
    return line[:best_pos].rstrip()


def _strip_multiline_c(content: str) -> str:
    return _C_BLOCK_RE.sub("", content)


def _strip_python_docstrings(content: str) -> str:
    """Remove triple-quoted strings appearing right after a ``def``/``class``
    header (skipping blank lines)."""
    lines = content.split("\n")
    out: list[str] = []
    i = 0
    n = len(lines)
    while i < n:
        out.append(lines[i])
        if not _PY_DEF_RE.match(lines[i]) or not lines[i].rstrip().endswith(":"):
            i += 1
            continue
        # Header line consumed; scan blanks then optional docstring.
        i += 1
        blanks: list[str] = []
        while i < n and not lines[i].strip():
            blanks.append(lines[i])
            i += 1
        if i >= n:
            out.extend(blanks)
            continue
        ds_line = lines[i].lstrip()
        quote = None
        for q in ('"""', "'''"):
            if ds_line.startswith(q):
                quote = q
                break
        if quote is None:
            out.extend(blanks)
            continue
        # Found docstring; skip blanks (drop them) and all docstring lines.
        rest = ds_line[len(quote):]
        if quote in rest:
            i += 1
            continue
        i += 1
        while i < n:
            if quote in lines[i]:
                i += 1
                break
            i += 1
    return "\n".join(out)


def _process_block(content: str, style: dict, level: str) -> str:
    if level == "aggressive":
        if style.get("block"):
            content = _strip_multiline_c(content)
        if style.get("docstring"):
            content = _strip_python_docstrings(content)

    lines = content.split("\n")
    out: list[str] = []
    line_markers = style["line"]
    block_for_full_line = style.get("block")
    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith("#!"):
            out.append(line)
            continue
        is_full = False
        for marker in line_markers:
            if stripped.startswith(marker):
                is_full = True
                break
        if not is_full and block_for_full_line:
            if stripped.startswith(block_for_full_line[0]) and stripped.rstrip().endswith(
                block_for_full_line[1]
            ):
                is_full = True
        if is_full:
            continue
        if level == "lite":
            out.append(line)
        else:
            out.append(_strip_trailing(line, style))
    return "\n".join(out)


def compress(text: str, *, level: str = "standard") -> EngineResult:
    validate_level(level)
    if level == "off" or not text or "```" not in text:
        return EngineResult(
            text=text, original_chars=len(text), compressed_chars=len(text)
        )

    lines = text.split("\n")
    out_lines: list[str] = []
    in_fence = False
    fence_style: dict | None = None
    buf: list[str] = []
    stripped_lines = 0

    def flush_block() -> None:
        nonlocal stripped_lines, buf
        if fence_style is None:
            out_lines.extend(buf)
        else:
            content = "\n".join(buf)
            processed = _process_block(content, fence_style, level)
            stripped_lines += content.count("\n") - processed.count("\n")
            out_lines.extend(processed.split("\n"))
        buf = []

    for line in lines:
        m = _FENCE_RE.match(line)
        if m:
            if in_fence:
                flush_block()
                in_fence = False
                fence_style = None
            else:
                in_fence = True
                fence_style = _detect_style(m.group("info"))
            out_lines.append(line)
            continue
        if in_fence:
            buf.append(line)
        else:
            out_lines.append(line)

    if in_fence:
        # Unclosed fence: pass-through, do not process.
        out_lines.extend(buf)

    out_text = "\n".join(out_lines)
    note = f"stripped {stripped_lines} comment lines" if stripped_lines else ""
    return make_result(text, out_text, note)
