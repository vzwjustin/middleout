"""Caveman-style compression: drop articles, filler, pleasantries.

Pure-stdlib. Operates only on prose; passes code blocks, URLs, and identifiers
through untouched. Output is deterministic.
"""

from __future__ import annotations

import re

_LEVELS = ("lite", "standard", "aggressive", "ultra")

# Words considered "filler" — dropped at every level.
_FILLER = {
    "very", "really", "just", "quite", "actually", "basically",
    "literally", "simply", "essentially",
}

# Articles — dropped at every level (but only when safe: not at sentence start
# capitalized, and not adjacent to punctuation that would change meaning).
_ARTICLES = {"the", "a", "an"}

# Pleasantries (multi-word phrases handled via regex).
_PLEASANTRY_PATTERNS = [
    (re.compile(r"\bplease\s+", re.IGNORECASE), ""),
    (re.compile(r"\bthanks?(?:\s+you)?\b[,.!]?\s*", re.IGNORECASE), ""),
    (re.compile(r"\bthank you\b[,.!]?\s*", re.IGNORECASE), ""),
    (re.compile(r"\bcould you\b\s*", re.IGNORECASE), ""),
    (re.compile(r"\bwould you\b\s*", re.IGNORECASE), ""),
    (re.compile(r"\bcan you\b\s*", re.IGNORECASE), ""),
]

# Phrase collapses (standard level).
_PHRASE_COLLAPSES = [
    (re.compile(r"\bin order to\b", re.IGNORECASE), "to"),
    (re.compile(r"\bmake sure to\b", re.IGNORECASE), "ensure"),
    (re.compile(r"\bmake sure that\b", re.IGNORECASE), "ensure"),
    (re.compile(r"\byou should\b", re.IGNORECASE), "do"),
    (re.compile(r"\bin terms of\b", re.IGNORECASE), "re"),
    (re.compile(r"\bas well as\b", re.IGNORECASE), "and"),
    (re.compile(r"\bdue to the fact that\b", re.IGNORECASE), "because"),
    (re.compile(r"\bat this point in time\b", re.IGNORECASE), "now"),
]

# Aggressive abbreviations (whole-word, case-preserving lower-only).
_AGGRESSIVE_ABBR = {
    "function": "fn",
    "functions": "fns",
    "return": "ret",
    "returns": "rets",
    "should": "shd",
    "would": "wd",
    "could": "cd",
    "implementation": "impl",
    "configuration": "cfg",
    "documentation": "doc",
    "parameter": "param",
    "parameters": "params",
    "argument": "arg",
    "arguments": "args",
    "variable": "var",
    "variables": "vars",
    "between": "btwn",
    "approximately": "~",
}

# Ultra-level: drop these conjunctions/copulas when surrounded by safe context.
_ULTRA_DROP = {"is", "are", "was", "were", "am", "be", "been", "being", "and", "or"}

_CODE_FENCE_RE = re.compile(r"```")
_URL_RE = re.compile(r"https?://\S+")
# Identifier-ish tokens: camelCase, snake_case, paths, dot-paths.
_IDENT_RE = re.compile(r"\b(?:[a-z][a-zA-Z0-9]*[A-Z][a-zA-Z0-9_]*|[A-Za-z_][A-Za-z0-9_]*_[A-Za-z0-9_]*|[A-Za-z0-9_]+\.[A-Za-z0-9_.]+|/[^\s]+|\.{1,2}/[^\s]+)\b")


def _is_code_line(line: str) -> bool:
    return line.startswith("    ") or line.startswith("\t")


def _protect(text: str) -> tuple[str, list[str]]:
    """Replace URLs/identifiers with placeholders. Returns text + placeholder list."""
    placeholders: list[str] = []

    def stash(match: re.Match) -> str:
        placeholders.append(match.group(0))
        return f"\x00{len(placeholders) - 1}\x00"

    text = _URL_RE.sub(stash, text)
    text = _IDENT_RE.sub(stash, text)
    return text, placeholders


def _restore(text: str, placeholders: list[str]) -> str:
    def unstash(match: re.Match) -> str:
        idx = int(match.group(1))
        return placeholders[idx]

    return re.sub(r"\x00(\d+)\x00", unstash, text)


def _process_prose(text: str, level: str) -> str:
    text, placeholders = _protect(text)

    if level in ("standard", "aggressive", "ultra"):
        for pat, repl in _PLEASANTRY_PATTERNS:
            text = pat.sub(repl, text)
        for pat, repl in _PHRASE_COLLAPSES:
            text = pat.sub(repl, text)

    tokens = re.split(r"(\s+)", text)
    out: list[str] = []
    for tok in tokens:
        if not tok or tok.isspace():
            out.append(tok)
            continue
        # Strip trailing punctuation for matching.
        m = re.match(r"^(\W*)(.*?)(\W*)$", tok, re.DOTALL)
        if not m:
            out.append(tok)
            continue
        lead, core, trail = m.group(1), m.group(2), m.group(3)
        low = core.lower()

        # Lite+: drop filler.
        if low in _FILLER:
            continue
        # Lite+: drop articles (only when fully lowercase to avoid sentence starts).
        if low in _ARTICLES and core.islower():
            continue
        # Aggressive: abbreviate.
        if level in ("aggressive", "ultra") and low in _AGGRESSIVE_ABBR:
            core = _AGGRESSIVE_ABBR[low]
        # Aggressive: drop "that" before clauses.
        if level in ("aggressive", "ultra") and low == "that" and core.islower():
            continue
        # Ultra: drop copulas/conjunctions.
        if level == "ultra" and low in _ULTRA_DROP and core.islower():
            continue

        out.append(f"{lead}{core}{trail}")

    text = "".join(out)
    # Collapse runs of whitespace introduced by deletions, but preserve newlines.
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    return _restore(text, placeholders)


def compress_caveman(text: str, level: str = "standard") -> str:
    """Caveman-style compression. Drops articles, filler, pleasantries."""
    if level not in _LEVELS:
        raise ValueError(f"caveman level must be one of {_LEVELS}, got {level!r}")
    if not text:
        return text

    # Split on code fences; alternating segments are prose / code.
    parts = _CODE_FENCE_RE.split(text)
    rebuilt: list[str] = []
    for i, segment in enumerate(parts):
        if i % 2 == 1:
            # Inside a fenced code block — pass through.
            rebuilt.append(segment)
            continue
        # Process line-by-line so we can skip indented code lines.
        lines = segment.split("\n")
        processed = [line if _is_code_line(line) else _process_prose(line, level) for line in lines]
        rebuilt.append("\n".join(processed))
    return "```".join(rebuilt)


if __name__ == "__main__":
    sample = (
        "Hello, could you please make sure to return the value of the function? "
        "Actually, it really is very important that you should call myFunction() "
        "and return the configuration. Thanks!\n\n"
        "Here is some code:\n"
        "```\n"
        "def the_function():\n"
        "    return the_value  # do not touch this\n"
        "```\n"
        "In order to test, visit https://example.com/path and run the implementation."
    )
    for lvl in _LEVELS:
        print(f"\n--- {lvl} ---")
        print(compress_caveman(sample, lvl))
        print(f"(orig={len(sample)} compressed={len(compress_caveman(sample, lvl))})")
