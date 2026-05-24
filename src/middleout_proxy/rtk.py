"""RTK (Rust Token Killer) - dictionary-based phrase abbreviation.

Pure-stdlib. Whole-word replacements only. Skips fenced code blocks, URLs,
and identifier-ish tokens (camelCase, snake_case, dotted paths).
"""

from __future__ import annotations

import re

_LEVELS = ("minimal", "standard", "aggressive")

# minimal: ~30 high-frequency programming phrases.
_MINIMAL = {
    "function": "fn",
    "functions": "fns",
    "return": "ret",
    "returns": "rets",
    "import": "imp",
    "imports": "imps",
    "implementation": "impl",
    "implementations": "impls",
    "configuration": "cfg",
    "configurations": "cfgs",
    "documentation": "doc",
    "parameter": "param",
    "parameters": "params",
    "argument": "arg",
    "arguments": "args",
    "variable": "var",
    "variables": "vars",
    "directory": "dir",
    "directories": "dirs",
    "database": "db",
    "databases": "dbs",
    "application": "app",
    "applications": "apps",
    "environment": "env",
    "environments": "envs",
    "repository": "repo",
    "repositories": "repos",
    "command": "cmd",
    "commands": "cmds",
    "object": "obj",
    "objects": "objs",
    "request": "req",
    "response": "resp",
}

# standard: + ~50 more common English compressions.
_STANDARD = {
    "approximately": "~",
    "because": "bc",
    "without": "w/o",
    "with": "w/",
    "between": "btwn",
    "before": "b4",
    "after": "aft",
    "through": "thru",
    "though": "tho",
    "although": "altho",
    "however": "hwvr",
    "therefore": "thus",
    "something": "smth",
    "someone": "s1",
    "anyone": "any1",
    "everyone": "evry1",
    "people": "ppl",
    "about": "abt",
    "around": "arnd",
    "really": "rly",
    "should": "shd",
    "would": "wd",
    "could": "cd",
    "number": "num",
    "numbers": "nums",
    "message": "msg",
    "messages": "msgs",
    "package": "pkg",
    "packages": "pkgs",
    "version": "ver",
    "versions": "vers",
    "service": "svc",
    "services": "svcs",
    "context": "ctx",
    "contexts": "ctxs",
    "reference": "ref",
    "references": "refs",
    "performance": "perf",
    "operation": "op",
    "operations": "ops",
    "execute": "exec",
    "executes": "execs",
    "execution": "exec",
    "production": "prod",
    "development": "dev",
    "testing": "test",
    "example": "ex",
    "examples": "exs",
    "different": "diff",
    "difference": "diff",
    "specific": "spec",
    "specification": "spec",
    "previous": "prev",
    "current": "cur",
    "minimum": "min",
    "maximum": "max",
}

# aggressive: + ~80 more (longer phrases, more ambiguous).
_AGGRESSIVE = {
    "as soon as possible": "ASAP",
    "for your information": "FYI",
    "in my opinion": "IMO",
    "by the way": "BTW",
    "for example": "e.g.",
    "that is": "i.e.",
    "and so on": "etc",
    "et cetera": "etc",
    "in other words": "iow",
    "as a result": "thus",
    "in addition": "also",
    "in conclusion": "thus",
    "on the other hand": "OTOH",
    "in the meantime": "meanwhile",
    "make sure": "ensure",
    "make sure to": "ensure",
    "in order to": "to",
    "due to": "from",
    "according to": "per",
    "regardless of": "despite",
    "in spite of": "despite",
    "as well as": "and",
    "such as": "like",
    "depending on": "per",
    "based on": "per",
    "in case of": "if",
    "in terms of": "re",
    "with respect to": "re",
    "with regard to": "re",
    "necessary": "needed",
    "additional": "extra",
    "approximately": "~",
    "approximately equal": "~",
    "currently": "now",
    "recently": "lately",
    "frequently": "often",
    "occasionally": "sometimes",
    "immediately": "now",
    "subsequently": "then",
    "consequently": "thus",
    "particularly": "esp",
    "especially": "esp",
    "generally": "usually",
    "typically": "usually",
    "primarily": "mainly",
    "fundamentally": "basically",
    "essentially": "basically",
    "absolutely": "yes",
    "definitely": "yes",
    "certainly": "yes",
    "probably": "prob",
    "possibly": "maybe",
    "process": "proc",
    "processes": "procs",
    "module": "mod",
    "modules": "mods",
    "library": "lib",
    "libraries": "libs",
    "directory": "dir",
    "schedule": "sched",
    "validate": "chk",
    "validation": "chk",
    "deprecated": "dep",
    "asynchronous": "async",
    "synchronous": "sync",
    "concurrent": "concur",
    "transaction": "tx",
    "transactions": "txs",
    "category": "cat",
    "categories": "cats",
    "language": "lang",
    "languages": "langs",
    "client": "cli",
    "clients": "clis",
    "server": "srv",
    "servers": "srvs",
    "framework": "fw",
    "frameworks": "fws",
    "interface": "iface",
    "interfaces": "ifaces",
    "structure": "struct",
    "structures": "structs",
    "string": "str",
    "strings": "strs",
    "integer": "int",
    "integers": "ints",
    "boolean": "bool",
    "booleans": "bools",
    "regular expression": "regex",
    "regular expressions": "regexes",
}


def _build_dict(level: str) -> dict[str, str]:
    d: dict[str, str] = {}
    d.update(_MINIMAL)
    if level in ("standard", "aggressive"):
        d.update(_STANDARD)
    if level == "aggressive":
        d.update(_AGGRESSIVE)
    return d


_CODE_FENCE_RE = re.compile(r"```")
_URL_RE = re.compile(r"https?://\S+")
_IDENT_RE = re.compile(r"\b(?:[a-z][a-zA-Z0-9]*[A-Z][a-zA-Z0-9_]*|[A-Za-z_][A-Za-z0-9_]*_[A-Za-z0-9_]*|[A-Za-z0-9_]+\.[A-Za-z0-9_.]+|/[^\s]+|\.{1,2}/[^\s]+)\b")


def _is_code_line(line: str) -> bool:
    return line.startswith("    ") or line.startswith("\t")


def _protect(text: str) -> tuple[str, list[str]]:
    placeholders: list[str] = []

    def stash(match: re.Match) -> str:
        placeholders.append(match.group(0))
        return f"\x00{len(placeholders) - 1}\x00"

    text = _URL_RE.sub(stash, text)
    text = _IDENT_RE.sub(stash, text)
    return text, placeholders


def _restore(text: str, placeholders: list[str]) -> str:
    return re.sub(r"\x00(\d+)\x00", lambda m: placeholders[int(m.group(1))], text)


def _apply_dict(text: str, mapping: dict[str, str]) -> str:
    # Sort longest phrases first to avoid prefix collisions.
    items = sorted(mapping.items(), key=lambda kv: -len(kv[0]))
    for src, dst in items:
        # Whole-word boundary; case-insensitive but only replaces when token is fully lowercase.
        pattern = re.compile(r"\b" + re.escape(src) + r"\b")
        text = pattern.sub(dst, text)
    return text


def compress_rtk(text: str, level: str = "minimal") -> str:
    """RTK token-killer. Replaces common phrases/words with abbreviations."""
    if level not in _LEVELS:
        raise ValueError(f"rtk level must be one of {_LEVELS}, got {level!r}")
    if not text:
        return text

    mapping = _build_dict(level)
    parts = _CODE_FENCE_RE.split(text)
    rebuilt: list[str] = []
    for i, segment in enumerate(parts):
        if i % 2 == 1:
            rebuilt.append(segment)
            continue
        lines = segment.split("\n")
        new_lines: list[str] = []
        for line in lines:
            if _is_code_line(line):
                new_lines.append(line)
                continue
            protected, placeholders = _protect(line)
            replaced = _apply_dict(protected, mapping)
            new_lines.append(_restore(replaced, placeholders))
        rebuilt.append("\n".join(new_lines))
    return "```".join(rebuilt)


if __name__ == "__main__":
    sample = (
        "The implementation of the function returns a configuration object. "
        "The documentation describes the parameters and arguments in detail. "
        "Without proper validation, the application will fail because the "
        "database connection requires specific environment variables. "
        "For example, you should make sure to set the variable before "
        "starting the service. In order to test, run the command.\n\n"
        "```python\n"
        "def the_function():\n"
        "    return configuration  # untouched\n"
        "```\n"
        "Visit https://example.com/docs for more information."
    )
    for lvl in _LEVELS:
        out = compress_rtk(sample, lvl)
        print(f"\n--- {lvl} ---")
        print(out)
        print(f"(orig={len(sample)} compressed={len(out)})")
