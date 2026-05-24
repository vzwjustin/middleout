"""Collapse oversized JSON arrays and objects.

Stdlib-only. Deterministic. The whole input must parse as JSON via
``json.loads``; if it doesn't, the text is returned unchanged (per spec, we
do not try to extract JSON from arbitrary prose).

Levels:
  off         identity
  lite        arrays only, threshold >=50 items
  standard    arrays >=20 items, objects >=50 keys
  aggressive  arrays >=10 items, objects >=20 keys

An over-sized array is rewritten to ``head 3 + marker + tail 2`` where the
marker reports how many items were dropped and a frequency-count of their
JSON types (``int=24, str=2``). An over-sized object becomes
``head 5 + marker_key + tail 3`` keys.

Nested structures are walked and collapsed at every depth.
"""

from __future__ import annotations

import json

from .base import EngineResult, make_result, validate_level

NAME = "json_collapse"


class _Marker(str):
    """Marker string that serializes as a normal JSON string."""


_OMITTED_KEY = "__middleout_omitted__"


def _level_config(level: str) -> tuple[int, int | None]:
    """Return (array_threshold, object_threshold_or_None)."""
    if level == "lite":
        return 50, None
    if level == "standard":
        return 20, 50
    return 10, 20  # aggressive


def _type_name(value: object) -> str:
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    if isinstance(value, list):
        return "list"
    if isinstance(value, dict):
        return "dict"
    if value is None:
        return "null"
    return type(value).__name__


def _type_counts(items: list) -> str:
    counts: dict[str, int] = {}
    for v in items:
        t = _type_name(v)
        counts[t] = counts.get(t, 0) + 1
    # Sorted by type name for determinism.
    return ", ".join(f"{t}={n}" for t, n in sorted(counts.items()))


class _Collapser:
    def __init__(self, arr_th: int, obj_th: int | None) -> None:
        self.arr_th = arr_th
        self.obj_th = obj_th
        self.collapses = 0

    def walk(self, obj: object) -> object:
        if isinstance(obj, list):
            return self._walk_list(obj)
        if isinstance(obj, dict):
            return self._walk_dict(obj)
        return obj

    def _walk_list(self, items: list) -> list:
        walked = [self.walk(v) for v in items]
        if len(walked) >= self.arr_th:
            head = walked[:3]
            tail = walked[-2:]
            omitted_original = items[3 : len(items) - 2]
            omitted = len(walked) - 5
            marker_text = (
                f"[... {omitted} items omitted; types: "
                f"{_type_counts(omitted_original)} ...]"
            )
            self.collapses += 1
            return [*head, _Marker(marker_text), *tail]
        return walked

    def _walk_dict(self, d: dict) -> dict:
        walked = {k: self.walk(v) for k, v in d.items()}
        if self.obj_th is not None and len(walked) >= self.obj_th:
            keys = list(walked.keys())
            head_keys = keys[:5]
            tail_keys = keys[-3:]
            omitted = len(keys) - 8
            result: dict = {k: walked[k] for k in head_keys}
            result[_OMITTED_KEY] = _Marker(
                f"[... {omitted} keys omitted ...]"
            )
            for k in tail_keys:
                result[k] = walked[k]
            self.collapses += 1
            return result
        return walked


def _serialize(obj: object) -> str:
    # Stable separators give deterministic output regardless of the input's
    # original whitespace.
    return json.dumps(obj, separators=(", ", ": "), ensure_ascii=False)


def compress(text: str, *, level: str = "standard") -> EngineResult:
    validate_level(level)
    if level == "off" or not text:
        return EngineResult(
            text=text, original_chars=len(text), compressed_chars=len(text)
        )

    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return EngineResult(
            text=text, original_chars=len(text), compressed_chars=len(text)
        )

    arr_th, obj_th = _level_config(level)
    collapser = _Collapser(arr_th, obj_th)
    new_struct = collapser.walk(parsed)

    if collapser.collapses == 0:
        return EngineResult(
            text=text, original_chars=len(text), compressed_chars=len(text)
        )

    out_text = _serialize(new_struct)
    # Don't fight the input: if serialization happens to be longer than the
    # source (e.g. forced reformatting of an already-compact doc), give up.
    if len(out_text) >= len(text):
        return EngineResult(
            text=text, original_chars=len(text), compressed_chars=len(text)
        )

    note = f"{collapser.collapses} structures collapsed"
    return make_result(text, out_text, note)
