"""Cache wall: the single source of truth for what the proxy is allowed to mutate.

The Anthropic prompt cache hashes the byte-identical prefix of an outgoing request
up through the last `cache_control` marker. Any byte change left of that marker
invalidates the cache and forces a full prefill on the next request. This module
defines the proxy-wide invariant that protects that prefix.

Model
-----
A request payload is split into two regions:

  [ system | tools | messages...up-to-wall ]  +  [ messages...after-wall, current user turn ]
  ---------------- protected ----------------     ---------------- volatile ----------------

The "wall" is the position of the **last** `cache_control` marker in canonical
processing order (system blocks, then tools, then messages in order). Every byte
at or before the wall is sacred. Every byte after the wall is fair game for
compression / dedupe / transformation.

When the incoming request has no `cache_control` markers at all, the proxy may
(at the caller's option) auto-insert one after `[system][tools]` so the prefix
becomes cacheable upstream. This is the only sanctioned way the proxy modifies
the prefix; even then, the inserted marker carries the byte position the client
*would* have set if it had known to.

API
---
- `compute_wall(payload, *, auto_insert=False) -> CacheWall` — inspect payload,
  optionally insert a wall, return descriptor.
- `CacheWall.is_protected(kind, msg_idx, block_idx) -> bool` — the protection
  oracle every engine must consult before touching a block.
- `CacheWall.split(payload) -> tuple[ProtectedView, VolatileView]` — convenience
  for callers that only want the tail.
- `assert_prefix_unchanged(original_bytes, outgoing_bytes, wall)` — invariant
  assertion for tests and dev-mode runtime checks.

The legacy `_payload_cache_protection` / `_is_block_protected` helpers in
`compression.py` remain in place for the existing engines; this module is the
forward-looking replacement. Both must agree on every input — see
`tests/test_cache_wall.py::test_parity_with_legacy_protection`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable


# Canonical kinds of blocks that participate in the prefix. Ordered by Anthropic
# processing order: system first, then tools, then messages.
_KIND_SYSTEM = "system"
_KIND_TOOLS = "tools"
_KIND_MESSAGE = "message"

_KIND_ORDER = {_KIND_SYSTEM: 0, _KIND_TOOLS: 1, _KIND_MESSAGE: 2}


@dataclass(frozen=True)
class WallMarker:
    """Identifies one cache_control marker location in a payload.

    `msg_idx` is None for system/tools markers, an int index into `messages` for
    message markers. `block_idx` is the index within that list.
    """

    kind: str
    msg_idx: int | None
    block_idx: int

    def _order_key(self) -> tuple[int, int, int]:
        # Bigger = later in processing order.
        return (_KIND_ORDER[self.kind], self.msg_idx if self.msg_idx is not None else -1, self.block_idx)

    def __le__(self, other: "WallMarker") -> bool:  # type: ignore[override]
        return self._order_key() <= other._order_key()


@dataclass
class CacheWall:
    """Descriptor of the cache-stable prefix of a request payload.

    `marker` is None when the payload has no cache_control anywhere — in that
    case every block is volatile. `auto_inserted` is True if the proxy inserted
    the marker (vs. it being client-supplied).
    """

    marker: WallMarker | None
    auto_inserted: bool = False
    all_markers: tuple[WallMarker, ...] = field(default_factory=tuple)

    @property
    def has_marker(self) -> bool:
        return self.marker is not None

    def is_protected(
        self,
        *,
        kind: str,
        msg_idx: int | None,
        block_idx: int,
    ) -> bool:
        """True if (kind, msg_idx, block_idx) is at or before the wall.

        Protected blocks must NEVER be mutated. Engines consult this oracle on
        every block before deciding whether to touch it.
        """
        if self.marker is None:
            return False
        m = self.marker
        if kind not in _KIND_ORDER:
            return False
        order = _KIND_ORDER[kind]
        m_order = _KIND_ORDER[m.kind]
        if order < m_order:
            return True
        if order > m_order:
            return False
        # Same kind — compare within-kind position.
        if kind == _KIND_SYSTEM or kind == _KIND_TOOLS:
            return block_idx <= m.block_idx
        # message
        if msg_idx is None or m.msg_idx is None:
            return False
        if msg_idx < m.msg_idx:
            return True
        if msg_idx > m.msg_idx:
            return False
        return block_idx <= m.block_idx


def _find_all_markers(payload: dict[str, Any]) -> list[WallMarker]:
    found: list[WallMarker] = []

    system = payload.get("system")
    if isinstance(system, list):
        for i, block in enumerate(system):
            if isinstance(block, dict) and "cache_control" in block:
                found.append(WallMarker(_KIND_SYSTEM, None, i))

    tools = payload.get("tools")
    if isinstance(tools, list):
        for i, tool in enumerate(tools):
            if isinstance(tool, dict) and "cache_control" in tool:
                found.append(WallMarker(_KIND_TOOLS, None, i))

    messages = payload.get("messages")
    if isinstance(messages, list):
        for mi, message in enumerate(messages):
            if not isinstance(message, dict):
                continue
            content = message.get("content")
            if isinstance(content, list):
                for bi, block in enumerate(content):
                    if isinstance(block, dict) and "cache_control" in block:
                        found.append(WallMarker(_KIND_MESSAGE, mi, bi))

    return found


def _last_marker(markers: Iterable[WallMarker]) -> WallMarker | None:
    best: WallMarker | None = None
    best_key: tuple[int, int, int] | None = None
    for m in markers:
        key = m._order_key()
        if best_key is None or key > best_key:
            best, best_key = m, key
    return best


def compute_wall(payload: dict[str, Any], *, auto_insert: bool = False) -> CacheWall:
    """Inspect a payload, return the cache wall descriptor.

    With `auto_insert=True`, if no marker exists, the proxy stamps an ephemeral
    breakpoint on the last item of `[system][tools]` (whichever exists last) so
    the [system][tools] prefix becomes cacheable. The mutation is reflected in
    the returned `CacheWall.auto_inserted=True` and the payload itself is
    modified in place — callers that need byte-identity to the original request
    must snapshot the bytes before calling with `auto_insert=True`.
    """
    markers = _find_all_markers(payload)
    last = _last_marker(markers)

    if last is not None or not auto_insert:
        return CacheWall(marker=last, auto_inserted=False, all_markers=tuple(markers))

    inserted = _auto_insert_breakpoint(payload)
    if inserted is None:
        # Nothing to anchor a wall to — payload has no system/tools.
        return CacheWall(marker=None, auto_inserted=False, all_markers=())
    return CacheWall(marker=inserted, auto_inserted=True, all_markers=(inserted,))


def _auto_insert_breakpoint(payload: dict[str, Any]) -> WallMarker | None:
    """Mutate `payload` to insert a cache_control breakpoint after [system][tools].

    Strategy: insert on the *last* element of `tools` if present, otherwise on
    the last element of `system` if present and list-shaped. If `system` is a
    plain string, we cannot attach cache_control there without changing the
    schema, so we promote it to a single-block list `[{"type":"text","text":...,
    "cache_control":{"type":"ephemeral"}}]`. Returns the inserted marker, or
    None if there is nothing to anchor to.
    """
    breakpoint = {"type": "ephemeral"}

    tools = payload.get("tools")
    if isinstance(tools, list) and tools:
        last = tools[-1]
        if isinstance(last, dict):
            last["cache_control"] = breakpoint
            return WallMarker(_KIND_TOOLS, None, len(tools) - 1)

    system = payload.get("system")
    if isinstance(system, list) and system:
        last = system[-1]
        if isinstance(last, dict):
            last["cache_control"] = breakpoint
            return WallMarker(_KIND_SYSTEM, None, len(system) - 1)
    if isinstance(system, str) and system:
        payload["system"] = [
            {"type": "text", "text": system, "cache_control": breakpoint},
        ]
        return WallMarker(_KIND_SYSTEM, None, 0)

    return None


def iter_volatile_blocks(
    payload: dict[str, Any],
    wall: CacheWall,
) -> Iterable[tuple[str, int | None, int, dict[str, Any] | str]]:
    """Yield `(kind, msg_idx, block_idx, block)` for every volatile (post-wall) block.

    System and tools blocks are intentionally never yielded — even if a future
    marker placement somehow ended up earlier in messages, system/tools always
    sit in the prefix region and are not eligible for compression.
    """
    messages = payload.get("messages")
    if not isinstance(messages, list):
        return

    for mi, message in enumerate(messages):
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if isinstance(content, list):
            for bi, block in enumerate(content):
                if wall.is_protected(kind=_KIND_MESSAGE, msg_idx=mi, block_idx=bi):
                    continue
                yield _KIND_MESSAGE, mi, bi, block
        elif isinstance(content, str):
            if wall.is_protected(kind=_KIND_MESSAGE, msg_idx=mi, block_idx=0):
                continue
            yield _KIND_MESSAGE, mi, 0, content


def assert_prefix_unchanged(
    original: bytes,
    outgoing: bytes,
    wall: CacheWall,
    *,
    prefix_len: int | None = None,
) -> None:
    """Assert that no byte left of the wall changed between original and outgoing.

    `prefix_len` is the byte length of the prefix in the encoded request. When
    None (the common case), and the wall is present, we conservatively require
    the entire `original` to equal `outgoing` up to the minimum length of the
    two — this is the strongest invariant and the cheapest check.

    Used in tests and in dev-mode runtime checks; not on the hot path.
    """
    if not wall.has_marker:
        return
    if prefix_len is None:
        prefix_len = min(len(original), len(outgoing))
    if original[:prefix_len] != outgoing[:prefix_len]:
        # Find first differing byte for the error message.
        for i, (a, b) in enumerate(zip(original[:prefix_len], outgoing[:prefix_len])):
            if a != b:
                raise CacheWallViolation(
                    f"Cache prefix mutated at byte {i}: "
                    f"original={original[max(0,i-16):i+16]!r}, "
                    f"outgoing={outgoing[max(0,i-16):i+16]!r}"
                )


class CacheWallViolation(AssertionError):
    """Raised when a byte left of the cache wall changed.

    A violation here means the next upstream request will miss prompt cache and
    force a full prefill. Always a bug, never a perf regression — fix at root.
    """


__all__ = [
    "CacheWall",
    "CacheWallViolation",
    "WallMarker",
    "assert_prefix_unchanged",
    "compute_wall",
    "iter_volatile_blocks",
]
