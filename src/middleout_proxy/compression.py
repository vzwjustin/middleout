from __future__ import annotations

import copy
import hashlib
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from threading import Lock
from typing import Any

from .caveman import compress_caveman
from .config import Settings
from .jl import RequestSketchIndex
from .rtk import compress_rtk


@dataclass
class CompressionEvent:
    path: str
    mode: str
    original_chars: int
    compressed_chars: int
    sha256: str
    note: str = ""
    sample_before: str | None = None
    sample_after: str | None = None

    @property
    def chars_saved(self) -> int:
        return max(0, self.original_chars - self.compressed_chars)


@dataclass
class CompressionAudit:
    endpoint: str
    events: list[CompressionEvent] = field(default_factory=list)
    cache_hits: int = 0
    cache_misses: int = 0
    protected_blocks: int = 0

    @property
    def original_chars(self) -> int:
        return sum(e.original_chars for e in self.events)

    @property
    def compressed_chars(self) -> int:
        return sum(e.compressed_chars for e in self.events)

    @property
    def chars_saved(self) -> int:
        return sum(e.chars_saved for e in self.events)

    @property
    def touched(self) -> bool:
        return bool(self.events)

    def to_dict(self) -> dict[str, Any]:
        return {
            "endpoint": self.endpoint,
            "events": [asdict(e) for e in self.events],
            "original_chars": self.original_chars,
            "compressed_chars": self.compressed_chars,
            "chars_saved": self.chars_saved,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "protected_blocks": self.protected_blocks,
        }


def sha256_short(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:16]


def middle_out_text(
    text: str,
    *,
    max_chars: int,
    min_omission_chars: int,
    head_fraction: float,
) -> str:
    """Lossy middle-out compaction preserving the beginning and end of a long text."""
    if len(text) <= max_chars:
        return text

    digest = sha256_short(text)
    marker_template = (
        "\n\n[... middle-out compressed locally: omitted {omitted} chars; "
        "original_chars={original}; sha256={digest}; not reversible by the model ...]\n\n"
    )

    # Initial marker estimate; then stabilize the marker length below.
    marker = marker_template.format(omitted=len(text) - max_chars, original=len(text), digest=digest)
    budget = max(128, max_chars - len(marker))
    head_chars = max(64, int(budget * head_fraction))
    tail_chars = max(64, budget - head_chars)
    if head_chars + tail_chars >= len(text) - min_omission_chars:
        return text

    # FIX H (bug-hunter): marker length can flip a digit when `omitted` changes,
    # which shifts the budget which shifts head/tail which shifts the real
    # omitted count again. Iterate to a fixed point (max 3 rounds in practice)
    # so the audit's `omitted` value in the marker actually matches the chars
    # we're omitting.
    for _ in range(3):
        omitted = len(text) - head_chars - tail_chars
        new_marker = marker_template.format(omitted=omitted, original=len(text), digest=digest)
        if len(new_marker) == len(marker):
            marker = new_marker
            break
        marker = new_marker
        budget = max(128, max_chars - len(marker))
        head_chars = max(64, int(budget * head_fraction))
        tail_chars = max(64, budget - head_chars)

    return text[:head_chars].rstrip() + marker + text[-tail_chars:].lstrip()


def duplicate_marker(text: str, *, record_path: str, similarity: float) -> str:
    digest = sha256_short(text)
    return (
        "[Near-duplicate content omitted locally by JL-style request sketch. "
        f"Similar to earlier block at {record_path}; similarity={similarity:.3f}; "
        f"original_chars={len(text)}; sha256={digest}.]"
    )



def _payload_cache_protection(payload: dict[str, Any]) -> dict[str, Any] | None:
    """Return descriptor of last Anthropic cache_control marker in processing order, or None.

    Anthropic prompt caching keys on the byte-identical prefix up through a cache_control
    marker. Mutating any block at or before such a marker invalidates the cache. We walk
    system blocks first, then messages in order, and remember the last marker we see.
    """
    last_kind: str | None = None
    last_msg: int | None = None
    last_block: int = -1

    system = payload.get("system")
    if isinstance(system, list):
        for i, block in enumerate(system):
            if isinstance(block, dict) and "cache_control" in block:
                last_kind, last_msg, last_block = "system", None, i

    messages = payload.get("messages")
    if isinstance(messages, list):
        for mi, message in enumerate(messages):
            if not isinstance(message, dict):
                continue
            content = message.get("content")
            if isinstance(content, list):
                for bi, block in enumerate(content):
                    if isinstance(block, dict) and "cache_control" in block:
                        last_kind, last_msg, last_block = "message", mi, bi

    if last_kind is None:
        return None
    return {"kind": last_kind, "msg_idx": last_msg, "block_idx": last_block}


def _is_block_protected(
    protection: dict[str, Any] | None,
    *,
    kind: str,
    msg_idx: int | None,
    block_idx: int,
) -> bool:
    """True if (kind, msg_idx, block_idx) is at or before the protected cache prefix."""
    if protection is None:
        return False
    last_kind = protection["kind"]
    last_msg = protection["msg_idx"]
    last_block = protection["block_idx"]

    if kind == "system":
        if last_kind == "system":
            return block_idx <= last_block
        return True
    if last_kind == "system":
        return False
    if msg_idx is None or last_msg is None:
        return False
    if msg_idx < last_msg:
        return True
    if msg_idx > last_msg:
        return False
    return block_idx <= last_block


class _CompressionResultCache:
    """Bounded LRU cache for deterministic post-JL compression output.

    Keyed by sha256 of input text + every parameter that influences output. Independent of
    Anthropic's native prompt cache; only avoids local CPU work for repeated text.
    """

    def __init__(self, max_entries: int) -> None:
        self.max_entries = max(0, int(max_entries))
        self._data: OrderedDict[str, str] = OrderedDict()
        self._lock = Lock()
        self.hits = 0
        self.misses = 0

    def _enabled(self) -> bool:
        return self.max_entries > 0

    def get(self, key: str) -> str | None:
        if not self._enabled():
            return None
        with self._lock:
            if key not in self._data:
                self.misses += 1
                return None
            self._data.move_to_end(key)
            self.hits += 1
            return self._data[key]

    def put(self, key: str, value: str) -> None:
        if not self._enabled():
            return
        with self._lock:
            if key in self._data:
                self._data.move_to_end(key)
                self._data[key] = value
                return
            self._data[key] = value
            while len(self._data) > self.max_entries:
                self._data.popitem(last=False)

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "size": len(self._data),
                "max_entries": self.max_entries,
                "hits": self.hits,
                "misses": self.misses,
            }


class PayloadCompressor:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.result_cache = _CompressionResultCache(
            settings.compression_cache_size if settings.compression_cache_enabled else 0
        )
        self._protection: dict[str, Any] | None = None

    def compress_request_payload(
        self,
        payload: dict[str, Any],
        *,
        endpoint: str,
        jl_dedupe: bool | None = None,
        caveman: dict | None = None,
        rtk: dict | None = None,
        json_aware: dict | None = None,
        lsh: dict | None = None,
        force_enabled: bool = False,
    ) -> tuple[dict[str, Any], CompressionAudit]:
        # The static `settings.input_compression_enabled` gate used to short-
        # circuit here, which broke the runtime toggle: an operator who flipped
        # input compression OFF at startup but turned it back ON via
        # `/settings` would never see compression happen because the static
        # gate fired first. Callers that *want* to honor the static gate can
        # opt in via the `force_enabled=False` parameter; the server passes
        # the runtime decision in and lets us run unconditionally.
        audit = CompressionAudit(endpoint=endpoint)
        if not force_enabled and not self.settings.input_compression_enabled:
            return payload, audit

        self._jl_active = self.settings.jl_dedupe_enabled if jl_dedupe is None else jl_dedupe
        if caveman is None:
            self._caveman_active = {
                "enabled": self.settings.caveman_enabled,
                "level": self.settings.caveman_level,
            }
        else:
            self._caveman_active = {
                "enabled": bool(caveman.get("enabled", self.settings.caveman_enabled)),
                "level": str(caveman.get("level", self.settings.caveman_level)),
            }
        if rtk is None:
            self._rtk_active = {
                "enabled": self.settings.rtk_enabled,
                "level": self.settings.rtk_level,
            }
        else:
            self._rtk_active = {
                "enabled": bool(rtk.get("enabled", self.settings.rtk_enabled)),
                "level": str(rtk.get("level", self.settings.rtk_level)),
            }
        # json_aware: per-text-block JSON minify + whitespace coalesce.
        # Defaults are conservative: off, "safe" level.
        if json_aware is None:
            self._json_aware_active = {
                "enabled": getattr(self.settings, "json_aware_enabled", False),
                "level": getattr(self.settings, "json_aware_level", "safe"),
            }
        else:
            self._json_aware_active = {
                "enabled": bool(json_aware.get("enabled", False)),
                "level": str(json_aware.get("level", "safe")),
            }
        # lsh: cross-block near-duplicate dedupe inside each message's content list.
        if lsh is None:
            self._lsh_active = {
                "enabled": getattr(self.settings, "lsh_enabled", False),
                "level": getattr(self.settings, "lsh_level", "standard"),
            }
        else:
            self._lsh_active = {
                "enabled": bool(lsh.get("enabled", False)),
                "level": str(lsh.get("level", "standard")),
            }
        working = copy.deepcopy(payload)
        sketch_index = RequestSketchIndex(
            dims=self.settings.jl_dims, shingle_tokens=self.settings.jl_shingle_tokens
        )

        self._protection = (
            _payload_cache_protection(working) if self.settings.preserve_anthropic_cache else None
        )

        if self.settings.compress_system and "system" in working:
            system_value = working["system"]
            if isinstance(system_value, str) and self._protection is not None:
                audit.protected_blocks += 1
            else:
                working["system"] = self._compress_content_value(
                    system_value,
                    path="system",
                    audit=audit,
                    sketch_index=sketch_index,
                    allow_tool_result=False,
                    kind="system",
                    msg_idx=None,
                )

        messages = working.get("messages")
        if isinstance(messages, list):
            for i, message in enumerate(messages):
                if not isinstance(message, dict) or "content" not in message:
                    continue
                role = message.get("role", "message")
                content_value = message["content"]
                if isinstance(content_value, str) and _is_block_protected(
                    self._protection, kind="message", msg_idx=i, block_idx=0
                ):
                    audit.protected_blocks += 1
                    continue
                message["content"] = self._compress_content_value(
                    content_value,
                    path=f"messages[{i}].{role}.content",
                    audit=audit,
                    sketch_index=sketch_index,
                    allow_tool_result=self.settings.compress_tool_results,
                    kind="message",
                    msg_idx=i,
                )

        return working, audit

    def compress_response_payload(
        self, payload: dict[str, Any], *, endpoint: str, force_enabled: bool = False,
    ) -> tuple[dict[str, Any], CompressionAudit]:
        # Same runtime-gate concern as compress_request_payload — callers that
        # already decided "yes, run" via a runtime flag pass `force_enabled=True`.
        audit = CompressionAudit(endpoint=endpoint)
        if not force_enabled and not self.settings.output_compression_enabled:
            return payload, audit

        working = copy.deepcopy(payload)
        content = working.get("content")
        if isinstance(content, list):
            for i, block in enumerate(content):
                if isinstance(block, dict) and block.get("type") == "text" and isinstance(
                    block.get("text"), str
                ):
                    block["text"] = self._compress_response_text(
                        block["text"], path=f"response.content[{i}].text", audit=audit
                    )
        return working, audit

    def _compress_content_value(
        self,
        value: Any,
        *,
        path: str,
        audit: CompressionAudit,
        sketch_index: RequestSketchIndex,
        allow_tool_result: bool,
        kind: str = "message",
        msg_idx: int | None = None,
    ) -> Any:
        if isinstance(value, str):
            return self._compress_text_with_dedupe(
                value, path=path, audit=audit, sketch_index=sketch_index
            )

        if not isinstance(value, list):
            return value

        # LSH PRE-PASS: cross-block near-duplicate dedupe within this list.
        # Runs BEFORE per-block compression so the rest of the pipeline sees
        # the already-deduped list. Protected blocks are excluded from
        # replacement but still participate as match targets.
        lsh_cfg = getattr(self, "_lsh_active", None) or {"enabled": False, "level": "standard"}
        if lsh_cfg.get("enabled") and value:
            try:
                from .lsh_dedupe import dedupe_blocks as _lsh_dedupe
                protected_idx = {
                    i for i in range(len(value))
                    if _is_block_protected(
                        self._protection, kind=kind, msg_idx=msg_idx, block_idx=i
                    )
                }
                new_value, stats = _lsh_dedupe(
                    value, level=lsh_cfg.get("level", "standard"), protected=protected_idx
                )
                if stats.get("replaced", 0) > 0:
                    # Reassign value to the deduped list and emit one audit event
                    # per replacement for the dashboard. The original lengths
                    # are recoverable from the marker strings.
                    for i, (old, new) in enumerate(zip(value, new_value, strict=False)):
                        if old != new:
                            old_text = old.get("text") if isinstance(old, dict) else None
                            new_text = new.get("text") if isinstance(new, dict) else None
                            if not isinstance(old_text, str):
                                old_text = ""
                            if not isinstance(new_text, str):
                                new_text = ""
                            audit.events.append(
                                self._event(
                                    path=f"{path}[{i}]",
                                    mode="lsh-near-duplicate",
                                    original=old_text,
                                    compressed=new_text,
                                    digest=sha256_short(old_text),
                                    note=f"level={lsh_cfg.get('level', 'standard')}",
                                )
                            )
                    value = new_value
            except (ImportError, ValueError):
                pass

        for i, block in enumerate(value):
            block_path = f"{path}[{i}]"
            if not isinstance(block, dict):
                continue

            if _is_block_protected(
                self._protection, kind=kind, msg_idx=msg_idx, block_idx=i
            ):
                audit.protected_blocks += 1
                continue

            block_type = block.get("type")
            if block_type == "text" and isinstance(block.get("text"), str):
                block["text"] = self._compress_text_with_dedupe(
                    block["text"], path=f"{block_path}.text", audit=audit, sketch_index=sketch_index
                )
            elif block_type == "tool_result" and allow_tool_result:
                content = block.get("content")
                block["content"] = self._compress_tool_result_content(
                    content,
                    path=f"{block_path}.tool_result.content",
                    audit=audit,
                    sketch_index=sketch_index,
                )
        return value

    def _compress_tool_result_content(
        self,
        content: Any,
        *,
        path: str,
        audit: CompressionAudit,
        sketch_index: RequestSketchIndex,
    ) -> Any:
        if isinstance(content, str):
            return self._compress_text_with_dedupe(
                content, path=path, audit=audit, sketch_index=sketch_index
            )
        if isinstance(content, list):
            for i, item in enumerate(content):
                if isinstance(item, dict) and item.get("type") == "text" and isinstance(
                    item.get("text"), str
                ):
                    item["text"] = self._compress_text_with_dedupe(
                        item["text"],
                        path=f"{path}[{i}].text",
                        audit=audit,
                        sketch_index=sketch_index,
                    )
        return content

    def _compress_text_with_dedupe(
        self,
        text: str,
        *,
        path: str,
        audit: CompressionAudit,
        sketch_index: RequestSketchIndex,
    ) -> str:
        original_digest = sha256_short(text)

        if getattr(self, '_jl_active', self.settings.jl_dedupe_enabled) and len(text) >= self.settings.jl_min_chars:
            best_record, best_score = sketch_index.find_best(text)
            if best_record is not None and best_score >= self.settings.jl_similarity_threshold:
                replacement = duplicate_marker(
                    text, record_path=best_record.path, similarity=best_score
                )
                audit.events.append(
                    self._event(
                        path=path,
                        mode="jl-near-duplicate",
                        original=text,
                        compressed=replacement,
                        digest=original_digest,
                        note=f"matched {best_record.path} ({best_record.digest})",
                    )
                )
                return replacement
            sketch_index.add(text=text, path=path, digest=original_digest)

        cache_key = self._build_cache_key(text)
        cached = self.result_cache.get(cache_key)
        if cached is not None:
            audit.cache_hits += 1
            if cached != text:
                audit.events.append(
                    self._event(
                        path=path,
                        mode="cache-hit",
                        original=text,
                        compressed=cached,
                        digest=original_digest,
                        note="local-lru",
                    )
                )
            return cached
        audit.cache_misses += 1

        compressed = self._compress_text(text, path=path, audit=audit, digest=original_digest)
        # json_aware runs BEFORE the prose engines so caveman/rtk see already-
        # minified JSON (less work, cleaner output). Cache key already includes
        # json_aware enabled+level so cached entries don't bleed across configs.
        compressed = self._apply_json_aware(compressed, path=path, audit=audit)
        compressed = self._apply_caveman(compressed, path=path, audit=audit)
        compressed = self._apply_rtk(compressed, path=path, audit=audit)
        self.result_cache.put(cache_key, compressed)
        return compressed

    def _build_cache_key(self, text: str) -> str:
        """Stable key for the post-JL compression pipeline. Includes every output-determining param."""
        cav = getattr(self, "_caveman_active", None) or {
            "enabled": self.settings.caveman_enabled,
            "level": self.settings.caveman_level,
        }
        rtk = getattr(self, "_rtk_active", None) or {
            "enabled": self.settings.rtk_enabled,
            "level": self.settings.rtk_level,
        }
        ja = getattr(self, "_json_aware_active", None) or {"enabled": False, "level": "safe"}
        parts = (
            sha256_short(text),
            str(len(text)),
            str(self.settings.max_text_chars),
            str(self.settings.min_omission_chars),
            f"{self.settings.head_fraction:.4f}",
            "cav1" if cav.get("enabled") else "cav0",
            str(cav.get("level", "standard")),
            "rtk1" if rtk.get("enabled") else "rtk0",
            str(rtk.get("level", "minimal")),
            "ja1" if ja.get("enabled") else "ja0",
            str(ja.get("level", "safe")),
        )
        return "|".join(parts)

    def _apply_json_aware(self, text: str, *, path: str, audit: CompressionAudit) -> str:
        cfg = getattr(self, "_json_aware_active", None) or {"enabled": False, "level": "safe"}
        if not cfg.get("enabled"):
            return text
        try:
            from .json_aware import compress as _ja_compress
            out, _stats = _ja_compress(text, level=cfg.get("level", "safe"))
        except (ValueError, ImportError):
            return text
        if out != text:
            audit.events.append(
                self._event(
                    path=path,
                    mode="json-aware",
                    original=text,
                    compressed=out,
                    digest=sha256_short(text),
                    note=f"level={cfg.get('level', 'safe')}",
                )
            )
        return out

    def _apply_caveman(self, text: str, *, path: str, audit: CompressionAudit) -> str:
        cfg = getattr(self, "_caveman_active", None) or {
            "enabled": self.settings.caveman_enabled,
            "level": self.settings.caveman_level,
        }
        if not cfg.get("enabled"):
            return text
        try:
            out = compress_caveman(text, level=cfg.get("level", "standard"))
        except ValueError:
            return text
        if out != text:
            audit.events.append(
                self._event(
                    path=path,
                    mode="caveman",
                    original=text,
                    compressed=out,
                    digest=sha256_short(text),
                    note=f"level={cfg.get('level', 'standard')}",
                )
            )
        return out

    def _apply_rtk(self, text: str, *, path: str, audit: CompressionAudit) -> str:
        cfg = getattr(self, "_rtk_active", None) or {
            "enabled": self.settings.rtk_enabled,
            "level": self.settings.rtk_level,
        }
        if not cfg.get("enabled"):
            return text
        try:
            out = compress_rtk(text, level=cfg.get("level", "minimal"))
        except ValueError:
            return text
        if out != text:
            audit.events.append(
                self._event(
                    path=path,
                    mode="rtk",
                    original=text,
                    compressed=out,
                    digest=sha256_short(text),
                    note=f"level={cfg.get('level', 'minimal')}",
                )
            )
        return out


    def _compress_response_text(self, text: str, *, path: str, audit: CompressionAudit) -> str:
        compressed = middle_out_text(
            text,
            max_chars=self.settings.output_max_text_chars,
            min_omission_chars=self.settings.min_omission_chars,
            head_fraction=self.settings.head_fraction,
        )
        if compressed != text:
            audit.events.append(
                self._event(
                    path=path,
                    mode="middle-out-response",
                    original=text,
                    compressed=compressed,
                    digest=sha256_short(text),
                )
            )
        return compressed

    def _compress_text(
        self, text: str, *, path: str, audit: CompressionAudit, digest: str | None = None
    ) -> str:
        compressed = middle_out_text(
            text,
            max_chars=self.settings.max_text_chars,
            min_omission_chars=self.settings.min_omission_chars,
            head_fraction=self.settings.head_fraction,
        )
        if compressed != text:
            audit.events.append(
                self._event(
                    path=path,
                    mode="middle-out",
                    original=text,
                    compressed=compressed,
                    digest=digest or sha256_short(text),
                )
            )
        return compressed

    def _event(
        self,
        *,
        path: str,
        mode: str,
        original: str,
        compressed: str,
        digest: str,
        note: str = "",
    ) -> CompressionEvent:
        sample_before = sample_after = None
        if self.settings.log_text_samples:
            sample_before = original[:500]
            sample_after = compressed[:500]
        return CompressionEvent(
            path=path,
            mode=mode,
            original_chars=len(original),
            compressed_chars=len(compressed),
            sha256=digest,
            note=note,
            sample_before=sample_before,
            sample_after=sample_after,
        )
