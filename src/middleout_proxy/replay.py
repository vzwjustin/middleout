"""Request capture/replay store.

Stores a redacted copy of each captured request as a single-line JSONL file
under a directory. The integration layer can replay these locally without
calling the upstream — this module never touches the network.

**Security**: ``authorization``, ``x-api-key``, ``anthropic-api-key``,
``cookie``, and ``set-cookie`` headers are always redacted before they hit
disk. This redaction is non-negotiable; tests assert it.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any

# Header names whose values are scrubbed on the way to disk. Lower-cased.
_REDACTED_HEADERS: frozenset[str] = frozenset(
    {
        "authorization",
        "x-api-key",
        "anthropic-api-key",
        "cookie",
        "set-cookie",
    }
)
_REDACTION_PLACEHOLDER = "<redacted>"
_CAPTURE_SUFFIX = ".jsonl"


def _redact_headers(headers: dict[str, str]) -> dict[str, str]:
    """Return a copy of ``headers`` with sensitive values replaced."""
    out: dict[str, str] = {}
    for key, value in headers.items():
        if key.lower() in _REDACTED_HEADERS:
            out[key] = _REDACTION_PLACEHOLDER
        else:
            out[key] = value
    return out


def _capture_id_for(now: float) -> str:
    """Build a sortable, mostly-unique capture id from a timestamp."""
    stamp = datetime.fromtimestamp(now, tz=timezone.utc).strftime("%Y%m%d-%H%M%S")
    suffix = os.urandom(4).hex()
    return f"{stamp}-{suffix}"


class RequestCapture:
    """File-backed circular buffer for captured requests.

    Each captured request becomes a single-line JSONL file under
    ``capture_dir``. When the number of capture files exceeds ``max_files``,
    the oldest files are deleted (oldest is determined by the file name's
    time-prefixed stem, which sorts naturally).

    Set ``max_files=0`` to disable capture entirely; :meth:`capture` will then
    return ``None`` without touching the disk.
    """

    def __init__(self, capture_dir: Path | str, *, max_files: int = 1000) -> None:
        self.capture_dir = Path(capture_dir)
        self.max_files = int(max_files)
        self._lock = Lock()

    # -- public API --------------------------------------------------------

    def capture(
        self,
        *,
        method: str,
        path: str,
        headers: dict[str, str],
        body: bytes,
        response_status: int | None,
        response_body: bytes | None,
    ) -> Path | None:
        """Persist one request capture to disk and return the path.

        Returns ``None`` if capture is disabled (``max_files <= 0``).
        """
        if self.max_files <= 0:
            return None

        body_bytes = body if isinstance(body, (bytes, bytearray)) else b""
        body_bytes = bytes(body_bytes)
        resp_bytes: bytes | None = None
        if isinstance(response_body, (bytes, bytearray)):
            resp_bytes = bytes(response_body)

        record: dict[str, Any] = {
            "ts": time.time(),
            "method": str(method),
            "path": str(path),
            "headers": _redact_headers(dict(headers or {})),
            "body_b64": base64.b64encode(body_bytes).decode("ascii"),
            "body_sha256": hashlib.sha256(body_bytes).hexdigest(),
            "status": None if response_status is None else int(response_status),
            "response_body_b64": (
                base64.b64encode(resp_bytes).decode("ascii") if resp_bytes is not None else None
            ),
        }

        with self._lock:
            self.capture_dir.mkdir(parents=True, exist_ok=True)
            capture_id = _capture_id_for(record["ts"])
            file_path = self.capture_dir / f"{capture_id}{_CAPTURE_SUFFIX}"
            # On the off chance of a collision (same microsecond + same random suffix),
            # add a small numeric tail until the path is free.
            collision = 0
            while file_path.exists():
                collision += 1
                file_path = self.capture_dir / f"{capture_id}-{collision}{_CAPTURE_SUFFIX}"

            file_path.write_text(
                json.dumps(record, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            self._rotate_locked()
        return file_path

    def list(self) -> list[Path]:
        """Return all capture files, oldest-first by name."""
        if not self.capture_dir.exists():
            return []
        return sorted(self.capture_dir.glob(f"*{_CAPTURE_SUFFIX}"))

    def load(self, capture_id: str) -> dict[str, Any]:
        """Load a single capture by id (filename stem).

        Raises:
            FileNotFoundError: if no capture with that id exists.
            ValueError: if the file is malformed.
        """
        path = self.capture_dir / f"{capture_id}{_CAPTURE_SUFFIX}"
        if not path.exists():
            raise FileNotFoundError(f"capture {capture_id!r} not found in {self.capture_dir}")
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            raise ValueError(f"capture {capture_id!r} is empty")
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"capture {capture_id!r} is not valid JSON: {exc}") from exc
        if not isinstance(data, dict):
            raise ValueError(f"capture {capture_id!r} did not deserialize to an object")
        return data

    def purge(self) -> int:
        """Delete every capture file. Returns the number removed."""
        if not self.capture_dir.exists():
            return 0
        removed = 0
        with self._lock:
            for path in self.capture_dir.glob(f"*{_CAPTURE_SUFFIX}"):
                try:
                    path.unlink()
                    removed += 1
                except FileNotFoundError:
                    continue
        return removed

    # -- internals ---------------------------------------------------------

    def _rotate_locked(self) -> None:
        """Delete oldest captures until the count fits ``max_files``. Caller holds the lock.

        Sorting is by mtime so the chronologically-oldest captures get evicted first.
        Sorting purely by name would order by the random 8-hex suffix when multiple
        captures land in the same wall-clock second — which is the wrong "oldest".
        """
        files = list(self.capture_dir.glob(f"*{_CAPTURE_SUFFIX}"))
        if len(files) <= self.max_files:
            return
        files.sort(key=lambda p: (p.stat().st_mtime_ns, p.name))
        while len(files) > self.max_files:
            oldest = files.pop(0)
            try:
                oldest.unlink()
            except FileNotFoundError:
                continue


__all__ = ["RequestCapture"]
