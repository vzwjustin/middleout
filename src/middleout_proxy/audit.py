from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any

from .compression import CompressionAudit
from .config import Settings


@dataclass
class ProxyStats:
    started_at: float = field(default_factory=time.time)
    requests_total: int = 0
    compressed_requests: int = 0
    chars_saved_in: int = 0
    chars_saved_out: int = 0
    upstream_errors: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    protected_blocks: int = 0

    def snapshot(self) -> dict[str, Any]:
        data = asdict(self)
        data["uptime_s"] = round(time.time() - self.started_at, 3)
        return data


class AuditLogger:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.stats = ProxyStats()
        self._lock = Lock()
        self._log_path: Path | None = None
        if settings.audit_enabled:
            settings.audit_log_dir.mkdir(parents=True, exist_ok=True)
            self._log_path = settings.audit_log_dir / "audit.jsonl"

    def record(
        self,
        *,
        method: str,
        path: str,
        status_code: int | None,
        request_audit: CompressionAudit,
        response_audit: CompressionAudit | None = None,
        request_id: str | None = None,
        error: str | None = None,
    ) -> None:
        with self._lock:
            self.stats.requests_total += 1
            if request_audit.chars_saved or (response_audit and response_audit.chars_saved):
                self.stats.compressed_requests += 1
            self.stats.chars_saved_in += request_audit.chars_saved
            self.stats.cache_hits += request_audit.cache_hits
            self.stats.cache_misses += request_audit.cache_misses
            self.stats.protected_blocks += request_audit.protected_blocks
            if response_audit:
                self.stats.chars_saved_out += response_audit.chars_saved
                self.stats.cache_hits += response_audit.cache_hits
                self.stats.cache_misses += response_audit.cache_misses
            if status_code is not None and status_code >= 500:
                self.stats.upstream_errors += 1
            if error:
                self.stats.upstream_errors += 1

        if not self._log_path:
            return

        entry = {
            "ts": time.time(),
            "method": method,
            "path": path,
            "status_code": status_code,
            "request_id": request_id,
            "request_compression": request_audit.to_dict(),
            "response_compression": response_audit.to_dict() if response_audit else None,
            "error": error,
        }
        with self._lock:
            with self._log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
