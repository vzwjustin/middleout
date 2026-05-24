"""Tests for audit.py: ProxyStats + AuditLogger."""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

from middleout_proxy.audit import AuditLogger, ProxyStats
from middleout_proxy.compression import CompressionAudit, CompressionEvent
from middleout_proxy.config import Settings


def _make_audit_with_savings(chars_saved: int = 100) -> CompressionAudit:
    """Build a CompressionAudit whose `chars_saved` is positive."""
    audit = CompressionAudit(endpoint="v1/messages")
    audit.events.append(
        CompressionEvent(
            path="messages[0].user.content",
            mode="middle-out",
            original_chars=1000,
            compressed_chars=1000 - chars_saved,
            sha256="abcdef0123456789",
        )
    )
    return audit


def test_proxy_stats_snapshot_includes_uptime():
    stats = ProxyStats()
    snap = stats.snapshot()
    assert "uptime_s" in snap
    assert snap["uptime_s"] >= 0.0
    # Snapshot must also expose the core counters.
    for key in (
        "requests_total",
        "compressed_requests",
        "chars_saved_in",
        "chars_saved_out",
        "upstream_errors",
        "cache_hits",
        "cache_misses",
        "protected_blocks",
        "started_at",
    ):
        assert key in snap


def test_proxy_stats_uptime_grows():
    stats = ProxyStats()
    snap1 = stats.snapshot()
    time.sleep(0.01)
    snap2 = stats.snapshot()
    assert snap2["uptime_s"] >= snap1["uptime_s"]


def test_audit_logger_writes_one_jsonl_line_per_record(tmp_audit_dir: Path):
    settings = Settings(audit_enabled=True, audit_log_dir=tmp_audit_dir)
    logger = AuditLogger(settings)
    for i in range(3):
        logger.record(
            method="POST",
            path="v1/messages",
            status_code=200,
            request_audit=CompressionAudit(endpoint="v1/messages"),
            request_id=f"req-{i}",
        )
    log_path = tmp_audit_dir / "audit.jsonl"
    assert log_path.exists()
    lines = [line for line in log_path.read_text().splitlines() if line.strip()]
    assert len(lines) == 3
    # Each line must be valid JSON with expected keys.
    for line in lines:
        entry = json.loads(line)
        assert entry["method"] == "POST"
        assert entry["path"] == "v1/messages"
        assert entry["status_code"] == 200


def test_audit_logger_increments_counters(tmp_audit_dir: Path):
    settings = Settings(audit_enabled=True, audit_log_dir=tmp_audit_dir)
    logger = AuditLogger(settings)
    audit = _make_audit_with_savings(chars_saved=250)
    audit.cache_hits = 1
    audit.cache_misses = 2
    audit.protected_blocks = 3

    logger.record(
        method="POST",
        path="v1/messages",
        status_code=200,
        request_audit=audit,
    )
    snap = logger.stats.snapshot()
    assert snap["requests_total"] == 1
    assert snap["compressed_requests"] == 1
    assert snap["chars_saved_in"] == 250
    assert snap["cache_hits"] == 1
    assert snap["cache_misses"] == 2
    assert snap["protected_blocks"] == 3

    # A second record with chars_saved == 0 must not increment compressed_requests.
    logger.record(
        method="POST",
        path="v1/messages",
        status_code=200,
        request_audit=CompressionAudit(endpoint="v1/messages"),
    )
    snap2 = logger.stats.snapshot()
    assert snap2["requests_total"] == 2
    assert snap2["compressed_requests"] == 1


def test_audit_logger_upstream_errors_on_5xx_and_error_param(tmp_audit_dir: Path):
    settings = Settings(audit_enabled=True, audit_log_dir=tmp_audit_dir)
    logger = AuditLogger(settings)

    # 5xx response increments upstream_errors.
    logger.record(
        method="POST",
        path="v1/messages",
        status_code=500,
        request_audit=CompressionAudit(endpoint="v1/messages"),
    )
    assert logger.stats.snapshot()["upstream_errors"] == 1

    # `error=...` increments upstream_errors as well, even without a status code.
    logger.record(
        method="POST",
        path="v1/messages",
        status_code=None,
        request_audit=CompressionAudit(endpoint="v1/messages"),
        error="ConnectError: dns lookup failed",
    )
    assert logger.stats.snapshot()["upstream_errors"] == 2


def test_audit_logger_handles_none_response_audit_cleanly(tmp_audit_dir: Path):
    settings = Settings(audit_enabled=True, audit_log_dir=tmp_audit_dir)
    logger = AuditLogger(settings)
    logger.record(
        method="POST",
        path="v1/messages",
        status_code=200,
        request_audit=CompressionAudit(endpoint="v1/messages"),
        response_audit=None,
    )
    log_path = tmp_audit_dir / "audit.jsonl"
    line = log_path.read_text().splitlines()[0]
    entry = json.loads(line)
    assert entry["response_compression"] is None


def test_audit_logger_disabled_creates_no_file(tmp_audit_dir: Path):
    settings = Settings(audit_enabled=False, audit_log_dir=tmp_audit_dir)
    logger = AuditLogger(settings)
    logger.record(
        method="POST",
        path="v1/messages",
        status_code=200,
        request_audit=CompressionAudit(endpoint="v1/messages"),
    )
    log_path = tmp_audit_dir / "audit.jsonl"
    assert not log_path.exists()
    # Counters still update in memory.
    assert logger.stats.snapshot()["requests_total"] == 1


def test_audit_logger_thread_safe_requests_total(tmp_audit_dir: Path):
    settings = Settings(audit_enabled=True, audit_log_dir=tmp_audit_dir)
    logger = AuditLogger(settings)

    def worker():
        for _ in range(100):
            logger.record(
                method="POST",
                path="v1/messages",
                status_code=200,
                request_audit=CompressionAudit(endpoint="v1/messages"),
            )

    threads = [threading.Thread(target=worker) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert logger.stats.snapshot()["requests_total"] == 1000
