from __future__ import annotations

import io
import json
from contextlib import redirect_stderr

import pytest
from fastapi.testclient import TestClient

from middleout_proxy.audit import AuditLogger
from middleout_proxy.compression import CompressionAudit, CompressionEvent
from middleout_proxy.config import Settings


def _make_audit(mode: str, original: int, compressed: int, *, endpoint: str = "v1/messages") -> CompressionAudit:
    ev = CompressionEvent(
        path="messages[0].user.content",
        mode=mode,
        original_chars=original,
        compressed_chars=compressed,
        sha256="deadbeef" * 2,
    )
    a = CompressionAudit(endpoint=endpoint)
    a.events.append(ev)
    return a


@pytest.fixture
def client():
    """TestClient against the live server module. We DO NOT reload the module
    because reload() rebinds class objects (e.g. StrictSubscriptionAuthError),
    which then fail isinstance checks in test_strict_subscription_auth.py when
    those tests run after this fixture.

    To keep state isolated across tests, we reset the module-level AuditLogger's
    stats container at fixture teardown.
    """
    import middleout_proxy.server as srv
    saved_stats = srv.audit_logger.stats
    # Fresh stats for this test.
    from middleout_proxy.audit import ProxyStats
    srv.audit_logger.stats = ProxyStats(
        window_minutes=int(saved_stats.window_minutes),
        recent_max=int(saved_stats.recent_max),
    )
    saved_log_path = srv.audit_logger._log_path
    srv.audit_logger._log_path = None  # avoid mutating real .middleout-logs/audit.jsonl
    try:
        yield TestClient(srv.app), srv
    finally:
        srv.audit_logger.stats = saved_stats
        srv.audit_logger._log_path = saved_log_path


def test_timeseries_empty_on_cold_start(client):
    c, _srv = client
    resp = c.get("/stats/timeseries")
    assert resp.status_code == 200
    body = resp.json()
    assert body["window_minutes"] == 60
    assert body["buckets"] == []


def test_recent_empty_on_cold_start(client):
    c, _srv = client
    resp = c.get("/stats/recent?n=50")
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 50
    assert body["items"] == []


def test_recent_never_includes_raw_text_fields(client):
    """Even with MIDDLEOUT_LOG_TEXT_SAMPLES=true, /stats/recent must stay hash-only."""
    c, srv = client
    # Drive a synthetic record with a poisoned audit summary containing samples.
    poisoned_event = CompressionEvent(
        path="messages[0].user.content",
        mode="middle-out",
        original_chars=1000,
        compressed_chars=200,
        sha256="cafebabe" * 2,
        sample_before="SECRET PROMPT TEXT THAT MUST NOT LEAK",
        sample_after="SECRET COMPRESSED TEXT THAT MUST NOT LEAK",
    )
    a = CompressionAudit(endpoint="v1/messages")
    a.events.append(poisoned_event)
    srv.audit_logger.record(
        method="POST",
        path="v1/messages",
        status_code=200,
        request_audit=a,
        latency_ms=42.0,
        bytes_in=1234,
        bytes_out=5678,
    )
    body = c.get("/stats/recent?n=10").json()
    assert body["count"] == 10
    assert len(body["items"]) == 1
    item = body["items"][0]
    # The recent record contains a request_audit summary with events list.
    serialized = json.dumps(item)
    assert "SECRET PROMPT TEXT" not in serialized
    assert "SECRET COMPRESSED TEXT" not in serialized
    assert "sample_before" not in serialized
    assert "sample_after" not in serialized
    # But hash + stats are present.
    assert "cafebabecafebabe" in serialized
    assert item["chars_saved_in"] == 800


def test_timeseries_bucketing_orders_by_minute():
    settings = Settings(audit_enabled=False)
    al = AuditLogger(settings)
    base = 1_700_000_000
    t1 = base
    t2 = base + 60
    t3 = base + 120
    al.record(
        method="POST", path="v1/messages", status_code=200,
        request_audit=_make_audit("middle-out", 1000, 200),
        latency_ms=10.0, bytes_in=500, bytes_out=300, now=t1,
    )
    al.record(
        method="POST", path="v1/messages", status_code=200,
        request_audit=_make_audit("middle-out", 800, 150),
        latency_ms=20.0, bytes_in=400, bytes_out=200, now=t1 + 5,
    )
    al.record(
        method="POST", path="v1/messages", status_code=500,
        request_audit=_make_audit("middle-out", 0, 0),
        latency_ms=120.0, bytes_in=100, bytes_out=0, now=t2,
    )
    al.record(
        method="POST", path="v1/messages", status_code=200,
        request_audit=_make_audit("jl-near-duplicate", 5000, 200),
        latency_ms=8.0, bytes_in=300, bytes_out=100, now=t3,
    )
    buckets = al.stats.timeseries(now=t3 + 1)
    minute_ts = [b["minute_ts"] for b in buckets]
    assert minute_ts == sorted(minute_ts)
    assert len(buckets) == 3
    assert buckets[0]["requests"] == 2
    assert buckets[0]["chars_saved_in"] == 800 + 650
    assert buckets[1]["errors"] == 1
    assert buckets[2]["engines"].get("jl-near-duplicate") == 4800


def test_timeseries_evicts_past_60_min():
    settings = Settings(audit_enabled=False)
    al = AuditLogger(settings)
    t0 = 1_700_000_000
    al.record(
        method="POST", path="v1/messages", status_code=200,
        request_audit=_make_audit("middle-out", 1000, 200),
        latency_ms=10.0, bytes_in=500, bytes_out=300, now=t0,
    )
    assert len(al.stats.timeseries(now=t0 + 5)) == 1
    later = t0 + 61 * 60
    assert al.stats.timeseries(now=later) == []


def test_structured_logging_emits_json_line_on_stderr():
    settings = Settings(audit_enabled=False, log_json=True)
    al = AuditLogger(settings)
    buf = io.StringIO()
    with redirect_stderr(buf):
        al.record(
            method="POST", path="v1/messages", status_code=200,
            request_audit=_make_audit("middle-out", 4000, 1500),
            latency_ms=123.45, bytes_in=999, bytes_out=2000,
            request_id="req_test_123", now=1_700_000_000.5,
        )
    out = buf.getvalue().strip()
    assert out, "expected a structured log line on stderr"
    lines = [ln for ln in out.splitlines() if ln.strip()]
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert set(rec.keys()) == {
        "ts","method","path","status","ms","model",
        "chars_saved_input","chars_saved_output","engines_active","request_id",
    }
    assert rec["method"] == "POST"
    assert rec["path"] == "v1/messages"
    assert rec["status"] == 200
    assert rec["ms"] == 123.45
    assert rec["chars_saved_input"] == 2500
    assert rec["chars_saved_output"] == 0
    assert rec["engines_active"] == ["middle-out"]
    assert rec["request_id"] == "req_test_123"
    # model was not passed; should serialize as None.
    assert rec["model"] is None
    from datetime import datetime
    datetime.fromisoformat(rec["ts"])


def test_structured_logging_default_off_writes_nothing():
    settings = Settings(audit_enabled=False, log_json=False)
    al = AuditLogger(settings)
    buf = io.StringIO()
    with redirect_stderr(buf):
        al.record(
            method="POST", path="v1/messages", status_code=200,
            request_audit=_make_audit("middle-out", 4000, 1500),
            latency_ms=10.0, bytes_in=1, bytes_out=1, now=1_700_000_000.0,
        )
    assert buf.getvalue() == ""


def test_proxy_stats_engine_attribution_extensible():
    settings = Settings(audit_enabled=False)
    al = AuditLogger(settings)
    custom = CompressionAudit(endpoint="v1/messages")
    custom.events.append(CompressionEvent(
        path="x", mode="future-engine-xyz",
        original_chars=500, compressed_chars=100, sha256="aaaa"*4,
    ))
    al.record(
        method="POST", path="v1/messages", status_code=200,
        request_audit=custom, latency_ms=5.0, bytes_in=100, bytes_out=80,
        now=1_700_000_000.0,
    )
    snap = al.stats.snapshot()
    assert snap["engines_total"] == {"future-engine-xyz": 400}
    buckets = al.stats.timeseries(now=1_700_000_000.5)
    assert buckets[0]["engines"] == {"future-engine-xyz": 400}
