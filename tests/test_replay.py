import base64
import json
import time

import pytest

from middleout_proxy.replay import RequestCapture


def _payload_bytes() -> bytes:
    return json.dumps({"model": "claude-test", "messages": [{"role": "user", "content": "hi"}]}).encode()


def test_capture_writes_jsonl_file(tmp_path):
    capture = RequestCapture(tmp_path / "captures", max_files=10)
    body = _payload_bytes()
    path = capture.capture(
        method="POST",
        path="v1/messages",
        headers={"content-type": "application/json"},
        body=body,
        response_status=200,
        response_body=b'{"ok": true}',
    )
    assert path is not None
    assert path.suffix == ".jsonl"
    assert path.exists()
    raw = path.read_text(encoding="utf-8")
    assert raw.endswith("\n")
    record = json.loads(raw)
    assert record["method"] == "POST"
    assert record["path"] == "v1/messages"
    assert record["status"] == 200
    assert base64.b64decode(record["body_b64"]) == body
    assert record["body_sha256"] == record["body_sha256"]  # presence
    assert isinstance(record["headers"], dict)


def test_authorization_header_is_redacted(tmp_path):
    capture = RequestCapture(tmp_path, max_files=5)
    path = capture.capture(
        method="POST",
        path="v1/messages",
        headers={
            "Authorization": "Bearer super-secret-oauth-token",
            "X-Api-Key": "sk-must-not-leak",
            "anthropic-api-key": "another-secret",
            "Cookie": "session=do-not-leak",
            "Set-Cookie": "session=do-not-leak",
            "content-type": "application/json",
            "anthropic-version": "2023-06-01",
        },
        body=b"{}",
        response_status=200,
        response_body=None,
    )
    record = json.loads(path.read_text(encoding="utf-8"))
    # ALL of these must be redacted; non-negotiable.
    assert record["headers"]["Authorization"] == "<redacted>"
    assert record["headers"]["X-Api-Key"] == "<redacted>"
    assert record["headers"]["anthropic-api-key"] == "<redacted>"
    assert record["headers"]["Cookie"] == "<redacted>"
    assert record["headers"]["Set-Cookie"] == "<redacted>"
    # And the raw secret must not appear anywhere in the file.
    raw_text = path.read_text(encoding="utf-8")
    assert "super-secret-oauth-token" not in raw_text
    assert "sk-must-not-leak" not in raw_text
    assert "another-secret" not in raw_text
    # Non-sensitive headers passed through unchanged.
    assert record["headers"]["content-type"] == "application/json"
    assert record["headers"]["anthropic-version"] == "2023-06-01"


def test_load_roundtrip(tmp_path):
    capture = RequestCapture(tmp_path, max_files=5)
    body = b"original body bytes"
    response = b"resp body bytes"
    path = capture.capture(
        method="POST",
        path="v1/messages",
        headers={"content-type": "application/json"},
        body=body,
        response_status=201,
        response_body=response,
    )
    capture_id = path.stem
    loaded = capture.load(capture_id)
    assert loaded["method"] == "POST"
    assert loaded["status"] == 201
    assert base64.b64decode(loaded["body_b64"]) == body
    assert base64.b64decode(loaded["response_body_b64"]) == response


def test_load_missing_raises_file_not_found(tmp_path):
    capture = RequestCapture(tmp_path, max_files=5)
    with pytest.raises(FileNotFoundError):
        capture.load("nope-not-here")


def test_rotation_drops_oldest_past_max_files(tmp_path):
    capture = RequestCapture(tmp_path, max_files=3)
    paths = []
    for i in range(7):
        # Force distinct file names by sleeping a hair and using distinct bodies.
        time.sleep(0.005)
        path = capture.capture(
            method="POST",
            path="v1/messages",
            headers={},
            body=f"body-{i}".encode(),
            response_status=200,
            response_body=None,
        )
        paths.append(path)
    remaining = capture.list()
    assert len(remaining) == 3, f"expected 3 files after rotation, got {len(remaining)}"
    # The 3 surviving files should be the 3 most-recently-created.
    for survivor in remaining:
        assert survivor in paths[-3:], f"unexpected survivor: {survivor}"
    for evicted in paths[:-3]:
        assert not evicted.exists(), f"old capture should have been removed: {evicted}"


def test_purge_clears_directory(tmp_path):
    capture = RequestCapture(tmp_path, max_files=5)
    for _i in range(3):
        time.sleep(0.002)
        capture.capture(
            method="GET",
            path="v1/models",
            headers={},
            body=b"",
            response_status=200,
            response_body=b"[]",
        )
    assert len(capture.list()) == 3
    removed = capture.purge()
    assert removed == 3
    assert capture.list() == []
    # Second purge on an empty dir is a no-op.
    assert capture.purge() == 0


def test_disabled_capture_returns_none(tmp_path):
    capture = RequestCapture(tmp_path, max_files=0)
    result = capture.capture(
        method="POST",
        path="v1/messages",
        headers={"authorization": "Bearer x"},
        body=b"{}",
        response_status=200,
        response_body=None,
    )
    assert result is None
    # And nothing should be written.
    assert not (tmp_path / "captures").exists() or list((tmp_path).glob("*.jsonl")) == []


def test_list_on_missing_directory_returns_empty(tmp_path):
    capture = RequestCapture(tmp_path / "does-not-exist", max_files=10)
    assert capture.list() == []
    assert capture.purge() == 0


def test_response_body_none_serializes_as_null(tmp_path):
    capture = RequestCapture(tmp_path, max_files=5)
    path = capture.capture(
        method="POST",
        path="v1/messages",
        headers={},
        body=b"{}",
        response_status=None,
        response_body=None,
    )
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["status"] is None
    assert record["response_body_b64"] is None
