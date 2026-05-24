import threading

import pytest

from middleout_proxy.budget import UsageBudget


def test_under_limit_is_ok():
    budget = UsageBudget(char_limit=1000, token_limit=500)
    budget.record(chars=100, tokens=20)
    assert budget.exceeded() is False
    remaining = budget.remaining()
    assert remaining["chars"] == 900
    assert remaining["tokens"] == 480


def test_at_limit_is_exceeded():
    budget = UsageBudget(char_limit=100, token_limit=50)
    budget.record(chars=100, tokens=0)
    assert budget.exceeded() is True
    remaining = budget.remaining()
    assert remaining["chars"] == 0


def test_over_limit_is_exceeded():
    budget = UsageBudget(char_limit=100, token_limit=50)
    budget.record(chars=200, tokens=0)
    assert budget.exceeded() is True
    # remaining clamps to 0 — no negative numbers leak out.
    assert budget.remaining()["chars"] == 0


def test_token_axis_independent_of_char_axis():
    budget = UsageBudget(char_limit=1000, token_limit=10)
    budget.record(chars=10, tokens=10)
    assert budget.exceeded() is True  # token axis tripped
    assert budget.remaining()["chars"] == 990
    assert budget.remaining()["tokens"] == 0


def test_reset_clears_counters():
    budget = UsageBudget(char_limit=100, token_limit=100)
    budget.record(chars=200, tokens=200)
    assert budget.exceeded() is True
    budget.reset()
    assert budget.exceeded() is False
    assert budget.remaining()["chars"] == 100
    assert budget.remaining()["tokens"] == 100


def test_none_limit_means_unlimited():
    budget = UsageBudget()
    budget.record(chars=10**9, tokens=10**9)
    assert budget.exceeded() is False
    assert budget.remaining() == {"chars": None, "tokens": None}


def test_mixed_none_limit():
    budget = UsageBudget(char_limit=100, token_limit=None)
    budget.record(chars=50, tokens=10**6)
    assert budget.exceeded() is False
    assert budget.remaining() == {"chars": 50, "tokens": None}
    budget.record(chars=60, tokens=0)
    assert budget.exceeded() is True


def test_snapshot_shape():
    budget = UsageBudget(char_limit=100, token_limit=50)
    budget.record(chars=10, tokens=5)
    snap = budget.snapshot()
    assert snap == {
        "chars_used": 10,
        "tokens_used": 5,
        "char_limit": 100,
        "token_limit": 50,
        "exceeded": False,
    }


def test_negative_record_rejected():
    budget = UsageBudget(char_limit=100, token_limit=50)
    with pytest.raises(ValueError):
        budget.record(chars=-1, tokens=0)
    with pytest.raises(ValueError):
        budget.record(chars=0, tokens=-1)


def test_negative_limit_rejected():
    with pytest.raises(ValueError):
        UsageBudget(char_limit=-1)
    with pytest.raises(ValueError):
        UsageBudget(token_limit=-1)


def test_thread_safety_smoke():
    # 10 threads each incrementing chars=1 and tokens=1 a thousand times -> 10000 each.
    budget = UsageBudget()
    workers = 10
    iterations = 1000

    def worker() -> None:
        for _ in range(iterations):
            budget.record(chars=1, tokens=1)

    threads = [threading.Thread(target=worker) for _ in range(workers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    snap = budget.snapshot()
    expected = workers * iterations
    assert snap["chars_used"] == expected
    assert snap["tokens_used"] == expected


def test_thread_safety_with_limit_and_reset():
    budget = UsageBudget(char_limit=100_000, token_limit=100_000)
    workers = 8
    iterations = 500
    stop = threading.Event()

    def add_worker() -> None:
        while not stop.is_set():
            budget.record(chars=1, tokens=1)

    def query_worker() -> None:
        for _ in range(iterations):
            budget.remaining()
            budget.exceeded()
            budget.snapshot()

    threads: list[threading.Thread] = []
    for _ in range(workers // 2):
        threads.append(threading.Thread(target=add_worker))
    for _ in range(workers // 2):
        threads.append(threading.Thread(target=query_worker))

    for t in threads:
        t.start()
    # Let it churn briefly then signal stop.
    for t in threads:
        if t.name.endswith("query_worker") or "query" in t.name:
            t.join()
    stop.set()
    for t in threads:
        t.join()

    # No assertion on exact totals — just that we got here without deadlock/exception.
    snap = budget.snapshot()
    assert snap["chars_used"] >= 0
    assert snap["tokens_used"] >= 0
