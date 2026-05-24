import asyncio

import pytest

from middleout_proxy.rate_limit import RequestLimiter, TokenBucket


def _run(coro):
    return asyncio.run(coro)


def test_bucket_starts_at_capacity():
    bucket = TokenBucket(capacity=3, refill_per_second=1.0)
    assert bucket.available == 3.0


def test_bucket_acquire_decrements():
    bucket = TokenBucket(capacity=3, refill_per_second=0.001)

    async def run():
        assert await bucket.acquire(1) is True
        assert await bucket.acquire(1) is True
        assert await bucket.acquire(1) is True

    _run(run())
    # Available is now ~0; refill rate is tiny enough that it's still close to 0.
    assert bucket.available < 1.0


def test_exhausted_bucket_returns_false():
    bucket = TokenBucket(capacity=2, refill_per_second=0.001)

    async def run():
        assert await bucket.acquire(1) is True
        assert await bucket.acquire(1) is True
        # Bucket is empty; further acquires fail.
        assert await bucket.acquire(1) is False
        assert await bucket.acquire(1) is False

    _run(run())


def test_bucket_refills_over_time():
    bucket = TokenBucket(capacity=2, refill_per_second=200.0)

    async def run():
        assert await bucket.acquire(2) is True  # drain
        assert await bucket.acquire(1) is False
        # 200 tokens/sec * 0.05s = 10 tokens of refill, capped at capacity=2.
        await asyncio.sleep(0.05)
        assert await bucket.acquire(1) is True

    _run(run())


def test_reset_restores_capacity():
    bucket = TokenBucket(capacity=5, refill_per_second=0.001)

    async def run():
        for _ in range(5):
            assert await bucket.acquire(1) is True
        assert await bucket.acquire(1) is False
        bucket.reset()
        assert bucket.available == 5.0
        assert await bucket.acquire(1) is True

    _run(run())


def test_bucket_rejects_invalid_construction():
    with pytest.raises(ValueError):
        TokenBucket(capacity=0, refill_per_second=1.0)
    with pytest.raises(ValueError):
        TokenBucket(capacity=10, refill_per_second=0.0)


def test_bucket_zero_cost_always_succeeds():
    bucket = TokenBucket(capacity=1, refill_per_second=0.001)

    async def run():
        # Even with an empty bucket, a zero-cost acquire is a no-op success.
        assert await bucket.acquire(1) is True
        assert await bucket.acquire(0) is True

    _run(run())


def test_request_limiter_per_client_independence():
    limiter = RequestLimiter(capacity=2, refill_per_second=0.001)

    async def run():
        # Client A uses up both tokens.
        assert await limiter.check("client-a-hash-aaaa") is True
        assert await limiter.check("client-a-hash-aaaa") is True
        assert await limiter.check("client-a-hash-aaaa") is False
        # Client B is independent and still has full capacity.
        assert await limiter.check("client-b-hash-bbbb") is True
        assert await limiter.check("client-b-hash-bbbb") is True
        assert await limiter.check("client-b-hash-bbbb") is False
        stats = limiter.stats()
        assert stats["active_buckets"] == 2
        assert stats["capacity"] == 2

    _run(run())


def test_request_limiter_refills_per_client():
    limiter = RequestLimiter(capacity=1, refill_per_second=200.0)

    async def run():
        assert await limiter.check("c1-abc") is True
        assert await limiter.check("c1-abc") is False
        await asyncio.sleep(0.05)
        assert await limiter.check("c1-abc") is True

    _run(run())


def test_request_limiter_stats_shape():
    limiter = RequestLimiter(capacity=10, refill_per_second=2.0)

    async def run():
        await limiter.check("client-zzzzzzzzzzzz")
        return limiter.stats()

    stats = _run(run())
    assert stats["active_buckets"] == 1
    assert stats["capacity"] == 10
    assert stats["refill_per_second"] == 2.0
    assert isinstance(stats["oldest_created_at"], float)


def test_request_limiter_rejects_empty_client_key():
    limiter = RequestLimiter(capacity=10, refill_per_second=1.0)

    async def run():
        with pytest.raises(ValueError):
            await limiter.check("")

    _run(run())
