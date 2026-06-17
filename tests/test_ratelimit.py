from scholar_rag.serve.ratelimit import RateLimiter


class Clock:
    """A controllable time source so windows can be tested without sleeping."""

    def __init__(self, t: float = 1000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


def test_per_ip_limit_trips_then_resets():
    clock = Clock()
    rl = RateLimiter(per_ip_min=3, global_min=100, global_day=1000, now_fn=clock)

    assert all(rl.check("1.1.1.1") is None for _ in range(3))  # 3 allowed
    assert rl.check("1.1.1.1") is not None  # 4th blocked
    assert rl.check("2.2.2.2") is None  # a different IP is unaffected

    clock.t += 61  # slide past the 1-minute window
    assert rl.check("1.1.1.1") is None  # allowed again


def test_global_minute_cap():
    clock = Clock()
    rl = RateLimiter(per_ip_min=100, global_min=2, global_day=1000, now_fn=clock)

    assert rl.check("a") is None
    assert rl.check("b") is None
    blocked = rl.check("c")  # global/min exhausted regardless of IP
    assert blocked and "busy" in blocked


def test_global_daily_ceiling():
    clock = Clock()
    rl = RateLimiter(per_ip_min=100, global_min=100, global_day=2, now_fn=clock)

    assert rl.check("a") is None
    assert rl.check("b") is None
    blocked = rl.check("c")
    assert blocked and "daily" in blocked

    clock.t += 60  # still within the day → still blocked
    assert rl.check("d") is not None

    clock.t += 86_400  # next day → allowed
    assert rl.check("e") is None
