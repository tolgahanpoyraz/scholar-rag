"""In-memory sliding-window rate limiting for the LLM endpoints.

The Space runs a single uvicorn process, so a dict + lock is enough — no Redis.
Three independent caps, any of which can trip (→ HTTP 429):

  - per IP / minute  — stops one client from hammering
  - global / minute  — smooths bursts
  - global / day     — a hard ceiling on OpenRouter spend (the main credit guard)

All three are env-configurable so they can be tuned on the Space without a
redeploy (SCHOLAR_RAG_RATE_PER_IP_MIN / _GLOBAL_MIN / _GLOBAL_DAY).
"""
from __future__ import annotations

import os
import threading
import time
from collections import deque
from collections.abc import Callable

from fastapi import Request

_MINUTE = 60
_DAY = 86_400


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def client_ip(request: Request) -> str:
    """Real client IP. Behind the HF Spaces proxy the caller is in
    X-Forwarded-For (first hop); fall back to the socket peer."""
    fwd = request.headers.get("x-forwarded-for")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


class RateLimiter:
    def __init__(
        self,
        per_ip_min: int | None = None,
        global_min: int | None = None,
        global_day: int | None = None,
        now_fn: Callable[[], float] = time.time,
    ) -> None:
        self.per_ip_min = per_ip_min if per_ip_min is not None else _int_env("SCHOLAR_RAG_RATE_PER_IP_MIN", 8)
        self.global_min = global_min if global_min is not None else _int_env("SCHOLAR_RAG_RATE_GLOBAL_MIN", 30)
        self.global_day = global_day if global_day is not None else _int_env("SCHOLAR_RAG_RATE_GLOBAL_DAY", 500)
        self._now = now_fn
        self._lock = threading.Lock()
        self._ip: dict[str, deque[float]] = {}
        self._gmin: deque[float] = deque()
        self._gday: deque[float] = deque()

    @staticmethod
    def _prune(dq: deque[float], cutoff: float) -> None:
        while dq and dq[0] <= cutoff:
            dq.popleft()

    def check(self, ip: str) -> str | None:
        """Return None if the request is allowed (and record it), else a short
        human-readable reason for the 429."""
        now = self._now()
        with self._lock:
            self._prune(self._gday, now - _DAY)
            if len(self._gday) >= self.global_day:
                return "daily request limit reached — try again tomorrow"

            self._prune(self._gmin, now - _MINUTE)
            if len(self._gmin) >= self.global_min:
                return "the service is busy right now — try again shortly"

            ipq = self._ip.get(ip)
            if ipq is None:
                ipq = self._ip[ip] = deque()
            self._prune(ipq, now - _MINUTE)
            if len(ipq) >= self.per_ip_min:
                return "too many requests — slow down a moment"

            now_t = now
            self._gday.append(now_t)
            self._gmin.append(now_t)
            ipq.append(now_t)
            return None
