from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

from scholar_rag.models import RawDocument

@dataclass(frozen=True)
class DiscoveredDoc:
    source: str
    source_doc_id: str
    title: str
    authors: list[str]
    abstract: str | None
    categories: list[str]

class RateLimiter:
    """arXiv's terms require <= 1 request / 3 seconds."""

    def __init__(self, min_interval_s: float) -> None:
        self._min_interval = min_interval_s
        self._lock = threading.Lock()
        self._last_call = 0.0

    def wait(self) -> None:
        with self._lock:
            elapsed = time.monotonic() - self._last_call
            sleep_for = self._min_interval - elapsed
            if sleep_for > 0:
                time.sleep(sleep_for)
            self._last_call = time.monotonic()

class DocumentSource(ABC):
    name: str

    @abstractmethod
    def discover(self, query: str, max_results: int = 50) -> list[DiscoveredDoc]:
        """Search the source's metadata."""

    @abstractmethod
    def fetch(selfself, doc_id: str) -> RawDocument:
        """Retrieve and parse the full content of one document."""
