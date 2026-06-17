from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator

from scholar_rag.models import Answer, RetrievedChunk

class Generator(ABC):
    @abstractmethod
    def generate(self, query: str, chunks: list[RetrievedChunk]) -> Answer:
        """Synthesize an answer to `query` grounded in `chunks`"""

    def generate_stream(self, query: str, chunks: list[RetrievedChunk]) -> Iterator[str]:
        """Yield the answer text in pieces as it is produced.

        Default: produce the whole answer, then yield it as a single piece — so
        any Generator works with the streaming endpoint even if it can't stream
        natively. OpenRouterGenerator overrides this with real token streaming.
        """
        yield self.generate(query, chunks).text