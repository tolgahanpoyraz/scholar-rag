from __future__ import annotations

from abc import ABC, abstractmethod

import tiktoken

class TokenCounter(ABC):
    @abstractmethod
    def count(self, text: str) -> int:
        """Number of tokens in `text`"""

class TiktokenCounter(TokenCounter):
    def __init__(self, encoding_name: str = "cl100k_base") -> None:
        self._enc = tiktoken.get_encoding(encoding_name)

    def count(self, text: str) -> int:
        return len(self._enc.encode(text))