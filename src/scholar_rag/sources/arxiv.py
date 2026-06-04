from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

from scholar_rag.sources.pdf_text import extract_text_from_bytes
import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from scholar_rag.models import RawDocument
from scholar_rag.sources.base import DiscoveredDoc, DocumentSource, RateLimiter

_NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom"
}

_QUERY_URL = "http://export.arxiv.org/api/query"
_PDF_URL = "https://arxiv.org/pdf/{doc_id}"

_VERSION_RE = re.compile(r"v\d+$")

_USER_AGENT = "scholar-rag/0.1 (research project; contact: mail@example.com)"

_RETRYABLE_STATUS = {429, 500, 502, 503, 504}

class _RetryableHTTP(Exception):
    """Raised to signal Tenacity that a response status is worth retrying."""

def _clean_id(raw_id: str) -> str:
    """'http://arxiv.org/abs/2401.01234v2' -> '2401.01234'."""
    bare = raw_id.rsplit("/", 1)[-1]
    return _VERSION_RE.sub("", bare)

def _text(entry: ET.Element, path: str) -> str | None:
    el = entry.find(path, _NS)
    if el is None or el.text is None:
        return None
    return " ".join(el.text.split())

class ArxivSource(DocumentSource):
    name = "arxiv"

    def __init__(
            self,
            client: httpx.Client | None = None,
            rate_limiter: RateLimiter | None = None,
    ) -> None:
        self._client = client or httpx.Client(
            headers={"User-Agent": _USER_AGENT},
            timeout=30.0,
            follow_redirects=True,
        )
        self._rate = rate_limiter or RateLimiter(min_interval_s=3.0)

    def discover(self, query: str, max_results: int = 50) -> list[DiscoveredDoc]:
        self._rate.wait()
        resp = self._get(
            _QUERY_URL,
            params={
                "search_query": query,
                "start": 0,
                "max_results": max_results,
                "sortBy": "submittedDate",
                "sortOrder": "descending",
            },
        )
        return self._parse_feed(resp.text)

    def _parse_feed(self, xml: str) -> list[DiscoveredDoc]:
        root = ET.fromstring(xml)
        out: list[DiscoveredDoc] = []
        for entry in root.findall("atom:entry", _NS):
            raw_id = _text(entry, "atom:id")
            title = _text(entry, "atom:title")
            if raw_id is None or title is None:
                continue
            authors = [
                name for a in entry.findall("atom:author", _NS)
                if (name := _text(a, "atom:name")) is not None
            ]
            categories = [
                term for c in entry.findall("atom:category", _NS)
                if (term := c.get("term")) is not None
            ]
            out.append(
                DiscoveredDoc(
                    source=self.name,
                    source_doc_id=_clean_id(raw_id),
                    title=title,
                    authors=authors,
                    abstract=_text(entry, "atom:summary"),
                    categories=categories,
                )
            )
        return out

    def fetch(self, doc_id: str) -> RawDocument:
        doc_id = _clean_id(doc_id)
        self._rate.wait()
        resp = self._get(url=_PDF_URL.format(doc_id=doc_id))
        full_text = extract_text_from_bytes(resp.content)
        return RawDocument(
            source=self.name,
            source_doc_id=doc_id,
            title="",
            full_text=full_text,
            fetched_at=datetime.now(timezone.utc),
            raw={"pdf_bytes_len": len(resp.content)}
        )

    @retry(
        retry=retry_if_exception_type((_RetryableHTTP, httpx.TimeoutException)),
        wait=wait_exponential_jitter(initial=3, max=60),
        stop=stop_after_attempt(6),
        reraise=True,
    )
    def _get(self, url: str, params: dict | None = None) -> httpx.Response:
        resp = self._client.get(url, params=params)
        if resp.status_code in _RETRYABLE_STATUS:
            raise _RetryableHTTP(f"{resp.status_code} from {url}")
        resp.raise_for_status()
        return resp