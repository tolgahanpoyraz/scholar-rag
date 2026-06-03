from __future__ import annotations

import httpx

from scholar_rag.sources.arxiv import ArxivSource, _clean_id
from scholar_rag.sources.base import RateLimiter

_FAKE_FEED = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"
      xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/2401.01234v2</id>
    <title>Edge Reconstruction Numbers of Split Graphs</title>
    <summary>We classify the ERN trichotomy for split graphs.</summary>
    <author><name>T. Poyraz</name></author>
    <author><name>Y. Zhao</name></author>
    <category term="math.CO" scheme="http://arxiv.org/schemas/atom"/>
    <category term="cs.DM" scheme="http://arxiv.org/schemas/atom"/>
  </entry>
</feed>"""

def _client_returning(xml: str) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=xml)
    return httpx.Client(transport=httpx.MockTransport(handler))

def _no_wait_limiter() -> RateLimiter:
    return RateLimiter(min_interval_s=0.0)

def test_clean_id_strips_url_and_version():
    assert _clean_id("http://arxiv.org/abs/2401.01234v2") == "2401.01234"
    assert _clean_id("2401.01234") == "2401.01234"

def test_discover_parses_entry_fields():
    src = ArxivSource(client=_client_returning(_FAKE_FEED),
                      rate_limiter=_no_wait_limiter())
    docs = src.discover("cat:math.CO", max_results=10)

    assert len(docs) == 1
    d = docs[0]
    assert d.source == "arxiv"
    assert d.source_doc_id == "2401.01234"
    assert d.title == "Edge Reconstruction Numbers of Split Graphs"
    assert d.authors == ["T. Poyraz", "Y. Zhao"]
    assert d.categories == ["math.CO", "cs.DM"]
    assert d.abstract is not None and "trichotomy" in d.abstract

def test_discover_skips_malformed_entries():
    broken = _FAKE_FEED.replace("<id>http://arxiv.org/abs/2401.01234v2</id>", "")
    broken = broken.replace(
        "<title>Edge Reconstruction Numbers of Split Graphs</title>", ""
    )
    src = ArxivSource(client=_client_returning(broken),
                      rate_limiter=_no_wait_limiter())
    assert src.discover("cat:math.CO") == []