"""Tests for agent/rss.py — offline mocked suite + @pytest.mark.live real-feed test."""

from unittest.mock import MagicMock, patch

import pytest
import requests

from agent.rss import fetch_recent

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SAMPLE_RSS = """\
<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>Bitovi Blog</title>
    <item>
      <title>Article One</title>
      <link>https://www.bitovi.com/blog/article-one</link>
      <description>First article summary.</description>
      <pubDate>Mon, 02 Jun 2026 10:00:00 GMT</pubDate>
    </item>
    <item>
      <title>Article Two</title>
      <link>https://www.bitovi.com/blog/article-two</link>
      <description>Second article summary.</description>
      <pubDate>Sun, 01 Jun 2026 08:00:00 GMT</pubDate>
    </item>
    <item>
      <title>Article Three</title>
      <link>https://www.bitovi.com/blog/article-three</link>
      <description>Third article summary.</description>
      <pubDate>Sat, 31 May 2026 12:00:00 GMT</pubDate>
    </item>
    <item>
      <title>Article Four</title>
      <link>https://www.bitovi.com/blog/article-four</link>
      <description>Fourth article summary.</description>
      <pubDate>Fri, 30 May 2026 09:00:00 GMT</pubDate>
    </item>
    <item>
      <title>Article Five</title>
      <link>https://www.bitovi.com/blog/article-five</link>
      <description>Fifth article summary.</description>
      <pubDate>Thu, 29 May 2026 07:00:00 GMT</pubDate>
    </item>
  </channel>
</rss>
"""

_EMPTY_RSS = """\
<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>Bitovi Blog</title>
  </channel>
</rss>
"""


def _mock_ok(body: str = _SAMPLE_RSS) -> MagicMock:
    resp = MagicMock(spec=requests.Response)
    resp.text = body
    resp.raise_for_status = MagicMock()
    return resp


# ---------------------------------------------------------------------------
# Offline mocked tests
# ---------------------------------------------------------------------------


class TestFetchRecentHappyPath:
    def test_returns_four_keys_per_item(self) -> None:
        with patch("agent.rss.requests.get", return_value=_mock_ok()):
            items = fetch_recent(None)
        assert items
        assert {"title", "link", "description", "pubDate"} <= set(items[0].keys())

    def test_default_limit_is_three(self) -> None:
        with patch("agent.rss.requests.get", return_value=_mock_ok()):
            items = fetch_recent(None)
        assert len(items) == 3

    def test_limit_respected(self) -> None:
        with patch("agent.rss.requests.get", return_value=_mock_ok()):
            items = fetch_recent(None, limit=2)
        assert len(items) == 2

    def test_feed_order_preserved(self) -> None:
        with patch("agent.rss.requests.get", return_value=_mock_ok()):
            items = fetch_recent(None, limit=5)
        assert items[0]["title"] == "Article One"
        assert items[4]["title"] == "Article Five"

    def test_correct_values_extracted(self) -> None:
        with patch("agent.rss.requests.get", return_value=_mock_ok()):
            items = fetch_recent(None, limit=1)
        assert items[0]["title"] == "Article One"
        assert items[0]["link"] == "https://www.bitovi.com/blog/article-one"
        assert items[0]["description"] == "First article summary."
        assert items[0]["pubDate"] == "Mon, 02 Jun 2026 10:00:00 GMT"


class TestFetchRecentUrlRouting:
    def test_slug_none_uses_global_feed(self) -> None:
        with patch("agent.rss.requests.get", return_value=_mock_ok()) as mock_get:
            fetch_recent(None)
        url = mock_get.call_args[0][0]
        assert url.endswith("/blog/rss.xml")
        assert "/topic/" not in url

    def test_slug_uses_topic_feed(self) -> None:
        with patch("agent.rss.requests.get", return_value=_mock_ok()) as mock_get:
            fetch_recent("devops")
        url = mock_get.call_args[0][0]
        assert "/blog/topic/devops/rss.xml" in url


class TestFetchRecentFailClosed:
    def test_raise_for_status_error_returns_empty(self) -> None:
        resp = _mock_ok()
        resp.raise_for_status.side_effect = requests.HTTPError("404")
        with patch("agent.rss.requests.get", return_value=resp):
            assert fetch_recent(None) == []

    def test_requests_get_raises_returns_empty(self) -> None:
        with patch("agent.rss.requests.get", side_effect=requests.ConnectionError("offline")):
            assert fetch_recent(None) == []

    def test_malformed_xml_returns_empty(self) -> None:
        resp = _mock_ok("<this is not xml>>>")
        with patch("agent.rss.requests.get", return_value=resp):
            assert fetch_recent(None) == []

    def test_empty_feed_returns_empty_list(self) -> None:
        with patch("agent.rss.requests.get", return_value=_mock_ok(_EMPTY_RSS)):
            assert fetch_recent(None) == []


# ---------------------------------------------------------------------------
# Live test — deselected by default; run via: make test-live
# ---------------------------------------------------------------------------


@pytest.mark.live
def test_real_bitovi_feed() -> None:
    items = fetch_recent(None)
    assert items, "expected at least one item from the real Bitovi RSS feed"
    assert len(items) <= 3
    assert {"title", "link", "description", "pubDate"} <= set(items[0].keys())
    from email.utils import parsedate_to_datetime

    parsedate_to_datetime(items[0]["pubDate"])  # must not raise — valid RFC-822
