"""Offline unit tests for ingest/fetcher.py."""

from unittest.mock import MagicMock, patch

import pytest
import requests

from ingest.fetcher import OffSiteRedirectError, fetch_html


def _mock_response(
    text: str = "<html>ok</html>", final_url: str = "https://example.com/page"
) -> MagicMock:
    """Build a minimal mock :class:`requests.Response`."""
    resp = MagicMock(spec=requests.Response)
    resp.text = text
    resp.url = final_url
    resp.raise_for_status = MagicMock()
    return resp


class TestFetchHtml:
    def test_returns_text_on_same_host(self) -> None:
        """Successful fetch with no redirect returns the response text."""
        mock_resp = _mock_response(
            text="<html>content</html>", final_url="https://example.com/page"
        )
        with patch("ingest.fetcher.requests.get", return_value=mock_resp):
            result = fetch_html("https://example.com/page")
        assert result == "<html>content</html>"

    def test_off_site_redirect_raises(self) -> None:
        """A redirect to a different host must raise OffSiteRedirectError."""
        mock_resp = _mock_response(final_url="https://stealjs.com/")
        with (
            patch("ingest.fetcher.requests.get", return_value=mock_resp),
            pytest.raises(OffSiteRedirectError),
        ):
            fetch_html("https://www.bitovi.com/blog/stealjs-script-manager")

    def test_same_host_different_path_does_not_raise(self) -> None:
        """Redirect within the same host (path change only) must succeed."""
        mock_resp = _mock_response(
            text="article",
            final_url="https://example.com/blog/article-canonical",
        )
        with patch("ingest.fetcher.requests.get", return_value=mock_resp):
            result = fetch_html("https://example.com/blog/article")
        assert result == "article"

    def test_raise_for_status_is_called(self) -> None:
        """HTTP error status (4xx/5xx) must propagate via raise_for_status."""
        mock_resp = _mock_response(final_url="https://example.com/page")
        with patch("ingest.fetcher.requests.get", return_value=mock_resp):
            fetch_html("https://example.com/page")
        mock_resp.raise_for_status.assert_called_once()

    def test_apex_to_www_redirect_does_not_raise(self) -> None:
        """apex → www redirect (same site) must NOT raise OffSiteRedirectError."""
        mock_resp = _mock_response(
            text="<html>article</html>",
            final_url="https://www.bitovi.com/blog/article",
        )
        with patch("ingest.fetcher.requests.get", return_value=mock_resp):
            result = fetch_html("https://bitovi.com/blog/article")
        assert result == "<html>article</html>"

    def test_off_site_redirect_error_is_requests_exception(self) -> None:
        """OffSiteRedirectError must be a subclass of requests.RequestException."""
        assert issubclass(OffSiteRedirectError, requests.RequestException)
