"""Offline unit tests for ingest/discovery.py.

All network calls are mocked — no live fetches.  Tests demonstrate that
:func:`discover_article_urls` correctly reconciles two URL sources and raises
:exc:`DiscoveryMismatchError` (with a useful diff) when they diverge.
"""

from unittest.mock import patch

import pytest

from ingest.discovery import DiscoveryMismatchError, discover_article_urls


class TestDiscoverArticleUrls:
    """Tests for the reconciliation logic in discover_article_urls."""

    def test_returns_union_when_sets_match(self) -> None:
        """When both sources agree, the union is returned (no error)."""
        urls = [
            "https://www.bitovi.com/blog/article-a",
            "https://www.bitovi.com/blog/article-b",
        ]
        with (
            patch("ingest.discovery.list_sitemap_urls", return_value=urls),
            patch("ingest.discovery.list_paginator_urls", return_value=urls),
        ):
            result = discover_article_urls(strict=True)
        assert result == urls

    def test_raises_on_mismatch_strict_true(self) -> None:
        """Any set-difference raises DiscoveryMismatchError in strict mode."""
        sitemap = [
            "https://www.bitovi.com/blog/article-a",
            "https://www.bitovi.com/blog/article-b",
            "https://www.bitovi.com/blog/article-sitemap-only",
        ]
        paginator = [
            "https://www.bitovi.com/blog/article-a",
            "https://www.bitovi.com/blog/article-b",
            "https://www.bitovi.com/blog/article-paginator-only",
        ]
        with (
            patch("ingest.discovery.list_sitemap_urls", return_value=sitemap),
            patch("ingest.discovery.list_paginator_urls", return_value=paginator),
            pytest.raises(DiscoveryMismatchError) as exc_info,
        ):
            discover_article_urls(strict=True)

        err = exc_info.value
        assert "https://www.bitovi.com/blog/article-sitemap-only" in err.sitemap_only
        assert "https://www.bitovi.com/blog/article-paginator-only" in err.paginator_only

    def test_mismatch_error_message_lists_both_diffs(self) -> None:
        """The error message must name both sides of the diff."""
        sitemap = ["https://www.bitovi.com/blog/only-in-sitemap"]
        paginator = ["https://www.bitovi.com/blog/only-in-paginator"]
        with (
            patch("ingest.discovery.list_sitemap_urls", return_value=sitemap),
            patch("ingest.discovery.list_paginator_urls", return_value=paginator),
            pytest.raises(DiscoveryMismatchError) as exc_info,
        ):
            discover_article_urls(strict=True)

        msg = str(exc_info.value)
        assert "only-in-sitemap" in msg
        assert "only-in-paginator" in msg

    def test_strict_false_returns_union_and_does_not_raise(self) -> None:
        """strict=False returns the union without raising."""
        sitemap = ["https://www.bitovi.com/blog/article-a"]
        paginator = [
            "https://www.bitovi.com/blog/article-a",
            "https://www.bitovi.com/blog/article-b",
        ]
        with (
            patch("ingest.discovery.list_sitemap_urls", return_value=sitemap),
            patch("ingest.discovery.list_paginator_urls", return_value=paginator),
        ):
            result = discover_article_urls(strict=False)

        assert "https://www.bitovi.com/blog/article-a" in result
        assert "https://www.bitovi.com/blog/article-b" in result

    def test_union_preserves_sitemap_order_first(self) -> None:
        """Sitemap-order URLs come first; paginator-only appended after."""
        sitemap = [
            "https://www.bitovi.com/blog/article-c",
            "https://www.bitovi.com/blog/article-a",
        ]
        paginator = [
            "https://www.bitovi.com/blog/article-a",
            "https://www.bitovi.com/blog/article-c",
            "https://www.bitovi.com/blog/article-new",
        ]
        with (
            patch("ingest.discovery.list_sitemap_urls", return_value=sitemap),
            patch("ingest.discovery.list_paginator_urls", return_value=paginator),
        ):
            result = discover_article_urls(strict=False)

        # Sitemap order preserved first
        assert result[0] == "https://www.bitovi.com/blog/article-c"
        assert result[1] == "https://www.bitovi.com/blog/article-a"
        # Paginator-only appended
        assert "https://www.bitovi.com/blog/article-new" in result

    def test_no_duplicates_in_union(self) -> None:
        """The union must not contain duplicate URLs."""
        shared = ["https://www.bitovi.com/blog/article-x"] * 1
        with (
            patch("ingest.discovery.list_sitemap_urls", return_value=shared),
            patch("ingest.discovery.list_paginator_urls", return_value=shared),
        ):
            result = discover_article_urls(strict=True)
        assert len(result) == len(set(result))


class TestDiscoveryMismatchError:
    def test_attributes_accessible(self) -> None:
        err = DiscoveryMismatchError(
            {"https://www.bitovi.com/blog/a"}, {"https://www.bitovi.com/blog/b"}
        )
        assert "https://www.bitovi.com/blog/a" in err.sitemap_only
        assert "https://www.bitovi.com/blog/b" in err.paginator_only

    def test_is_runtime_error(self) -> None:
        err = DiscoveryMismatchError(set(), set())
        assert isinstance(err, RuntimeError)
