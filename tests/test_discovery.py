"""Offline unit tests for ingest/discovery.py.

All network calls are mocked — no live fetches.  Tests cover:

- URL normalization and article-path filtering (new helpers).
- :func:`reconcile` — the pure function that detects real article-list
  discrepancies and surfaces raw-paginator repetitions.
- :func:`discover_article_urls` — asymmetric strict gate and union ordering.
- :class:`DiscoveryMismatchError` — attributes and inheritance.
"""

from unittest.mock import patch

import pytest

from ingest.discovery import (
    DiscoveryMismatchError,
    ReconcileReport,
    _is_article_url,
    _normalize_url,
    discover_article_urls,
    reconcile,
)

_CANONICAL = "www.bitovi.com"
_BASE = f"https://{_CANONICAL}"


# ---------------------------------------------------------------------------
# _normalize_url
# ---------------------------------------------------------------------------


class TestNormalizeUrl:
    """_normalize_url collapses www/apex variants, resolves relative paths,
    and returns None for off-site hrefs."""

    def test_absolute_www_unchanged(self) -> None:
        url = f"{_BASE}/blog/angular-tips"
        assert _normalize_url(url, _CANONICAL) == url

    def test_relative_path_resolved(self) -> None:
        result = _normalize_url("/blog/angular-tips", _CANONICAL)
        assert result == f"{_BASE}/blog/angular-tips"

    def test_apex_collapsed_to_canonical(self) -> None:
        """https://bitovi.com/blog/x should become https://www.bitovi.com/blog/x."""
        result = _normalize_url("https://bitovi.com/blog/canjs-4.0", _CANONICAL)
        assert result == f"{_BASE}/blog/canjs-4.0"

    def test_trailing_slash_stripped(self) -> None:
        result = _normalize_url(f"{_BASE}/blog/angular-tips/", _CANONICAL)
        assert result == f"{_BASE}/blog/angular-tips"

    def test_query_and_fragment_stripped(self) -> None:
        result = _normalize_url(f"{_BASE}/blog/tips?utm=1#section", _CANONICAL)
        assert result == f"{_BASE}/blog/tips"

    def test_off_site_returns_none(self) -> None:
        assert _normalize_url("https://stealjs.com/docs", _CANONICAL) is None

    def test_figma_returns_none(self) -> None:
        assert _normalize_url("https://figma.com/blog/schema-2025", _CANONICAL) is None

    def test_empty_string_returns_none(self) -> None:
        assert _normalize_url("", _CANONICAL) is None

    def test_fragment_only_returns_none(self) -> None:
        assert _normalize_url("#section", _CANONICAL) is None

    def test_http_scheme_forced_to_https(self) -> None:
        result = _normalize_url("http://www.bitovi.com/blog/article", _CANONICAL)
        # apex check doesn't apply; http host still matches www — collapses to https
        assert result is not None
        assert result.startswith("https://")


# ---------------------------------------------------------------------------
# _is_article_url
# ---------------------------------------------------------------------------


class TestIsArticleUrl:
    """_is_article_url accepts dotted/html slugs and rejects topic/page paths."""

    def test_plain_slug_accepted(self) -> None:
        assert _is_article_url(f"{_BASE}/blog/angular-tips") is True

    def test_dotted_slug_accepted(self) -> None:
        """Slugs like canjs-4.0, donejs-2.0 must match (dots allowed)."""
        assert _is_article_url(f"{_BASE}/blog/canjs-4.0") is True

    def test_html_suffix_accepted(self) -> None:
        assert _is_article_url(f"{_BASE}/blog/ie-11-and-angular-overview.html") is True

    def test_nodejs_dotted_slug_accepted(self) -> None:
        assert _is_article_url(f"{_BASE}/blog/node.js-consulting-101") is True

    def test_topic_path_rejected(self) -> None:
        assert _is_article_url(f"{_BASE}/blog/topic/react") is False

    def test_page_path_rejected(self) -> None:
        assert _is_article_url(f"{_BASE}/blog/page/2") is False

    def test_blog_root_rejected(self) -> None:
        assert _is_article_url(f"{_BASE}/blog") is False

    def test_non_blog_path_rejected(self) -> None:
        assert _is_article_url(f"{_BASE}/services/angular") is False


# ---------------------------------------------------------------------------
# reconcile
# ---------------------------------------------------------------------------

_ART_A = f"{_BASE}/blog/article-a"
_ART_B = f"{_BASE}/blog/article-b"
_ART_C = f"{_BASE}/blog/article-c"
_FEATURED = f"{_BASE}/blog/the-complete-guide-to-devops"


class TestReconcile:
    """reconcile() is a pure function — no network calls, no side effects."""

    def test_returns_reconcile_report(self) -> None:
        report = reconcile([_ART_A], [_ART_A])
        assert isinstance(report, ReconcileReport)

    def test_union_contains_all_articles(self) -> None:
        report = reconcile([_ART_A, _ART_B], [_ART_A, _ART_B, _ART_C])
        assert set(report.union) == {_ART_A, _ART_B, _ART_C}

    def test_no_duplicates_in_union(self) -> None:
        report = reconcile([_ART_A, _ART_B], [_ART_A, _ART_B])
        assert len(report.union) == len(set(report.union))

    def test_sitemap_order_preserved_first(self) -> None:
        report = reconcile([_ART_B, _ART_A], [_ART_A, _ART_B, _ART_C])
        # sitemap order: B then A
        assert report.union[0] == _ART_B
        assert report.union[1] == _ART_A
        # paginator-only appended after
        assert report.union[2] == _ART_C

    def test_paginator_only_appended_in_discovery_order(self) -> None:
        """Paginator-only additions follow sitemap block, in first-seen crawl order."""
        report = reconcile([], [_ART_C, _ART_A, _ART_B])
        assert report.union == [_ART_C, _ART_A, _ART_B]

    def test_sitemap_only_populated(self) -> None:
        report = reconcile([_ART_A, _ART_B], [_ART_A])
        assert report.sitemap_only == {_ART_B}

    def test_paginator_only_populated(self) -> None:
        report = reconcile([_ART_A], [_ART_A, _ART_C])
        assert report.paginator_only == {_ART_C}

    def test_both_sources_agree_empty_diffs(self) -> None:
        report = reconcile([_ART_A, _ART_B], [_ART_A, _ART_B])
        assert report.sitemap_only == set()
        assert report.paginator_only == set()

    def test_raw_count_reflects_repeats(self) -> None:
        """paginator_raw_count = number of raw links including repeats."""
        raw = [_FEATURED, _ART_A, _FEATURED, _ART_B, _FEATURED]
        report = reconcile([_ART_A, _ART_B, _FEATURED], raw)
        assert report.paginator_raw_count == 5

    def test_repeated_map_captures_featured_links(self) -> None:
        """URLs appearing >1x in raw paginator are featured-link repeats, not dupes."""
        raw = [_FEATURED, _ART_A, _FEATURED, _ART_B, _FEATURED]
        report = reconcile([_ART_A, _ART_B, _FEATURED], raw)
        assert report.repeated[_FEATURED] == 3
        assert _ART_A not in report.repeated
        assert _ART_B not in report.repeated

    def test_paginator_unique_collapses_repeats(self) -> None:
        """paginator_unique has no duplication regardless of raw repeat count."""
        raw = [_FEATURED] * 10 + [_ART_A, _ART_B]
        report = reconcile([_ART_A, _ART_B, _FEATURED], raw)
        assert report.paginator_unique == {_FEATURED, _ART_A, _ART_B}

    def test_empty_inputs_produce_empty_report(self) -> None:
        report = reconcile([], [])
        assert report.union == []
        assert report.repeated == {}
        assert report.sitemap_only == set()
        assert report.paginator_only == set()

    def test_within_page_repeats_also_counted(self) -> None:
        """Within-page repeats (same URL twice in one page's raw output) are captured."""
        raw = [_ART_A, _ART_A, _ART_B]  # _ART_A appeared twice on a single page
        report = reconcile([_ART_A, _ART_B], raw)
        assert report.repeated[_ART_A] == 2
        assert len(report.union) == 2  # only 2 unique articles, no duplication


# ---------------------------------------------------------------------------
# discover_article_urls — asymmetric strict gate
# ---------------------------------------------------------------------------


class TestDiscoverArticleUrls:
    """discover_article_urls raises only on sitemap-only discrepancies (asymmetric gate)."""

    def _patch(
        self,
        sitemap: list[str],
        paginator_raw: list[str],
    ) -> tuple[object, object]:
        """Context manager that patches both discovery functions."""
        return (
            patch("ingest.discovery.list_sitemap_urls", return_value=sitemap),
            patch("ingest.discovery._list_paginator_raw", return_value=paginator_raw),
        )

    def test_matching_sources_return_union(self) -> None:
        urls = [_ART_A, _ART_B]
        with (
            patch("ingest.discovery.list_sitemap_urls", return_value=urls),
            patch("ingest.discovery._list_paginator_raw", return_value=urls),
        ):
            result = discover_article_urls(strict=True)
        assert result == urls

    def test_paginator_only_does_not_raise_in_strict_mode(self) -> None:
        """paginator_only articles are a NORMAL state — must NOT trigger an error."""
        with (
            patch("ingest.discovery.list_sitemap_urls", return_value=[_ART_A]),
            patch("ingest.discovery._list_paginator_raw", return_value=[_ART_A, _ART_B]),
        ):
            result = discover_article_urls(strict=True)
        assert _ART_B in result  # included in union

    def test_paginator_only_included_in_union(self) -> None:
        with (
            patch("ingest.discovery.list_sitemap_urls", return_value=[_ART_A]),
            patch("ingest.discovery._list_paginator_raw", return_value=[_ART_A, _ART_B, _ART_C]),
        ):
            result = discover_article_urls(strict=False)
        assert set(result) == {_ART_A, _ART_B, _ART_C}

    def test_sitemap_only_raises_in_strict_mode(self) -> None:
        """sitemap_only is the suspicious direction — must raise when strict=True."""
        with (
            patch("ingest.discovery.list_sitemap_urls", return_value=[_ART_A, _ART_B]),
            patch("ingest.discovery._list_paginator_raw", return_value=[_ART_A]),
            pytest.raises(DiscoveryMismatchError) as exc_info,
        ):
            discover_article_urls(strict=True)
        assert _ART_B in exc_info.value.sitemap_only

    def test_sitemap_only_does_not_raise_when_strict_false(self) -> None:
        with (
            patch("ingest.discovery.list_sitemap_urls", return_value=[_ART_A, _ART_B]),
            patch("ingest.discovery._list_paginator_raw", return_value=[_ART_A]),
        ):
            result = discover_article_urls(strict=False)
        assert _ART_A in result
        assert _ART_B in result

    def test_no_duplicates_in_result(self) -> None:
        shared = [_ART_A, _ART_B]
        with (
            patch("ingest.discovery.list_sitemap_urls", return_value=shared),
            patch("ingest.discovery._list_paginator_raw", return_value=shared),
        ):
            result = discover_article_urls(strict=True)
        assert len(result) == len(set(result))

    def test_featured_repeats_do_not_inflate_union(self) -> None:
        """A URL repeated 10× in raw paginator must appear exactly once in the union."""
        raw = [_FEATURED] * 10 + [_ART_A]
        with (
            patch("ingest.discovery.list_sitemap_urls", return_value=[_ART_A, _FEATURED]),
            patch("ingest.discovery._list_paginator_raw", return_value=raw),
        ):
            result = discover_article_urls(strict=True)
        assert result.count(_FEATURED) == 1
        assert result.count(_ART_A) == 1

    def test_sitemap_order_is_first_in_result(self) -> None:
        with (
            patch("ingest.discovery.list_sitemap_urls", return_value=[_ART_B, _ART_A]),
            patch("ingest.discovery._list_paginator_raw", return_value=[_ART_A, _ART_B, _ART_C]),
        ):
            result = discover_article_urls(strict=False)
        assert result[0] == _ART_B
        assert result[1] == _ART_A
        assert result[2] == _ART_C

    def test_mismatch_error_message_includes_sitemap_only(self) -> None:
        with (
            patch("ingest.discovery.list_sitemap_urls", return_value=[_ART_A, _ART_B]),
            patch("ingest.discovery._list_paginator_raw", return_value=[_ART_A]),
            pytest.raises(DiscoveryMismatchError) as exc_info,
        ):
            discover_article_urls(strict=True)
        assert "article-b" in str(exc_info.value)


# ---------------------------------------------------------------------------
# DiscoveryMismatchError
# ---------------------------------------------------------------------------


class TestDiscoveryMismatchError:
    def test_attributes_accessible(self) -> None:
        err = DiscoveryMismatchError({_ART_A}, {_ART_B})
        assert _ART_A in err.sitemap_only
        assert _ART_B in err.paginator_only

    def test_is_runtime_error(self) -> None:
        assert isinstance(DiscoveryMismatchError(set(), set()), RuntimeError)

    def test_empty_sets_does_not_crash(self) -> None:
        err = DiscoveryMismatchError(set(), set())
        assert str(err) is not None
