"""Offline unit tests for ingest/extract.py and ingest/clean.py.

All tests use the committed HTML fixture at ``tests/fixtures/article_sample.html``
— no network requests, no OpenAI calls.  Demonstrates SRP testability: each
extraction function is exercised in isolation.
"""

from pathlib import Path

import pytest
from bs4 import BeautifulSoup

from ingest.clean import clean_html
from ingest.extract import (
    build_document,
    extract_categories,
    extract_metadata,
    select_blogposting,
)

# ---------------------------------------------------------------------------
# Fixture loading
# ---------------------------------------------------------------------------

_FIXTURE = Path(__file__).parent / "fixtures" / "article_sample.html"
_ARTICLE_URL = (
    "https://www.bitovi.com/blog/how-i-rewrote-a-clients-esling-config-with-a-single-ai-command"
)

_NO_LD_HTML = """
<html><head>
  <meta property="og:title" content="OG Title Article">
  <meta property="og:description" content="OG description text.">
  <meta property="article:published_time" content="2024-01-15T10:00:00Z">
  <meta name="author" content="OG Author">
</head><body>
  <span id="hs_cos_wrapper_post_body"><p>Article body here.</p></span>
  <a href="https://www.bitovi.com/blog/topic/react">react</a>
</body></html>
"""

_NO_LD_NO_OG_HTML = """
<html><head><title>H1 Fallback Article</title></head><body>
  <h1>H1 Fallback Title</h1>
  <span id="hs_cos_wrapper_post_body"><p>Content here.</p></span>
</body></html>
"""


@pytest.fixture(scope="module")
def sample_soup() -> BeautifulSoup:
    html = _FIXTURE.read_text(encoding="utf-8")
    return BeautifulSoup(html, "lxml")


@pytest.fixture(scope="module")
def sample_html() -> str:
    return _FIXTURE.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# select_blogposting
# ---------------------------------------------------------------------------


class TestSelectBlogposting:
    def test_returns_correct_block_not_organization(self, sample_soup: BeautifulSoup) -> None:
        """Block [0] is Organization — must NOT be returned."""
        bp = select_blogposting(sample_soup, _ARTICLE_URL)
        assert bp.get("@type") == "BlogPosting"
        expected = "How I Rewrote a Client's ESLint Config With a Single AI Command"
        assert bp.get("headline") == expected

    def test_does_not_return_graph_related_post(self, sample_soup: BeautifulSoup) -> None:
        """Block [2] @graph contains a RELATED BlogPosting (Amy Cutlip). Must NOT be chosen."""
        bp = select_blogposting(sample_soup, _ARTICLE_URL)
        assert bp.get("headline") != "How I Used AI Agents to Unify Prettier & ESLint in a Monorepo"

    def test_correct_author_and_date(self, sample_soup: BeautifulSoup) -> None:
        bp = select_blogposting(sample_soup, _ARTICLE_URL)
        author_raw = bp.get("author", {})
        assert isinstance(author_raw, dict)
        assert author_raw.get("name") == "Kyle Nazario"
        assert bp.get("datePublished") == "2025-08-15 19:39:36"

    def test_returns_empty_dict_when_no_blogposting(self) -> None:
        soup = BeautifulSoup("<html><body></body></html>", "lxml")
        bp = select_blogposting(soup, "https://example.com/blog/article")
        assert bp == {}


# ---------------------------------------------------------------------------
# extract_metadata — BlogPosting path
# ---------------------------------------------------------------------------


class TestExtractMetadataBlogPosting:
    def test_title_from_blogposting(self, sample_soup: BeautifulSoup) -> None:
        meta = extract_metadata(sample_soup, _ARTICLE_URL)
        assert meta["title"] == "How I Rewrote a Client's ESLint Config With a Single AI Command"

    def test_description_from_blogposting_not_organization(
        self, sample_soup: BeautifulSoup
    ) -> None:
        """Description must come from BlogPosting, NOT the generic Organization tagline."""
        meta = extract_metadata(sample_soup, _ARTICLE_URL)
        assert "Bitovi is a UX, UI design" not in meta["description"]
        assert "ESLint" in meta["description"]

    def test_author_from_nested_person(self, sample_soup: BeautifulSoup) -> None:
        meta = extract_metadata(sample_soup, _ARTICLE_URL)
        assert meta["author"] == "Kyle Nazario"

    def test_published_at(self, sample_soup: BeautifulSoup) -> None:
        meta = extract_metadata(sample_soup, _ARTICLE_URL)
        assert meta["published_at"] == "2025-08-15 19:39:36"

    def test_all_keys_present(self, sample_soup: BeautifulSoup) -> None:
        meta = extract_metadata(sample_soup, _ARTICLE_URL)
        assert set(meta.keys()) == {"title", "description", "author", "published_at"}


# ---------------------------------------------------------------------------
# extract_metadata — OpenGraph fallback
# ---------------------------------------------------------------------------


class TestExtractMetadataOGFallback:
    def test_og_title_used_when_no_blogposting(self) -> None:
        soup = BeautifulSoup(_NO_LD_HTML, "lxml")
        meta = extract_metadata(soup, "https://www.bitovi.com/blog/some-article")
        assert meta["title"] == "OG Title Article"

    def test_og_description_and_author(self) -> None:
        soup = BeautifulSoup(_NO_LD_HTML, "lxml")
        meta = extract_metadata(soup, "https://www.bitovi.com/blog/some-article")
        assert meta["description"] == "OG description text."
        assert meta["author"] == "OG Author"

    def test_og_published_at(self) -> None:
        soup = BeautifulSoup(_NO_LD_HTML, "lxml")
        meta = extract_metadata(soup, "https://www.bitovi.com/blog/some-article")
        assert meta["published_at"] == "2024-01-15T10:00:00Z"


# ---------------------------------------------------------------------------
# extract_metadata — h1/title fallback
# ---------------------------------------------------------------------------


class TestExtractMetadataH1Fallback:
    def test_h1_title_used_as_last_resort(self) -> None:
        soup = BeautifulSoup(_NO_LD_NO_OG_HTML, "lxml")
        meta = extract_metadata(soup, "https://www.bitovi.com/blog/some-article")
        assert meta["title"] == "H1 Fallback Title"

    def test_empty_description_and_author_in_h1_fallback(self) -> None:
        soup = BeautifulSoup(_NO_LD_NO_OG_HTML, "lxml")
        meta = extract_metadata(soup, "https://www.bitovi.com/blog/some-article")
        assert meta["description"] == ""
        assert meta["author"] == ""
        assert meta["published_at"] == ""


# ---------------------------------------------------------------------------
# extract_categories
# ---------------------------------------------------------------------------


class TestExtractCategories:
    def test_correct_slugs_from_topic_links(self, sample_soup: BeautifulSoup) -> None:
        cats = extract_categories(sample_soup)
        assert cats == ",ai,frontend-engineering,"

    def test_delimited_format(self, sample_soup: BeautifulSoup) -> None:
        cats = extract_categories(sample_soup)
        assert cats.startswith(",") and cats.endswith(",")
        assert len(cats) > 1

    def test_empty_sentinel_when_no_topic_links(self) -> None:
        soup = BeautifulSoup("<html><body><a href='/other'>link</a></body></html>", "lxml")
        assert extract_categories(soup) == ","

    def test_deduplication(self) -> None:
        html = """<html><body>
          <a href="https://www.bitovi.com/blog/topic/react">react</a>
          <a href="https://www.bitovi.com/blog/topic/react">react again</a>
          <a href="https://www.bitovi.com/blog/topic/angular">angular</a>
        </body></html>"""
        soup = BeautifulSoup(html, "lxml")
        assert extract_categories(soup) == ",react,angular,"


# ---------------------------------------------------------------------------
# clean_html
# ---------------------------------------------------------------------------


class TestCleanHtml:
    def test_extracts_only_post_body(self, sample_html: str) -> None:
        """Body content present; nav/footer/pixels must be stripped."""
        result = clean_html(sample_html)
        # Article body content preserved
        assert "refactored some code for a Bitovi client" in result
        assert "SWP uses a monorepo" in result

    def test_strips_facebook_pixel(self, sample_html: str) -> None:
        result = clean_html(sample_html)
        assert "facebook.com/tr" not in result

    def test_strips_nav(self, sample_html: str) -> None:
        result = clean_html(sample_html)
        assert "hs_cos_wrapper_header" not in result

    def test_strips_footer(self, sample_html: str) -> None:
        result = clean_html(sample_html)
        assert "© 2026 Bitovi" not in result

    def test_fallback_returns_original_when_no_selector(self) -> None:
        plain = "Just some plain text with no HTML structure."
        result = clean_html(plain)
        assert result == plain


# ---------------------------------------------------------------------------
# build_document (integration — offline)
# ---------------------------------------------------------------------------


class TestBuildDocument:
    def test_all_six_metadata_keys(self, sample_html: str) -> None:
        doc = build_document(_ARTICLE_URL, sample_html)
        assert set(doc.metadata.keys()) >= {
            "source_url",
            "title",
            "description",
            "author",
            "published_at",
            "categories",
        }

    def test_source_url_populated(self, sample_html: str) -> None:
        doc = build_document(_ARTICLE_URL, sample_html)
        assert doc.metadata["source_url"] == _ARTICLE_URL

    def test_title_non_empty(self, sample_html: str) -> None:
        doc = build_document(_ARTICLE_URL, sample_html)
        assert doc.metadata["title"] != ""

    def test_clean_body_in_page_content(self, sample_html: str) -> None:
        doc = build_document(_ARTICLE_URL, sample_html)
        assert "refactored some code for a Bitovi client" in doc.page_content
        assert "facebook.com/tr" not in doc.page_content
        assert "© 2026 Bitovi" not in doc.page_content
