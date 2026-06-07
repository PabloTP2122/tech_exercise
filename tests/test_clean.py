"""Unit tests for ingest/clean.py — Markdown body extraction."""

from ingest.clean import clean_body, clean_html

_ARTICLE_HTML = """\
<html><body>
<nav>Site navigation — must be stripped</nav>
<div id="hs_cos_wrapper_post_body">
  <h2>Getting started</h2>
  <p>Run <code>git diff</code> to inspect the changes.</p>
  <pre><code>function greet(name) {
  return `Hello, ${name}!`;
}</code></pre>
  <p>Pass <code>.eslintrc.json</code> as the config path.</p>
</div>
<footer>Footer — must be stripped</footer>
</body></html>
"""

_NO_SELECTOR_HTML = "<html><body><p>No article body here.</p></body></html>"


def test_inline_code_uses_backticks() -> None:
    """Inline <code> elements must become backtick spans, not plain text."""
    result = clean_html(_ARTICLE_HTML)
    assert "`git diff`" in result


def test_inline_code_stays_on_same_line() -> None:
    """Inline <code> must not be split onto its own line (get_text regression)."""
    result = clean_html(_ARTICLE_HTML)
    lines = result.splitlines()
    # "git diff" and "Run" must appear on the same line (inline, not fragmented).
    assert any("git diff" in line and "Run" in line for line in lines), (
        "Inline code was split from its surrounding sentence.\n"
        f"Lines containing 'git diff': {[ln for ln in lines if 'git diff' in ln]}"
    )


def test_pre_block_becomes_fenced_code() -> None:
    """<pre><code> blocks must produce fenced Markdown code blocks."""
    result = clean_html(_ARTICLE_HTML)
    assert "```" in result


def test_nav_and_footer_excluded() -> None:
    """Content outside the body selector must not appear in the output."""
    result = clean_html(_ARTICLE_HTML)
    assert "Site navigation" not in result
    assert "Footer" not in result


def test_fallback_returns_html_unchanged_when_selector_absent() -> None:
    """When the body selector is missing, clean_html returns the input unchanged."""
    result = clean_html(_NO_SELECTOR_HTML)
    assert result == _NO_SELECTOR_HTML


def test_clean_body_returns_blog_body_origin_when_selector_present() -> None:
    """clean_body must report 'blog_body' when the selector is found."""
    result = clean_body(_ARTICLE_HTML)
    assert result.content_origin == "blog_body"


def test_clean_body_returns_full_page_fallback_origin_when_selector_absent() -> None:
    """clean_body must report 'full_page_fallback' and return html intact when absent."""
    result = clean_body(_NO_SELECTOR_HTML)
    assert result.content_origin == "full_page_fallback"
    assert result.text == _NO_SELECTOR_HTML
