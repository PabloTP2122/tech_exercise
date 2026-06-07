"""Offline unit tests for ingest/load_vectorstore.py helpers."""

from ingest.load_vectorstore import wrap_as_data

_SAMPLE = "Run `git diff` to inspect the changes."


def test_wrap_adds_open_delimiter() -> None:
    assert wrap_as_data(_SAMPLE).startswith("<DATA_SOURCE>\n")


def test_wrap_adds_close_delimiter() -> None:
    assert wrap_as_data(_SAMPLE).endswith("\n</DATA_SOURCE>")


def test_wrap_preserves_text_verbatim() -> None:
    """Body text must be unchanged inside the delimiters."""
    result = wrap_as_data(_SAMPLE)
    assert _SAMPLE in result


def test_wrap_empty_string() -> None:
    assert wrap_as_data("") == "<DATA_SOURCE>\n\n</DATA_SOURCE>"
