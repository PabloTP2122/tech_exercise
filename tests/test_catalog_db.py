"""Offline unit tests for ingest/catalog_db._parse_published_at.

Covers all four date formats observed in the Bitovi blog JSON-LD, plus the
failure cases, and verifies every parsed result is timezone-aware (required
for the recency SQL route that orders by the ``published_at TIMESTAMPTZ``
column).
"""

from datetime import UTC, datetime

import pytest

from ingest.catalog_db import _parse_published_at, _split_category_slugs


class TestParseDatePublishedAt:
    def test_blogposting_space_format(self) -> None:
        """Primary Bitovi JSON-LD format: 'YYYY-MM-DD HH:MM:SS' (no T, no tz)."""
        dt = _parse_published_at("2025-08-15 19:39:36")
        assert dt == datetime(2025, 8, 15, 19, 39, 36, tzinfo=UTC)

    def test_iso_with_utc_offset(self) -> None:
        """ISO 8601 with explicit +00:00 offset."""
        dt = _parse_published_at("2024-01-15T10:00:00+00:00")
        assert dt is not None
        assert dt.year == 2024 and dt.month == 1 and dt.day == 15
        assert dt.hour == 10
        assert dt.tzinfo is not None

    def test_iso_no_tz_gets_utc(self) -> None:
        """Timezone-naive ISO string must be assigned UTC, not left naive."""
        dt = _parse_published_at("2024-01-15T10:00:00")
        assert dt == datetime(2024, 1, 15, 10, 0, 0, tzinfo=UTC)

    def test_date_only_parses_as_midnight_utc(self) -> None:
        """YYYY-MM-DD parses as midnight UTC."""
        dt = _parse_published_at("2025-08-15")
        assert dt == datetime(2025, 8, 15, 0, 0, 0, tzinfo=UTC)

    def test_empty_string_returns_none(self) -> None:
        assert _parse_published_at("") is None

    def test_garbage_returns_none(self) -> None:
        assert _parse_published_at("not-a-date") is None

    def test_partial_date_returns_none(self) -> None:
        assert _parse_published_at("2025-08") is None

    @pytest.mark.parametrize(
        "value",
        [
            "2025-08-15 19:39:36",
            "2024-01-15T10:00:00+00:00",
            "2024-01-15T10:00:00",
            "2025-08-15",
        ],
    )
    def test_all_valid_formats_produce_tz_aware_datetime(self, value: str) -> None:
        """Every successfully parsed date must carry tzinfo (TIMESTAMPTZ safety)."""
        dt = _parse_published_at(value)
        assert dt is not None, f"Expected a datetime for {value!r}, got None"
        assert dt.tzinfo is not None, f"Expected tz-aware datetime for {value!r}"


# ---------------------------------------------------------------------------
# _split_category_slugs
# ---------------------------------------------------------------------------


class TestSplitCategorySlugs:
    """_split_category_slugs splits delimited rows and validates each slug."""

    def test_basic_row(self) -> None:
        assert _split_category_slugs([",ai,devops,"]) == ["ai", "devops"]

    def test_multiple_rows_deduplicated(self) -> None:
        result = _split_category_slugs([",ai,devops,", ",devops,react,"])
        assert result == ["ai", "devops", "react"]

    def test_sorted_output(self) -> None:
        result = _split_category_slugs([",react,ai,devops,"])
        assert result == ["ai", "devops", "react"]

    def test_empty_input(self) -> None:
        assert _split_category_slugs([]) == []

    def test_empty_string_row(self) -> None:
        assert _split_category_slugs([""]) == []

    def test_drops_non_conforming_slug(self) -> None:
        """Slugs with spaces, special chars, or uppercase are dropped (ADR-0008 S4)."""
        result = _split_category_slugs([",ai,inject this,"])
        assert result == ["ai"]
        assert "inject this" not in result

    def test_drops_slug_with_comma(self) -> None:
        """A stored slug containing a literal comma must not split into phantom categories."""
        result = _split_category_slugs([",a,b,"])
        assert result == ["a", "b"]

    def test_drops_slug_with_percent(self) -> None:
        result = _split_category_slugs([",ai,a%b,"])
        assert result == ["ai"]

    def test_drops_slug_with_single_quote(self) -> None:
        result = _split_category_slugs([",ai,o'reilly,"])
        assert result == ["ai"]

    def test_drops_uppercase_slug(self) -> None:
        result = _split_category_slugs([",AI,devops,"])
        assert result == ["devops"]

    def test_hyphenated_slug_accepted(self) -> None:
        result = _split_category_slugs([",frontend-engineering,project-management,"])
        assert result == ["frontend-engineering", "project-management"]

    def test_numbers_in_slug_accepted(self) -> None:
        result = _split_category_slugs([",web3,angular17,"])
        assert result == ["angular17", "web3"]
