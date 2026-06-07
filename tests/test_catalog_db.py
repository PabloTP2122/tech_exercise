"""Offline unit tests for ingest/catalog_db._parse_published_at.

Covers all four date formats observed in the Bitovi blog JSON-LD, plus the
failure cases, and verifies every parsed result is timezone-aware (required
for the recency SQL route that orders by the ``published_at TIMESTAMPTZ``
column).
"""

from datetime import UTC, datetime

import pytest

from ingest.catalog_db import _parse_published_at


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
