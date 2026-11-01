"""Tests for config entry migration."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from custom_components.qiara import async_migrate_entry


def _entry(minor_version: int, data: dict) -> MagicMock:
    entry = MagicMock(version=1, minor_version=minor_version, data=data)
    return entry


@pytest.mark.asyncio
async def test_migrate_1_1_scopes_unique_id_by_host():
    """1.1 entries were keyed by customer id alone; 1.2 adds the host."""
    hass = MagicMock()
    entry = _entry(1, {"customer_id": 5016, "host": "192.0.2.25", "port": 64218})

    assert await async_migrate_entry(hass, entry)

    hass.config_entries.async_update_entry.assert_called_once_with(
        entry, unique_id="5016@192.0.2.25", minor_version=2
    )


@pytest.mark.asyncio
async def test_migrate_legacy_base_url_entry():
    """Entries that only stored base_url get their host from it."""
    hass = MagicMock()
    entry = _entry(1, {"customer_id": 7, "base_url": "https://lwip.localdomain:64218/api/v1"})

    assert await async_migrate_entry(hass, entry)

    hass.config_entries.async_update_entry.assert_called_once_with(
        entry, unique_id="7@lwip.localdomain", minor_version=2
    )


@pytest.mark.asyncio
async def test_migrate_1_2_untouched():
    """An entry already at 1.2 is left alone."""
    hass = MagicMock()
    entry = _entry(2, {"customer_id": 5016, "host": "192.0.2.25"})

    assert await async_migrate_entry(hass, entry)

    hass.config_entries.async_update_entry.assert_not_called()
