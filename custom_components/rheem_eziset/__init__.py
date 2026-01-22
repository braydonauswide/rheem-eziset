"""
Custom integration to integrate rheem_eziset with Home Assistant.

For more details about this integration, please refer to
https://github.com/braydonauswide/rheem-eziset
"""

from __future__ import annotations

import asyncio

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_HOST
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import device_registry as dr, entity_registry as er

from .api import RheemEziSETApi
from .const import (
    CONF_SCAN_INTERVAL,
    DEFAULT_BATHFILL_PRESETS,
    DEFAULT_SCAN_INTERVAL,
    DOMAIN,
    LOGGER,
    PLATFORMS,
    PRESET_SLOTS,
)
from .coordinator import RheemEziSETDataUpdateCoordinator
from .util import to_int


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up this integration using UI."""
    LOGGER.debug("Setting up entry for device: %s", entry.title)

    hass.data.setdefault(DOMAIN, {})

    host = entry.data.get(CONF_HOST)
    api = RheemEziSETApi(hass=hass, host=host)
    scan_interval = entry.options.get(CONF_SCAN_INTERVAL, DEFAULT_SCAN_INTERVAL)

    coordinator = RheemEziSETDataUpdateCoordinator(hass, api=api, update_interval=scan_interval, config_entry=entry)
    try:
        await coordinator.async_config_entry_first_refresh()
    except Exception as err:  # pylint: disable=broad-except
        LOGGER.warning("%s - Initial refresh failed: %s. Proceeding to register entry so Options/UI remain usable.", DOMAIN, err)
        # Ensure platforms can still set up even if the first refresh failed.
        # Many entities read coordinator.data during __init__.
        coordinator.data = coordinator.data or {}
    # Ensure coordinator.data is always a dict (some entities assume .data is not None)
    coordinator.data = coordinator.data or {}

    hass.data[DOMAIN][entry.entry_id] = coordinator

    # Register services once
    if not hass.data[DOMAIN].get("services_registered"):
        _register_services(hass)
        hass.data[DOMAIN]["services_registered"] = True

    platforms_to_load = [platform for platform in PLATFORMS if entry.options.get(platform, True)]
    coordinator.platforms.extend(platforms_to_load)

    if platforms_to_load:
        await hass.config_entries.async_forward_entry_setups(entry, platforms_to_load)

    entry.async_on_unload(entry.add_update_listener(async_reload_entry))

    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Handle removal of an entry."""
    coordinator: RheemEziSETDataUpdateCoordinator = hass.data[DOMAIN].get(entry.entry_id)
    platforms = coordinator.platforms if coordinator else PLATFORMS

    # Cancel request worker task to prevent "Task was destroyed but it is pending" errors
    if coordinator and coordinator.api:
        api = coordinator.api
        if api._request_worker_task and not api._request_worker_task.done():
            api._request_worker_task.cancel()
            try:
                await api._request_worker_task
            except asyncio.CancelledError:
                pass
            api._request_worker_task = None

    unloaded = await hass.config_entries.async_unload_platforms(entry, platforms)
    if unloaded:
        hass.data[DOMAIN].pop(entry.entry_id, None)
    return unloaded


async def async_reload_entry(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Reload config entry."""
    await async_unload_entry(hass, entry)
    await async_setup_entry(hass, entry)


def _resolve_entry_from_target(hass: HomeAssistant, device_id: str | None, entity_id: str | None) -> ConfigEntry:
    """Resolve config entry from device_id or entity_id."""
    if device_id:
        dev_reg = dr.async_get(hass)
        device = dev_reg.async_get(device_id)
        if not device:
            raise HomeAssistantError("Device not found")
        if not device.config_entries:
            raise HomeAssistantError("Device has no config entry")
        entry_id = next(iter(device.config_entries))
        entry = hass.config_entries.async_get_entry(entry_id)
        if not entry or entry.domain != DOMAIN:
            raise HomeAssistantError("Device does not belong to rheem_eziset")
        return entry

    if entity_id:
        ent_reg = er.async_get(hass)
        ent = ent_reg.async_get(entity_id)
        if not ent:
            raise HomeAssistantError("Entity not found")
        entry = hass.config_entries.async_get_entry(ent.config_entry_id)
        if not entry or entry.domain != DOMAIN:
            raise HomeAssistantError("Entity does not belong to rheem_eziset")
        return entry

    raise HomeAssistantError("Must provide device_id or entity_id")


def _register_services(hass: HomeAssistant) -> None:
    """Register preset management services."""

    async def _async_update_options(entry: ConfigEntry, new_options: dict) -> None:
        merged = {**entry.options, **new_options}
        # validate duplicates among enabled
        enabled_pairs: set[tuple[int, int]] = set()
        for idx in range(1, PRESET_SLOTS + 1):
            enabled = merged.get(f"bathfill_preset_{idx}_enabled", False)
            temp_raw = merged.get(f"bathfill_preset_{idx}_temp")
            vol_raw = merged.get(f"bathfill_preset_{idx}_vol")
            temp = to_int(temp_raw)
            vol = to_int(vol_raw)
            if not enabled:
                continue
            if temp in (None, 0) or vol in (None, 0):
                raise HomeAssistantError(f"Preset {idx} missing temp/vol")
            pair = (temp, vol)
            if pair in enabled_pairs:
                raise HomeAssistantError("Enabled presets must have unique temperature/volume pairs")
            enabled_pairs.add(pair)

        hass.config_entries.async_update_entry(entry, options=merged)
        await hass.config_entries.async_reload(entry.entry_id)

    async def async_set_bathfill_preset(call):
        """Set/enable a bath fill preset slot."""
        entry = _resolve_entry_from_target(hass, call.data.get("device_id"), call.data.get("entity_id"))
        slot = call.data["slot"]
        if slot < 1 or slot > PRESET_SLOTS:
            raise HomeAssistantError("Invalid slot; must be between 1 and 6")
        enabled_key = f"bathfill_preset_{slot}_enabled"
        name_key = f"bathfill_preset_{slot}_name"
        temp_key = f"bathfill_preset_{slot}_temp"
        vol_key = f"bathfill_preset_{slot}_vol"

        current_enabled = bool(entry.options.get(enabled_key, False))
        enabled = call.data.get("enabled", current_enabled)

        name = call.data.get("name")
        if name is None:
            name = entry.options.get(name_key) or f"Preset {slot}"
        else:
            name = name or f"Preset {slot}"

        temp_provided = "temp" in call.data
        vol_provided = "vol" in call.data
        temp = to_int(call.data.get("temp")) if temp_provided else None
        vol = to_int(call.data.get("vol")) if vol_provided else None

        if temp_provided and temp in (None, 0):
            raise HomeAssistantError("Invalid temp; must be a number > 0")
        if vol_provided and vol in (None, 0):
            raise HomeAssistantError("Invalid volume; must be a number > 0")

        merged_temp = temp if temp_provided else entry.options.get(temp_key)
        merged_vol = vol if vol_provided else entry.options.get(vol_key)
        if enabled and (merged_temp in (None, 0) or merged_vol in (None, 0)):
            raise HomeAssistantError("When enabling a preset, temp and vol are required and must be > 0")

        options_update = {
            enabled_key: enabled,
            name_key: name,
        }
        if temp_provided:
            options_update[temp_key] = temp
        if vol_provided:
            options_update[vol_key] = vol
        await _async_update_options(entry, options_update)

    async def async_disable_bathfill_preset(call):
        """Disable a bath fill preset slot."""
        entry = _resolve_entry_from_target(hass, call.data.get("device_id"), call.data.get("entity_id"))
        slot = call.data["slot"]
        if slot < 1 or slot > PRESET_SLOTS:
            raise HomeAssistantError("Invalid slot; must be between 1 and 6")
        options_update = {f"bathfill_preset_{slot}_enabled": False}
        await _async_update_options(entry, options_update)

    async def async_reset_bathfill_presets(call):
        """Reset bath fill presets to defaults."""
        entry = _resolve_entry_from_target(hass, call.data.get("device_id"), call.data.get("entity_id"))
        new_opts: dict = {}
        for idx in range(1, PRESET_SLOTS + 1):
            defaults = DEFAULT_BATHFILL_PRESETS.get(idx, {"enabled": False, "name": f"Preset {idx}", "temp": None, "vol": None})
            new_opts.update(
                {
                    f"bathfill_preset_{idx}_enabled": bool(defaults.get("enabled", False)),
                    f"bathfill_preset_{idx}_name": defaults.get("name", f"Preset {idx}"),
                    f"bathfill_preset_{idx}_temp": defaults.get("temp"),
                    f"bathfill_preset_{idx}_vol": defaults.get("vol"),
                }
            )
        await _async_update_options(entry, new_opts)

    hass.services.async_register(DOMAIN, "set_bathfill_preset", async_set_bathfill_preset)
    hass.services.async_register(DOMAIN, "disable_bathfill_preset", async_disable_bathfill_preset)
    hass.services.async_register(DOMAIN, "reset_bathfill_presets", async_reset_bathfill_presets)
