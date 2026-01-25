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
from homeassistant.setup import async_setup_component
from homeassistant.helpers.event import async_track_state_change_event
from homeassistant.util import slugify
from homeassistant.components import input_select

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


def _build_presets(entry: ConfigEntry) -> list[dict]:
    """Build enabled preset list from options (fallback to defaults)."""
    presets: list[dict] = []
    opts = entry.options
    use_defaults = not any(key.startswith("bathfill_preset_") for key in opts)
    LOGGER.debug(
        "%s preset build start entry_id=%s use_defaults=%s options_keys=%s",
        DOMAIN,
        entry.entry_id,
        use_defaults,
        [k for k in opts.keys() if k.startswith("bathfill_preset_")],
    )

    def _format_label(name: str, slot: int, temp: int | None, vol: int | None) -> str:
        """Format a stable, user-friendly label for an enabled preset."""
        safe_name = name.strip() or f"Preset {slot}"
        temp_part = f"{temp}C" if temp is not None else "temp?"
        vol_part = f"{vol}L" if vol is not None else "vol?"
        return f"{safe_name} (Slot {slot}, {temp_part}, {vol_part})"

    for idx in range(1, PRESET_SLOTS + 1):
        defaults = DEFAULT_BATHFILL_PRESETS.get(idx) if use_defaults else None
        enabled_raw = opts.get(f"bathfill_preset_{idx}_enabled")
        enabled = bool(enabled_raw) if enabled_raw is not None else bool(defaults and defaults.get("enabled", False))
        if not enabled:
            continue
        name = opts.get(f"bathfill_preset_{idx}_name") or (defaults.get("name") if defaults else f"Preset {idx}")
        temp = to_int(opts.get(f"bathfill_preset_{idx}_temp")) or (defaults.get("temp") if defaults else None)
        vol = to_int(opts.get(f"bathfill_preset_{idx}_vol")) or (defaults.get("vol") if defaults else None)
        if temp is None or vol is None:
            continue
        label = _format_label(str(name), idx, int(temp), int(vol))
        presets.append({"name": str(name), "temp": int(temp), "vol": int(vol), "slot": idx, "label": label})
    if not presets and use_defaults and DEFAULT_BATHFILL_PRESETS:
        # Fallback: synthesize defaults to keep bath fill usable
        for idx, defaults in DEFAULT_BATHFILL_PRESETS.items():
            label = _format_label(str(defaults.get("name", f"Preset {idx}")), idx, defaults.get("temp"), defaults.get("vol"))
            presets.append(
                {
                    "name": str(defaults.get("name", f"Preset {idx}")),
                    "temp": int(defaults.get("temp", 0) or 0),
                    "vol": int(defaults.get("vol", 0) or 0),
                    "slot": idx,
                    "label": label,
                }
            )
        LOGGER.warning("%s - No presets enabled; synthesized defaults for bath fill", DOMAIN)
    if not presets:
        LOGGER.warning(
            "%s - No enabled bath presets after build; use_defaults=%s options_present=%s",
            DOMAIN,
            use_defaults,
            any(key.startswith("bathfill_preset_") for key in opts),
        )
    else:
        LOGGER.debug("%s preset build complete count=%d labels=%s", DOMAIN, len(presets), [p.get("label") for p in presets])
    return presets


async def _ensure_bath_profile_input_select(
    hass: HomeAssistant, entry: ConfigEntry, coordinator: RheemEziSETDataUpdateCoordinator
) -> str | None:
    """Create/update input_select for bath profiles and wire coordinator state."""
    # Ensure the input_select domain is loaded
    result = await async_setup_component(hass, "input_select", {})
    if not result:
        LOGGER.warning("%s - Failed to load input_select helper domain", DOMAIN)

    presets = _build_presets(entry)
    options = [p["label"] for p in presets]
    placeholder = "No presets configured"
    displayed_options = options or [placeholder]

    name = f"{entry.title} Bath Profile Input"
    entity_id = f"input_select.{slugify(name)}"
    existing_state = hass.states.get(entity_id)

    # Preserve the user's last choice when possible.
    prior_option = getattr(coordinator, "bath_profile_current", None)
    initial = None
    for candidate in (existing_state.state if existing_state else None, prior_option):
        if candidate in options:
            initial = candidate
            break
    if initial is None and options:
        initial = options[0]

    try:
        if existing_state:
            # Entity exists, just update options
            await hass.services.async_call(
                "input_select",
                "set_options",
                {"entity_id": entity_id, "options": displayed_options},
                blocking=True,
            )
            # Re-apply last selection if still valid, otherwise fall back to first option.
            if initial or displayed_options:
                await hass.services.async_call(
                    "input_select",
                    "select_option",
                    {"entity_id": entity_id, "option": initial or displayed_options[0]},
                    blocking=True,
                )
        else:
            # Entity doesn't exist - create via storage collection API
            # In HA 2026.1+, input_select uses storage collections
            # Try to access the storage collection through the integration's data
            try:
                # Access input_select integration storage collection
                # The collection is typically accessible via hass.data after component setup
                input_select_data = hass.data.get("input_select")
                
                # Check multiple possible locations for the storage collection
                storage_collection = None
                if input_select_data:
                    # Try different attribute names that storage collections might use
                    for attr in ("async_create_item", "create_item", "_storage_collection", "storage_collection"):
                        if hasattr(input_select_data, attr):
                            storage_collection = getattr(input_select_data, attr)
                            if callable(storage_collection):
                                break
                            elif hasattr(storage_collection, "async_create_item"):
                                storage_collection = storage_collection.async_create_item
                                break
                
                if storage_collection and callable(storage_collection):
                    # Create item via storage collection
                    item_id = slugify(name)
                    await storage_collection(
                        {
                            "id": item_id,
                            "name": name,
                            "options": displayed_options,
                            "initial": initial or (displayed_options[0] if displayed_options else None),
                            "icon": "mdi:bathtub",
                        }
                    )
                    LOGGER.debug("%s - Created input_select helper via storage collection: %s", DOMAIN, entity_id)
                else:
                    # Storage collection not accessible - use REST API as fallback
                    from homeassistant.helpers.aiohttp_client import async_get_clientsession
                    from aiohttp import ClientTimeout
                    
                    # Try helper creation via REST API
                    internal_url = str(hass.config.internal_url) if hass.config.internal_url else "http://127.0.0.1:8123"
                    url = f"{internal_url}/api/config/config_entries/helper/create"
                    
                    payload = {
                        "type": "input_select",
                        "name": name,
                        "options": displayed_options,
                        "initial": initial or (displayed_options[0] if displayed_options else None),
                        "icon": "mdi:bathtub",
                    }
                    
                    session = async_get_clientsession(hass)
                    async with session.post(url, json=payload, timeout=ClientTimeout(total=10)) as resp:
                        if resp.status == 200:
                            LOGGER.debug("%s - Created input_select helper via REST API: %s", DOMAIN, entity_id)
                        else:
                            error_text = await resp.text()
                            raise HomeAssistantError(f"Failed to create helper via REST API: HTTP {resp.status} - {error_text}")
            except (AttributeError, KeyError, ImportError, Exception, HomeAssistantError) as err:  # pylint: disable=broad-except
                # All creation methods failed - log clear instructions and continue without helper
                LOGGER.warning(
                    "%s - Cannot create input_select helper programmatically (%s). "
                    "Bath fill will work using coordinator state, but Bath Profile selection helper must be created manually. "
                    "Go to: Settings > Devices & Services > Helpers > Add Helper > Dropdown. "
                    "Name: '%s', Options: %s, Initial: %s, Icon: mdi:bathtub",
                    DOMAIN,
                    err,
                    name,
                    displayed_options,
                    initial or displayed_options[0] if displayed_options else None,
                )
                # Don't raise - allow integration to continue without helper
                # Coordinator state will be set below, so bath fill will work
                # Set coordinator state even though helper creation failed
                current_slot = next((p.get("slot") for p in presets if p.get("label") == initial), None) if initial and presets else None
                coordinator.bath_profile_options = presets  # type: ignore[attr-defined]
                coordinator.bath_profile_current = initial  # type: ignore[attr-defined]
                coordinator.bath_profile_current_slot = current_slot  # type: ignore[attr-defined]
                return None
        LOGGER.debug(
            "%s input_select helper ready entity_id=%s options=%s initial=%s existing=%s",
            DOMAIN,
            entity_id,
            displayed_options,
            initial,
            bool(existing_state),
        )
    except Exception as err:  # pylint: disable=broad-except
        LOGGER.error("%s - Failed to create/update input_select helper: %s", DOMAIN, err, exc_info=True)
        # Still set coordinator state even if helper creation failed
        # This allows the switch to work even if input_select is unavailable
        current_slot = next((p.get("slot") for p in presets if p.get("label") == initial), None) if initial and presets else None
        coordinator.bath_profile_options = presets  # type: ignore[attr-defined]
        coordinator.bath_profile_current = initial  # type: ignore[attr-defined]
        coordinator.bath_profile_current_slot = current_slot  # type: ignore[attr-defined]
        return None

    # Track coordinator state
    current_slot = next((p.get("slot") for p in presets if p.get("label") == initial), None) if initial else None
    coordinator.bath_profile_options = presets  # type: ignore[attr-defined]
    coordinator.bath_profile_current = initial  # type: ignore[attr-defined]
    coordinator.bath_profile_current_slot = current_slot  # type: ignore[attr-defined]

    # Listen for user selections to update coordinator state
    async def _handle_state_change(event):
        new_state = event.data.get("new_state")
        if not new_state:
            return
        option = new_state.state
        if option in options:
            coordinator.bath_profile_current = option  # type: ignore[attr-defined]
            coordinator.bath_profile_current_slot = next(  # type: ignore[attr-defined]
                (p.get("slot") for p in presets if p.get("label") == option),
                None,
            )

    unsub = async_track_state_change_event(hass, [entity_id], _handle_state_change)
    entry.async_on_unload(unsub)

    return entity_id


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

    # Initialize coordinator bath profile state to safe defaults before helper setup
    coordinator.bath_profile_options = []  # type: ignore[attr-defined]
    coordinator.bath_profile_current = None  # type: ignore[attr-defined]
    coordinator.bath_profile_current_slot = None  # type: ignore[attr-defined]

    # Create/refresh bath profile input_select helper (replaces old select entity)
    await _ensure_bath_profile_input_select(hass, entry, coordinator)

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

    # Best-effort cleanup of helper
    try:
        name = f"{entry.title} Bath Profile Input"
        entity_id = f"input_select.{slugify(name)}"
        await hass.services.async_call(
            "input_select",
            "remove",
            {"entity_id": entity_id},
            blocking=True,
        )
    except Exception:  # pylint: disable=broad-except
        LOGGER.debug("%s - best-effort input_select removal failed", DOMAIN)

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
