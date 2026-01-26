"""Switch platform for rheem_eziset (bath fill control)."""
from __future__ import annotations

import time

from homeassistant.components.switch import SwitchEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.exceptions import HomeAssistantError

from .const import DOMAIN, LOGGER
from .coordinator import RheemEziSETDataUpdateCoordinator
from .entity import RheemEziSETEntity
from .util import is_one, to_int, to_float


async def async_setup_entry(hass, entry: ConfigEntry, async_add_entities):
    """Add switches for bath fill control."""
    coordinator: RheemEziSETDataUpdateCoordinator = hass.data[DOMAIN][entry.entry_id]

    entities: list[SwitchEntity] = []
    entities.append(BathFillSwitch(coordinator, entry))

    async_add_entities(entities, True)


class BathFillSwitch(RheemEziSETEntity, SwitchEntity):
    """Single start/stop switch for bath fill (uses selected profile)."""

    _attr_should_poll = False

    def __init__(self, coordinator: RheemEziSETDataUpdateCoordinator, entry: ConfigEntry) -> None:
        """Initialize."""
        super().__init__(coordinator, entry)
        self._attr_unique_id = f"{entry.entry_id}-bathfill"
        self._attr_name = "Bath Fill"
        self._attr_is_on = False
        self._pending_until = 0.0
        self._desired_on = False

    def _current_profile(self) -> tuple[int | None, int | None]:
        """Return (temp, vol) for current profile selection."""
        try:
            presets = getattr(self.coordinator, "bath_profile_options", []) or []
            if not isinstance(presets, list):
                LOGGER.warning("%s - bath_profile_options is not a list: %s", DOMAIN, type(presets))
                presets = []
            current_name = getattr(self.coordinator, "bath_profile_current", None)
            current_slot = getattr(self.coordinator, "bath_profile_current_slot", None)
            LOGGER.debug(
                "%s current_profile presets=%d current_name=%s current_slot=%s",
                DOMAIN,
                len(presets),
                current_name,
                current_slot,
            )
            chosen = None
            if current_slot is not None:
                chosen = next((p for p in presets if isinstance(p, dict) and p.get("slot") == current_slot), None)
            if chosen is None and current_name:
                chosen = next((p for p in presets if isinstance(p, dict) and (p.get("label") == current_name or p.get("name") == current_name)), None)
            
            # Fallback: if coordinator selection is missing, read from state machine
            # Check base entity_id first, then check for suffixes (_2, _3, etc.) if needed
            if chosen is None and self.hass:
                try:
                    from homeassistant.components.input_select import DOMAIN as INPUT_SELECT_DOMAIN
                    entry_id_prefix = self.entry.entry_id[:8].lower()
                    base_entity_id = f"{INPUT_SELECT_DOMAIN}.rheem_{entry_id_prefix}_bath_profile"
                    
                    # Try base entity_id first
                    state = self.hass.states.get(base_entity_id)
                    if not state or not state.state:
                        # If base entity not found, try with suffixes (storage collection may have created duplicates)
                        for suffix in ["_2", "_3", "_4", "_5"]:
                            suffixed_entity_id = f"{base_entity_id}{suffix}"
                            state = self.hass.states.get(suffixed_entity_id)
                            if state and state.state:
                                LOGGER.debug("%s - Found entity with suffix: %s", DOMAIN, suffixed_entity_id)
                                break
                    
                    if state and state.state:
                        selected_option = state.state
                        chosen = next((p for p in presets if isinstance(p, dict) and (p.get("label") == selected_option or p.get("name") == selected_option)), None)
                        if chosen:
                            LOGGER.debug("%s - Resolved profile from state machine: %s", DOMAIN, selected_option)
                except Exception as fallback_err:  # pylint: disable=broad-except
                    LOGGER.debug("%s - Fallback state read failed: %s", DOMAIN, fallback_err)
            
            if chosen is None and presets:
                chosen = presets[0] if isinstance(presets[0], dict) else None
            if not chosen or not isinstance(chosen, dict):
                LOGGER.warning("%s - No valid bath profile resolved; presets=%s", DOMAIN, presets)
                return None, None
            return to_int(chosen.get("temp")), to_int(chosen.get("vol"))
        except Exception as err:  # pylint: disable=broad-except
            LOGGER.error("%s - Error in _current_profile: %s", DOMAIN, err, exc_info=True)
            return None, None

    def _is_active(self) -> bool:
        data = self.coordinator.data or {}
        api = self.coordinator.api
        return bool(api._bathfill_engaged(data))

    async def async_turn_on(self, **kwargs):
        """Start bath fill using selected profile."""
        temp, vol = self._current_profile()
        if temp is None or vol is None:
            raise HomeAssistantError(
                "No valid bath profile selected. Configure bath fill presets in Options and ensure the Bath Profile input_select is created."
            )
        await self.coordinator.api.async_start_bath_fill(temp, vol, origin="user", entity_id=self.entity_id)
        self._attr_is_on = True
        self._desired_on = True
        self._pending_until = time.monotonic() + 30
        self.async_write_ha_state()
        # Fast temperature confirmation: poll once immediately (honors rate limits inside api)
        try:
            data = await self.coordinator.api.async_get_info_only()
            if data:
                self.coordinator.async_set_updated_data(data)
        except Exception as err:  # pylint: disable=broad-except
            LOGGER.debug("%s bathfill_start immediate poll failed: %s", DOMAIN, err)
        await self.coordinator.async_schedule_fast_refresh("bathfill_start")

    async def async_turn_off(self, **kwargs):
        """Cancel bath fill."""
        await self.coordinator.api.async_cancel_bath_fill(origin="user", entity_id=self.entity_id)
        self._attr_is_on = False
        self._desired_on = False
        self._pending_until = time.monotonic() + 10
        self.async_write_ha_state()
        await self.coordinator.async_schedule_fast_refresh("bathfill_cancel")

    def _handle_coordinator_update(self) -> None:
        engaged = self._is_active()
        if engaged:
            if self._desired_on:
                self._attr_is_on = True
            else:
                # user requested off; keep off even if device still reports engaged
                self._attr_is_on = False
        else:
            now = time.monotonic()
            if now <= self._pending_until:
                # keep optimistic during short confirmation window
                pass
            else:
                self._attr_is_on = False
        super()._handle_coordinator_update()
