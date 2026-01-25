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
        presets = getattr(self.coordinator, "bath_profile_options", []) or []
        current_name = getattr(self.coordinator, "bath_profile_current", None)
        current_slot = getattr(self.coordinator, "bath_profile_current_slot", None)
        chosen = None
        if current_slot is not None:
            chosen = next((p for p in presets if p.get("slot") == current_slot), None)
        if chosen is None and current_name:
            chosen = next((p for p in presets if p.get("label") == current_name or p.get("name") == current_name), None)
        if chosen is None and presets:
            chosen = presets[0]
        if not chosen:
            return None, None
        return to_int(chosen.get("temp")), to_int(chosen.get("vol"))

    def _is_active(self) -> bool:
        data = self.coordinator.data or {}
        api = self.coordinator.api
        return bool(api._bathfill_engaged(data))

    async def async_turn_on(self, **kwargs):
        """Start bath fill using selected profile."""
        temp, vol = self._current_profile()
        if temp is None or vol is None:
            raise HomeAssistantError("No valid bath profile selected. Configure bath fill presets in Options.")
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
