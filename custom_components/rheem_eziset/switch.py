"""Switch platform for rheem_eziset (bath fill presets + exit control)."""
from __future__ import annotations

import time

from homeassistant.components.switch import SwitchEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.restore_state import RestoreEntity

from .const import DOMAIN
from .coordinator import RheemEziSETDataUpdateCoordinator
from .entity import RheemEziSETEntity
from .util import is_one, to_int, to_float


async def async_setup_entry(hass, entry: ConfigEntry, async_add_entities):
    """Add switches for bath fill + exit control."""
    coordinator: RheemEziSETDataUpdateCoordinator = hass.data[DOMAIN][entry.entry_id]

    entities: list[SwitchEntity] = []
    # Global exit control
    entities.append(ExitBathFillSwitch(coordinator, entry))
    entities.append(AutoExitBathFillSwitch(coordinator, entry))
    entities.append(BathFillSwitch(coordinator, entry))

    async_add_entities(entities, True)


class ExitBathFillSwitch(RheemEziSETEntity, SwitchEntity):
    """Momentary switch to exit bath fill."""

    _attr_should_poll = False
    _attr_is_on = False
    _attr_name = "Exit Bath Fill"

    def __init__(self, coordinator: RheemEziSETDataUpdateCoordinator, entry: ConfigEntry) -> None:
        """Initialize the exit bath fill switch."""
        super().__init__(coordinator, entry)
        self._attr_unique_id = f"{entry.entry_id}-bathfill-exit"

    async def async_turn_on(self, **kwargs):
        """Trigger cancel."""
        api = self.coordinator.api
        now = time.monotonic()
        lockout_until = getattr(api, "_lockout_until_monotonic", None)
        if lockout_until and now < lockout_until:
            raise HomeAssistantError("Device lockout suspected; cannot send cancel right now.")

        await api.async_cancel_bath_fill(origin="user", entity_id=self.entity_id)
        # momentary
        self._attr_is_on = False
        self.async_write_ha_state()
        await self.coordinator.async_schedule_fast_refresh("bathfill_exit")

    async def async_turn_off(self, **kwargs):
        """No-op for momentary switch."""
        self._attr_is_on = False
        self.async_write_ha_state()


class AutoExitBathFillSwitch(RheemEziSETEntity, RestoreEntity, SwitchEntity):
    """Toggle for automatic bath fill exit on completion."""

    _attr_should_poll = False
    _attr_name = "Auto Exit Bath Fill"

    def __init__(self, coordinator: RheemEziSETDataUpdateCoordinator, entry: ConfigEntry) -> None:
        """Initialize the auto-exit switch."""
        super().__init__(coordinator, entry)
        self._attr_unique_id = f"{entry.entry_id}-bathfill-auto-exit"
        self._attr_is_on = False

    async def async_added_to_hass(self) -> None:
        """Restore previous state."""
        await super().async_added_to_hass()
        if (state := await self.async_get_last_state()) is not None:
            self._attr_is_on = state.state == "on"
            self.coordinator.api.set_auto_exit_enabled(self._attr_is_on)

    async def async_turn_on(self, **kwargs):
        """Enable auto-exit."""
        self._attr_is_on = True
        self.coordinator.api.set_auto_exit_enabled(True)
        self.async_write_ha_state()

    async def async_turn_off(self, **kwargs):
        """Disable auto-exit."""
        self._attr_is_on = False
        self.coordinator.api.set_auto_exit_enabled(False)
        self.async_write_ha_state()


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
        chosen = None
        for p in presets:
            if p.get("name") == current_name:
                chosen = p
                break
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
            raise HomeAssistantError("No valid bath profile selected.")
        await self.coordinator.api.async_start_bath_fill(temp, vol, origin="user", entity_id=self.entity_id)
        self._attr_is_on = True
        self._desired_on = True
        self._pending_until = time.monotonic() + 30
        self.async_write_ha_state()
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
