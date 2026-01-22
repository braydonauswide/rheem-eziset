"""Select platform for bath fill profile."""
from __future__ import annotations

from homeassistant.components.select import SelectEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.helpers.restore_state import RestoreEntity

from .const import DEFAULT_BATHFILL_PRESETS, DOMAIN, PRESET_SLOTS
from .coordinator import RheemEziSETDataUpdateCoordinator
from .entity import RheemEziSETEntity
from .util import to_int


async def async_setup_entry(hass, entry: ConfigEntry, async_add_entities):
    """Add the bath profile selector."""
    coordinator: RheemEziSETDataUpdateCoordinator = hass.data[DOMAIN][entry.entry_id]
    async_add_entities([BathProfileSelect(coordinator, entry)], True)


class BathProfileSelect(RheemEziSETEntity, RestoreEntity, SelectEntity):
    """Select the active bath fill profile."""

    _attr_should_poll = False
    _attr_has_entity_name = True
    _attr_name = "Bath Profile"

    def __init__(self, coordinator: RheemEziSETDataUpdateCoordinator, entry: ConfigEntry) -> None:
        """Initialize the selector."""
        super().__init__(coordinator, entry)
        self._attr_unique_id = f"{entry.entry_id}-bath_profile"
        self._presets = self._build_presets(entry)
        self._attr_options = [p["name"] for p in self._presets]
        self._attr_current_option = self._attr_options[0] if self._attr_options else None
        # Persist selection on coordinator for other entities (bath fill switch)
        coordinator.bath_profile_options = self._presets  # type: ignore[attr-defined]
        coordinator.bath_profile_current = self._attr_current_option  # type: ignore[attr-defined]

    async def async_added_to_hass(self) -> None:
        """Restore last state."""
        await super().async_added_to_hass()
        if (state := await self.async_get_last_state()) and state.state in self._attr_options:
            self._attr_current_option = state.state
            self.coordinator.bath_profile_current = state.state  # type: ignore[attr-defined]

    def _build_presets(self, entry: ConfigEntry) -> list[dict]:
        """Build enabled preset list from options (fallback to defaults)."""
        presets: list[dict] = []
        opts = entry.options
        use_defaults = not any(key.startswith("bathfill_preset_") for key in opts)

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
            presets.append({"name": str(name), "temp": int(temp), "vol": int(vol), "slot": idx})
        return presets

    async def async_select_option(self, option: str) -> None:
        """Handle selection."""
        if option not in self._attr_options:
            raise ValueError(f"Invalid option {option}")
        self._attr_current_option = option
        self.coordinator.bath_profile_current = option  # type: ignore[attr-defined]
        self.async_write_ha_state()

    @property
    def current_option(self):
        """Return current option."""
        return self._attr_current_option
