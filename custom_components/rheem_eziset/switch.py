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
            
            if not presets:
                LOGGER.warning("%s - No presets available", DOMAIN)
                return None, None
            
            chosen = None
            selected_option = None
            
            # PRIMARY: Read directly from state machine (most authoritative source)
            if self.hass:
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
                        selected_option = state.state.strip()  # Strip whitespace
                        LOGGER.info("%s - Reading profile from state machine: '%s'", DOMAIN, selected_option)
                except Exception as state_read_err:  # pylint: disable=broad-except
                    LOGGER.debug("%s - State machine read failed: %s", DOMAIN, state_read_err)
            
            # Match selected option to preset (case-insensitive, flexible matching)
            if selected_option:
                LOGGER.info("%s - Attempting to match selected option: '%s' against %d presets", 
                          DOMAIN, selected_option, len(presets))
                LOGGER.debug("%s - Available preset labels: %s", DOMAIN, [p.get("label") for p in presets])
                LOGGER.debug("%s - Available preset names: %s", DOMAIN, [p.get("name") for p in presets])
                
                # Try exact match first (label or name)
                chosen = next(
                    (p for p in presets if isinstance(p, dict) and (
                        p.get("label", "").strip() == selected_option or
                        p.get("name", "").strip() == selected_option
                    )),
                    None
                )
                if chosen:
                    LOGGER.info("%s - Exact match found: '%s' -> slot=%d, temp=%d, vol=%d (label: '%s')", 
                              DOMAIN, selected_option, chosen.get("slot"), chosen.get("temp"), chosen.get("vol"), chosen.get("label"))
                
                # Try case-insensitive match if exact match failed
                if not chosen:
                    selected_lower = selected_option.lower()
                    chosen = next(
                        (p for p in presets if isinstance(p, dict) and (
                            p.get("label", "").strip().lower() == selected_lower or
                            p.get("name", "").strip().lower() == selected_lower
                        )),
                        None
                    )
                    if chosen:
                        LOGGER.info("%s - Case-insensitive match found: '%s' -> slot=%d, temp=%d, vol=%d (label: '%s')", 
                                  DOMAIN, selected_option, chosen.get("slot"), chosen.get("temp"), chosen.get("vol"), chosen.get("label"))
                
                # OPTIMIZATION: Try partial/fuzzy match if still no match (handles cases where user sees different format)
                # Match by name prefix (e.g., "test" matches "test (48C, 10L)")
                # IMPORTANT: Prefer longest/best match to avoid matching wrong profile when multiple profiles have similar names
                # However, if multiple profiles match, we need to be more careful - prefer exact name matches
                if not chosen:
                    selected_lower = selected_option.lower()
                    # Try matching name prefix (before any parentheses or special chars)
                    selected_name_part = selected_lower.split('(')[0].split('[')[0].strip()
                    if selected_name_part:
                        # Find all potential matches and prefer the best one (longest name match or exact prefix match)
                        potential_matches = []
                        for p in presets:
                            if not isinstance(p, dict):
                                continue
                            preset_name = p.get("name", "").strip().lower()
                            preset_label = p.get("label", "").strip().lower()
                            
                            # Check if name or label starts with selected name part
                            name_match = preset_name.startswith(selected_name_part) if preset_name else False
                            label_match = preset_label.startswith(selected_name_part) if preset_label else False
                            
                            if name_match or label_match:
                                # Calculate match quality: prefer exact name match, then longest match
                                match_quality = 0
                                if preset_name == selected_name_part:
                                    match_quality = 100  # Exact name match (highest priority)
                                elif preset_name.startswith(selected_name_part):
                                    # For prefix matches, prefer longer names (more specific)
                                    # But also check if the selected option contains temp/vol info that matches
                                    match_quality = len(preset_name)  # Longer names are better matches
                                    # Bonus: if selected option contains temp/vol that matches preset, boost quality
                                    if selected_lower and preset_label:
                                        preset_temp = p.get("temp")
                                        preset_vol = p.get("vol")
                                        # Check if selected option mentions temp/vol that matches
                                        if preset_temp and f"{preset_temp}c" in selected_lower:
                                            match_quality += 10
                                        if preset_vol and f"{preset_vol}l" in selected_lower:
                                            match_quality += 10
                                elif label_match:
                                    match_quality = 50  # Label match is less preferred
                                
                                potential_matches.append((match_quality, p))
                        
                        # Sort by match quality (highest first) and take the best match
                        if potential_matches:
                            potential_matches.sort(key=lambda x: x[0], reverse=True)
                            chosen = potential_matches[0][1]
                            LOGGER.info("%s - Partial name match found: '%s' -> slot=%d, temp=%d, vol=%d (matched name: '%s', label: '%s', quality=%d, total_matches=%d)", 
                                      DOMAIN, selected_option, chosen.get("slot"), chosen.get("temp"), chosen.get("vol"), 
                                      chosen.get("name"), chosen.get("label"), potential_matches[0][0], len(potential_matches))
                            # If multiple matches, log them for debugging
                            if len(potential_matches) > 1:
                                LOGGER.warning("%s - Multiple partial matches found for '%s': %s. Selected best match (quality=%d).", 
                                             DOMAIN, selected_option, 
                                             [f"{p[1].get('label')} (quality={p[0]})" for p in potential_matches[:3]], 
                                             potential_matches[0][0])
                
                if not chosen:
                    LOGGER.warning("%s - No match found for selected option: '%s'. Available presets: %s", 
                                 DOMAIN, selected_option, [f"{p.get('label')} (name: {p.get('name')})" for p in presets])
            
            # FALLBACK 1: Use coordinator state (if state machine read failed)
            if not chosen:
                current_name = getattr(self.coordinator, "bath_profile_current", None)
                current_slot = getattr(self.coordinator, "bath_profile_current_slot", None)
                LOGGER.debug("%s - Falling back to coordinator state: name='%s', slot=%s", DOMAIN, current_name, current_slot)
                
                if current_slot is not None:
                    chosen = next((p for p in presets if isinstance(p, dict) and p.get("slot") == current_slot), None)
                if not chosen and current_name:
                    current_name_clean = current_name.strip()
                    chosen = next(
                        (p for p in presets if isinstance(p, dict) and (
                            p.get("label", "").strip() == current_name_clean or
                            p.get("name", "").strip() == current_name_clean
                        )),
                        None
                    )
            
            # FALLBACK 2: Use first preset if nothing matched
            # NOTE: This fallback should rarely be needed if matching logic is correct
            # If it's triggered, it means the selected option doesn't match any preset
            # which could indicate a mismatch between input_select options and presets
            if not chosen and presets:
                chosen = presets[0] if isinstance(presets[0], dict) else None
                LOGGER.warning(
                    "%s - No profile matched for selected option '%s', using first preset as fallback: %s (slot=%d, temp=%d, vol=%d). "
                    "This may indicate a mismatch between input_select options and presets. Available presets: %s",
                    DOMAIN, selected_option, chosen.get("label") if chosen else "None",
                    chosen.get("slot") if chosen else None, chosen.get("temp") if chosen else None, chosen.get("vol") if chosen else None,
                    [f"{p.get('label')} (name: {p.get('name')})" for p in presets]
                )
            
            if not chosen or not isinstance(chosen, dict):
                LOGGER.error("%s - No valid bath profile resolved; presets=%s, selected_option='%s'", 
                           DOMAIN, [p.get("label") for p in presets], selected_option)
                return None, None
            
            temp = to_int(chosen.get("temp"))
            vol = to_int(chosen.get("vol"))
            LOGGER.info("%s - Selected profile: '%s' (slot=%d) -> temp=%s, vol=%s", 
                      DOMAIN, chosen.get("label"), chosen.get("slot"), temp, vol)
            return temp, vol
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
        # OPTIMIZATION: Fast refresh is already scheduled via post_write_callback, so immediate poll is redundant
        # Fast refresh will start immediately and provide updates, so we can skip the extra poll here
        # This reduces unnecessary requests and improves efficiency
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
