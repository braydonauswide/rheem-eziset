"""
Custom integration to integrate rheem_eziset with Home Assistant.

For more details about this integration, please refer to
https://github.com/braydonauswide/rheem-eziset
"""

from __future__ import annotations

import asyncio

from homeassistant.config_entries import ConfigEntry, ConfigEntryState
from homeassistant.const import CONF_HOST, CONF_ICON, CONF_ID, CONF_NAME, CONF_OPTIONS
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import device_registry as dr, entity_registry as er
from homeassistant.helpers.entity_component import DATA_INSTANCES
from homeassistant.setup import async_setup_component
from homeassistant.helpers.event import async_track_state_change_event
from homeassistant.util import slugify

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


def _resolve_input_select_entity_id_after_add(
    hass: HomeAssistant,
    domain: str,
    object_id: str,
    base_entity_id: str,
) -> str | None:
    """Resolve the actual entity_id HA assigned (may have suffix e.g. _2). Returns None if not found."""
    prefix = f"{domain}.{object_id}"
    # Prefer entity registry: find entry with our object_id base
    ent_reg = er.async_get(hass)
    for ent_id in ent_reg.entities:
        if not ent_id.startswith(prefix):
            continue
        # entity_id is prefix or prefix + "_N"
        if hass.states.get(ent_id):
            return ent_id
    # Fallback: scan states for domain.object_id*
    for state in hass.states.async_all(domain):
        eid = state.entity_id
        if eid.startswith(prefix) and (eid == prefix or eid[len(prefix) :].startswith("_")):
            return eid
    return None


def _get_input_select_storage_collection(hass: HomeAssistant, domain: str):
    """Get the storage collection for input_select domain."""
    if domain not in hass.data:
        return None
    
    domain_data = hass.data[domain]
    
    # Try different possible locations for the storage collection
    # It might be the domain_data itself, or under a key
    if hasattr(domain_data, 'async_create_item'):
        return domain_data
    
    if isinstance(domain_data, dict):
        # Try common keys where storage collection might be stored
        for key in ["collection", "items", "storage", "async_create_item"]:
            candidate = domain_data.get(key)
            if candidate and hasattr(candidate, 'async_create_item'):
                return candidate
    
    return None


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
        return f"{safe_name} ({temp_part}, {vol_part})"

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
    """Create/update input_select entity for bath profiles using EntityComponent."""
    from homeassistant.components.input_select import InputSelect, DOMAIN as INPUT_SELECT_DOMAIN
    
    # Ensure input_select domain is loaded
    await async_setup_component(hass, INPUT_SELECT_DOMAIN, {})
    
    # Build presets and options
    LOGGER.info("%s - Building presets for entry_id=%s, options=%s", DOMAIN, entry.entry_id, list(entry.options.keys()))
    presets = _build_presets(entry)
    LOGGER.info("%s - _build_presets returned %d presets: %s", DOMAIN, len(presets), [p.get("label", p.get("name")) for p in presets])
    options = [p.get("label", p.get("name", "")) for p in presets if p.get("label")]
    if not options and presets:
        options = [p.get("name", "") for p in presets if p.get("name")]
    
    LOGGER.info("%s - Built %d options from presets: %s", DOMAIN, len(options), options)
    
    if not options:
        LOGGER.error("%s - No presets available for input_select entity (presets=%s, entry.options=%s)", DOMAIN, presets, entry.options)
        # Still set coordinator state
        coordinator.bath_profile_options = presets  # type: ignore[attr-defined]
        coordinator.bath_profile_current = None  # type: ignore[attr-defined]
        coordinator.bath_profile_current_slot = None  # type: ignore[attr-defined]
        return None
    
    # Use stable entity_id format based on entry_id (not entry.title)
    entry_id_prefix = entry.entry_id[:8].lower()
    object_id = f"rheem_{entry_id_prefix}_bath_profile"
    entity_id = f"{INPUT_SELECT_DOMAIN}.{object_id}"
    name = "Bath Profile"
    
    # Get the input_select EntityComponent (with retry for timing)
    component = None
    for attempt in range(3):
        component = hass.data.get(DATA_INSTANCES, {}).get(INPUT_SELECT_DOMAIN)
        if component:
            LOGGER.debug("%s - Found input_select EntityComponent on attempt %d", DOMAIN, attempt + 1)
            break
        if attempt < 2:
            await asyncio.sleep(0.1)
    
    if not component:
        LOGGER.error("%s - input_select EntityComponent not found after setup (hass.data keys: %s)", DOMAIN, list(hass.data.get(DATA_INSTANCES, {}).keys()))
        # Set coordinator state even if entity creation failed
        coordinator.bath_profile_options = presets  # type: ignore[attr-defined]
        coordinator.bath_profile_current = options[0] if options else None  # type: ignore[attr-defined]
        current_slot = next((p.get("slot") for p in presets if p.get("label") == options[0] or p.get("name") == options[0]), None) if options and presets else None
        coordinator.bath_profile_current_slot = current_slot  # type: ignore[attr-defined]
        return None
    
    # Check for existing entity (including any with _2, _3 suffixes from previous failed removals)
    existing_entity = component.get_entity(entity_id)
    existing_state = hass.states.get(entity_id)
    
    # Also check for entities with suffixes (storage collection may have created duplicates)
    entity_with_suffix = None
    for suffix in ["", "_2", "_3", "_4", "_5"]:
        if suffix:
            suffixed_id = f"{entity_id}{suffix}"
            suffixed_entity = component.get_entity(suffixed_id)
            if suffixed_entity:
                entity_with_suffix = (suffixed_id, suffixed_entity)
                LOGGER.info("%s - Found entity with suffix: %s", DOMAIN, suffixed_id)
                break
    
    # If entity exists (with or without suffix), remove it/them before creating
    entities_to_remove = []
    if existing_entity:
        entities_to_remove.append((entity_id, existing_entity))
    if entity_with_suffix:
        entities_to_remove.append(entity_with_suffix)
    
    # Also check storage collection for any items with matching base name
    storage_collection = _get_input_select_storage_collection(hass, INPUT_SELECT_DOMAIN)
    if storage_collection:
        try:
            # Try to get all items from storage to find any with matching base name
            if hasattr(storage_collection, 'async_items'):
                items = await storage_collection.async_items()
                for item in items:
                    item_id = item.get("id", "") if isinstance(item, dict) else getattr(item, "id", "")
                    # Check if this item matches our base object_id (with or without suffix)
                    if item_id and (item_id == object_id or item_id.startswith(f"{object_id}_")):
                        # Try to find the corresponding entity
                        potential_entity_id = f"{INPUT_SELECT_DOMAIN}.{item_id}"
                        potential_entity = component.get_entity(potential_entity_id)
                        if potential_entity and (potential_entity_id, potential_entity) not in entities_to_remove:
                            entities_to_remove.append((potential_entity_id, potential_entity))
                            LOGGER.info("%s - Found entity in storage to remove: %s", DOMAIN, potential_entity_id)
        except Exception as list_err:
            LOGGER.debug("%s - Could not list storage items: %s", DOMAIN, list_err)
    
    if entities_to_remove:
        LOGGER.info("%s - Found %d existing entity/entities, removing before recreation: %s", 
                   DOMAIN, len(entities_to_remove), [eid for eid, _ in entities_to_remove])
        
        # Remove via storage collection first (more reliable for persistence)
        if storage_collection and hasattr(storage_collection, 'async_delete_item'):
            for eid, ent in entities_to_remove:
                try:
                    # Extract object_id from entity_id
                    obj_id = eid.replace(f"{INPUT_SELECT_DOMAIN}.", "")
                    # Also try with just the base object_id and variations
                    for obj_id_variant in [obj_id, object_id, f"{object_id}_2", f"{object_id}_3", f"{object_id}_4", f"{object_id}_5"]:
                        try:
                            await storage_collection.async_delete_item(obj_id_variant)
                            LOGGER.info("%s - Removed from storage: %s", DOMAIN, obj_id_variant)
                        except Exception:
                            pass  # Item might not exist, continue
                except Exception as storage_remove_err:
                    LOGGER.debug("%s - Storage removal attempt failed: %s", DOMAIN, storage_remove_err)
        
        # Also remove entities via entity method
        for eid, ent in entities_to_remove:
            try:
                await ent.async_remove()
                LOGGER.info("%s - Removed entity via async_remove: %s", DOMAIN, eid)
            except Exception as remove_err:
                LOGGER.warning("%s - Failed to remove entity %s: %s", DOMAIN, eid, remove_err)
        
        # Wait for removals to complete
        await asyncio.sleep(1.0)  # Increased wait time for storage cleanup
        
        # Re-check to ensure entities are gone
        existing_entity = component.get_entity(entity_id)
        existing_state = hass.states.get(entity_id)
        if existing_entity or existing_state:
            LOGGER.warning("%s - Entity still exists after removal (entity=%s, state=%s), will try to recreate anyway", 
                         DOMAIN, existing_entity is not None, existing_state is not None)
        
        # Clear flags so we go through creation path
        existing_entity = None
        existing_state = None
    
    # Create new entity if it doesn't exist (or was removed for recreation)
    if not existing_entity:
        # Create new entity - PRIMARY: Use storage collection API (matches HA's internal approach)
        # FALLBACK: Use InputSelect.from_yaml() if storage collection not available
        
        # Prepare item data matching YAML structure
        item_data = {
            "id": object_id,  # Storage collection uses "id", not CONF_ID
            "name": name,
            "options": options,
            "initial": options[0] if options else None,
            "icon": "mdi:bathtub",
        }
        
        entity_created = False
        
        # PRIMARY APPROACH: Storage Collection API
        try:
            LOGGER.info("%s - Attempting to create input_select via storage collection API: %s", DOMAIN, entity_id)
            storage_collection = _get_input_select_storage_collection(hass, INPUT_SELECT_DOMAIN)
            
            if storage_collection:
                LOGGER.info("%s - Found storage collection, creating entity with data: %s", DOMAIN, item_data)
                try:
                    # Final cleanup: delete any remaining items with matching base name
                    try:
                        if hasattr(storage_collection, 'async_items'):
                            items = await storage_collection.async_items()
                            for item in items:
                                item_id = item.get("id", "") if isinstance(item, dict) else getattr(item, "id", "")
                                if item_id and (item_id == object_id or item_id.startswith(f"{object_id}_")):
                                    try:
                                        await storage_collection.async_delete_item(item_id)
                                        LOGGER.info("%s - Final cleanup: removed storage item: %s", DOMAIN, item_id)
                                        await asyncio.sleep(0.2)  # Brief pause between deletions
                                    except Exception as del_err:
                                        LOGGER.debug("%s - Could not delete storage item %s: %s", DOMAIN, item_id, del_err)
                    except Exception:
                        pass  # If we can't list/delete, continue with creation
                    
                    await storage_collection.async_create_item(item_data)
                    LOGGER.info("%s - async_create_item call completed for: %s", DOMAIN, entity_id)
                except Exception as create_err:
                    LOGGER.error("%s - async_create_item failed: %s", DOMAIN, create_err, exc_info=True)
                    raise
                
                # Verify entity was created and is accessible
                verification_success = False
                for verify_attempt in range(10):
                    await asyncio.sleep(0.3)
                    verify_state = hass.states.get(entity_id)
                    if verify_state:
                        LOGGER.info(
                            "%s - Verified entity exists in state machine: %s (attempt %d, state=%s)",
                            DOMAIN,
                            entity_id,
                            verify_attempt + 1,
                            verify_state.state,
                        )
                        verification_success = True
                        entity_created = True
                        break
                    elif verify_attempt == 9:
                        LOGGER.warning(
                            "%s - Entity created via storage collection but not yet in state machine after 10 attempts: %s",
                            DOMAIN,
                            entity_id,
                        )
                
                # If verification failed, mark as not created so we try fallback
                if not verification_success:
                    LOGGER.warning("%s - Storage collection creation succeeded but verification failed, will try fallback", DOMAIN)
                    entity_created = False
            else:
                LOGGER.warning("%s - Storage collection not found, will try fallback method", DOMAIN)
        except Exception as storage_err:  # pylint: disable=broad-except
            LOGGER.warning(
                "%s - Storage collection creation failed, trying fallback: %s",
                DOMAIN,
                storage_err,
            )
            entity_created = False
        
        # FALLBACK APPROACH: InputSelect.from_yaml() if storage collection failed
        if not entity_created:
            try:
                LOGGER.info("%s - Attempting fallback: InputSelect.from_yaml()", DOMAIN)
                # Config should match YAML structure: just name, options, initial, icon (no ID)
                config = {
                    CONF_NAME: name,
                    CONF_OPTIONS: options,
                    "initial": options[0] if options else None,
                    CONF_ICON: "mdi:bathtub",
                }
                
                input_select_entity = None
                created_entity_id = None
                
                # Try multiple InputSelect.from_yaml() approaches
                try:
                    LOGGER.debug("%s - Attempting InputSelect.from_yaml(object_id, config)", DOMAIN)
                    input_select_entity = InputSelect.from_yaml(object_id, config)
                    if hasattr(input_select_entity, 'entity_id'):
                        created_entity_id = input_select_entity.entity_id
                        LOGGER.debug("%s - InputSelect.from_yaml(object_id, config) succeeded: %s", DOMAIN, created_entity_id)
                except TypeError:
                    try:
                        LOGGER.debug("%s - Attempting InputSelect.from_yaml(config) with ID in config", DOMAIN)
                        config_with_id = {CONF_ID: object_id, **config}
                        input_select_entity = InputSelect.from_yaml(config_with_id)
                        if hasattr(input_select_entity, 'entity_id'):
                            created_entity_id = input_select_entity.entity_id
                            LOGGER.debug("%s - InputSelect.from_yaml(config_with_id) succeeded: %s", DOMAIN, created_entity_id)
                    except Exception as e2:
                        try:
                            LOGGER.debug("%s - Attempting InputSelect.from_yaml(config) without ID", DOMAIN)
                            input_select_entity = InputSelect.from_yaml(config)
                            if hasattr(input_select_entity, 'entity_id'):
                                created_entity_id = input_select_entity.entity_id
                                LOGGER.debug("%s - InputSelect.from_yaml(config) succeeded: %s", DOMAIN, created_entity_id)
                            else:
                                input_select_entity.entity_id = entity_id
                                created_entity_id = entity_id
                                LOGGER.debug("%s - Set entity_id manually: %s", DOMAIN, entity_id)
                        except Exception as e3:
                            LOGGER.error("%s - All InputSelect.from_yaml approaches failed: %s, %s", DOMAIN, e2, e3, exc_info=True)
                            raise
                
                if not input_select_entity:
                    raise RuntimeError("Failed to create InputSelect entity with any approach")
                
                # Use the created entity_id or fall back to expected
                if created_entity_id and created_entity_id != entity_id:
                    LOGGER.warning("%s - Entity ID mismatch: created=%s, expected=%s, using created", DOMAIN, created_entity_id, entity_id)
                    entity_id = created_entity_id
                
                # Ensure the entity has hass reference
                if not hasattr(input_select_entity, 'hass') or input_select_entity.hass is None:
                    input_select_entity.hass = hass
                
                # Add entity to component
                LOGGER.debug("%s - Adding entity to component: %s", DOMAIN, entity_id)
                await component.async_add_entities([input_select_entity], update_before_add=True)
                LOGGER.info("%s - Created input_select entity via InputSelect.from_yaml: %s", DOMAIN, entity_id)
                
                # Verify entity was actually created and is accessible (HA may assign suffixed entity_id e.g. _2)
                for verify_attempt in range(10):
                    await asyncio.sleep(0.3)
                    verify_state = hass.states.get(entity_id)
                    if not verify_state and verify_attempt >= 2:
                        # Resolve actual entity_id in case HA assigned a suffix
                        resolved_id = _resolve_input_select_entity_id_after_add(
                            hass, INPUT_SELECT_DOMAIN, object_id, entity_id
                        )
                        if resolved_id and resolved_id != entity_id:
                            entity_id = resolved_id
                            verify_state = hass.states.get(entity_id)
                            LOGGER.info(
                                "%s - Resolved actual entity_id after add: %s",
                                DOMAIN,
                                entity_id,
                            )
                    if verify_state:
                        LOGGER.info(
                            "%s - Verified entity exists in state machine: %s (attempt %d, state=%s)",
                            DOMAIN,
                            entity_id,
                            verify_attempt + 1,
                            verify_state.state,
                        )
                        entity_created = True
                        break
                    elif verify_attempt == 9:
                        resolved_id = _resolve_input_select_entity_id_after_add(
                            hass, INPUT_SELECT_DOMAIN, object_id, entity_id
                        )
                        if resolved_id:
                            entity_id = resolved_id
                            entity_created = True
                            LOGGER.info(
                                "%s - Resolved actual entity_id after 10 attempts: %s",
                                DOMAIN,
                                entity_id,
                            )
                            break
                        LOGGER.error("%s - Entity created but not found in state machine after 10 attempts: %s", DOMAIN, entity_id)
                        all_input_selects = [s for s in hass.states.async_all() if s.entity_id.startswith("input_select.")]
                        LOGGER.warning("%s - All input_select entities found: %s", DOMAIN, [s.entity_id for s in all_input_selects])
                        component_entity = component.get_entity(entity_id)
                        if component_entity:
                            LOGGER.warning("%s - Entity exists in component but not in state machine", DOMAIN)
                        else:
                            LOGGER.error("%s - Entity not found in component either", DOMAIN)
            except Exception as fallback_err:  # pylint: disable=broad-except
                LOGGER.error(
                    "%s - Fallback InputSelect.from_yaml() also failed: %s",
                    DOMAIN,
                    fallback_err,
                    exc_info=True,
                )
                # Set coordinator state even if entity creation failed
                coordinator.bath_profile_options = presets  # type: ignore[attr-defined]
                coordinator.bath_profile_current = options[0] if options else None  # type: ignore[attr-defined]
                current_slot = next((p.get("slot") for p in presets if p.get("label") == options[0] or p.get("name") == options[0]), None) if options and presets else None
                coordinator.bath_profile_current_slot = current_slot  # type: ignore[attr-defined]
                return None
            # Set coordinator state even if entity creation failed
            coordinator.bath_profile_options = presets  # type: ignore[attr-defined]
            coordinator.bath_profile_current = options[0] if options else None  # type: ignore[attr-defined]
            current_slot = next((p.get("slot") for p in presets if p.get("label") == options[0] or p.get("name") == options[0]), None) if options and presets else None
            coordinator.bath_profile_current_slot = current_slot  # type: ignore[attr-defined]
            return None
    
    # Initialize coordinator state
    initial = options[0] if options else None
    # Preserve existing selection if entity already exists
    if existing_state and existing_state.state in options:
        initial = existing_state.state
    current_slot = next((p.get("slot") for p in presets if p.get("label") == initial or p.get("name") == initial), None) if initial and presets else None
    coordinator.bath_profile_options = presets  # type: ignore[attr-defined]
    coordinator.bath_profile_current = initial  # type: ignore[attr-defined]
    coordinator.bath_profile_current_slot = current_slot  # type: ignore[attr-defined]
    
    # Listen for state changes to update coordinator
    # Note: options list is captured in closure; if presets change, entity is recreated/updated
    async def _handle_state_change(event):
        new_state = event.data.get("new_state")
        if not new_state:
            return
        option = new_state.state
        if not option:
            return
        option = option.strip()  # Strip whitespace for matching
        # Get current presets (they may have changed)
        current_presets = getattr(coordinator, "bath_profile_options", []) or []
        
        LOGGER.info("%s - State change detected: option='%s', available presets: %s", 
                   DOMAIN, option, [p.get("label") for p in current_presets])
        
        matched_preset = None
        
        # Try exact match first (label or name)
        matched_preset = next(
            (p for p in current_presets if isinstance(p, dict) and (
                p.get("label", "").strip() == option or
                p.get("name", "").strip() == option
            )),
            None
        )
        if matched_preset:
            LOGGER.info("%s - State change: exact match found for '%s' -> slot=%d", DOMAIN, option, matched_preset.get("slot"))
        
        # Try case-insensitive match if exact match failed
        if not matched_preset:
            option_lower = option.lower()
            matched_preset = next(
                (p for p in current_presets if isinstance(p, dict) and (
                    p.get("label", "").strip().lower() == option_lower or
                    p.get("name", "").strip().lower() == option_lower
                )),
                None
            )
            if matched_preset:
                LOGGER.info("%s - State change: case-insensitive match found for '%s' -> slot=%d", DOMAIN, option, matched_preset.get("slot"))
        
        # OPTIMIZATION: Try partial/fuzzy match with best match selection (same logic as switch.py)
        if not matched_preset:
            option_lower = option.lower()
            selected_name_part = option_lower.split('(')[0].split('[')[0].strip()
            if selected_name_part:
                potential_matches = []
                for p in current_presets:
                    if not isinstance(p, dict):
                        continue
                    preset_name = p.get("name", "").strip().lower()
                    preset_label = p.get("label", "").strip().lower()
                    
                    name_match = preset_name.startswith(selected_name_part) if preset_name else False
                    label_match = preset_label.startswith(selected_name_part) if preset_label else False
                    
                    if name_match or label_match:
                        match_quality = 0
                        if preset_name == selected_name_part:
                            match_quality = 100  # Exact name match
                        elif preset_name.startswith(selected_name_part):
                            match_quality = len(preset_name)  # Longer names are better matches
                        elif label_match:
                            match_quality = 50  # Label match is less preferred
                        potential_matches.append((match_quality, p))
                
                if potential_matches:
                    potential_matches.sort(key=lambda x: x[0], reverse=True)
                    matched_preset = potential_matches[0][1]
                    LOGGER.info("%s - State change: partial match found for '%s' -> slot=%d (quality=%d)", 
                              DOMAIN, option, matched_preset.get("slot"), potential_matches[0][0])
        
        if matched_preset:
            coordinator.bath_profile_current = option  # type: ignore[attr-defined]
            coordinator.bath_profile_current_slot = matched_preset.get("slot")  # type: ignore[attr-defined]
            LOGGER.info("%s - State change: updated profile to '%s' (slot=%d, temp=%d, vol=%d)", 
                       DOMAIN, option, matched_preset.get("slot"), matched_preset.get("temp"), matched_preset.get("vol"))
        else:
            LOGGER.warning("%s - State change: option '%s' not found in presets. Available: %s", 
                          DOMAIN, option, [f"{p.get('label')} (name: {p.get('name')})" for p in current_presets])
    
    unsub = async_track_state_change_event(hass, [entity_id], _handle_state_change)
    entry.async_on_unload(unsub)
    
    LOGGER.debug(
        "%s input_select entity ready: %s options=%s initial=%s",
        DOMAIN,
        entity_id,
        options,
        initial,
    )
    
    return entity_id


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up this integration using UI."""
    LOGGER.debug("Setting up entry for device: %s", entry.title)

    hass.data.setdefault(DOMAIN, {})

    host = entry.data.get(CONF_HOST)
    api = RheemEziSETApi(hass=hass, host=host)
    api._config_entry = entry  # Store reference for session persistence
    # Restore session ID from storage if available
    try:
        await api._restore_session_id(entry)
    except Exception as err:  # pylint: disable=broad-except
        LOGGER.debug("%s - Failed to restore session ID on startup: %s", DOMAIN, err)
    scan_interval = entry.options.get(CONF_SCAN_INTERVAL, DEFAULT_SCAN_INTERVAL)

    coordinator = RheemEziSETDataUpdateCoordinator(hass, api=api, update_interval=scan_interval, config_entry=entry)
    try:
        if entry.state == ConfigEntryState.SETUP_IN_PROGRESS:
            await coordinator.async_config_entry_first_refresh()
        else:
            # Reload: entry is already LOADED; first refresh only valid during initial setup
            await coordinator.async_refresh()
    except Exception as err:  # pylint: disable=broad-except
        LOGGER.warning("%s - Initial refresh failed: %s. Proceeding to register entry so Options/UI remain usable. Will retry in background.", DOMAIN, err)
        # Ensure platforms can still set up even if the first refresh failed.
        # Many entities read coordinator.data during __init__.
        coordinator.data = coordinator.data or {}
        # Schedule a background retry to fetch data as soon as possible
        async def _retry_initial_refresh():
            """Retry initial data fetch in background."""
            try:
                await asyncio.sleep(2.0)  # Brief delay to allow setup to complete
                await coordinator.async_refresh()
                LOGGER.info("%s - Background retry of initial refresh succeeded", DOMAIN)
            except Exception as retry_err:
                LOGGER.debug("%s - Background retry of initial refresh failed: %s", DOMAIN, retry_err)
                # Coordinator will continue with scheduled updates
        hass.async_create_task(_retry_initial_refresh())
    # Ensure coordinator.data is always a dict (some entities assume .data is not None)
    coordinator.data = coordinator.data or {}

    hass.data[DOMAIN][entry.entry_id] = coordinator

    # Initialize coordinator bath profile state to safe defaults
    coordinator.bath_profile_options = []  # type: ignore[attr-defined]
    coordinator.bath_profile_current = None  # type: ignore[attr-defined]
    coordinator.bath_profile_current_slot = None  # type: ignore[attr-defined]

    # Create/update bath profile input_select helper
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
    from homeassistant.components.input_select import DOMAIN as INPUT_SELECT_DOMAIN
    
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

    # Remove input_select entity if it exists (check base and suffixes)
    try:
        entry_id_prefix = entry.entry_id[:8].lower()
        object_id = f"rheem_{entry_id_prefix}_bath_profile"
        base_entity_id = f"{INPUT_SELECT_DOMAIN}.{object_id}"
        
        component = hass.data.get(DATA_INSTANCES, {}).get(INPUT_SELECT_DOMAIN)
        if component:
            # Try to remove base entity and any with suffixes
            entities_to_remove = []
            for suffix in ["", "_2", "_3", "_4", "_5"]:
                entity_id = f"{base_entity_id}{suffix}" if suffix else base_entity_id
                existing_entity = component.get_entity(entity_id)
                if existing_entity:
                    entities_to_remove.append((entity_id, existing_entity))
            
            for entity_id, existing_entity in entities_to_remove:
                try:
                    # Use entity's async_remove method (more reliable than component.async_remove_entity)
                    await existing_entity.async_remove()
                    LOGGER.info("%s - Removed input_select entity: %s", DOMAIN, entity_id)
                except Exception as entity_remove_err:
                    LOGGER.warning(
                        "%s - Error removing input_select entity %s: %s",
                        DOMAIN,
                        entity_id,
                        entity_remove_err,
                    )
    except Exception as cleanup_err:  # pylint: disable=broad-except
        LOGGER.warning(
            "%s - Error during input_select cleanup in unload: %s",
            DOMAIN,
            cleanup_err,
        )

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
        # Schedule reload as a background task to avoid blocking the service call
        # This prevents 500 errors if reload takes time or encounters issues
        async def _reload_after_update():
            try:
                await asyncio.sleep(0.1)  # Brief delay to ensure options are saved
                await hass.config_entries.async_reload(entry.entry_id)
            except Exception as reload_err:
                LOGGER.error(
                    "%s - Failed to reload entry after options update: %s",
                    DOMAIN,
                    reload_err,
                    exc_info=True,
                )
                # Don't re-raise - allow service to complete successfully
                # The options are saved, reload can be retried manually if needed
        
        hass.async_create_task(_reload_after_update())

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
