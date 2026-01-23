---
name: home-assistant-integration
description: Develop Home Assistant custom integrations following established patterns. Use when creating or modifying HA integrations, entities, coordinators, config flows, or services. Covers integration structure, DataUpdateCoordinator pattern, entity platforms, config/options flows, service registration, and common pitfalls.
---

# Home Assistant Integration Development

## Integration Structure

### Core Files

**`__init__.py`** - Integration entry point:
- `async_setup_entry()`: Initialize coordinator, register platforms, set up services
- `async_unload_entry()`: Clean up tasks, unload platforms
- Store coordinator in `hass.data[DOMAIN][entry.entry_id]`
- Always initialize `coordinator.data = coordinator.data or {}` for null safety

**`manifest.json`** - Integration metadata:
- `domain`: Unique identifier (matches folder name)
- `name`: Display name (can differ from domain)
- `version`: Semantic version
- `config_flow`: `true` for UI-based setup
- `iot_class`: `"local_polling"` for local devices

**`config_flow.py`** - Setup and options UI:
- `ConfigFlow`: User setup flow (`async_step_user()`)
- `OptionsFlowHandler`: Options flow for configuration
- Use `vol.Schema` for form validation
- Test connectivity before creating entry

### Coordinator Pattern

**`coordinator.py`** - Data polling and state management:

```python
class MyDataUpdateCoordinator(DataUpdateCoordinator):
    def __init__(self, hass, api, update_interval, config_entry):
        super().__init__(
            hass=hass,
            logger=LOGGER,
            name=DOMAIN,
            update_interval=timedelta(seconds=update_interval),
            config_entry=config_entry,
        )
        self.api = api
    
    async def _async_update_data(self):
        """Fetch data from API."""
        try:
            return await self.api.async_get_data()
        except Exception as exc:
            raise UpdateFailed(str(exc)) from exc
```

**Key patterns:**
- Store API client in `coordinator.api` (shared across entities)
- Use `update_interval` for regular polling
- Implement fast refresh after control operations
- Handle `UpdateFailed` for error propagation

### Entity Base Class

**`entity.py`** - Base entity with coordinator integration:

```python
class MyEntity(CoordinatorEntity):
    def __init__(self, coordinator, entry):
        super().__init__(coordinator)
        self.entry = entry
    
    @property
    def device_info(self):
        """Return device registry info."""
        return {
            "identifiers": {(DOMAIN, unique_id)},
            "name": device_name,
            "manufacturer": MANUFACTURER,
        }
    
    @property
    def available(self) -> bool:
        """Entity availability based on coordinator state."""
        return self.coordinator.last_update_success
```

**Null safety:** Always use `(coordinator.data or {}).get(...)` - `coordinator.data` can be `None` during init or after failures.

### Platform Entities

**Platform files** (`sensor.py`, `switch.py`, `water_heater.py`, etc.):

```python
async def async_setup_entry(hass, entry, async_add_entities):
    """Set up platform entities."""
    coordinator = hass.data[DOMAIN][entry.entry_id]
    entities = [MyEntity(coordinator, entry)]
    async_add_entities(entities, True)  # True = update before add

class MyEntity(MyBaseEntity, SensorEntity):
    def __init__(self, coordinator, entry):
        super().__init__(coordinator, entry)
        self._attr_unique_id = f"{entry.entry_id}-entity-name"
        self._attr_name = "Entity Name"
        self._attr_has_entity_name = True  # Use device name prefix
    
    @property
    def native_value(self):
        """Return sensor value."""
        data = self.coordinator.data or {}
        return data.get("field")
```

**Entity patterns:**
- Inherit from base entity + platform entity class
- Use `entry.entry_id` in `unique_id` for stability
- Set `_attr_has_entity_name = True` for device name prefixing
- Access data via `coordinator.data or {}`
- Use `async_write_ha_state()` after optimistic updates

## Config Flow

### User Flow

```python
class ConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    VERSION = 1
    
    async def async_step_user(self, user_input=None):
        if user_input is not None:
            # Validate no duplicates
            for entry in self._async_current_entries():
                if user_input[CONF_HOST] == entry.title:
                    return self.async_abort(reason="already_configured")
            
            # Test connectivity
            if await self._test_connection(user_input[CONF_HOST]):
                return self.async_create_entry(
                    title=user_input[CONF_HOST],
                    data=user_input
                )
            self._errors["base"] = "cannot_connect"
        
        return self.async_show_form(
            step_id="user",
            data_schema=vol.Schema({vol.Required(CONF_HOST): str}),
            errors=self._errors,
        )
```

### Options Flow

```python
@staticmethod
@callback
def async_get_options_flow(config_entry):
    return OptionsFlowHandler(config_entry)

class OptionsFlowHandler(config_entries.OptionsFlow):
    async def async_step_init(self, user_input=None):
        if user_input is not None:
            return self.async_create_entry(title="", data=user_input)
        
        schema = vol.Schema({
            vol.Optional(CONF_OPTION, default=current_value): int,
        })
        return self.async_show_form(step_id="init", data_schema=schema)
```

## Service Registration

Register services in `__init__.py`:

```python
def _register_services(hass: HomeAssistant) -> None:
    async def async_service_handler(call):
        """Handle service call."""
        entry = _resolve_entry_from_target(
            hass,
            call.data.get("device_id"),
            call.data.get("entity_id")
        )
        # Service logic here
    
    hass.services.async_register(DOMAIN, "service_name", async_service_handler)
```

**Entry resolution:**
- Use `device_registry` or `entity_registry` to find config entry
- Validate entry belongs to your domain
- Access coordinator via `hass.data[DOMAIN][entry.entry_id]`

## Common Patterns

### Fast Refresh After Control

```python
# In coordinator
async def async_schedule_fast_refresh(self, reason: str):
    """Fast polling after control operations."""
    # Bounded loop with rate limiting
    # Only update if data changed
    # Respect device rate limits
```

### Rate Limiting

For devices with strict rate limits:
- Use global request queue (one request at a time)
- Enforce minimum gap between requests
- Implement lockout/cooldown on failures
- Use soft failures for non-critical operations

### State Management

**Bath fill / session state:**
- Track active vs engaged states separately
- Use latches to handle stale device state
- Implement completion detection
- Handle auto-exit logic

### Entity Availability

```python
@property
def available(self) -> bool:
    """Check device availability."""
    # Respect lockout/cooldown
    # Check coordinator.last_update_success
    # Track consecutive failures
    return self.coordinator.last_update_success
```

## Common Pitfalls

### Null Safety
**Problem:** `coordinator.data` is `None` during init or after failures.

**Solution:** Always use `(coordinator.data or {}).get(...)`

### Changing unique_id
**Problem:** Changing `unique_id` breaks entity registry and HomeKit pairing.

**Solution:** Never change existing `unique_id` formats. Only add new entities.

### Not Releasing Sessions
**Problem:** Device sessions can lock device if not released.

**Solution:** Always call release after control operations, even on failure.

### Hard-coding Entity IDs
**Problem:** Entity IDs change based on device name.

**Solution:** Resolve from registry using `unique_id` patterns.

### Ignoring Rate Limits
**Problem:** Too many requests can cause device lockout.

**Solution:** Use unified request queue with rate limiting.

## Testing

Use `scripts/ha_self_test.py` pattern:
- Resolve entity IDs from registry
- Use Long-Lived Access Token (LLAT)
- Test control operations and services
- Handle flow-aware waiting (detect active flow)

## File Reference

| File | Purpose |
|------|---------|
| `__init__.py` | Integration setup, service registration |
| `coordinator.py` | Data polling, fast refresh |
| `entity.py` | Base entity class |
| `config_flow.py` | Setup and options UI |
| `manifest.json` | Integration metadata |
| `*_platform.py` | Entity platforms (sensor, switch, etc.) |
| `api.py` | HTTP client, rate limiting, control logic |
| `const.py` | Constants (domain, mode maps, icons) |
