"""Water heater platform for rheem_eziset."""
from __future__ import annotations

from homeassistant.config_entries import ConfigEntry
from homeassistant.components.water_heater import WaterHeaterEntity, WaterHeaterEntityFeature, STATE_GAS, STATE_OFF
from homeassistant.const import ATTR_TEMPERATURE, UnitOfTemperature, PRECISION_WHOLE

from .const import DOMAIN
from .coordinator import RheemEziSETDataUpdateCoordinator
from .entity import RheemEziSETEntity
from .util import to_int


async def async_setup_entry(hass, entry, async_add_devices):
    """Add water heater for passed config_entry in HA."""
    coordinator = hass.data[DOMAIN][entry.entry_id]

    water_heater = [RheemEziSETWaterHeater(coordinator, entry)]

    async_add_devices(water_heater, True)


class RheemEziSETWaterHeater(RheemEziSETEntity, WaterHeaterEntity):
    """rheem_eziset Water Heater class."""

    def __init__(
        self,
        coordinator: RheemEziSETDataUpdateCoordinator,
        entry: ConfigEntry,
    ) -> None:
        """Initialize the sensor."""
        super().__init__(coordinator, entry)
        self._attr_current_operation = STATE_OFF
        self._attr_operation_list = [STATE_GAS, STATE_OFF]
        self._attr_precision = PRECISION_WHOLE
        self._attr_supported_features = WaterHeaterEntityFeature.TARGET_TEMPERATURE
        self._attr_target_temperature = (self.coordinator.data or {}).get("tempMin")
        self._attr_current_temperature = None
        self._attr_has_entity_name = True
        self.rheem_target_temperature = (self.coordinator.data or {}).get("temp")
        self.rheem_current_temperature = (self.coordinator.data or {}).get("temp")
        self.entry = entry

    @property
    def name(self):
        """Return a name."""
        return "Water Heater"

    @property
    def unique_id(self):
        """Return a unique id."""
        return f"{self.entry.entry_id}-water-heater"

    @property
    def extra_state_attributes(self):
        """Return the optional entity specific state attributes."""
        import time
        from datetime import datetime, timezone
        
        data = {"target_temp_step": PRECISION_WHOLE}
        
        # Add last update timestamp
        if self.coordinator.last_update_success:
            data["last_update"] = datetime.now(timezone.utc).isoformat()
        
        # Add connection quality (based on recent success rate)
        api = self.coordinator.api
        endpoint_health = getattr(api, "_endpoint_health", {})
        healthy_endpoints = sum(1 for healthy in endpoint_health.values() if healthy)
        total_endpoints = len(endpoint_health) if endpoint_health else 3
        connection_quality = (healthy_endpoints / total_endpoints * 100) if total_endpoints > 0 else 100
        data["connection_quality"] = round(connection_quality, 1)
        
        # Add device lockout info if applicable
        lockout_until = getattr(api, "_lockout_until_monotonic", None)
        if lockout_until:
            now = time.monotonic()
            if now < lockout_until:
                data["device_lockout_until"] = lockout_until
                data["device_lockout_remaining_s"] = max(0, int(lockout_until - now))
        
        # Add session owner info
        owned_sid = getattr(api, "_owned_sid", None)
        data["session_owner"] = "integration" if owned_sid else "none"
        if owned_sid:
            data["session_id"] = owned_sid

        bathfill_target = to_int((self.coordinator.data or {}).get("bathfill_target_temp"))
        if bathfill_target is not None:
            data["bathfill_target_temperature"] = bathfill_target
        
        return data

    @property
    def precision(self) -> float:
        """Return the precision of the system."""
        return PRECISION_WHOLE

    @property
    def temperature_unit(self) -> str:
        """Return the unit of measurement used by the platform."""
        return UnitOfTemperature.CELSIUS

    @property
    def current_operation(self):
        """Return the state of the sensor."""
        mode_val = to_int((self.coordinator.data or {}).get("mode"))
        if mode_val in (15, 25, 30):
            return STATE_GAS
        else:
            return STATE_OFF

    @property
    def supported_features(self):
        """Return the Supported features of the water heater."""
        return self._attr_supported_features

    @property
    def min_temp(self):
        """Return the minimum temperature that can be set."""
        return (self.coordinator.data or {}).get("tempMin")

    @property
    def max_temp(self):
        """Return the maximum temperature that can be set."""
        return (self.coordinator.data or {}).get("tempMax")

    @property
    def current_temperature(self):
        """Return the current temperature."""
        data = self.coordinator.data or {}
        api = self.coordinator.api
        
        # First, check if device reports temperature (most authoritative)
        result = to_int(data.get("temp"))
        if result is not None:
            self.rheem_current_temperature = result
            return result
        
        # If device doesn't report temp, check if we're in bath fill mode
        # During bath fill, use the bath fill target temperature as fallback
        # (the device may not report temp during bath fill mode)
        bathfill_engaged = bool(api._bathfill_engaged(data))
        if bathfill_engaged:
            bathfill_target = to_int(data.get("bathfill_target_temp"))
            if bathfill_target is not None:
                # Update cache for consistency
                self.rheem_current_temperature = bathfill_target
                return bathfill_target
        
        # Last resort: use cached value
        return self.rheem_current_temperature

    @property
    def target_temperature(self):
        """Return the target temperature or the current temperature if there is no target."""
        # Check device-reported target first (most authoritative)
        data = self.coordinator.data or {}
        device_reqtemp = to_int(data.get("reqtemp"))
        device_settemp = to_int(data.get("setTemp"))
        device_target = device_reqtemp or device_settemp
        
        # Bath fill target takes precedence if bath fill is active
        bathfill_target = to_int(data.get("bathfill_target_temp"))
        if bathfill_target is not None:
            return bathfill_target
        
        # Use device-reported target if available (from latest poll or API response)
        if device_target is not None:
            # Update our cached value to match device
            if self.rheem_target_temperature != device_target:
                self.rheem_target_temperature = device_target
            return device_target
        
        # Fall back to cached value if set (user just changed it, waiting for device confirmation)
        if self.rheem_target_temperature is not None:
            return self.rheem_target_temperature
        
        # Last resort: use current temperature
        current = self.current_temperature
        if current is not None:
            return current
        return self.rheem_current_temperature

    async def async_set_temperature(self, **kwargs):
        """Set the target temperature of the water heater."""
        api = self.coordinator.api
        temp = kwargs.get(ATTR_TEMPERATURE)
        self.rheem_target_temperature = temp
        await api.async_set_temp(water_heater=self, temp=temp, origin="user", entity_id=self.entity_id)
        self.rheem_current_temperature = None
        await self.coordinator.async_schedule_fast_refresh("set_temp")