"""Sensor platform for rheem_eziset."""
from __future__ import annotations

from typing import Any

from homeassistant.config_entries import ConfigEntry
from homeassistant.components.sensor import SensorEntity, SensorStateClass, SensorDeviceClass
from homeassistant.helpers.entity import EntityCategory
from homeassistant.helpers import entity_registry as er
from homeassistant.const import UnitOfTime, UnitOfVolume, STATE_UNAVAILABLE, UnitOfTemperature, PERCENTAGE

from .const import BATHFILL_MODES, ICON_NAME, ICON_RAW, ICON_TAPON, ICON_TAPOFF, ICON_TIMER, ICON_TEMP, ICON_WATERHEATER, CONST_MODE_MAP, CONST_STATUS_MAP, DOMAIN, LOGGER
from .util import is_one, to_float, to_int
from .coordinator import RheemEziSETDataUpdateCoordinator
from .entity import RheemEziSETEntity

TIME_MINUTES = UnitOfTime.MINUTES
TIME_SECONDS = UnitOfTime.SECONDS
VOLUME_LITERS = UnitOfVolume.LITERS
CELSIUS = UnitOfTemperature.CELSIUS

SENSOR_MAP = [
    # ("description",        "key",          "unit",                             "icon",             "device_class",                         "state_class",                      "entity_category",              "enabled_default"), # pylint: disable=line-too-long
    ("Flow", "flow", f"{VOLUME_LITERS}/{TIME_MINUTES}", ICON_TAPON, None, SensorStateClass.MEASUREMENT, None, True),  # pylint: disable=line-too-long
    ("Status", "state", None, ICON_WATERHEATER, None, None, None, True),  # pylint: disable=line-too-long
    ("Mode", "mode", None, ICON_WATERHEATER, None, None, None, True),  # pylint: disable=line-too-long
    ("Status raw", "state", None, ICON_RAW, None, SensorStateClass.MEASUREMENT, EntityCategory.DIAGNOSTIC, True),  # pylint: disable=line-too-long
    ("Mode raw", "mode", None, ICON_RAW, None, SensorStateClass.MEASUREMENT, EntityCategory.DIAGNOSTIC, True),  # pylint: disable=line-too-long
    ("Heater error raw", "appErrCode", None, ICON_RAW, None, SensorStateClass.MEASUREMENT, EntityCategory.DIAGNOSTIC, True),  # pylint: disable=line-too-long
    ("Session timeout", "sTimeout", TIME_SECONDS, ICON_TIMER, None, SensorStateClass.MEASUREMENT, EntityCategory.DIAGNOSTIC, True),  # pylint: disable=line-too-long
    ("Current Temperature", "temp", CELSIUS, ICON_TEMP, SensorDeviceClass.TEMPERATURE, SensorStateClass.MEASUREMENT, EntityCategory.DIAGNOSTIC, True),  # pylint: disable=line-too-long
    ("Heater Name", "heaterName", None, ICON_NAME, None, None, EntityCategory.DIAGNOSTIC, True),  # pylint: disable=line-too-long
    ("Heater Model", "heaterModel", None, ICON_NAME, None, None, EntityCategory.DIAGNOSTIC, True),  # pylint: disable=line-too-long
]


class RheemEziSETHealthSensor(RheemEziSETEntity, SensorEntity):
    """Sensor for device health/connection status."""

    _attr_icon = "mdi:heart-pulse"
    _attr_entity_category = EntityCategory.DIAGNOSTIC
    _attr_translation_key = "health_status"

    def __init__(self, coordinator: RheemEziSETDataUpdateCoordinator, entry_id: str) -> None:
        """Initialize the health sensor."""
        super().__init__(coordinator, entry_id)
        self._attr_unique_id = f"{entry_id}-health"
        self._attr_name = f"{coordinator.api.host} Health Status"

    @property
    def native_value(self) -> str:
        """Return the health status."""
        data = self.coordinator.data or {}
        api = self.coordinator.api
        
        # Check lockout state
        lockout_until = getattr(api, "_lockout_until_monotonic", None)
        if lockout_until:
            import time
            now = time.monotonic()
            if now < lockout_until:
                # Check if health check is pending
                if getattr(api, "_health_check_pending", False):
                    return "recovering"
                return "unavailable"
        
        # Check endpoint health
        endpoint_health = getattr(api, "_endpoint_health", {})
        if not endpoint_health.get("info", True):
            return "unavailable"
        
        degraded_endpoints = [ep for ep, healthy in endpoint_health.items() if not healthy and ep != "info"]
        if degraded_endpoints:
            return "degraded"
        
        return "healthy"

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Return additional state attributes."""
        api = self.coordinator.api
        endpoint_health = getattr(api, "_endpoint_health", {})
        lockout_until = getattr(api, "_lockout_until_monotonic", None)
        
        attrs: dict[str, Any] = {
            "endpoint_info": endpoint_health.get("info", True),
            "endpoint_params": endpoint_health.get("params", True),
            "endpoint_version": endpoint_health.get("version", True),
        }
        
        if lockout_until:
            import time
            now = time.monotonic()
            if now < lockout_until:
                attrs["lockout_until"] = lockout_until
                attrs["lockout_remaining_s"] = max(0, lockout_until - now)
        
        return attrs


async def async_setup_entry(hass, entry, async_add_devices):
    """Add sensors for passed config_entry in HA."""
    coordinator: RheemEziSETDataUpdateCoordinator = hass.data[DOMAIN][entry.entry_id]

    # Add health status sensor
    health_sensor = RheemEziSETHealthSensor(coordinator, entry.entry_id)
    async_add_devices([health_sensor])

    # If earlier versions disabled diagnostic sensors by default, re-enable those that were
    # disabled_by=integration (does not override user-disabled entities).
    enable_by_default_descriptions = {
        "Status raw",
        "Mode raw",
        "Heater error raw",
        "Current Temperature",
        "Heater Name",
        "Heater Model",
        "Session timeout",
    }
    unique_ids_to_enable = {f"{entry.entry_id}-{desc}" for desc in enable_by_default_descriptions}
    registry = er.async_get(hass)
    for reg_entry in er.async_entries_for_config_entry(registry, entry.entry_id):
        if reg_entry.domain != "sensor":
            continue
        if reg_entry.unique_id not in unique_ids_to_enable:
            continue
        if reg_entry.disabled_by != er.RegistryEntryDisabler.INTEGRATION:
            continue
        registry.async_update_entity(reg_entry.entity_id, disabled_by=None)

    sensors = [
        RheemEziSETSensor(coordinator, entry, description, key, unit, icon, device_class, state_class, entity_category, enabled_default)
        for description, key, unit, icon, device_class, state_class, entity_category, enabled_default in SENSOR_MAP  # pylint: disable=line-too-long
    ]

    sensors.extend(
        [
            BathFillStatusSensor(coordinator, entry),
            QueuedChangeSensor(coordinator, entry),
            BathFillProgressSensor(coordinator, entry),
        ]
    )

    async_add_devices(sensors, True)


class RheemEziSETSensor(RheemEziSETEntity, SensorEntity):
    """rheem_eziset Sensor class."""

    def __init__(
        self,
        coordinator: RheemEziSETDataUpdateCoordinator,
        entry: ConfigEntry,
        description: str,
        key: str,
        unit: str,
        icon: str,
        device_class: str,
        state_class: str,
        entity_category: str,
        enabled_default: bool,
    ) -> None:
        """Initialize the sensor."""
        super().__init__(coordinator, entry)
        self.description = description
        self.key = key
        self._attr_native_unit_of_measurement = unit
        self._icon = icon
        self._device_class = device_class
        self._state_class = state_class
        self._attr_entity_category = entity_category
        self._attr_has_entity_name = True
        self._attr_entity_registry_enabled_default = enabled_default
        if description == "Flow":
            self._attr_suggested_display_precision = 1
        if description == "Bath Fill Progress":
            self._attr_suggested_display_precision = 0

    @property
    def native_value(self):
        """Return the native_value of the sensor."""
        result = (self.coordinator.data or {}).get(self.key, None)
        if self.description == "Status":
            try:
                val = to_int(result)
                if val in CONST_STATUS_MAP:
                    return CONST_STATUS_MAP[val][0]
                return STATE_UNAVAILABLE
            except Exception:  # pylint: disable=broad-except
                LOGGER.error("%s -  Unexpected result for status, result was %s", DOMAIN, result)
                return STATE_UNAVAILABLE
        elif self.description == "Mode":
            try:
                val = to_int(result)
                if val in CONST_MODE_MAP:
                    return CONST_MODE_MAP[val][0]
                return STATE_UNAVAILABLE
            except Exception:  # pylint: disable=broad-except
                LOGGER.error("%s -  Unexpected result for mode, result was %s", DOMAIN, result)
                return STATE_UNAVAILABLE
        else:
            return result

    @property
    def icon(self):
        """Return the icon with processing in the case of some sensors."""
        result = (self.coordinator.data or {}).get(self.key, STATE_UNAVAILABLE)
        if self.description == "Flow":
            try:
                if float(result) != 0:
                    return ICON_TAPON
                else:
                    return ICON_TAPOFF
            except Exception:  # pylint: disable=broad-except
                return ICON_TAPOFF
        elif self.description == "Status":
            try:
                val = to_int(result)
                if val in CONST_STATUS_MAP:
                    return CONST_STATUS_MAP[val][1]
            except Exception:  # pylint: disable=broad-except
                return self._icon
            return self._icon
        elif self.description == "Mode":
            try:
                val = to_int(result)
                if val in CONST_MODE_MAP:
                    return CONST_MODE_MAP[val][1]
            except Exception:  # pylint: disable=broad-except
                return self._icon
            return self._icon
        else:
            return self._icon

    @property
    def unique_id(self):
        """Return the unique id."""
        return f"{self.entry.entry_id}-{self.description}"

    @property
    def name(self):
        """Return the name."""
        return self.description


class BathFillStatusSensor(RheemEziSETEntity, SensorEntity):
    """Human-readable bath fill status."""

    _attr_has_entity_name = True
    _attr_name = "Bath Fill Status"

    def __init__(self, coordinator: RheemEziSETDataUpdateCoordinator, entry: ConfigEntry) -> None:
        """Initialize the bath fill status sensor."""
        super().__init__(coordinator, entry)
        self._attr_unique_id = f"{self.entry.entry_id}-bathfill-status"

    @property
    def native_value(self):
        """Return the human-readable bath fill status."""
        data = self.coordinator.data or {}
        mode_val = to_int(data.get("mode"))
        state_val = to_int(data.get("state"))
        flow_val = to_float(data.get("flow"))
        s_timeout = to_int(data.get("sTimeout"))
        api = self.coordinator.api
        bathfill_active = bool(api._bathfill_engaged(data))
        if bathfill_active:
            if mode_val == 35 or state_val == 3:
                return "complete_waiting_for_exit"
            if flow_val not in (0, None):
                return "filling"
            return "active"
        # not active
        if s_timeout not in (0, None):
            return "session_busy"
        return "idle"


class QueuedChangeSensor(RheemEziSETEntity, SensorEntity):
    """Expose queued change state (temp, bath fill start/exit, session timer, restore)."""

    _attr_has_entity_name = True
    _attr_name = "Queued Change"

    def __init__(self, coordinator: RheemEziSETDataUpdateCoordinator, entry: ConfigEntry) -> None:
        """Initialize the queued change sensor."""
        super().__init__(coordinator, entry)
        self._attr_unique_id = f"{self.entry.entry_id}-queued-temp"

    @property
    def native_value(self):
        """Return string describing queued change state; 'none' when no ops pending (complete)."""
        api = self.coordinator.api
        pending = getattr(api, "_pending_writes", {})
        info = self.coordinator.data or {}

        def fmt_temp(payload: dict[str, Any], kind: str) -> str:
            direction = payload.get("direction") or "up"
            requested = payload.get("temp") or payload.get("requested_temp")
            base = f"{kind}:{direction}"
            if requested is not None:
                base = f"{base}:{requested}"
            flow_val = to_float(info.get("flow"))
            mode_val = to_int(info.get("mode"))
            s_timeout = to_int(info.get("sTimeout"))
            bathfill = bool((to_int(info.get("mode")) in BATHFILL_MODES) or is_one(info.get("bathfillCtrl")))
            reason_parts = []
            if s_timeout not in (0, None):
                reason_parts.append("sTimeout")
            if flow_val not in (0, None) and direction == "up":
                reason_parts.append("flow")
            if bathfill and kind == "set_temp":
                reason_parts.append("bathfill_active")
            if mode_val not in (5, None) and direction == "up":
                reason_parts.append(f"mode{mode_val}")
            if reason_parts:
                return f"{base}:waiting({','.join(reason_parts)})"
            return f"{base}:applying"

        def fmt_bathfill_start() -> str:
            flow_val = to_float(info.get("flow"))
            mode_val = to_int(info.get("mode"))
            s_timeout = to_int(info.get("sTimeout"))
            bathfill = bool((to_int(info.get("mode")) in BATHFILL_MODES) or is_one(info.get("bathfillCtrl")))
            reason_parts = []
            if bathfill:
                reason_parts.append("bathfill_already_active")
            if s_timeout not in (0, None):
                reason_parts.append("sTimeout_active")
            if flow_val not in (0, None):
                reason_parts.append("flow_active")
            if mode_val not in (5, None):
                reason_parts.append(f"mode_busy_{mode_val}")
            if reason_parts:
                return f"bathfill_start:waiting({','.join(reason_parts)})"
            return "bathfill_start:applying"

        # Report first pending op in drain order
        if "bathfill_cancel" in pending:
            return "bathfill_cancel:applying"
        if "bathfill_set_temp" in pending:
            return fmt_temp(pending["bathfill_set_temp"], "setBathTemp")
        if "bathfill_start" in pending:
            return fmt_bathfill_start()
        if "set_session_timer" in pending:
            return "set_session_timer:applying"
        if "set_temp" in pending:
            return fmt_temp(pending["set_temp"], "set_temp")
        if "restore_temp" in pending:
            return "restore_temp:applying"
        return "none"


class BathFillProgressSensor(RheemEziSETEntity, SensorEntity):
    """Progress sensor that reports None when unknown during bath fill."""

    _attr_has_entity_name = True
    _attr_name = "Bath Fill Progress"
    _attr_native_unit_of_measurement = PERCENTAGE
    _attr_device_class = SensorDeviceClass.HUMIDITY
    _attr_state_class = SensorStateClass.MEASUREMENT
    _attr_suggested_display_precision = 0

    def __init__(self, coordinator: RheemEziSETDataUpdateCoordinator, entry: ConfigEntry) -> None:
        """Initialize the bath fill progress sensor."""
        super().__init__(coordinator, entry)
        self._attr_unique_id = f"{self.entry.entry_id}-bathfill-progress"

    @property
    def native_value(self):
        """Return progress percent or None when unknown during active bath fill."""
        data = self.coordinator.data or {}
        api = self.coordinator.api
        engaged = bool(api._bathfill_engaged(data))
        val = to_float(data.get("fillPercent"))
        if not engaged:
            return 0.0 if val is None else max(0.0, min(100.0, val))
        # engaged but no value: report unknown
        if val is None:
            return None
        return max(0.0, min(100.0, val))
