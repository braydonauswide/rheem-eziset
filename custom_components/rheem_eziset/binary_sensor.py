"""Binary Sensor platform for rheem_eziset."""
from __future__ import annotations

from homeassistant.config_entries import ConfigEntry
from homeassistant.components.binary_sensor import BinarySensorDeviceClass, BinarySensorEntity
from homeassistant.helpers.entity import EntityCategory

from .const import DOMAIN
from .coordinator import RheemEziSETDataUpdateCoordinator
from .entity import RheemEziSETEntity
from .util import is_one, to_float, to_int

BINARY_SENSOR_MAP = [
    # ("description", "key", "icon", "device_class", "entity_category"),
    ("Heater error", "appErrCode", None, BinarySensorDeviceClass.PROBLEM, EntityCategory.DIAGNOSTIC),  # pylint: disable=line-too-long
]


async def async_setup_entry(hass, entry, async_add_devices):
    """Add binary sensors for passed config_entry in HA."""
    coordinator: RheemEziSETDataUpdateCoordinator = hass.data[DOMAIN][entry.entry_id]

    binary_sensors = [RheemEziSETBinarySensor(coordinator, entry, description, key, icon, device_class, entity_category) for description, key, icon, device_class, entity_category in BINARY_SENSOR_MAP]
    binary_sensors.append(RheemEziSETProblemBinarySensor(coordinator, entry))
    binary_sensors.append(CanIncreaseTemperatureNowSensor(coordinator, entry))
    binary_sensors.append(BathFillAcceptedSensor(coordinator, entry))

    async_add_devices(binary_sensors, True)


class RheemEziSETBinarySensor(RheemEziSETEntity, BinarySensorEntity):
    """rheem_eziset Binary Sensor class."""

    def __init__(
        self,
        coordinator: RheemEziSETDataUpdateCoordinator,
        entry: ConfigEntry,
        description: str,
        key: str,
        icon: str,
        device_class: str,
        entity_category: str,
    ) -> None:
        """Initialize the sensor."""
        super().__init__(coordinator, entry)
        self.key = key
        self._attr_icon = icon
        self._attr_device_class = device_class
        self._attr_entity_category = entity_category
        self._attr_has_entity_name = True
        self._attr_unique_id = f"{self.entry.entry_id}-{description}"
        self._attr_name = description

    @property
    def is_on(self) -> bool:
        """Return True if a problem is detected."""
        if not self.coordinator.last_update_success:
            return False
        result = (self.coordinator.data or {}).get(self.key)
        val = to_int(result)
        return val != 0 if val is not None else False


class RheemEziSETProblemBinarySensor(RheemEziSETEntity, BinarySensorEntity):
    """rheem_eziset Binary Sensor class."""

    def __init__(
        self,
        coordinator: RheemEziSETDataUpdateCoordinator,
        entry: ConfigEntry,
    ) -> None:
        """Initialize the sensor."""
        super().__init__(coordinator, entry)
        self._attr_has_entity_name = True
        self._attr_device_class = BinarySensorDeviceClass.PROBLEM
        self._attr_entity_category = EntityCategory.DIAGNOSTIC
        self._attr_unique_id = f"{self.entry.entry_id}-connectivity-problem"
        self._attr_name = "Connectivity Problem"

    @property
    def is_on(self) -> bool:
        """Return True when connectivity problem is detected."""
        return bool(self.coordinator.problem_flag)

    @property
    def available(self) -> bool:
        """Connectivity problem sensor should always be available."""
        return True


class CanIncreaseTemperatureNowSensor(RheemEziSETEntity, BinarySensorEntity):
    """Binary sensor indicating whether temperature increase can be applied now."""

    _attr_has_entity_name = True
    _attr_device_class = None
    _attr_name = "Can Increase Temperature Now"

    def __init__(self, coordinator: RheemEziSETDataUpdateCoordinator, entry: ConfigEntry) -> None:
        """Initialize the can-increase indicator."""
        super().__init__(coordinator, entry)
        self._attr_unique_id = f"{self.entry.entry_id}-can-increase-temp"

    @property
    def is_on(self) -> bool:
        """Return True when a temp increase can be applied immediately."""
        data = self.coordinator.data or {}
        flow_val = to_float(data.get("flow"))
        mode_val = to_int(data.get("mode"))
        s_timeout = to_int(data.get("sTimeout"))
        sid_val = to_int(data.get("sid")) or getattr(self.coordinator.api, "_owned_sid", None)
        bathfill_ctrl = data.get("bathfillCtrl")
        api = self.coordinator.api
        bathfill_active = bool(api._bathfill_engaged(data))

        if bathfill_active:
            # For bathfill: need flow==0, a usable sid, and a known volume to safely apply increase.
            vol_val = to_int(data.get("reqbathvol")) or to_int(data.get("bathvol"))
            return flow_val in (0, None) and sid_val not in (None, 0) and vol_val not in (None, 0)

        # Normal heater temp increase path
        return s_timeout in (0, None) and flow_val in (0, None) and mode_val in (5, None)


class BathFillAcceptedSensor(RheemEziSETEntity, BinarySensorEntity):
    """Binary sensor indicating controller has accepted bath fill (ready to open tap)."""

    _attr_has_entity_name = True
    _attr_device_class = None
    _attr_name = "Bath fill accepted"

    def __init__(self, coordinator: RheemEziSETDataUpdateCoordinator, entry: ConfigEntry) -> None:
        """Initialize the accepted indicator."""
        super().__init__(coordinator, entry)
        self._attr_unique_id = f"{self.entry.entry_id}-bathfill-accepted"

    @property
    def is_on(self) -> bool:
        """Return True when bath fill start was accepted and pending/filling."""
        return bool(getattr(self.coordinator.api, "_bathfill_accepted", False))
