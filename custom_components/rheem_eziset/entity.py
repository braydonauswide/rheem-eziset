"""Sets up the basic entity template."""
from __future__ import annotations

from homeassistant.helpers.update_coordinator import CoordinatorEntity

from .const import DOMAIN, NAME, MANUFACTURER
from .coordinator import RheemEziSETDataUpdateCoordinator


class RheemEziSETEntity(CoordinatorEntity):
    """Basic entity definition used by all entities."""

    def __init__(self, coordinator: RheemEziSETDataUpdateCoordinator, entry):
        """Initialise the entity."""
        super().__init__(coordinator)
        self.entry = entry

    @property
    def device_info(self):
        """Defines the device information."""
        data = self.coordinator.data or {}
        return {
            "identifiers": {(DOMAIN, self.coordinator.api.host)},
            "name": data.get("heaterName", NAME),
            "manufacturer": MANUFACTURER,
            "sw_version": data.get("FWversion"),
        }

    @property
    def available(self) -> bool:
        """Return availability, tracking the coordinator's latest poll result.

        When the device is unreachable the coordinator fails the update
        (last_update_success is False), so entities go unavailable within ~1
        poll interval rather than reporting the last known values as if live.
        On the next successful poll they recover automatically. The connectivity
        problem is surfaced by binary_sensor.rheem_connectivity_problem, driven
        off coordinator.problem_flag.
        """
        return self.coordinator.last_update_success

    @property
    def should_poll(self) -> bool:
        """Device should not poll because this is handled by async requests in the api."""
        return False
