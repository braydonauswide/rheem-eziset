"""DataUpdateCoordinator for rheem_eziset."""
from __future__ import annotations

from datetime import timedelta
import asyncio
import time

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator, UpdateFailed

from .api import RheemEziSETApi
from .const import DOMAIN, LOGGER


class RheemEziSETDataUpdateCoordinator(DataUpdateCoordinator):
    """Class to manage fetching data from the API."""

    def __init__(
        self,
        hass: HomeAssistant,
        api: RheemEziSETApi,
        update_interval: int,
        config_entry: ConfigEntry,
    ) -> None:
        """Initialize."""
        self.api = api
        self.platforms = []
        self._fast_refresh_task: asyncio.Task | None = None
        self.api.set_post_write_callback(self._post_write_refresh)

        super().__init__(
            hass=hass,
            logger=LOGGER,
            name=DOMAIN,
            update_interval=timedelta(seconds=update_interval),
            config_entry=config_entry,
        )

    async def _async_update_data(self):
        """Update basic data via client."""
        try:
            result = await self.api.async_get_data()
            LOGGER.debug("%s - Fetched data: %s", DOMAIN, result)
            return result
        except Exception as exception:
            raise UpdateFailed(str(exception)) from exception

    async def _post_write_refresh(self, reason: str) -> None:
        """Fast refresh after a queued write succeeds."""
        await self.async_schedule_fast_refresh(reason)

    async def async_schedule_fast_refresh(self, reason: str) -> None:
        """Run a bounded fast-refresh loop while writes are pending or within a settle window.
        
        Fast refresh provides rapid state updates after control operations to ensure entities
        reflect device state changes quickly. The loop:
        1. Respects rate limits via api._next_request_at and poll backoff
        2. Runs while writes are pending OR within settle_window (10s)
        3. Bounded by max_iters (20) to prevent infinite loops
        4. Uses lightweight async_get_info_only() (only getInfo.cgi, not params/version)
        5. Only updates entities if data actually changed (optimization)
        
        This ensures users see immediate feedback after control actions without hammering the device.
        """
        if self._fast_refresh_task and not self._fast_refresh_task.done():
            LOGGER.debug("%s debug fast_refresh skip pending reason=%s", DOMAIN, reason)
            return

        async def _run() -> None:
            start = time.monotonic()
            base_settle_window = 7.0  # Optimized from 8s - still sufficient for state to settle, improves responsiveness
            max_iters = 20  # Increased from 12 (allows more frequent polling during active operations)
            iters = 0
            try:
                # OPTIMIZATION: Start first poll immediately if rate limit allows, don't wait for initial delay
                # This provides faster initial feedback after control operations
                while True:
                    iters += 1
                    now = time.monotonic()
                    next_allowed = getattr(self.api, "_next_request_at", now)
                    poll_backoff = getattr(self.api, "_poll_backoff_until", 0.0) or 0.0
                    target = max(next_allowed, poll_backoff, now)
                    delay = max(0.0, target - now)
                    # Only wait if delay is significant (>0.1s) - small delays can be ignored for faster updates
                    if delay > 0.1:
                        await asyncio.sleep(delay)

                    try:
                        data = await self.api.async_get_info_only()
                        # OPTIMIZATION: Only compare key fields instead of entire dict for efficiency
                        # This avoids expensive deep dict comparisons while still detecting meaningful changes
                        current_data = self.data
                        if current_data:
                            # Check if critical fields changed (more efficient than full dict comparison)
                            # Only check fields that are present in at least one dict to avoid false positives
                            key_fields = ("temp", "reqtemp", "setTemp", "mode", "state", "flow", "fillPercent", "bathfillCtrl", "sTimeout", "sid")
                            data_changed = False
                            for key in key_fields:
                                old_val = current_data.get(key)
                                new_val = data.get(key)
                                # Only compare if at least one value exists (avoids None != None false positives)
                                if old_val is not None or new_val is not None:
                                    if old_val != new_val:
                                        data_changed = True
                                        break
                            
                            if data_changed:
                                self.async_set_updated_data(data)
                            else:
                                LOGGER.debug("%s debug fast_refresh: key fields unchanged, skipping update", DOMAIN)
                        else:
                            # No current data, always update
                            self.async_set_updated_data(data)
                        
                        # OPTIMIZATION: If bath fill is active, extend settle window for more frequent updates
                        # This ensures fill percent updates are more responsive during active bath fill
                        bathfill_active = bool(self.api._bathfill_active(data) if hasattr(self.api, "_bathfill_active") else False)
                        # Use extended settle window (30s) while bath fill is active for more frequent updates
                        settle_window = 30.0 if bathfill_active else base_settle_window
                    except Exception as err:  # pylint: disable=broad-except
                        LOGGER.debug("%s debug fast_refresh iteration failed (%s): %s", DOMAIN, type(err).__name__, err)
                        # On error, use base settle window
                        settle_window = base_settle_window

                    pending = bool(getattr(self.api, "_pending_writes", {}))
                    within_settle = (time.monotonic() - start) < settle_window
                    if not pending and not within_settle:
                        break
                    if iters >= max_iters:
                        break
            finally:
                self._fast_refresh_task = None

        self._fast_refresh_task = self.hass.async_create_task(_run())
