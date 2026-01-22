"""Adds config flow for Rheem EziSET."""
from __future__ import annotations

import traceback
import json
import time
import asyncio
import contextlib
import voluptuous as vol

from homeassistant import config_entries
from homeassistant.core import callback
from homeassistant.const import CONF_HOST

from .const import CONF_SCAN_INTERVAL, DEFAULT_BATHFILL_PRESETS, DEFAULT_SCAN_INTERVAL, DOMAIN, LOGGER, PRESET_SLOTS
from .api import RheemEziSETApi
from .util import is_one, to_int
from .manifest import manifest_version

DEBUG_LOG_PATH = "/config/custom_components/rheem_eziset/debug.log"
DEBUG_SESSION_ID = "debug-session"
DEBUG_RUN_ID = "pre-fix"


def _agent_log(hypothesis: str, location: str, message: str, data: dict) -> None:
    """Write NDJSON debug instrumentation into a file under custom_components (off the event loop)."""

    def _write() -> None:
        payload = {
            "sessionId": DEBUG_SESSION_ID,
            "runId": DEBUG_RUN_ID,
            "integrationVersion": manifest_version(),
            "hypothesisId": hypothesis,
            "location": location,
            "message": message,
            "data": data,
            "timestamp": int(time.time() * 1000),
        }
        try:
            with open(DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload) + "\n")
        except Exception:
            # Never let debug logging break the flow; best-effort only.
            return

    try:
        loop = asyncio.get_running_loop()
        task = loop.create_task(asyncio.to_thread(_write))

        def _drain(task: asyncio.Task) -> None:
            # Consume task result to silence "Task exception was never retrieved".
            with contextlib.suppress(Exception):
                task.result()

        task.add_done_callback(_drain)
    except Exception:
        with contextlib.suppress(Exception):
            _write()


SAFE_CONN_CLASS_LOCAL_POLL = getattr(config_entries, "CONN_CLASS_LOCAL_POLL", None)


class ConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Config flow for rheem_eziset."""

    VERSION = 1
    CONNECTION_CLASS = SAFE_CONN_CLASS_LOCAL_POLL

    def __init__(self):
        """Initialise the flow with no errors."""
        self._errors = {}

    async def async_step_user(self, user_input=None):
        """Handle a flow initialized by the user."""
        self._errors = {}

        if user_input is not None:
            # Don't allow duplicates
            current_entries = self._async_current_entries()
            for entry in current_entries:
                if user_input[CONF_HOST] == entry.title:
                    return self.async_abort(reason="host_already_exists")

            # Test connectivity
            valid = await self._test_host(user_input[CONF_HOST])

            if valid:
                return self.async_create_entry(title=user_input[CONF_HOST], data=user_input)
            else:
                self._errors["base"] = "connection"

            return await self._show_config_form(user_input)

        return await self._show_config_form(user_input)

    async def _show_config_form(self, user_input):  # pylint: disable=unused-argument
        return self.async_show_form(
            step_id="user",
            data_schema=vol.Schema({vol.Required(CONF_HOST): str}),
            errors=self._errors,
        )

    async def _test_host(self, host: str) -> bool:
        """Validate host."""
        try:
            api = RheemEziSETApi(self.hass, host=host)
            # Single lightweight call to avoid bursts
            await api._enqueue_request("getInfo.cgi")  # pylint: disable=protected-access
            return True
        except Exception as ex:  # pylint: disable=broad-except
            LOGGER.error(
                "%s Exception in connection: %s - traceback: %s",
                DOMAIN,
                ex,
                traceback.format_exc(),
            )
        return False

    @staticmethod
    @callback
    def async_get_options_flow(config_entry: config_entries.ConfigEntry) -> config_entries.OptionsFlow:
        """Create the options flow."""
        try:
            # region agent log
            _agent_log(
                hypothesis="H2-config-flow",
                location="config_flow.py:async_get_options_flow:pre",
                message="Create OptionsFlowHandler",
                data={"entry_id": config_entry.entry_id},
            )
            # endregion
            LOGGER.debug("%s debug options_flow pre entry_id=%s", DOMAIN, config_entry.entry_id)
            return OptionsFlowHandler(config_entry)
        except Exception:  # pylint: disable=broad-except
            LOGGER.exception("%s - Failed to create OptionsFlowHandler, falling back to SafeFallbackOptionsFlow", DOMAIN)
            # region agent log
            _agent_log(
                hypothesis="H2-config-flow",
                location="config_flow.py:async_get_options_flow:error",
                message="OptionsFlowHandler init failed",
                data={"entry_id": config_entry.entry_id},
            )
            # endregion
            LOGGER.debug("%s debug options_flow fallback entry_id=%s", DOMAIN, config_entry.entry_id)
            return SafeFallbackOptionsFlow(config_entry)


class OptionsFlowHandler(config_entries.OptionsFlow):
    """Options flow for Rheem EziSET."""

    def __init__(self, config_entry: config_entries.ConfigEntry) -> None:
        """Initialise the options flow."""
        # Keep a local reference (OptionsFlow exposes config_entry as read-only).
        # Also set a best-effort internal attribute that some HA versions use.
        self._config_entry = config_entry
        self._entry = config_entry
        self._errors: dict[str, str] = {}

    def _add_suggested_schema(self, schema: vol.Schema, defaults: dict) -> vol.Schema:
        """Wrapper for add_suggested_values_to_schema with graceful fallback."""
        try:
            return self.add_suggested_values_to_schema(schema, defaults)  # type: ignore[attr-defined]
        except Exception:
            return schema

    def _get_default(self, user_input: dict, key: str, fallback: object) -> object:
        """Return default value, preferring user input then existing options."""
        return user_input.get(key) if key in user_input else self._entry.options.get(key, fallback)

    def _preset_keys(self, idx: int) -> tuple[str, str, str, str]:
        """Return keys for enabled/name/temp/vol."""
        return (
            f"bathfill_preset_{idx}_enabled",
            f"bathfill_preset_{idx}_name",
            f"bathfill_preset_{idx}_temp",
            f"bathfill_preset_{idx}_vol",
        )

    def _device_engaged(self) -> bool:
        """Check if bath fill is engaged from cached coordinator data."""
        hass = getattr(self, "hass", None)
        if hass is None:
            return False
        coordinator = hass.data.get(DOMAIN, {}).get(self._entry.entry_id)
        if not coordinator or not getattr(coordinator, "data", None):
            return False
        data = coordinator.data
        mode_val = to_int(data.get("mode"))
        bathfill_ctrl = data.get("bathfillCtrl")
        api = getattr(coordinator, "api", None)
        engaged = bool(api and getattr(api, "_bathfill_engaged", None) and api._bathfill_engaged(data))
        _agent_log(
            hypothesis="H2-config-flow",
            location="config_flow.py:_device_engaged",
            message="Device engaged check",
            data={"entry_id": self._entry.entry_id, "mode": mode_val, "bathfillCtrl": bathfill_ctrl, "engaged": engaged},
        )
        return engaged

    async def async_step_init(self, user_input=None):
        """Handle an option flow."""
        self._errors = {}
        existing = {**self._entry.data, **self._entry.options}
        use_defaults = not any(key.startswith("bathfill_preset_") for key in self._entry.options)
        # region agent log
        _agent_log(
            hypothesis="H2-config-flow",
            location="config_flow.py:async_step_init:start",
            message="Options init start",
            data={"has_user_input": user_input is not None, "use_defaults": use_defaults, "entry_id": self._entry.entry_id},
        )
        # endregion
        LOGGER.debug(
            "%s debug options_init start entry_id=%s has_user_input=%s use_defaults=%s",
            DOMAIN,
            self._entry.entry_id,
            user_input is not None,
            use_defaults,
        )

        try:
            if user_input is not None:
                # Validate device not engaged
                if self._device_engaged():
                    self._errors["base"] = "bathfill_engaged"
                # Validate enabled presets have values + no duplicates
                enabled_pairs: set[tuple[int, int]] = set()
                for idx in range(1, PRESET_SLOTS + 1):
                    enabled_key, _, temp_key, vol_key = self._preset_keys(idx)
                    if not user_input.get(enabled_key, False):
                        continue
                    temp = user_input.get(temp_key)
                    volume = user_input.get(vol_key)
                    # We model "unset" as 0 for schema serialization safety
                    if temp in (None, 0) or volume in (None, 0):
                        self._errors["base"] = "preset_missing_values"
                        break
                    pair = (temp, volume)
                    if pair in enabled_pairs:
                        self._errors["base"] = "duplicate_presets"
                        break
                    enabled_pairs.add(pair)

                if not self._errors:
                    _agent_log(
                        hypothesis="H2-config-flow",
                        location="config_flow.py:async_step_init:create_entry",
                        message="Options create entry",
                        data={"entry_id": self._entry.entry_id, "user_input_keys": list(user_input.keys())},
                    )
                    return self.async_create_entry(title="", data=user_input)

            schema_dict: dict = {
                # Ensure a valid int default for schema serialization (avoid None).
                vol.Optional(
                    CONF_SCAN_INTERVAL,
                    description={
                        "suggested_value": max(
                            2,
                            to_int(existing.get(CONF_SCAN_INTERVAL)) or DEFAULT_SCAN_INTERVAL,
                        )
                    },
                    default=max(2, to_int(existing.get(CONF_SCAN_INTERVAL)) or DEFAULT_SCAN_INTERVAL),
                ): vol.All(vol.Coerce(int), vol.Range(min=2)),
            }

            for idx in range(1, PRESET_SLOTS + 1):
                enabled_key, name_key, temp_key, vol_key = self._preset_keys(idx)
                defaults = DEFAULT_BATHFILL_PRESETS.get(idx) if use_defaults else None
                existing_enabled = existing.get(enabled_key)
                if isinstance(existing_enabled, bool):
                    enabled_default = existing_enabled
                elif existing_enabled is None:
                    enabled_default = bool(defaults and defaults.get("enabled", False))
                else:
                    enabled_default = is_one(existing_enabled)

                existing_name = existing.get(name_key)
                if isinstance(existing_name, str) and existing_name.strip():
                    name_default = existing_name
                else:
                    name_default = str(defaults.get("name")) if defaults else f"Preset {idx}"

                schema_dict[vol.Optional(enabled_key, default=enabled_default)] = bool
                schema_dict[vol.Optional(name_key, default=name_default)] = str
                existing_temp = to_int(existing.get(temp_key))
                existing_vol = to_int(existing.get(vol_key))
                default_temp = existing_temp if existing_temp is not None else (defaults.get("temp") if defaults else 0)
                default_vol = existing_vol if existing_vol is not None else (defaults.get("vol") if defaults else 0)
                schema_dict[vol.Optional(temp_key, default=default_temp)] = vol.Coerce(int)
                schema_dict[vol.Optional(vol_key, default=default_vol)] = vol.Coerce(int)

            schema = vol.Schema(schema_dict)
            schema = self._add_suggested_schema(schema, existing)
            _agent_log(
                hypothesis="H2-config-flow",
                location="config_flow.py:async_step_init:show_form",
                message="Options show form",
                data={"entry_id": self._entry.entry_id, "errors": self._errors, "schema_keys": list(schema_dict.keys())},
            )
            return self.async_show_form(
                step_id="init",
                data_schema=schema,
                errors=self._errors,
            )

        except Exception as exc:  # pylint: disable=broad-except
            # Surface unexpected errors as form error instead of a 500
            self._errors["base"] = "unknown"
            LOGGER.exception("%s - Unexpected error in options flow", DOMAIN)
            # region agent log
            _agent_log(
                hypothesis="H2-config-flow",
                location="config_flow.py:async_step_init:exception",
                message="Options init exception",
                data={"entry_id": self._entry.entry_id, "error": repr(exc)},
            )
            # endregion
            LOGGER.debug("%s debug options_init exception entry_id=%s", DOMAIN, self._entry.entry_id)

        # Fallback: show a minimal form if schema construction failed unexpectedly
        try:
            _agent_log(
                hypothesis="H2-config-flow",
                location="config_flow.py:async_step_init:fallback",
                message="Options fallback form",
                data={"entry_id": self._entry.entry_id, "errors": self._errors},
            )
            return self.async_show_form(step_id="init", data_schema=vol.Schema({}), errors=self._errors)
        except Exception as exc:  # pylint: disable=broad-except
            LOGGER.exception("%s - OptionsFlow fallback form failed", DOMAIN)
            # region agent log
            _agent_log(
                hypothesis="H2-config-flow",
                location="config_flow.py:async_step_init:fallback_exception",
                message="Options fallback exception",
                data={"entry_id": self._entry.entry_id, "error": repr(exc)},
            )
            # endregion
            LOGGER.debug("%s debug options_init fallback exception entry_id=%s", DOMAIN, self._entry.entry_id)
            return self.async_abort(reason="unknown")


class SafeFallbackOptionsFlow(config_entries.OptionsFlow):
    """Minimal fallback flow to avoid 500s if the main flow fails at creation."""

    def __init__(self, config_entry: config_entries.ConfigEntry) -> None:
        """Initialise the fallback options flow."""
        # Best-effort for HA internals
        self._config_entry = config_entry
        # Store privately; OptionsFlow exposes config_entry as read-only.
        self._entry = config_entry

    async def async_step_init(self, user_input=None):
        """Show minimal fallback form."""
        return self.async_show_form(
            step_id="init",
            data_schema=vol.Schema({}),
            errors={"base": "unknown"},
        )
