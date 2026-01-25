"""Constants for Rheem EziSET."""
from logging import Logger, getLogger

from homeassistant.const import Platform

LOGGER: Logger = getLogger(__package__)

# Component Data
NAME = "Rheem EziSET Water Heater"
IDPREFIX = "rheem_water_heater_"
DOMAIN = "rheem_eziset"
MANUFACTURER = "Rheem"
P26.1.4"  # Note: Use manifest_version() from manifest.py for runtime version
PLATFORMS: list[Platform] = [
    Platform.SENSOR,
    Platform.BINARY_SENSOR,
    Platform.NUMBER,
    Platform.WATER_HEATER,
    Platform.SWITCH,
    Platform.SELECT,
]

# SCAN INTERVAL
CONF_SCAN_INTERVAL = "scan_interval"
DEFAULT_SCAN_INTERVAL = 3  # seconds (still well above MIN_REQUEST_GAP 1.5s)

# Presets
PRESET_SLOTS = 6

# Default bath fill presets (used when the user has not configured any presets yet).
# These are intended to be HomeKit-friendly defaults that can be edited in Options.
DEFAULT_BATHFILL_PRESETS: dict[int, dict[str, object]] = {
    1: {"enabled": True, "name": "Children", "temp": 39, "vol": 110},
    2: {"enabled": True, "name": "Adults", "temp": 43, "vol": 140},
}

# Mode Dictionary
CONST_MODE_MAP = {
    5: ("Idle", "mdi:water-boiler-auto"),
    10: ("Heating Control Mode", "mdi:thermometer"),
    15: ("Heating (Conventional Mode)", "mdi:fire"),
    20: ("Idle (Bath Fill Mode Waiting for Tap)", "mdi:bathtub-outline"),
    25: ("Heating (Bath Fill Mode)", "mdi:bathtub"),
    30: ("Heating (Bath Fill Mode - Flowing)", "mdi:bathtub"),
    35: ("Idle (Bath Fill Mode Complete)", "mdi:water-boiler-off"),
}

# Status Dictionary
CONST_STATUS_MAP = {
    1: ("Idle", "mdi:water-boiler-auto"),
    2: ("Heating", "mdi:water-boiler"),
    3: ("Bath Fill Complete (Off)", "mdi:water-boiler-off"),
}

# Icon Dictionary
ICON_NAME = "mdi:label"
ICON_RAW = "mdi:raw"
ICON_TAPON = "mdi:water-pump"
ICON_TAPOFF = "mdi:water-pump-off"
ICON_TEMP = "mdi:thermometer"
ICON_TIMER = "mdi:timer"
ICON_WATERHEATER = "mdi:water-boiler"
