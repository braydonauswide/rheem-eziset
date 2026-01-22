"""Manifest helpers."""

from __future__ import annotations

import json
from importlib import resources
from typing import Any


def manifest_version() -> str:
    """Return the integration version from manifest.json."""
    try:
        with resources.files(__package__).joinpath("manifest.json").open("r", encoding="utf-8") as f:
            data: dict[str, Any] = json.load(f)
            version = data.get("version")
            if isinstance(version, str):
                return version
    except Exception:
        pass
    return "unknown"
