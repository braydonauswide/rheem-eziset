"""Utility helpers for safe type coercion."""

from __future__ import annotations

from typing import Any


def to_int(val: Any) -> int | None:
    """Coerce a value to int, returning None on failure."""
    try:
        return int(float(val))
    except Exception:
        return None


def to_float(val: Any) -> float | None:
    """Coerce a value to float, returning None on failure."""
    try:
        return float(val)
    except Exception:
        return None


def is_one(val: Any) -> bool:
    """Return True when val represents 1, "1", or 1.0."""
    try:
        return int(float(val)) == 1
    except Exception:
        return False
