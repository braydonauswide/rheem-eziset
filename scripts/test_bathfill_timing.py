#!/usr/bin/env python3
"""Test script to find optimal timing for bath fill operations.

Tests:
1. Bath fill start to complete (full cycle)
2. Complete to exit
3. Bath fill start to exit (full cycle)
4. Fill percent update frequency during active bath fill

This script helps find the "sweet spot" for polling intervals and exit timing.
"""

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime
from typing import Any

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    from homeassistant_api import Client, RESTClient
except ImportError:
    print("Error: homeassistant_api not installed. Install with: pip install homeassistant-api")
    sys.exit(1)


async def get_entity_state(rest: RESTClient, entity_id: str) -> dict[str, Any] | None:
    """Get current state of an entity."""
    try:
        state = await rest.get_state(entity_id)
        return state
    except Exception as e:
        print(f"  Error getting state for {entity_id}: {e}")
        return None


async def wait_for_state(
    rest: RESTClient,
    entity_id: str,
    target_state: str,
    timeout_s: float = 60.0,
    poll_interval_s: float = 1.0,
) -> tuple[bool, float]:
    """Wait for entity to reach target state. Returns (success, elapsed_time)."""
    start = time.monotonic()
    while True:
        state = await get_entity_state(rest, entity_id)
        if state and state.state == target_state:
            elapsed = time.monotonic() - start
            return True, elapsed
        
        elapsed = time.monotonic() - start
        if elapsed >= timeout_s:
            return False, elapsed
        
        await asyncio.sleep(poll_interval_s)


async def monitor_fill_percent(
    rest: RESTClient,
    entity_id: str,
    duration_s: float,
    poll_interval_s: float = 1.0,
) -> list[tuple[float, float]]:
    """Monitor fill percent during bath fill. Returns list of (timestamp, fill_percent)."""
    start = time.monotonic()
    readings = []
    
    while True:
        elapsed = time.monotonic() - start
        if elapsed >= duration_s:
            break
        
        state = await get_entity_state(rest, entity_id)
        if state:
            fill_percent = state.attributes.get("native_value")
            if fill_percent is not None:
                readings.append((elapsed, float(fill_percent)))
        
        await asyncio.sleep(poll_interval_s)
    
    return readings


async def test_bathfill_start_to_complete(
    rest: RESTClient,
    ids: dict[str, str],
    poll_interval_s: float = 1.0,
) -> dict[str, Any]:
    """Test: Bath fill start to complete (full cycle)."""
    print("\n=== TEST: Bath Fill Start to Complete ===")
    
    # Turn on bath fill
    print("1. Starting bath fill...")
    start_time = time.monotonic()
    await rest.call_service("switch", "turn_on", {"entity_id": ids["bathfill_switch"]})
    
    # Wait for switch to turn on
    success, elapsed = await wait_for_state(rest, ids["bathfill_switch"], "on", timeout_s=30.0, poll_interval_s=poll_interval_s)
    if not success:
        return {"error": "Bath fill switch did not turn on", "elapsed": elapsed}
    
    switch_on_time = time.monotonic() - start_time
    print(f"   Switch ON: {switch_on_time:.2f}s")
    
    # Monitor fill percent until completion
    print("2. Monitoring fill percent until completion...")
    fill_readings = []
    max_wait = 600.0  # 10 minutes max
    start_monitor = time.monotonic()
    
    while True:
        elapsed = time.monotonic() - start_monitor
        if elapsed >= max_wait:
            return {"error": "Bath fill did not complete within timeout", "elapsed": elapsed}
        
        state = await get_entity_state(rest, ids["bathfill_progress"])
        if state:
            fill_percent = state.attributes.get("native_value")
            if fill_percent is not None:
                fill_readings.append((elapsed, float(fill_percent)))
                if float(fill_percent) >= 100.0:
                    break
        
        await asyncio.sleep(poll_interval_s)
    
    complete_time = time.monotonic() - start_time
    print(f"   Complete: {complete_time:.2f}s (switch_on: {switch_on_time:.2f}s)")
    
    return {
        "success": True,
        "total_time": complete_time,
        "switch_on_time": switch_on_time,
        "fill_to_complete_time": complete_time - switch_on_time,
        "fill_readings": fill_readings,
        "poll_interval_s": poll_interval_s,
    }


async def test_complete_to_exit(
    rest: RESTClient,
    ids: dict[str, str],
    poll_interval_s: float = 1.0,
) -> dict[str, Any]:
    """Test: Complete to exit (turn off at completion)."""
    print("\n=== TEST: Complete to Exit ===")
    
    # Ensure bath fill is at completion (100%)
    state = await get_entity_state(rest, ids["bathfill_progress"])
    if not state or state.attributes.get("native_value", 0) < 100.0:
        return {"error": "Bath fill not at completion (100%)"}
    
    # Turn off bath fill
    print("1. Turning off bath fill at completion...")
    start_time = time.monotonic()
    await rest.call_service("switch", "turn_off", {"entity_id": ids["bathfill_switch"]})
    
    # Wait for switch to turn off
    success, elapsed = await wait_for_state(rest, ids["bathfill_switch"], "off", timeout_s=120.0, poll_interval_s=poll_interval_s)
    if not success:
        return {"error": "Bath fill switch did not turn off", "elapsed": elapsed}
    
    exit_time = time.monotonic() - start_time
    print(f"   Exit complete: {exit_time:.2f}s")
    
    # Verify device is idle
    await asyncio.sleep(2.0)  # Brief wait for device to settle
    status_state = await get_entity_state(rest, ids["bathfill_status"])
    status = status_state.state if status_state else "unknown"
    
    return {
        "success": True,
        "exit_time": exit_time,
        "final_status": status,
        "poll_interval_s": poll_interval_s,
    }


async def test_bathfill_start_to_exit(
    rest: RESTClient,
    ids: dict[str, str],
    poll_interval_s: float = 1.0,
) -> dict[str, Any]:
    """Test: Bath fill start to exit (full cycle with immediate exit)."""
    print("\n=== TEST: Bath Fill Start to Exit (Full Cycle) ===")
    
    # Turn on bath fill
    print("1. Starting bath fill...")
    start_time = time.monotonic()
    await rest.call_service("switch", "turn_on", {"entity_id": ids["bathfill_switch"]})
    
    # Wait briefly for switch to turn on
    await asyncio.sleep(2.0)
    
    # Immediately turn off
    print("2. Immediately turning off bath fill...")
    await rest.call_service("switch", "turn_off", {"entity_id": ids["bathfill_switch"]})
    
    # Wait for switch to turn off
    success, elapsed = await wait_for_state(rest, ids["bathfill_switch"], "off", timeout_s=120.0, poll_interval_s=poll_interval_s)
    if not success:
        return {"error": "Bath fill switch did not turn off", "elapsed": elapsed}
    
    exit_time = time.monotonic() - start_time
    print(f"   Full cycle (start to exit): {exit_time:.2f}s")
    
    return {
        "success": True,
        "total_time": exit_time,
        "poll_interval_s": poll_interval_s,
    }


async def test_fill_percent_update_frequency(
    rest: RESTClient,
    ids: dict[str, str],
    test_duration_s: float = 60.0,
    poll_intervals: list[float] = [0.5, 1.0, 2.0, 3.0],
) -> dict[str, Any]:
    """Test: Fill percent update frequency with different poll intervals."""
    print("\n=== TEST: Fill Percent Update Frequency ===")
    
    results = {}
    
    for poll_interval in poll_intervals:
        print(f"\nTesting poll interval: {poll_interval}s")
        
        # Ensure bath fill is active
        state = await get_entity_state(rest, ids["bathfill_switch"])
        if not state or state.state != "on":
            print("  Bath fill not active, skipping...")
            continue
        
        # Monitor fill percent
        readings = await monitor_fill_percent(rest, ids["bathfill_progress"], test_duration_s, poll_interval)
        
        if readings:
            update_count = len(readings)
            update_rate = update_count / test_duration_s
            fill_changes = sum(1 for i in range(1, len(readings)) if readings[i][1] != readings[i-1][1])
            
            results[f"poll_{poll_interval}s"] = {
                "poll_interval_s": poll_interval,
                "update_count": update_count,
                "update_rate_per_sec": update_rate,
                "fill_changes": fill_changes,
                "readings": readings[:10],  # First 10 readings
            }
            
            print(f"  Updates: {update_count} ({update_rate:.2f}/sec), Fill changes: {fill_changes}")
        else:
            print("  No readings collected")
        
        await asyncio.sleep(2.0)  # Brief pause between tests
    
    return results


async def discover_entity_ids(rest: RESTClient, entry_id_prefix: str) -> dict[str, str]:
    """Discover entity IDs from registry."""
    registry = await rest.get_entity_registry()
    ids = {}
    
    for entity in registry:
        unique_id = entity.get("unique_id", "")
        entity_id = entity.get("entity_id", "")
        
        if entry_id_prefix.lower() in unique_id.lower():
            if "bathfill" in unique_id and "switch" in unique_id:
                ids["bathfill_switch"] = entity_id
            elif "bathfill" in unique_id and "progress" in unique_id:
                ids["bathfill_progress"] = entity_id
            elif "bathfill" in unique_id and "status" in unique_id:
                ids["bathfill_status"] = entity_id
    
    return ids


async def main():
    parser = argparse.ArgumentParser(description="Test bath fill timing and optimization")
    parser.add_argument("--url", default="http://localhost:8123", help="Home Assistant URL")
    parser.add_argument("--token", help="Long-lived access token (or set HA_TOKEN env var)")
    parser.add_argument("--entry-id-prefix", required=True, help="Config entry ID prefix (e.g., 01KFD1XQ)")
    parser.add_argument("--test", choices=["start_to_complete", "complete_to_exit", "start_to_exit", "fill_percent", "all"], default="all", help="Test to run")
    parser.add_argument("--poll-interval", type=float, default=1.0, help="Poll interval in seconds")
    parser.add_argument("--output", help="Output JSON file for results")
    
    args = parser.parse_args()
    
    token = args.token or os.getenv("HA_TOKEN")
    if not token:
        print("Error: --token required or set HA_TOKEN environment variable")
        sys.exit(1)
    
    async with Client(args.url, token) as client:
        rest = client.rest
        
        # Discover entity IDs
        print("Discovering entity IDs...")
        ids = await discover_entity_ids(rest, args.entry_id_prefix)
        
        if not ids:
            print(f"Error: No entities found for entry ID prefix: {args.entry_id_prefix}")
            print("Available entities:")
            registry = await rest.get_entity_registry()
            for entity in registry[:20]:
                print(f"  {entity.get('entity_id')} - {entity.get('unique_id')}")
            sys.exit(1)
        
        print(f"Found entities: {ids}")
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "entry_id_prefix": args.entry_id_prefix,
            "poll_interval_s": args.poll_interval,
        }
        
        # Run tests
        if args.test in ("start_to_complete", "all"):
            results["start_to_complete"] = await test_bathfill_start_to_complete(rest, ids, args.poll_interval)
        
        if args.test in ("complete_to_exit", "all"):
            results["complete_to_exit"] = await test_complete_to_exit(rest, ids, args.poll_interval)
        
        if args.test in ("start_to_exit", "all"):
            results["start_to_exit"] = await test_bathfill_start_to_exit(rest, ids, args.poll_interval)
        
        if args.test in ("fill_percent", "all"):
            results["fill_percent_frequency"] = await test_fill_percent_update_frequency(rest, ids, test_duration_s=60.0, poll_intervals=[0.5, 1.0, 2.0, 3.0])
        
        # Output results
        if args.output:
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to: {args.output}")
        else:
            print("\n=== RESULTS ===")
            print(json.dumps(results, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
