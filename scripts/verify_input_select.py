#!/usr/bin/env python3
"""Quick verification script for input_select entity existence."""
import asyncio
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from scripts.ha_self_test import HaRestClient, HaWsClient, ensure_rheem_entry
except ImportError:
    print("Error: Could not import test utilities")
    sys.exit(1)


async def verify_input_select_entity():
    """Verify the input_select entity exists."""
    ha_url = os.environ.get("HA_URL", "http://localhost:8123")
    token = os.environ.get("HA_TOKEN")
    heater_host = os.environ.get("HEATER_HOST")
    
    if not token:
        print("Error: HA_TOKEN environment variable not set")
        print("Set it with: export HA_TOKEN='your-long-lived-access-token'")
        sys.exit(1)
    
    rest = HaRestClient(ha_url, token)
    
    # Test HA connection
    try:
        await rest.get_json("/api/")
        print("✓ Connected to Home Assistant")
    except Exception as e:
        print(f"✗ Failed to connect to HA: {e}")
        sys.exit(1)
    
    # Get rheem entry (discover heater_host if not provided)
    try:
        async with HaWsClient(ha_url, token) as ws:
            # If heater_host not provided, try to find existing entry
            if not heater_host:
                entries_result = await ws.call({"type": "config_entries/get"})
                # API may return list directly or dict with result
                entries = entries_result if isinstance(entries_result, list) else entries_result.get("result", [])
                rheem_entries = [e for e in entries if e.get("domain") == "rheem_eziset"]
                if rheem_entries:
                    entry = rheem_entries[0]
                    print(f"✓ Found existing rheem_eziset entry: {entry.get('entry_id')}")
                else:
                    print("✗ No rheem_eziset entry found and HEATER_HOST not provided")
                    print("  Set HEATER_HOST or create a config entry first")
                    sys.exit(1)
            else:
                entry = await ensure_rheem_entry(ws, heater_host)
            entry_id = entry.get("entry_id") or entry.get("entryId")
            if not entry_id:
                print("✗ Could not determine entry_id from entry")
                sys.exit(1)
            
            entry_id_prefix = entry_id[:8].lower()
            expected_entity_id = f"input_select.rheem_{entry_id_prefix}_bath_profile"
            
            print(f"✓ Found rheem_eziset entry: {entry_id}")
            print(f"  Expected entity_id: {expected_entity_id}")
            
            # Verify entity exists
            try:
                state = await rest.get_state(expected_entity_id)
                options = state.get("attributes", {}).get("options", [])
                current_selection = state.get("state")
                
                print(f"✓ Entity exists: {expected_entity_id}")
                print(f"  Current selection: {current_selection}")
                print(f"  Options count: {len(options)}")
                if options:
                    print(f"  Options: {options[:3]}{'...' if len(options) > 3 else ''}")
                
                # Verify it's a valid input_select
                if state.get("entity_id") == expected_entity_id:
                    print("✓ Entity ID matches expected format")
                else:
                    print(f"⚠ Entity ID mismatch: {state.get('entity_id')} != {expected_entity_id}")
                
                # Check if selection is valid
                if current_selection in options:
                    print("✓ Current selection is valid")
                else:
                    print(f"⚠ Current selection '{current_selection}' not in options")
                
                return True
                
            except Exception as e:
                print(f"✗ Entity not found: {expected_entity_id}")
                print(f"  Error: {e}")
                
                # Try to find any input_select entities with "bath" or "profile"
                try:
                    all_states = await rest.get_json("/api/states")
                    matching = [
                        s for s in all_states
                        if s.get("entity_id", "").startswith("input_select.")
                        and ("bath" in s.get("entity_id", "").lower() or "profile" in s.get("entity_id", "").lower())
                    ]
                    if matching:
                        print(f"\nFound {len(matching)} similar input_select entities:")
                        for m in matching:
                            print(f"  - {m.get('entity_id')}")
                    else:
                        print("\nNo matching input_select entities found")
                except Exception:
                    pass
                
                return False
                
    except Exception as e:
        print(f"✗ Failed to get entry: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(verify_input_select_entity())
    sys.exit(0 if success else 1)
