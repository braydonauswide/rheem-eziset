# Agents Guide

Welcome! This repo is a custom Home Assistant integration for the Rheem EziSET water heater. It targets HA ≥ 2026.1.0 and is branded "Rheem EziSET HomeKit" (display name only; domain stays `rheem_eziset`).

## What this integration does
- Talks to the Rheem EziSET local HTTP API to read temps/modes and control the heater.
- Adds Bath Fill presets and an Exit Bath Fill control that can be exposed cleanly to Apple Home via HomeKit Bridge.

## Critical constraints (must-read)
- **Device DoS protection:** The heater refuses connections if it gets > ~1 request/sec; sustained bursts can require a power-cycle. Enforce a global per-device scheduler: **MIN_REQUEST_GAP ≈ 1.5s** between any requests, one in-flight request at a time, backoff on repeated failures.
- **Single API instance:** All entities must use the shared `coordinator.api` so the rate limiter is truly global.
- **Session safety:** Always release control (`heatingCtrl=0`) after `setTemp`/`setSessionTimer`. Bath fill start/stop must honor `sid`, `mode`, `flow`, and `sTimeout` prechecks.
- **Bath Fill completion:** Default is **no auto-exit** at mode 35; the active preset stays ON until the user turns it OFF (safer if a tap is open). An Exit Bath Fill switch exists for "stuck" cases.

## Architecture Overview

### Global Request Queue System
All HTTP requests (both polls and writes) go through a unified `asyncio.Queue` managed by `_request_worker()` in `api.py`. This ensures:
- **Serialization**: Only one request in-flight at a time
- **Rate limiting**: `MIN_REQUEST_GAP = 1.5s` between request start times
- **Timeout handling**: `REQUEST_TIMEOUT = 25.0s` (increased from 15s for reliability)
- **Soft failures**: Non-critical requests (e.g., session release) use `allow_read_timeout=True` to treat timeouts as soft failures

**Key components:**
- `_request_queue`: `asyncio.Queue` for all HTTP requests
- `_request_worker()`: Background task that processes requests sequentially
- `_enqueue_request()`: Public API to enqueue any HTTP request
- `_next_request_at`: Monotonic timestamp tracking when next request can start

### Single API Instance
- Each device has exactly one `RheemEziSETApi` instance stored in `coordinator.api`
- All entities access this shared instance: `self.coordinator.api`
- This ensures the rate limiter is truly global across all entities for a device

### Request Serialization
- Polls (`getInfo.cgi`) and writes (`ctrl.cgi`, `set.cgi`) all go through the same queue
- The worker honors `MIN_REQUEST_GAP` and respects lockout/cooldown periods
- Control operations can be deferred if device is in lockout or cooldown

## Bath Fill State Management

### Active Detection (`_bathfill_active`)
Returns `True` when the device reports bath fill is active:
- Device mode is in `{20, 25, 30, 35}` (bath fill modes), OR
- `bathfillCtrl == 1` (device confirms bath fill control is active)

**Note:** Mode 35 is "Idle (Bath Fill Mode Complete)" - the device is waiting for exit but still considers bath fill active.

### Engaged Detection (`_bathfill_engaged`)
Returns `True` when bath fill should be considered engaged (including completion until exit):
1. Device is actively in bath fill mode (`_bathfill_active` returns `True`), OR
2. State latches indicate a known bath fill session (`_bathfill_latched` or `_completion_latched`), OR
3. Completion markers are present (`mode == 35` or `state == 3`) unless explicitly ignored

**Why latches?** The device may report stale completion markers (`state == 3`) for a long time after exit. Latches track when we've actually started/stopped a session, providing a more accurate view than device state alone.

### State Latches

**`_bathfill_latched`**
- Set to `True` when bath fill start succeeds
- Cleared to `False` when bath fill exit succeeds and device returns to idle
- Purpose: Track that we've initiated a bath fill session, even if device state is unclear

**`_completion_latched`**
- Set to `True` when completion is detected (`mode == 35` or `state == 3`)
- Cleared to `False` when device returns to idle (`mode == 5`, `flow == 0`) and bath fill is no longer active
- Purpose: Track that bath fill has completed, even if device keeps reporting completion state

**`_ignore_completion_state`**
- Set to `True` on successful bath fill exit
- Prevents stale completion markers (`state == 3`) from making `_bathfill_engaged` return `True`
- Reset to `False` when a new bath fill starts
- Purpose: Suppress persistent completion markers after exit

### Completion Handling
- When completion is detected (`mode == 35` or `state == 3`), `fillPercent` is forced to `100.0` if device reports less
- When device returns to idle and bath fill is not engaged, `fillPercent` is reset to `0.0`
- If `_ignore_completion_state` is `True` and device is not actively filling, `fillPercent` is reset to `0.0` for UI consistency

**Important:** All entities should use `api._bathfill_engaged(data)` for consistent engagement detection, not manual mode/state checks.

## Session Management

### Session ID (`_owned_sid`)
- Tracks the session ID for control operations
- Set when we successfully acquire a session (e.g., `setTemp`, `bathfillCtrl=1`)
- Cleared when session is released or another controller takes over
- Used in `getInfo.cgi?sid={_owned_sid}` for sid-aware polling

### Session Timeout (`sTimeout`)
- Device session timeout in seconds (0 = no active session)
- Must be `0` or `None` before starting bath fill or setting temperature
- Default session timer: **600s** (prevents bath fill timeout during user actions like opening/closing tap)

### Session Release
- Always call `heatingCtrl=0` after `setTemp`/`setSessionTimer` operations
- Release requests use `allow_read_timeout=True` to treat timeouts as soft failures
- Best-effort: If release fails, we continue (device will timeout the session naturally)

### Session Safety Prechecks
Before starting bath fill or setting temperature:
- `flow == 0`: No water flow (tap closed)
- `mode == 5`: Device is idle
- `sTimeout == 0`: No active session held by another controller

## Auto-Exit Logic

### Trigger Conditions
Auto-exit triggers when **all** of these are true:
- `_auto_exit_enabled` is `True` (user-controlled via "Auto Exit Bath Fill" switch)
- Bath fill is engaged (`_bathfill_engaged` returns `True`)
- Either:
  - `fillPercent >= 80.0` AND `flow == 0`, OR
  - Completion reached (`mode == 35` or `state == 3`) AND `flow == 0`

### Duplicate Prevention
- Uses `_auto_exit_latched_sid` to track which session ID we've already attempted to exit
- Prevents duplicate exit attempts for the same session
- Cleared when bath fill is no longer engaged

### User Control
- Default: Auto-exit is **disabled** (safer if tap is open)
- User can enable via "Auto Exit Bath Fill" switch
- State persists across HA restarts (uses `RestoreEntity`)

## Rate Limiting & Device Protection

### Request Throttling
- `MIN_REQUEST_GAP = 1.5s`: Minimum time between request start times
- Applies to **all** requests (polls and writes) through the unified queue
- Worker calculates `wait_s` to honor the gap before starting each request

### Timeout Handling
- `REQUEST_TIMEOUT = 25.0s`: Total timeout for HTTP requests (increased from 15s)
- Poll timeouts are treated as soft failures (device may be temporarily unresponsive)
- Control timeouts trigger cooldown periods to avoid hammering the device

### Lockout Detection
- After `LOCKOUT_CONSEC_FAILURES = 3` consecutive failures, device enters lockout
- Lockout uses exponential backoff: `COOLDOWN_SCHEDULE_S = [10, 30, 60, 180]` seconds
- `_lockout_until_monotonic`: Monotonic timestamp when lockout expires
- All requests are blocked during lockout (except best-effort session releases)

### Control Cooldown
- `_control_cooldown_until`: Additional cooldown for control operations after timeouts
- Set via `_apply_control_cooldown()` with jitter (base + random 0-50%)
- Prevents rapid retries of failed control operations
- Separate from lockout (can be active even if device isn't in full lockout)

### Soft Failures
- `allow_read_timeout=True`: For non-critical requests (e.g., session release)
- Timeout is logged but doesn't increment failure count or trigger lockout
- Allows graceful degradation when device is slow but not completely unresponsive

## Entity State Synchronization

### Null Safety
- `coordinator.data` can be `None` during initial setup or after coordinator failures
- **Always** use `(coordinator.data or {}).get(...)` when accessing data in properties
- `__init__.py` sets `coordinator.data = coordinator.data or {}` but this happens after entity `__init__` runs

### Consistent Engagement Detection
- All entities should call `api._bathfill_engaged(data)` instead of manual mode/state checks
- This ensures consistent behavior across all entities
- Entities using this: `binary_sensor.py`, `switch.py`, `sensor.py`, `config_flow.py`

### Fast Refresh
- Coordinator schedules fast polls after control operations via `async_schedule_fast_refresh()`
- Bounded fast-refresh loop runs while writes are pending or within a settle window
- Prevents entities from showing stale state after user actions

### Entity Updates
- Entities implement `_handle_coordinator_update()` to react to data changes
- Switch entities use optimistic state updates during confirmation windows
- Sensors update immediately when coordinator data changes

## Debug Logging

### Format
- NDJSON (newline-delimited JSON) format in `debug.log`
- File path: `/config/custom_components/rheem_eziset/debug.log` (absolute path for HA container)
- Cleared on integration startup (via `_init_debug_log()`)
- 24-hour retention: Entries older than 24 hours are pruned automatically

### Structured Fields
Each log entry contains:
- `isoTime`: ISO 8601 timestamp (UTC)
- `integrationVersion`: Version from `manifest.json` (via `manifest_version()`)
- `sessionId`: Session identifier (currently `"debug-session"`)
- `runId`: Run identifier (currently `"pre-fix"`)
- `hypothesisId`: Hypothesis identifier (e.g., `"action-trace"`, `"H1-rate-limit-or-network"`)
- `location`: Code location (e.g., `"api.py:bathfill_start:ok"`)
- `message`: Log message (e.g., `"HTTP ok"`, `"action trace"`)
- `data`: Structured data (varies by log type)
- `timestamp`: Unix timestamp in milliseconds

### Action Traces
Control operations log structured traces:
- `action`: Operation name (e.g., `"bathfill_start"`, `"set_temp"`)
- `stage`: Stage of operation (e.g., `"start"`, `"ok"`, `"error"`)
- `control_seq_id`: Sequence identifier for grouping related operations
- `snapshot`: Device state snapshot at time of log
- `entity_snapshot`: Entity state snapshot (if `entity_id` provided)

### Log Reading
- Test runner uses `DebugLogTail` class for incremental reading
- Tracks `_last_ts_ms` and `_last_count_at_ts` to read only new entries
- Handles log rotation/clearing by resetting cursor when timestamps decrease

## Testing

### Test Runner
- Location: `scripts/ha_self_test.py`
- Purpose: Local development testing against running HA instance (e.g., Docker on port 8123)
- Authentication: Uses Long-Lived Access Token (LLAT)

### Test Suites
- `discover`: List all entities for a config entry
- `core`: Run both control and service tests
- `core_controls`: Temperature, session timer, bath fill start/cancel
- `core_services`: Preset service operations
- `extended`: Physical/interactive scenarios
- `flow_block`: Test that flow>0 blocks bath fill start
- `completion_progress_reset`: Test auto-exit and progress reset
- `no_auto_exit_progress`: Test manual exit and progress reset

### Interactive Mode
- `--interactive` flag enables user-guided scenarios
- Waits for user actions (e.g., opening/closing tap) instead of skipping
- Uses longer timeouts (900s) for user actions

### Flow-Aware Waiting
- `_wait_for_idle_controls()` detects active flow and waits instead of failing
- Extends timeout while flow is active
- Prevents false failures when someone is using hot water during automated tests

### Entity ID Resolution
- Test runner resolves entity IDs from registry using `unique_id` patterns
- Never hard-code entity IDs (they can change based on device name)
- `EntityIds` dataclass centralizes all entity ID references

## Common Pitfalls

### Accessing `coordinator.data` Without Null Guard
**Problem:** `coordinator.data` can be `None` during entity `__init__` or after coordinator failures.

**Solution:** Always use `(coordinator.data or {}).get(...)` in properties.

**Example:**
```python
# ❌ Bad
mode_val = to_int(self.coordinator.data.get("mode"))

# ✅ Good
mode_val = to_int((self.coordinator.data or {}).get("mode"))
```

### Changing `unique_id` Formats
**Problem:** Changing `unique_id` formats breaks HomeKit pairing and entity registry.

**Solution:** Never change existing `unique_id` formats. Only add new entities with new formats.

### Not Using `api._bathfill_engaged()`
**Problem:** Manual mode/state checks lead to inconsistent state reporting across entities.

**Solution:** Always use `api._bathfill_engaged(data)` for engagement detection.

**Example:**
```python
# ❌ Bad
engaged = mode_val in (20, 25, 30, 35) or state_val == 3

# ✅ Good
engaged = bool(api._bathfill_engaged(data))
```

### Hard-Coding Entity IDs in Tests
**Problem:** Entity IDs can change based on device name or HA configuration.

**Solution:** Resolve entity IDs from registry using `unique_id` patterns.

**Example:**
```python
# ❌ Bad
await rest.call_service("switch", "turn_on", {"entity_id": "switch.bath_fill"})

# ✅ Good
await rest.call_service("switch", "turn_on", {"entity_id": ids.bathfill_v2_switch})
```

### Not Releasing Sessions
**Problem:** Failing to release sessions can leave device in locked state.

**Solution:** Always call `heatingCtrl=0` after control operations, even if operation fails.

### Ignoring Device Rate Limits
**Problem:** Sending requests too fast can cause device lockout requiring power-cycle.

**Solution:** All requests must go through the unified queue which enforces `MIN_REQUEST_GAP`.

## File Structure

### Core Files
- **`api.py`**: HTTP client, rate limiting, control logic, state management, bath fill operations
- **`coordinator.py`**: Data polling, fast refresh scheduling, coordinator lifecycle
- **`__init__.py`**: Integration setup, service registration, platform loading

### Entity Platforms
- **`water_heater.py`**: Main water heater entity (temperature control)
- **`sensor.py`**: Various sensors (flow, status, mode, bath fill progress, etc.)
- **`binary_sensor.py`**: Binary sensors (bath fill active, connectivity problem, etc.)
- **`switch.py`**: Switches (bath fill presets, exit control, auto-exit toggle)
- **`number.py`**: Number entity (session timeout)
- **`select.py`**: Select entity (bath profile v2)

### Configuration
- **`config_flow.py`**: Configuration and options UI flow
- **`const.py`**: Constants (domain, mode maps, status maps, icons)
- **`manifest.json`**: Integration metadata (version, domain, dependencies)
- **`manifest.py`**: Helper to read version from manifest.json

### Utilities
- **`entity.py`**: Base entity class with coordinator integration
- **`util.py`**: Utility functions (`to_int`, `to_float`, `is_one`)

### Testing
- **`scripts/ha_self_test.py`**: Automated test runner for local development

## Key Files Reference

| File | Purpose |
|------|---------|
| `custom_components/rheem_eziset/api.py` | HTTP client, rate limiting, control helpers, bath fill state management |
| `custom_components/rheem_eziset/coordinator.py` | Polling/coordinator, fast refresh scheduling |
| `custom_components/rheem_eziset/*.py` | Entity platforms (water heater, sensors, numbers, switches) |
| `custom_components/rheem_eziset/config_flow.py` | Config/options flow (preset slots, scan interval) |
| `custom_components/rheem_eziset/manifest.json` | Display name, version, domain |
| `hacs.json` | HACS metadata |
| `scripts/ha_self_test.py` | Automated test runner |

## HomeKit Guidance
- Expose only: preset switches + Exit Bath Fill (momentary switch). Keep diagnostic sensors out of HomeKit to reduce clutter and stay under the 150 accessory limit.
- Keep `unique_id`s stable; avoid renaming entity_ids after pairing to HomeKit.
- Recommend a dedicated HomeKit Bridge filtered to just the bath-fill entities; use `entity_config` `type: faucet` if you want faucet icons.

## Dev Workflow (Safe Testing)
- Dependencies: see `requirements.txt` (target HA 2025.8.x). Use a local venv and keep it git-ignored.
- Lint/format: `.ruff.toml` configured; caches are ignored in `.gitignore`.
- Testing with a live device: throttle yourself (1 req/sec max). If the device stops responding, wait for cooldown; power-cycle if needed.
- Sample HA config: `config/configuration.yaml` is the only tracked file under `config/`; everything else in `config/` should remain ignored.

## Release/Upgrade Notes
- Display name is "Rheem EziSET HomeKit"; **do not** change the domain or folder name.
- When adding new entities, never change existing `unique_id` formats.
- Version is managed in `manifest.json`; use `manifest_version()` from `manifest.py` for runtime version.