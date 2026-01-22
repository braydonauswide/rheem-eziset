# API Documentation

This document describes the public API methods available in the Rheem EziSET integration.

## Rate Limiting

**Critical Constraint:** The device has DoS protection and will refuse connections if it receives more than ~1 request/second. Sustained bursts can cause lockout requiring a power-cycle.

**Implementation:**
- `MIN_REQUEST_GAP = 1.5s`: Minimum time between request start times
- Applies to ALL requests (polls and writes) through a unified queue
- Global rate limiter ensures no entity can bypass the limit

## Core API Class

### `RheemEziSETApi`

Main API class for device communication. All entities share a single instance via `coordinator.api`.

#### `async_get_data() -> dict[str, Any]`

Fetches and merges device data from multiple endpoints with caching.

**Returns:** Merged dictionary containing:
- `getInfo.cgi` data (always fetched, critical)
- `getParams.cgi` data (cached for 60 minutes, non-critical)
- `version.cgi` data (cached for 24 hours, non-critical)

**Behavior:**
- Returns cached data if write lock is held (control operation in progress)
- Performs health check after lockout expires
- Uses graceful degradation: non-critical endpoint failures don't trigger lockout
- Only critical endpoint (`getInfo.cgi`) failures can cause lockout

**Raises:**
- `HomeAssistantError`: If device is in lockout or data is invalid

#### `async_set_temp(water_heater, temp: int, *, allow_bathfill_override: bool = False, ...) -> None`

Sets the water heater target temperature.

**Parameters:**
- `water_heater`: WaterHeaterEntity instance (can be None)
- `temp`: Target temperature in Celsius (integer)
- `allow_bathfill_override`: If True, allows temp change during active bath fill
- `control_seq_id`: Optional identifier for logging
- `origin`: Optional origin identifier ("user", "system", etc.)
- `entity_id`: Optional entity ID for logging

**Validation:**
- Checks device min/max temperature limits
- Rejects if bath fill is active (unless `allow_bathfill_override=True`)
- Coerces float inputs to int

**Raises:**
- `ConditionErrorMessage`: If validation fails (type: "minimum_temperature", "maximum_temperature", "bathfill_active")

#### `async_set_session_timer(number, session_timer: int) -> None`

Sets the session timeout for control operations.

**Parameters:**
- `number`: NumberEntity instance
- `session_timer`: Timeout in seconds (60-900)

**Validation:**
- Minimum: 60 seconds
- Maximum: 900 seconds

**Raises:**
- `ConditionErrorMessage`: If validation fails (type: "minimum_session_timer", "maximum_session_timer")

#### `async_start_bath_fill(temp: int, vol: int, *, origin: str | None = None, entity_id: str | None = None) -> None`

Starts a bath fill operation with specified temperature and volume.

**Parameters:**
- `temp`: Target temperature in Celsius (integer)
- `vol`: Target volume in liters (integer)
- `origin`: Optional origin identifier
- `entity_id`: Optional entity ID for logging

**Validation:**
- Checks device bath fill temp/vol limits (`bathtempMin/Max`, `bathvolMin/Max`)
- Prechecks: device must be idle (mode=5, flow=0, sTimeout=0)

**State Machine:**
1. Precheck device state
2. Acquire session (set session timer to 600s default)
3. Start bath fill (set `bathfillCtrl=1` with temp/vol)
4. Verify device entered bath fill mode
5. Set `_bathfill_latched=True`

**Raises:**
- `ConditionErrorMessage`: If validation fails or precheck fails

#### `async_cancel_bath_fill(*, origin: str | None = None, entity_id: str | None = None) -> None`

Cancels an active bath fill operation.

**Session Reacquisition:**
- If we own the session (`_owned_sid`), use it directly
- If no session ID, try to reacquire via `ctrl.cgi?sid=0&bathfillCtrl=0`
- Some firmware returns `sid=0` for "no-sid" cancel; treat as best-effort

**State Cleanup:**
- Clears `_bathfill_latched` and `_completion_latched` flags
- Sets `_ignore_completion_state=True` to suppress stale completion markers
- Restores previous temperature if `_pending_restore_temp` is set

**Raises:**
- `ConditionErrorMessage`: If cancel fails or device still active after cancel

## Helper Methods

### `_bathfill_active(data: Mapping[str, Any] | None = None) -> bool`

Returns `True` if device reports bath fill is active.

**Checks:**
- Device mode in `{20, 25, 30, 35}` (bath fill modes)
- OR `bathfillCtrl == 1` (device confirms control is active)

### `_bathfill_engaged(data: Mapping[str, Any] | None = None) -> bool`

Returns `True` if bath fill should be considered engaged (including completion until exit).

**Checks:**
1. Device is actively in bath fill mode (`_bathfill_active` returns `True`), OR
2. State latches indicate a known bath fill session (`_bathfill_latched` or `_completion_latched`), OR
3. Completion markers are present (`mode == 35` or `state == 3`) unless explicitly ignored

**Why latches?** The device may report stale completion markers for a long time after exit. Latches track when we've actually started/stopped a session.

## Error Types

### `ConditionErrorMessage`

Raised for user-actionable errors (validation failures, state conflicts).

**Common types:**
- `minimum_temperature`: Temperature below device minimum
- `maximum_temperature`: Temperature above device maximum
- `bathfill_active`: Operation blocked because bath fill is active
- `invalid_sTimeout`: Cannot take control (another controller has session)
- `invalid_mode`: Cannot take control (device not idle)
- `invalid_flow`: Cannot take control (water flow detected)

### `HomeAssistantError`

Raised for system-level errors (lockout, invalid data, connection failures).

**Common messages:**
- Device lockout suspected (with remaining time)
- Invalid getInfo payload
- Request failed for path

## Rate Limiting Details

### Request Throttling

All requests go through `_enqueue_request()` which:
1. Queues the request in `_request_queue`
2. Worker thread processes queue serially
3. Calculates `wait_s` to honor `MIN_REQUEST_GAP` (1.5s)
4. Applies additional delays for:
   - Control cooldown (after control failures)
   - Poll backoff (after poll timeouts)
   - Control backoff (after control timeouts)
   - Connection backoff (after connection errors, shorter than device errors)
   - Write lock delay (0.5s when write lock is held)
   - Session delay (0.5s when another controller has session, but NOT when we own it)

### Lockout Mechanism

After `LOCKOUT_CONSEC_FAILURES = 3` consecutive failures:
- Device enters lockout with exponential backoff: `[10s, 30s, 60s, 180s]`
- All requests blocked during lockout (except best-effort session releases)
- After lockout expires, health check probe is performed
- If health check succeeds, normal operations resume
- If health check fails, lockout extends by one level

### Connection vs Device Errors

**Connection errors** (`ClientConnectorError`):
- Shorter backoff: 5s, 10s, 20s max
- Don't count toward device lockout
- Indicate network issues, not device problems

**Device errors** (timeouts, HTTP errors):
- Longer backoff schedules
- Count toward device lockout
- Indicate device DoS/backpressure or device problems

## Session Management

### Session ID (`_owned_sid`)

Tracks the session ID for control operations:
- Set when we successfully acquire a session
- Cleared when session is released or another controller takes over
- Used in `getInfo.cgi?sid={_owned_sid}` for sid-aware polling

### Session Timeout (`sTimeout`)

Device session timeout in seconds (0 = no active session):
- Must be `0` or `None` before starting bath fill or setting temperature
- Default session timer: **600s** (prevents bath fill timeout during user actions)

### Session Release

Always call `heatingCtrl=0` after `setTemp`/`setSessionTimer` operations:
- Release requests use `allow_read_timeout=True` to treat timeouts as soft failures
- Best-effort: If release fails, we continue (device will timeout the session naturally)

## Caching Strategy

### Endpoint Caching

- **getInfo.cgi**: Always fetched (critical, no cache)
- **getParams.cgi**: Cached for 60 minutes (non-critical)
- **version.cgi**: Cached for 24 hours (non-critical)

### Graceful Degradation

If non-critical endpoints fail:
- Use cached data
- Log warning but continue operation
- Don't enter lockout
- Track endpoint health separately

Only critical endpoint (`getInfo.cgi`) failures can cause lockout.

## Fast Refresh

After control operations, coordinator schedules fast refresh:
- Bounded loop: runs while writes pending OR within settle window (10s)
- Max iterations: 20
- Uses lightweight `async_get_info_only()` (only getInfo.cgi)
- Respects rate limits via `_next_request_at`
- Only updates entities if data actually changed (optimization)

This ensures entities reflect device state changes quickly without hammering the device.
