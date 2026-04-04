# Rheem EziSET Device HTTP API — Definitive Reference

**Discovered via:** tcpdump packet capture of the official iOS app (macOS mirror mode)
**Date:** 2026-04-04
**Confidence:** Empirical ground truth — every statement in this document was directly observed in packet captures, not inferred.

---

## Table of Contents

1. [Overview](#1-overview)
2. [All Endpoints](#2-all-endpoints)
3. [JSON Response Schemas](#3-json-response-schemas)
4. [Mode Reference](#4-mode-reference)
5. [State Reference](#5-state-reference)
6. [Session Management](#6-session-management)
7. [Regular Heating Flow](#7-regular-heating-flow)
8. [Bath Fill Flow](#8-bath-fill-flow)
9. [sTimeout Behaviour](#9-stimeout-behaviour)
10. [Temperature Limits](#10-temperature-limits)
11. [Error States — Mode 99](#11-error-states--mode-99)
12. [Known Gotchas](#12-known-gotchas)
13. [What NOT To Do](#13-what-not-to-do)

---

## 1. Overview

The Rheem EziSET is a gas instantaneous water heater with a Wi-Fi/powerline bridge. The bridge exposes a minimal HTTP/1.0 API on the local network.

### Network

| Property | Value |
|---|---|
| Protocol | HTTP/1.0 |
| Port | 80 (default) |
| Request method | GET only — no POST requests exist |
| Connection model | New TCP connection per request (HTTP/1.0 — no keep-alive) |
| Rate limiting | None observed — but device has DoS protection; do not exceed ~1 req/sec combined |
| Authentication | None |
| Encoding | Plain text query string parameters; JSON response bodies |

### Endpoints summary

There are exactly **three endpoints**:

| Endpoint | Purpose |
|---|---|
| `GET /getInfo.cgi` | Poll device state (no session) |
| `GET /getInfo.cgi?sid=<sid>` | Poll device state (with active session) |
| `GET /ctrl.cgi?<params>` | Open/close sessions, start/stop bath fill |
| `GET /set.cgi?<params>` | Adjust temperature or volume during an active session |
| `GET /clearAppError.cgi?<params>` | Clear a device error state (mode 99) |

No other endpoints exist. There are no POST endpoints, no firmware update endpoints, and no undocumented paths observed.

---

## 2. All Endpoints

### 2.1 `GET /getInfo.cgi`

Poll device state without a session. Used when no session is open (mode 0 or 5) or as a fallback.

**Request:** No parameters required.

**Response:** Schema B (see Section 3).

**Example:**
```
GET /getInfo.cgi HTTP/1.0
```

---

### 2.2 `GET /getInfo.cgi?sid=<sid>`

Poll device state with an active session. This is the primary poll used during both heating and bath fill sessions.

**Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `sid` | integer | Session ID assigned by device on session open |

**Response:** Schema A (during bath fill) or Schema B (if session has expired/no bath fill active).

**Special case:** If the session has expired (e.g., after mode 99/appErrCode=8), the device returns:
```json
{"reason": "Session invalid!"}
```
This is not a JSON error — it is the device's way of indicating the sid is no longer valid.

**Example:**
```
GET /getInfo.cgi?sid=1554 HTTP/1.0
```

---

### 2.3 `GET /ctrl.cgi?<params>`

Opens or closes a control session. Also used to start and stop bath fill.

**Response:** Schema C (see Section 3).

#### Open a regular heating session

```
GET /ctrl.cgi?sid=0&heatingCtrl=1
```

| Parameter | Value | Description |
|---|---|---|
| `sid` | `0` | Always 0 when opening a new session |
| `heatingCtrl` | `1` | Request heating control |

Response includes the new `sid` assigned by the device.

#### Close a regular heating session

```
GET /ctrl.cgi?sid=<sid>&heatingCtrl=0
```

Response: `{"sid":0,"sTimeout":0,"heatingCtrl":0,"bathfillCtrl":0}`

#### Start a bath fill session

```
GET /ctrl.cgi?sid=0&bathfillCtrl=1&setBathTemp=<temp>&setBathVol=<vol>
```

| Parameter | Value | Description |
|---|---|---|
| `sid` | `0` | Always 0 when starting a new session |
| `bathfillCtrl` | `1` | Start bath fill |
| `setBathTemp` | integer (°C) | Requested bath temperature |
| `setBathVol` | integer (L) | Requested bath volume |

Response:
```json
{"sid": <new_sid>, "sTimeout": 300, "heatingCtrl": 0, "bathfillCtrl": 1, "reqbathtemp": <temp>, "reqbathvol": <vol>}
```

#### Exit a bath fill session

```
GET /ctrl.cgi?sid=<sid>&bathfillCtrl=0&setBathTemp=<temp>&setBathVol=<vol>
```

| Parameter | Value | Description |
|---|---|---|
| `sid` | active sid | The current session ID |
| `bathfillCtrl` | `0` | Exit/cancel bath fill |
| `setBathTemp` | integer (°C) | **Must be included** — echo of current bath temp setting |
| `setBathVol` | integer (L) | **Must be included** — echo of current bath volume setting |

Response: `{"sid":0,"sTimeout":0,"heatingCtrl":0,"bathfillCtrl":0}`

**Critical:** `setBathTemp` and `setBathVol` are required even on exit. Omitting them results in incorrect device behaviour. The iOS app always echoes the current settings on exit.

---

### 2.4 `GET /set.cgi?<params>`

Adjusts temperature or volume during an active session. This is the **only** correct endpoint for mid-session updates. Do not use `ctrl.cgi` for this purpose (see Section 13).

**This endpoint resets `sTimeout` to 300 on every call.**

#### Set bath temperature during session

```
GET /set.cgi?sid=<sid>&setBathTemp=<temp>
```

Response:
```json
{"bathtemp": <current>, "reqbathtemp": <new>, "sid": <sid>, "sTimeout": 300}
```

#### Set bath volume during session

```
GET /set.cgi?sid=<sid>&setBathVol=<vol>
```

Response:
```json
{"bathvol": <current>, "reqbathvol": <new>, "sid": <sid>, "sTimeout": 300}
```

#### Set regular heater temperature during session

```
GET /set.cgi?sid=<sid>&setTemp=<temp>
```

Note the parameter name is `setTemp` (not `setBathTemp`) for regular heating mode.

Response:
```json
{"temp": <current>, "reqtemp": <new>, "sid": <sid>, "sTimeout": 300}
```

The device confirms the change within 1–2 poll cycles (~200–400ms at 500ms polling).

---

### 2.5 `GET /clearAppError.cgi?<params>`

Clears a device error state. Used only for mode 99 / appErrCode=8 recovery.

```
GET /clearAppError.cgi?bathfill=0
```

| Parameter | Value | Description |
|---|---|---|
| `bathfill` | `0` | Signals bath fill error clear |

**Response:** Returns a full getInfo-style JSON response (not a simple ACK). Example on successful clear:
```json
{"heaterName": "Rheem", "mode": 0, ..., "appErrCode": 0, ...}
```

**Important behaviour:**
- Has **no effect** while `flow > 0` — the device ignores it
- After a successful call, the device transitions: mode=99 → mode=0 → (brief mode=99/appErrCode=64 flash) → mode=0 → mode=5
- Send exactly **one** call — do not retry

---

## 3. JSON Response Schemas

### Schema A — Bath Fill Session (modes 20/25/30/35)

Returned by `GET /getInfo.cgi?sid=<sid>` when a bath fill session is active.

| Field | Type | Description |
|---|---|---|
| `heaterName` | string | Device name (configurable via app, default `"Rheem"`) |
| `mode` | integer | Current device mode (see Section 4) |
| `bathtemp` | integer | Current bath temperature setpoint (°C) |
| `bathvol` | integer | Current bath volume setpoint (L) |
| `fillPercent` | integer | Bath fill progress (0–100, peaks at ~92–96%) |
| `flow` | number | Current water flow rate (L/min) |
| `wtemp` | integer | Current water outlet temperature (°C) |
| `intemp` | integer | Inlet/cold water temperature (°C) |
| `state` | integer | Device state within current mode (see Section 5) |
| `appErrCode` | integer | Application error code (0 = no error) |
| `sid` | integer | Current session ID |
| `sessionTimer` | integer | Seconds elapsed since session opened |
| `sTimeout` | integer | Seconds until session times out (see Section 9) |
| `heatingCtrl` | integer | Heating control flag (always 0 in observed traffic) |
| `bathfillCtrl` | integer | Bath fill control flag (1 = active) |

### Schema B — No Session (modes 0/5)

Returned by `GET /getInfo.cgi` (no sid) or when no session is active.

| Field | Type | Description |
|---|---|---|
| `heaterName` | string | Device name |
| `mode` | integer | Current device mode |
| `temp` | integer | Current heater temperature setpoint (°C) |
| `flow` | number | Current water flow rate (L/min) |
| `state` | integer | Device state |
| `appErrCode` | integer | Application error code (0 = no error) |
| `sTimeout` | integer | Always 0 when no session is active |
| `sessionTimer` | integer | Always 0 when no session is active |

### Schema C — ctrl.cgi Response

Returned by all `ctrl.cgi` calls.

| Field | Type | Description |
|---|---|---|
| `sid` | integer | New session ID (non-zero on open, 0 on close) |
| `sTimeout` | integer | Session timeout in seconds (300 on open, 0 on close) |
| `heatingCtrl` | integer | 1 if heating session open, 0 otherwise |
| `bathfillCtrl` | integer | 1 if bath fill session open, 0 otherwise |
| `reqbathtemp` | integer | (bath fill start only) Acknowledged bath temperature |
| `reqbathvol` | integer | (bath fill start only) Acknowledged bath volume |

---

## 4. Mode Reference

The `mode` field indicates the current operating state of the device.

### Regular Heating Modes

| Mode | Name | Description |
|---|---|---|
| `0` | Transitioning | Brief transient state during mode changes; not stable |
| `5` | Idle / Standby | No session open, no flow, heater ready |
| `10` | Heating Control Mode | Heating session open (`heatingCtrl=1`), tap closed |
| `15` | Heating (Conventional) | Heating session open, tap open, heater firing |
| `20` | Post-heat cooldown | After tap closes from mode=15; device cooling down |

### Bath Fill Modes

| Mode | Name | Description |
|---|---|---|
| `20` | Bath Fill — Waiting for Tap | Bath fill session open, tap not yet opened |
| `25` | Bath Fill — Heating/Waiting | Session open, tap detected (flow brief), heating beginning |
| `30` | Bath Fill — Active Fill | Tap open, water flowing, heater actively filling bath |
| `35` | Bath Fill — Complete | Target volume reached, tap closed, 120s ack window open |

**Note:** Mode `20` appears in both regular heating (post-heat cooldown) and bath fill (waiting for tap). Context is determined by whether a bath fill session is active.

### Error Mode

| Mode | Name | Description |
|---|---|---|
| `99` | Error / Locked | Device is in an error state; see Section 11 for types and recovery |

---

## 5. State Reference

The `state` field provides sub-state information within a mode.

| State | Description |
|---|---|
| `1` | Idle / Standby (within current mode) |
| `2` | Heater firing / burner active |
| `3` | Wind-down / tapering flow / exiting mode |

### State transitions during bath fill (mode=30)

- `state=2` — heater burner firing, water flowing at temperature
- `state=1` — burner off, blending continues (mixing valve active)
- `state=3` — fill nearly complete, flow tapering to target volume

### State at mode=35 (fill complete)

- `state=3` — fill is complete, device in wind-down, 120s ack window started

### State during mode transition (mode=0)

Brief sequence observed: `state=3` → `state=1` → `state=0` → device reaches mode=5.

---

## 6. Session Management

### Overview

The device uses a session model to prevent multiple controllers from conflicting. Only one session can be active at a time. The two session types are mutually exclusive.

| Session Type | Parameter | Purpose |
|---|---|---|
| `heatingCtrl` | `heatingCtrl=1` | Regular tap heating control; allows `setTemp` commands |
| `bathfillCtrl` | `bathfillCtrl=1` | Bath fill control; allows `setBathTemp`/`setBathVol` commands |

### Session ID (sid)

- When opening a new session, always send `sid=0` — the device assigns and returns a new sid.
- The returned sid must be included in all subsequent requests while that session is open.
- Sid values are positive integers (e.g., 1554). The exact values are device-assigned and opaque.
- Polling with the correct sid resets `sTimeout` on every poll, keeping the session alive indefinitely as long as polling continues.

### heatingCtrl Session

- Open: `ctrl.cgi?sid=0&heatingCtrl=1`
- Close: `ctrl.cgi?sid=<sid>&heatingCtrl=0`
- While open: device is in mode=10 (tap closed) or mode=15 (tap open)
- A heatingCtrl session cannot be opened while flow > 0 (device rejects it)
- Temperature commands use `set.cgi?sid=<sid>&setTemp=<N>`

### bathfillCtrl Session

- Open: `ctrl.cgi?sid=0&bathfillCtrl=1&setBathTemp=<N>&setBathVol=<N>`
- Close: `ctrl.cgi?sid=<sid>&bathfillCtrl=0&setBathTemp=<N>&setBathVol=<N>`
- While open: device progresses through modes 20 → 25 → 30 → 35
- Temperature/volume changes use `set.cgi?sid=<sid>&setBathTemp=<N>` or `set.cgi?sid=<sid>&setBathVol=<N>`
- **setBathTemp and setBathVol must be included even on close/exit**

### Session timeout (sTimeout)

- `sTimeout` starts at 300 seconds when a session is opened.
- Every `getInfo.cgi?sid=<sid>` poll resets sTimeout to 300 (it never counts down if you are polling).
- Every `set.cgi` call also resets sTimeout to 300.
- If polling stops, sTimeout counts down at ~1/sec and the session expires at 0.
- Exception: sTimeout is frozen at 300 during mode=30 (active fill) — see Section 9.
- Exception: sTimeout resets to 120 at mode=35 (fill complete) — see Section 9.

### Persistent session design (recommended for HA)

Since Home Assistant is the sole controller, the recommended design is:
- Open a `heatingCtrl=1` session on HA startup.
- Maintain it indefinitely via continuous polling — no session timeout occurs.
- Never release it unless switching to bath fill mode.
- On bath fill start: release heatingCtrl, open bathfillCtrl.
- On bath fill end: release bathfillCtrl, re-open heatingCtrl.

This eliminates session acquisition latency and ensures a valid sid is always available for immediate temperature commands.

### heatingCtrl field in requests

The `heatingCtrl` parameter appears in all `ctrl.cgi` requests but is always `0` except when explicitly opening a heating session. Its purpose beyond that is unknown. The iOS app always includes it.

---

## 7. Regular Heating Flow

### Step-by-step

1. **Open session:**
   ```
   GET /ctrl.cgi?sid=0&heatingCtrl=1
   ```
   Device returns new `sid`. Mode transitions to `10`.

2. **Poll state:**
   ```
   GET /getInfo.cgi?sid=<sid>
   ```
   Poll at ~500ms (1Hz). Resets sTimeout each call.

3. **Set temperature (tap closed, mode=10):**
   ```
   GET /set.cgi?sid=<sid>&setTemp=<N>
   ```
   No temperature limits apply while tap is closed. Values up to at least 47°C are accepted.

4. **User opens tap:**
   Device detects flow > 0. Mode transitions to `15`. Heater begins firing.

5. **Set temperature (tap open, mode=15):**
   ```
   GET /set.cgi?sid=<sid>&setTemp=<N>
   ```
   Temperature limits apply while flowing — see Section 10.

6. **Optional: release session mid-flow:**
   ```
   GET /ctrl.cgi?sid=<sid>&heatingCtrl=0
   ```
   - Device continues heating autonomously at the last set temperature.
   - `sTimeout` drops to 0, session is released.
   - Cannot open a new heatingCtrl session while flow > 0.
   - After tap closes (mode=5), a new session can be opened.

7. **User closes tap:**
   Mode returns to `10` (if session still held) or `5` (if session was released).

8. **Close session:**
   ```
   GET /ctrl.cgi?sid=<sid>&heatingCtrl=0
   ```
   Device returns `{"sid":0,"sTimeout":0,...}`. Mode transitions to `5`.

---

## 8. Bath Fill Flow

### Prerequisites (enforced by app/HA, not device)

- Tap closed (flow = 0)
- Device in mode=5 (idle)
- No existing session (`sTimeout` = 0)

### Step-by-step

1. **Start bath fill:**
   ```
   GET /ctrl.cgi?sid=0&bathfillCtrl=1&setBathTemp=<temp>&setBathVol=<vol>
   ```
   Device returns new `sid`, `sTimeout=300`, `bathfillCtrl=1`. Mode transitions to `20` or `25`.

2. **Poll state:**
   ```
   GET /getInfo.cgi?sid=<sid>
   ```
   Poll at ~500ms. Schema A is returned. `fillPercent` starts at 0.

3. **Mode=20 — Waiting for tap:**
   Device is ready, waiting for user to open the hot tap. `sTimeout` counts down from 300.

4. **Mode=25 — Tap detected / heating:**
   Flow briefly appears in the last mode=25 poll before transitioning to mode=30. Device begins heating to target temperature.

5. **Mode=30 — Active fill:**
   Water flowing. `fillPercent` increments. `sTimeout` is frozen at 300 (see Section 9).
   - `state=2`: burner firing
   - `state=1`: blending (burner off, mixing valve active)
   - `state=3`: flow tapering as target volume approached

6. **Optional: adjust temperature during fill:**
   ```
   GET /set.cgi?sid=<sid>&setBathTemp=<N>
   ```
   See Section 10 for safety restrictions on direction. Resets sTimeout to 300.

7. **Optional: adjust volume during fill:**
   ```
   GET /set.cgi?sid=<sid>&setBathVol=<N>
   ```
   Valid during mode=25 and mode=30.

8. **Mode=35 — Fill complete:**
   Target volume delivered. `sTimeout` resets to exactly `120` and begins counting down. `fillPercent` peaks at approximately 92–96% (not 100%) because the device accounts for water already in-pipe during wind-down.

9. **Exit within 120s window (clean exit):**
   ```
   GET /ctrl.cgi?sid=<sid>&bathfillCtrl=0&setBathTemp=<temp>&setBathVol=<vol>
   ```
   Device returns `{"sid":0,"sTimeout":0,"heatingCtrl":0,"bathfillCtrl":0}`. Mode transitions through `0` briefly then to `5`.

10. **If 120s window expires without exit:**
    Session is invalidated. Next `getInfo?sid=` returns `{"reason":"Session invalid!"}`. Sessionless getInfo returns mode=99 with appErrCode=8. See Section 11 for recovery.

### Cancel/abort during modes 20/25/30

To cancel before completion:
```
GET /ctrl.cgi?sid=<sid>&bathfillCtrl=0&setBathTemp=<temp>&setBathVol=<vol>
```

If the tap is open when cancel is sent (flow > 0), the device enters mode=99 with appErrCode=32. This is a self-clearing error — it clears the instant flow drops to 0 (user closes tap). No command is required to recover.

---

## 9. sTimeout Behaviour

`sTimeout` is the session timeout countdown in seconds. Its behaviour depends heavily on device mode.

| Mode | sTimeout Behaviour |
|---|---|
| `0` / `5` (no session) | Always `0` |
| `10` (heatingCtrl, tap closed) | Counts down ~1/sec; reset to 300 by every poll or set.cgi call |
| `15` (heatingCtrl, tap open) | Counts down ~1/sec; reset to 300 by every poll or set.cgi call |
| `20` (bath fill — waiting) | Counts down ~1/sec; reset to 300 by every poll or set.cgi call |
| `25` (bath fill — heating) | Counts down ~1/sec; reset to 300 by every poll or set.cgi call |
| `30` (bath fill — active fill) | **Frozen at exactly 300** — never ticks down during active water flow |
| `35` (bath fill — complete) | **Resets to exactly 120** when mode=35 is first entered, then counts down ~1/sec |
| `99` (error) | `0` (session already invalidated) |

### Implications

- During mode=30, continuous polling is not required to keep the session alive — sTimeout is frozen regardless.
- At mode=35, the controller has exactly 120 seconds to send the exit command. If polling stops (e.g., HA restart) and the 120s window expires, the device enters mode=99/appErrCode=8.
- The frozen sTimeout during mode=30 is a device safety feature: the fill cannot be accidentally abandoned mid-flow due to a timeout.

---

## 10. Temperature Limits

### Regular heating — while flowing (mode=15)

These limits are enforced by the **device firmware** — they are not app-level. The request reaches the device and is rejected.

| Limit | Value | Device response on violation |
|---|---|---|
| Maximum while flowing | **43°C** | `{"temp":43,"reqtemp":0,"reason":"Cannot set higher than 43 degC when water is flowing."}` |
| Minimum while flowing | **~37°C** | `reqtemp:0` (exact floor may vary by device) |

- With tap closed (mode=10): no limits enforced; accepted up to at least 47°C.
- These limits only apply during mode=15 (heatingCtrl session with tap open).

### Bath fill — temperature changes during session

The device itself accepts `setBathTemp` at any value at any time during an active session. However, the **app layer should enforce** a directional safety constraint:

| Mode | Allow temp increase? | Allow temp decrease? |
|---|---|---|
| `20` / `25` (waiting, no flow) | Yes | Yes |
| `30` (active fill, water flowing) | **No** — safety rule | Yes |
| `35` (fill complete) | No — session ending | No |

**Reason for mode=30 restriction:** Increasing bath temperature while hot water is already flowing at the target temperature could scald the user. The device does not enforce this — it is a required app-level safety constraint.

### Bath fill — volume limits

No device-enforced volume limits were observed. The iOS app allows user input; the device accepts whatever setBathVol value is provided.

---

## 11. Error States — Mode 99

Mode 99 is the device error/locked state. There are three distinct types, distinguished by `appErrCode`.

### Type 1 — appErrCode=32 — Cancel with tap open

| Property | Value |
|---|---|
| Trigger | Bath fill cancel command sent while `flow > 0` |
| Session state | Session invalidated by the cancel command |
| Recovery | **Self-clearing** — clears the instant `flow = 0` (user closes tap) |
| Command required | None |

**Behaviour:** When the user cancels bath fill while the hot tap is still open, the device enters mode=99/appErrCode=32. The moment the user closes the tap and flow drops to 0, the device automatically exits mode=99 and transitions to mode=5. No clearAppError.cgi call is needed or useful.

---

### Type 2 — appErrCode=8 — Session expired at mode=35

| Property | Value |
|---|---|
| Trigger | `sTimeout` counts to 0 while in mode=35 (120s ack window expired) |
| Session state | Session fully invalidated — `getInfo?sid=` returns `{"reason":"Session invalid!"}` |
| Recovery | Send exactly one `GET /clearAppError.cgi?bathfill=0` |

**Recovery sequence:**
1. Detect mode=99 + appErrCode=8 (from sessionless getInfo)
2. Send: `GET /clearAppError.cgi?bathfill=0`
3. Device returns: mode=0, appErrCode=0
4. Continue polling (sessionless)
5. Approximately 6 seconds later: **transient mode=99/appErrCode=64 flash** (see Type 3 below)
6. After the transient clears: mode=0 → mode=5
7. Total recovery time: approximately 7 seconds after clearAppError

**Do not:**
- Send clearAppError more than once
- Try to re-acquire a session during recovery
- Send ctrl.cgi during recovery

---

### Type 3 — appErrCode=64 — Transient post-recovery flash

| Property | Value |
|---|---|
| Trigger | Occurs ~6 seconds after successful clearAppError.cgi?bathfill=0 |
| Duration | Exactly one poll cycle (~500ms) |
| Recovery | None required — self-clears |

**This is not a new error.** It is a predictable transient that occurs during every Type 2 recovery. If your controller is in a "recovering from clearAppError" state, ignore any mode=99/appErrCode=64 observation. Treating it as a new error and sending another clearAppError.cgi will disrupt recovery.

**Full Type 2 recovery timeline:**

```
mode=99/appErrCode=8    → detected (session expired)
clearAppError.cgi sent  → response: mode=0/appErrCode=0
~500ms...               → mode=0 (polling)
...~5.5 seconds...      → mode=99/appErrCode=64 (one poll, IGNORE)
~500ms later            → mode=0
~1 second later         → mode=5 (device fully recovered)
```

### What the iOS app does during mode=99

- Does **not** send ctrl.cgi
- Does **not** try to re-acquire a session
- Does **not** send set.cgi
- Polls getInfo (sessionless) until user acknowledges
- When user taps exit: sends exactly one `clearAppError.cgi?bathfill=0`

---

## 12. Known Gotchas

### 12.1 fillPercent never reaches 100%

`fillPercent` peaks at approximately 92–96%, not 100%. The device accounts for the water volume that will flow out during the wind-down phase (after the burner cuts off, water is still in the pipes and flowing under pressure). The reported fill percent reflects the volume that has been metered, not including in-flight wind-down water. Do not wait for 100% — mode=35 is the completion signal.

### 12.2 The 120-second exit window at mode=35

The moment mode=35 is entered, a 120-second countdown begins. If no exit command is received within 120 seconds, the session expires and the device enters mode=99/appErrCode=8. This is a hard device constraint. Ensure your controller:
- Detects mode=35 promptly (requires active polling)
- Sends the exit command in response to mode=35, or prompts the user immediately
- Has a fallback to handle the mode=99/appErrCode=8 case

### 12.3 Transient mode=99/appErrCode=64 during recovery

As described in Section 11, a single-poll-cycle flash of mode=99/appErrCode=64 occurs approximately 6 seconds after sending clearAppError.cgi. Any error-detection logic must be suppressed during the recovery window to avoid mis-triggering a second clearAppError cycle.

### 12.4 Two polling clients in the iOS app

The iOS app runs two parallel polling clients simultaneously:
- One polling `getInfo.cgi?sid=<sid>` (session-aware, ~500ms)
- One polling `getInfo.cgi` (sessionless, ~500ms)

Both run throughout the entire bath fill session. The sessionless client acts as a fallback to detect when the session has become invalid. A HA implementation should do the same during bath fill to detect session expiry promptly.

### 12.5 Flow briefly appears in last mode=25 poll

In the final poll before mode=30, `flow` briefly shows a non-zero value while the mode field still shows `25`. This is the transition instant where the tap was just opened. Do not use a non-zero flow reading alone to infer mode=30 — always check the mode field.

### 12.6 mode=0 is a transient, not a stable state

After a bath fill session ends (either clean exit or recovery), the device passes through mode=0 briefly. Mode=0 polls may still show residual `bathtemp` and `state=3` values. Do not treat mode=0 as a stable state. Wait for mode=5.

### 12.7 sTimeout frozen at 300 during mode=30 is intentional

During active bath fill (mode=30), sTimeout always reads 300 regardless of how long the fill has been running and regardless of whether polling has paused. This is a safety design — the device cannot be accidentally abandoned mid-fill. Do not interpret sTimeout=300 as an anomaly.

### 12.8 heatingCtrl is always 0 in observed traffic

In all packet captures, `heatingCtrl` in getInfo responses is always 0 even during an active heatingCtrl session. Its exact semantic in getInfo responses is unclear. In ctrl.cgi requests it has a clear function (value=1 to open, 0 to close). Do not rely on the getInfo `heatingCtrl` field to determine if a heating session is open — use the sid and mode instead.

### 12.9 No rate limiting, but DoS protection exists

The device does not enforce a rate limit in the traditional sense, but if requests arrive faster than the device can process them (roughly more than 1/sec sustained), it enters a lockout mode where it stops responding. Recovery from lockout may require a power cycle of the bridge. The safe polling interval is 500ms (2 req/sec including both poll and control), matching the iOS app's observed behaviour.

---

## 13. What NOT To Do

### Do not use ctrl.cgi to change temperature or volume mid-session

**Wrong:**
```
GET /ctrl.cgi?sid=<sid>&bathfillCtrl=1&setBathTemp=42
```
This is the session **start** command. Sending it mid-session will restart the bath fill process, not update the temperature.

**Correct:**
```
GET /set.cgi?sid=<sid>&setBathTemp=42
```

This is one of the most impactful bugs an implementor can introduce. The original HA integration had this bug (Finding 5, 2026-04-04 audit).

---

### Do not omit setBathTemp and setBathVol from the exit command

**Wrong:**
```
GET /ctrl.cgi?sid=<sid>&bathfillCtrl=0
```

**Correct:**
```
GET /ctrl.cgi?sid=<sid>&bathfillCtrl=0&setBathTemp=<temp>&setBathVol=<vol>
```

The device expects the current bath settings echoed back on every bathfillCtrl command, including the exit. Omitting them causes incorrect behaviour.

---

### Do not run unnecessary pre-steps before bath fill start

The iOS app goes directly to:
```
GET /ctrl.cgi?sid=0&bathfillCtrl=1&setBathTemp=<N>&setBathVol=<N>
```

No preceding `heatingCtrl=1` setup, no `setSessionTimer` call, no intermediate poll. All such pre-steps are unnecessary and add 2–3 seconds of latency.

---

### Do not poll at less than 500ms interval

The iOS app polls at ~500ms. Polling faster than this increases the risk of triggering the DoS lockout. The device stops responding under sustained high-frequency requests.

---

### Do not treat sTimeout as reliable during mode=30

sTimeout is frozen at 300 during active fill. Any logic that relies on sTimeout counting down to detect session expiry will fail during mode=30.

---

### Do not send clearAppError.cgi more than once

Send exactly one `clearAppError.cgi?bathfill=0` when appErrCode=8 is detected. Multiple calls will not accelerate recovery and may interfere with it. The recovery sequence takes ~7 seconds; wait it out.

---

### Do not attempt to re-acquire a session during mode=99 recovery

After sending clearAppError.cgi, do not attempt ctrl.cgi calls to re-open a session until mode=5 is observed. The device is in a recovery/transition state and will not accept session opens.

---

### Do not confuse the two uses of mode=20

Mode=20 has two distinct meanings:
- In the context of a `bathfillCtrl` session: "waiting for tap" (start of bath fill).
- In the context of regular heating (no bath fill session): "post-heat cooldown" (after a heating session with tap closed).

The `bathfillCtrl` flag in the getInfo response distinguishes these cases.

---

### Do not use clearAppError.cgi while flow > 0

Calling `clearAppError.cgi` while flow is non-zero has no effect — the device ignores it. For appErrCode=32 (cancel with tap open), wait for the user to close the tap. The error self-clears at flow=0.

---

*End of document. All findings are based on direct packet capture of the Rheem EziSET iOS app, 2026-04-04.*
