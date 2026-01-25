"""All API calls belong here (async + rate limited)."""

from __future__ import annotations

import asyncio
import json
import contextlib
import time
import random
from datetime import datetime, timezone
from pathlib import Path
from collections.abc import Mapping, Awaitable, Callable
from typing import Any

from aiohttp import ClientError, ClientConnectorError, ClientTimeout
from homeassistant.components.number import NumberEntity
from homeassistant.components.water_heater import WaterHeaterEntity
from homeassistant.exceptions import ConditionErrorMessage, HomeAssistantError
from homeassistant.helpers.aiohttp_client import async_get_clientsession

from .const import DOMAIN, LOGGER
from .util import is_one, to_float, to_int
from .manifest import manifest_version

DEBUG_LOG_PATH = "/config/custom_components/rheem_eziset/debug.log"
DEBUG_SESSION_ID = "debug-session"
DEBUG_RUN_ID = "pre-fix"


def _agent_log(hypothesis: str, location: str, message: str, data: dict[str, Any]) -> None:
    """Write NDJSON debug instrumentation into a file under custom_components (off the event loop)."""

    def _write() -> None:
        payload = {
            "sessionId": DEBUG_SESSION_ID,
            "runId": DEBUG_RUN_ID,
            "integrationVersion": manifest_version(),
            "isoTime": datetime.now(timezone.utc).isoformat(),
            "hypothesisId": hypothesis,
            "location": location,
            "message": message,
            "data": data,
            "timestamp": int(time.time() * 1000),
        }
        with open(DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")
        cutoff_ms = int((time.time() - 24 * 3600) * 1000)
        try:
            lines = Path(DEBUG_LOG_PATH).read_text(encoding="utf-8").splitlines()
            kept: list[str] = []
            for line in lines:
                try:
                    obj = json.loads(line)
                    if obj.get("timestamp", 0) >= cutoff_ms:
                        kept.append(line)
                except Exception:
                    kept.append(line)
            if len(kept) != len(lines):
                Path(DEBUG_LOG_PATH).write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
        except Exception:
            pass

    try:
        loop = asyncio.get_running_loop()
        task = loop.create_task(asyncio.to_thread(_write))
        task.add_done_callback(lambda _: None)
    except Exception:
        with contextlib.suppress(Exception):
            _write()


class RheemEziSETApi:
    """Define the Rheem EziSET API (async, rate limited)."""

    MIN_REQUEST_GAP = 1.5  # seconds (between request start times)
    REQUEST_TIMEOUT = ClientTimeout(total=25.0)
    LOCKOUT_CONSEC_FAILURES = 3
    COOLDOWN_SCHEDULE_S = [10, 30, 60, 180]

    def __init__(self, hass, host: str, *, test_min_request_gap: float | None = None) -> None:
        """Initialise the basic parameters.
        
        Args:
            hass: Home Assistant instance
            host: Device hostname/IP
            test_min_request_gap: Optional override for MIN_REQUEST_GAP (testing only)
        """
        self.hass = hass
        self.host = host
        self.base_url = f"http://{host}/"
        
        # Allow test mode to override MIN_REQUEST_GAP
        self._effective_min_request_gap = test_min_request_gap if test_min_request_gap is not None else self.MIN_REQUEST_GAP

        self._request_lock = asyncio.Lock()
        self._write_lock = asyncio.Lock()
        self._next_request_at = 0.0
        self._failure_count = 0
        self._lockout_level = 0
        self._lockout_until_monotonic: float | None = None

        self._last_info: dict[str, Any] | None = None
        self._last_params: dict[str, Any] | None = None
        self._last_version: dict[str, Any] | None = None
        self._params_expires = 0.0
        self._version_expires = 0.0
        self._params_retry_at = 0.0
        self._version_retry_at = 0.0
        self._health_check_pending = False
        self._endpoint_health: dict[str, bool] = {"info": True, "params": True, "version": True}  # Track endpoint health
        self._endpoint_failure_count: dict[str, int] = {"info": 0, "params": 0, "version": 0}  # Track consecutive failures per endpoint

        self._owned_sid: int | None = None
        self._prev_temp_for_bathfill: int | None = None
        self._pending_restore_temp: int | None = None
        self._bathfill_target_temp: int | None = None
        self._cancel_user_settemp: bool = False
        # Default to a longer session timer so bath fill doesn't expire while waiting for user actions.
        self._control_session_timer: int = 600
        self._req_counter: int = 0
        self._poll_backoff_until: float = 0.0
        self._poll_failures: int = 0
        self._control_backoff_until: float = 0.0
        self._control_failures: int = 0
        self._pending_writes: dict[str, dict[str, Any]] = {}
        self._drain_lock = asyncio.Lock()
        self._drain_task: asyncio.Task | None = None
        self._post_write_callback: Callable[[str], Awaitable[None]] | None = None
        self._last_release_attempt: float = 0.0
        self._completion_latched: bool = False
        self._bathfill_latched: bool = False
        # When True, treat completion markers (state==3 / mode==35) as cleared until the next bath fill start.
        # Some devices keep reporting "Bath Fill Complete" for a long time after exit.
        self._ignore_completion_state: bool = False
        self._request_queue: asyncio.Queue[tuple[str, str | None, str | None, bool, asyncio.Future]] = asyncio.Queue()
        self._request_worker_task: asyncio.Task | None = None
        self._control_cooldown_until: float = 0.0
        self._connection_failures: int = 0
        self._connection_backoff_until: float = 0.0
        self._init_debug_log()

    # ---------------------------------------------------------
    # Debug helpers
    # ---------------------------------------------------------
    def _init_debug_log(self) -> None:
        """Reset debug log on startup and ensure file exists."""
        async def _reset() -> None:
            try:
                path = Path(DEBUG_LOG_PATH)
                path.parent.mkdir(parents=True, exist_ok=True)
                await self.hass.async_add_executor_job(path.write_text, "", "utf-8")
            except Exception:
                pass

        try:
            # Fire-and-forget so we do not block the event loop during setup.
            self.hass.async_create_task(_reset())
        except Exception:
            pass

    def _snapshot_state(self) -> dict[str, Any]:
        """Snapshot key device and internal state for logging."""
        data = self._last_info or {}
        return {
            "mode": data.get("mode"),
            "sTimeout": data.get("sTimeout"),
            "bathfillCtrl": data.get("bathfillCtrl"),
            "fillPercent": data.get("fillPercent"),
            "owned_sid": self._owned_sid,
            "cancel_user_settemp": self._cancel_user_settemp,
            "prev_temp_for_bathfill": self._prev_temp_for_bathfill,
            "pending_restore_temp": self._pending_restore_temp,
        }

    def _entity_snapshot(self, entity_id: str | None) -> dict[str, Any] | None:
        """Snapshot entity state/attributes if available."""
        if not entity_id:
            return None
        try:
            state = self.hass.states.get(entity_id)
            if state:
                return {"entity_id": entity_id, "state": state.state, "attributes": dict(state.attributes)}
        except Exception:
            return {"entity_id": entity_id, "state": None}
        return {"entity_id": entity_id, "state": None}

    def _with_transient_fields(self, payload: Mapping[str, Any] | None) -> dict[str, Any]:
        """Attach transient integration fields (e.g., bath fill target temp)."""
        data = dict(payload or {})
        target_temp = self._bathfill_target_temp
        if target_temp is not None:
            if self._bathfill_engaged(data):
                data["bathfill_target_temp"] = target_temp
            else:
                self._bathfill_target_temp = None
                data.pop("bathfill_target_temp", None)
        else:
            data.pop("bathfill_target_temp", None)
        return data

    def _log_action(self, action: str, stage: str, *, control_seq_id: str | None = None, extra: dict[str, Any] | None = None) -> None:
        """Emit an action log with snapshot + extras."""
        payload = {
            "action": action,
            "stage": stage,
            "control_seq_id": control_seq_id,
            "snapshot": self._snapshot_state(),
        }
        if extra:
            payload.update(extra)
        entity_id = payload.get("entity_id")
        if entity_id:
            entity_snapshot = self._entity_snapshot(entity_id)
            if entity_snapshot is not None:
                payload["entity_snapshot"] = entity_snapshot
        _agent_log(
            hypothesis="action-trace",
            location=f"api.py:{action}:{stage}",
            message="action trace",
            data=payload,
        )

    def _payload_summary(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Return a safe, loggable subset of a payload."""
        summary: dict[str, Any] = {}
        for key, val in payload.items():
            if isinstance(val, str | int | float | bool) or val is None:
                summary[key] = val
        return summary

    def _queue_log(self, stage: str, op: str, *, extra: dict[str, Any] | None = None) -> None:
        """Structured logging for queue events."""
        data = {"op": op, "pending_ops": list(self._pending_writes.keys())}
        if extra:
            data.update(extra)
        self._log_action("write_queue", stage, control_seq_id=op, extra=data)

    def _apply_control_cooldown(self, reason: str, *, base: float = 10.0) -> None:
        """Set a cooldown window for control requests to avoid hammering."""
        jitter = random.uniform(0.0, base * 0.5)
        cooldown = base + jitter
        until = time.monotonic() + cooldown
        self._control_cooldown_until = max(self._control_cooldown_until, until)
        self._next_request_at = max(self._next_request_at, until)
        self._log_action(
            "control_cooldown",
            "set",
            extra={"reason": reason, "cooldown_s": round(cooldown, 2), "until": until},
        )

    # ---------------------------------------------------------
    # Request queue (covers polls and writes)
    # ---------------------------------------------------------
    def _ensure_request_worker(self) -> None:
        """Ensure the request worker is running."""
        if self._request_worker_task and not self._request_worker_task.done():
            return
        try:
            loop = asyncio.get_running_loop()
            self._request_worker_task = loop.create_task(self._request_worker())
            self._request_worker_task.add_done_callback(lambda _: None)
        except Exception:
            self._request_worker_task = None

    async def _enqueue_request(
        self,
        path: str,
        *,
        op_id: str | None = None,
        control_seq_id: str | None = None,
        allow_read_timeout: bool = False,
    ) -> dict[str, Any]:
        """Enqueue any HTTP request (polls + writes) through the unified queue."""
        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        await self._request_queue.put((path, op_id, control_seq_id, allow_read_timeout, fut))
        self._ensure_request_worker()
        return await fut

    async def _request_worker(self) -> None:
        """Process requests sequentially honoring throttling."""
        try:
            while True:
                path, op_id, control_seq_id, allow_read_timeout, fut = await self._request_queue.get()
                if fut.cancelled():
                    self._request_queue.task_done()
                    continue
                try:
                    result = await self._throttled_get_json(
                        path,
                        op_id=op_id,
                        control_seq_id=control_seq_id,
                        allow_read_timeout=allow_read_timeout,
                    )
                    if not fut.cancelled():
                        fut.set_result(result)
                except Exception as err:  # pylint: disable=broad-except
                    if not fut.cancelled():
                        fut.set_exception(err)
                finally:
                    self._request_queue.task_done()
        except asyncio.CancelledError:
            # Task was cancelled (e.g., during integration unload)
            # Cancel any pending futures and exit cleanly
            while not self._request_queue.empty():
                try:
                    _, _, _, _, fut = self._request_queue.get_nowait()
                    if not fut.cancelled():
                        fut.cancel()
                    self._request_queue.task_done()
                except Exception:
                    pass
            raise

    def _enqueue_write(self, op: str, payload: dict[str, Any]) -> None:
        """Coalesce and enqueue a write operation (latest wins)."""
        replaced = op in self._pending_writes
        self._pending_writes[op] = payload
        self._queue_log("enqueue", op, extra={"replaced": replaced, "payload": self._payload_summary(payload)})
        self._schedule_drain()

    def _schedule_drain(self) -> None:
        """Schedule draining pending writes if not already running."""
        if self._drain_task and not self._drain_task.done():
            return
        try:
            loop = asyncio.get_running_loop()
            self._drain_task = loop.create_task(self._drain_pending_writes())
            self._drain_task.add_done_callback(lambda _: None)
        except Exception:
            self._drain_task = None

    def set_post_write_callback(self, callback: Callable[[str], Awaitable[None]]) -> None:
        """Register a callback to be invoked after a queued write succeeds."""
        self._post_write_callback = callback

    def _op_ready(self, op: str, payload: dict[str, Any], info: Mapping[str, Any]) -> tuple[bool, str | None, bool]:
        """Return (ready, reason, drop) for a pending op given cached info."""
        if self._write_lock.locked():
            return False, "write_lock_in_use", False

        mode_val = to_int(info.get("mode"))
        flow_val = to_float(info.get("flow"))
        s_timeout = to_int(info.get("sTimeout"))
        if op == "bathfill_cancel":
            # Allow cancel to execute even if not currently engaged; executor is idempotent.
            return True, None, False

        if op == "bathfill_set_temp":
            if not self._bathfill_active(info):
                return False, "bathfill_inactive", True
            direction = payload.get("direction") or "up"
            sid_val = to_int(info.get("sid")) or self._owned_sid
            if s_timeout not in (0, None) and (not self._owned_sid or sid_val != self._owned_sid):
                return False, "sTimeout_active", False
            if direction == "up":
                if flow_val not in (0, None):
                    return False, "flow_active", False
                if not sid_val and s_timeout in (0, None):
                    return True, None, False  # allow executor to reacquire sid
                if not sid_val:
                    return False, "sid_missing", False
                return True, None, False
            # direction down
            if not sid_val and s_timeout in (0, None):
                return True, None, False  # allow executor to reacquire sid
            if not sid_val:
                return False, "sid_missing", False
            return True, None, False

        if op == "bathfill_start":
            if self._bathfill_active(info):
                return False, "bathfill_already_active", True
            if s_timeout not in (0, None):
                return False, "sTimeout_active", False
            if flow_val not in (0, None):
                return False, "flow_active", False
            if mode_val not in (5, None):
                return False, f"mode_busy_{mode_val}", False
            return True, None, False

        if op in {"set_temp", "set_session_timer", "restore_temp"}:
            if s_timeout not in (0, None):
                return False, "sTimeout_active", False
            direction = payload.get("direction") or "up"
            if direction == "up":
                if flow_val not in (0, None):
                    return False, "flow_active", False
                if mode_val not in (5, None):
                    return False, f"mode_busy_{mode_val}", False
                if self._bathfill_active(info):
                    return False, "bathfill_active", False
            # direction down: allow even if mode/flow missing/busy; rely on executor checks
            return True, None, False

        return False, "unknown_op", False

    async def _drain_pending_writes(self) -> None:
        """Drain and execute pending writes in priority order."""
        async with self._drain_lock:
            try:
                while True:
                    if not self._pending_writes:
                        self._queue_log("empty", "write_queue")
                        break

                    info = self._last_info or {}
                    if not info:
                        self._queue_log("blocked", "write_queue", extra={"reason": "no_cached_info"})
                        break

                    selected_op: str | None = None
                    selected_payload: dict[str, Any] | None = None
                    blocked_reason: tuple[str, str | None] | None = None

                    for op in ("bathfill_cancel", "bathfill_set_temp", "bathfill_start", "set_session_timer", "set_temp", "restore_temp"):
                        if op not in self._pending_writes:
                            continue
                        payload = self._pending_writes[op]
                        ready, reason, drop = self._op_ready(op, payload, info)
                        if drop:
                            self._pending_writes.pop(op, None)
                            self._queue_log("drop", op, extra={"reason": reason})
                            continue
                        if ready:
                            selected_op = op
                            selected_payload = payload
                            break
                        if blocked_reason is None:
                            blocked_reason = (op, reason)

                    if selected_op is None or selected_payload is None:
                        if blocked_reason:
                            op, reason = blocked_reason
                            self._queue_log(
                                "blocked",
                                op,
                                extra={
                                    "reason": reason,
                                    "mode": to_int(info.get("mode")),
                                    "flow": to_float(info.get("flow")),
                                    "sTimeout": to_int(info.get("sTimeout")),
                                    "bathfillCtrl": info.get("bathfillCtrl"),
                                },
                            )
                            if isinstance(reason, str):
                                if reason == "sTimeout_active":
                                    self._apply_control_cooldown(reason, base=12.0)
                                elif reason.startswith("mode_busy") or reason in {"invalid_mode", "invalid_flow", "flow_active", "bathfill_active"}:
                                    self._apply_control_cooldown(reason, base=8.0)
                                elif reason == "write_lock_in_use":
                                    self._apply_control_cooldown(reason, base=5.0)
                        break

                    payload_summary = self._payload_summary(selected_payload)
                    self._queue_log("start", selected_op, extra={"payload": payload_summary})
                    try:
                        await self._execute_pending_op(selected_op, selected_payload)
                        self._pending_writes.pop(selected_op, None)
                        self._queue_log("ok", selected_op, extra={"payload": payload_summary})
                        if self._post_write_callback:
                            with contextlib.suppress(Exception):
                                await self._post_write_callback(f"write_queue:{selected_op}")
                    except ConditionErrorMessage as err:
                        err_type = getattr(err, "type", None)
                        permanent_conditions = {"bathfill_inactive", "bathfill_already_active"}
                        if err_type in permanent_conditions:
                            self._queue_log(
                                "drop",
                                selected_op,
                                extra={
                                    "error": type(err).__name__,
                                    "err_type": err_type,
                                    "reason": str(err),
                                    "payload": payload_summary,
                                },
                            )
                            self._pending_writes.pop(selected_op, None)
                            continue
                        # Keep pending; wait for next poll to retry.
                        self._queue_log(
                            "blocked",
                            selected_op,
                            extra={
                                "error": type(err).__name__,
                                "reason": str(err),
                                "payload": payload_summary,
                            },
                        )
                        if err_type in {
                            "invalid_sTimeout",
                            "invalid_mode",
                            "invalid_flow",
                            "flow_active",
                            "bathfill_start_busy",
                            "bathfill_exit_retry",
                        }:
                            self._apply_control_cooldown(err_type, base=12.0)
                        break
                    except HomeAssistantError as err:
                        # Likely device-side contention; keep pending and retry later.
                        if isinstance(err.__cause__, asyncio.TimeoutError):
                            self._queue_log(
                                "blocked",
                                selected_op,
                                extra={
                                    "error": type(err).__name__,
                                    "reason": "timeout",
                                    "payload": self._payload_summary(selected_payload),
                                },
                            )
                        else:
                            self._queue_log(
                                "error",
                                selected_op,
                                extra={
                                    "error": type(err).__name__,
                                    "reason": str(err),
                                    "payload": self._payload_summary(selected_payload),
                                },
                            )
                        break
                    except Exception as err:  # pylint: disable=broad-except
                        # Drop on unexpected errors to avoid livelock.
                        self._queue_log(
                            "error_drop",
                            selected_op,
                            extra={
                                "error": type(err).__name__,
                                "reason": str(err),
                                "payload": self._payload_summary(selected_payload),
                            },
                        )
                        self._pending_writes.pop(selected_op, None)
                        continue
            finally:
                self._drain_task = None
                self._queue_log("complete", "write_queue")

    async def _execute_pending_op(self, op: str, payload: dict[str, Any]) -> None:
        """Execute a pending op by name."""
        if op == "set_temp":
            await self._do_set_temp(**payload)
        elif op == "set_session_timer":
            await self._do_set_session_timer(**payload)
        elif op == "bathfill_start":
            await self._do_start_bath_fill(**payload)
        elif op == "bathfill_cancel":
            await self._do_cancel_bath_fill(**payload)
        elif op == "restore_temp":
            await self._do_restore_temp(**payload)
        elif op == "bathfill_set_temp":
            await self._do_bathfill_set_temp(**payload)
        else:
            raise HomeAssistantError(f"{DOMAIN} - Unknown queued operation: {op}")

    # ---------------------------------------------------------
    # Core request plumbing
    # ---------------------------------------------------------
    async def _throttled_get_json(self, path: str, *, op_id: str | None = None, control_seq_id: str | None = None, allow_read_timeout: bool = False) -> dict[str, Any]:
        """GET JSON with global rate limiting and lockout handling."""
        req_id = self._req_counter = self._req_counter + 1
        now = time.monotonic()
        if self._lockout_until_monotonic and now < self._lockout_until_monotonic:
            raise HomeAssistantError(
                f"{DOMAIN} - Device lockout suspected; backing off until {self._lockout_until_monotonic}"
            )

        async with self._request_lock:
            now = time.monotonic()
            # Calculate when the next request can start based on rate limiting constraints.
            # MIN_REQUEST_GAP (1.5s) is the core safety limit to prevent device DoS.
            # Additional delays apply for:
            # - Control cooldown: Prevents rapid retries after control failures
            # - Poll backoff: Aggressive backoff for poll timeouts (device DoS/backpressure)
            # - Control backoff: Backoff for control timeouts
            # - Connection backoff: Shorter backoff for network connection errors (separate from device errors)
            effective_next = max(self._next_request_at, self._control_cooldown_until)
            if op_id == "data-poll" and self._poll_backoff_until:
                effective_next = max(effective_next, self._poll_backoff_until)
            if op_id != "data-poll" and self._control_backoff_until:
                effective_next = max(effective_next, self._control_backoff_until)
            # Respect connection backoff for all operations
            if self._connection_backoff_until:
                effective_next = max(effective_next, self._connection_backoff_until)
            if op_id == "data-poll" and self._write_lock.locked():
                # Avoid colliding polls with in-flight writes (0.5s is sufficient).
                effective_next = max(effective_next, now + 0.5)
            if op_id == "data-poll":
                s_timeout = to_int((self._last_info or {}).get("sTimeout"))
                # Only add delay if session is active but we don't own it (another controller)
                if s_timeout not in (0, None) and not self._owned_sid:
                    # Soften poll cadence when another controller holds session.
                    effective_next = max(effective_next, now + 0.5)
            wait_s = max(0.0, effective_next - now)
            wait_start = time.monotonic()
            if wait_s:
                await asyncio.sleep(wait_s)
            wait_actual_ms = int((time.monotonic() - wait_start) * 1000)
            # reserve slot by start time
            next_request_at_before = self._next_request_at
            self._next_request_at = time.monotonic() + self._effective_min_request_gap

            session = async_get_clientsession(self.hass)
            url = f"{self.base_url}{path}"
            start = time.monotonic()
            # region agent log
            _agent_log(
                hypothesis="H1-rate-limit-or-network",
                location="api.py:_throttled_get_json:pre",
                message="HTTP start",
                data={
                    "req_id": req_id,
                    "op_id": op_id,
                    "control_seq_id": control_seq_id,
                    "host": self.host,
                    "url": url,
                    "timeout_total_s": getattr(self.REQUEST_TIMEOUT, "total", None),
                    "path": path,
                    "wait_s": wait_s,
                    "wait_actual_ms": wait_actual_ms,
                    "lockout_until": self._lockout_until_monotonic,
                    "next_request_at_before": next_request_at_before,
                    "next_request_at_after": self._next_request_at,
                    "write_lock_locked": self._write_lock.locked(),
                    "owned_sid": self._owned_sid,
                },
            )
            # endregion
            LOGGER.debug(
                "%s debug pre req=%s path=%s wait_s=%.3f lockout_until=%s op_id=%s ctrl_seq=%s",
                DOMAIN,
                id(self),
                path,
                wait_s,
                self._lockout_until_monotonic,
                op_id,
                control_seq_id,
            )
            try:
                async with session.get(url, timeout=self.REQUEST_TIMEOUT) as resp:
                    if resp.status != 200:
                        raise HomeAssistantError(f"{DOMAIN} - HTTP {resp.status} for {path}")
                    text = await resp.text()
                    try:
                        data = await resp.json(content_type=None)
                    except Exception as parse_err:
                        snippet = (text or "")[:200]
                        LOGGER.error("%s - JSON parse error for %s (%s): body snippet=%r", DOMAIN, url, type(parse_err).__name__, snippet)
                        raise
            except (ClientError, asyncio.TimeoutError, ValueError, json.JSONDecodeError, HomeAssistantError) as err:
                # Some requests (notably session-release) are best-effort: a read timeout is not necessarily a failure.
                # When allow_read_timeout=True, treat TimeoutError as a soft outcome (no failure count / no lockout).
                if allow_read_timeout and isinstance(err, asyncio.TimeoutError):
                    self._apply_control_cooldown("timeout_allowed", base=12.0)
                    _agent_log(
                        hypothesis="H1-rate-limit-or-network",
                        location="api.py:_throttled_get_json:timeout_allowed",
                        message="HTTP timeout allowed",
                        data={
                            "req_id": req_id,
                            "op_id": op_id,
                            "control_seq_id": control_seq_id,
                            "host": self.host,
                            "url": url,
                            "path": path,
                            "error_type": type(err).__name__,
                            "elapsed_ms": int((time.monotonic() - start) * 1000),
                            "write_lock_locked": self._write_lock.locked(),
                            "owned_sid": self._owned_sid,
                        },
                    )
                    return {"_timeout": True}

                # Distinguish connection errors from device errors
                is_connection_error = isinstance(err, ClientConnectorError)
                
                if is_connection_error:
                    # Connection errors (network issues) use shorter backoff to recover faster
                    # Don't count connection errors toward device lockout
                    self._connection_failures += 1
                    # Connection error backoff: 5s, 10s, 20s max (shorter than device errors)
                    conn_backoff = min(20.0, 5.0 * (2 ** min(self._connection_failures - 1, 2)))
                    jitter_conn = random.uniform(0.0, min(1.0, conn_backoff * 0.1))
                    self._connection_backoff_until = time.monotonic() + conn_backoff + jitter_conn
                    self._next_request_at = max(self._next_request_at, self._connection_backoff_until)
                    LOGGER.warning("%s - Connection error (network issue); backing off for %.1fs", DOMAIN, conn_backoff)
                    # Don't record failure for connection errors (avoid false lockouts from network issues)
                else:
                    # Device errors: record failure and apply device-specific backoff
                    await self._record_failure()
                    if op_id == "data-poll" and isinstance(err, asyncio.TimeoutError):
                        self._poll_failures += 1
                        # Poll timeouts often indicate device DoS/backpressure; back off aggressively.
                        # Sequence: 5s, 10s, 20s, 40s, 60s max.
                        backoff = min(60.0, 5.0 * (2 ** min(self._poll_failures - 1, 4)))
                        jitter = random.uniform(0.0, min(1.0, backoff * 0.1))
                        self._poll_backoff_until = time.monotonic() + backoff + jitter
                        self._next_request_at = max(self._next_request_at, self._poll_backoff_until)
                    if op_id != "data-poll" and isinstance(err, asyncio.TimeoutError):
                        # Back off further on control timeouts to avoid hammering.
                        self._control_failures += 1
                        # Control timeouts: 10s, 20s, 40s, 60s max.
                        ctrl_backoff = min(60.0, 10.0 * (2 ** min(self._control_failures - 1, 3)))
                        jitter_ctrl = random.uniform(0.0, min(1.0, ctrl_backoff * 0.1))
                        self._control_backoff_until = time.monotonic() + ctrl_backoff + jitter_ctrl
                        # Also set a short cooldown to avoid immediate requeue of another control.
                        self._control_cooldown_until = max(self._control_cooldown_until, self._control_backoff_until)
                        self._next_request_at = max(self._next_request_at, self._control_backoff_until)
                LOGGER.error("%s - HTTP/parse error for %s (%s): %s", DOMAIN, url, type(err).__name__, err)
                # region agent log
                _agent_log(
                    hypothesis="H1-rate-limit-or-network",
                    location="api.py:_throttled_get_json:error",
                    message="HTTP error",
                    data={
                        "req_id": req_id,
                        "op_id": op_id,
                        "control_seq_id": control_seq_id,
                        "host": self.host,
                        "url": url,
                        "path": path,
                        "error_type": type(err).__name__,
                        "elapsed_ms": int((time.monotonic() - start) * 1000),
                        "failure_count": self._failure_count,
                        "lockout_until": self._lockout_until_monotonic,
                        "write_lock_locked": self._write_lock.locked(),
                        "owned_sid": self._owned_sid,
                    },
                )
                # endregion
                LOGGER.debug(
                    "%s debug err req=%s path=%s error=%s elapsed_ms=%d failure_count=%d lockout_until=%s op_id=%s ctrl_seq=%s",
                    DOMAIN,
                    id(self),
                    path,
                    type(err).__name__,
                    int((time.monotonic() - start) * 1000),
                    self._failure_count,
                    self._lockout_until_monotonic,
                    op_id,
                    control_seq_id,
                )
                raise HomeAssistantError(f"{DOMAIN} - Request failed for {path}") from err

            # success
            if op_id == "data-poll":
                self._poll_failures = 0
                self._poll_backoff_until = 0.0
            else:
                self._control_failures = 0
                self._control_backoff_until = 0.0
            self._reset_failures()
            json_type = type(data).__name__
            key_count = len(data) if isinstance(data, Mapping) else None
            summary_keys = {}
            if isinstance(data, Mapping):
                for key in ("mode", "bathfillCtrl", "sid", "sTimeout", "fillPercent"):
                    if key in data:
                        summary_keys[key] = data.get(key)
            # region agent log
            _agent_log(
                hypothesis="H1-rate-limit-or-network",
                location="api.py:_throttled_get_json:ok",
                message="HTTP ok",
                data={
                    "req_id": req_id,
                    "op_id": op_id,
                    "control_seq_id": control_seq_id,
                    "host": self.host,
                    "url": url,
                    "path": path,
                    "elapsed_ms": int((time.monotonic() - start) * 1000),
                    "len": len(text),
                    "json_type": json_type,
                    "key_count": key_count,
                    "keys_summary": summary_keys,
                },
            )
            # endregion
            # Reset connection failure counter on successful request
            if self._connection_failures > 0:
                self._connection_failures = 0
                self._connection_backoff_until = 0.0
            LOGGER.debug(
                "%s debug ok req=%s path=%s elapsed_ms=%d bytes=%d",
                DOMAIN,
                id(self),
                path,
                int((time.monotonic() - start) * 1000),
                len(text),
            )
            return data if isinstance(data, Mapping) else {}

    async def _record_failure(self) -> None:
        """Track failures and enter lockout cooldown if needed."""
        self._failure_count += 1
        if self._failure_count >= self.LOCKOUT_CONSEC_FAILURES:
            level = min(self._lockout_level, len(self.COOLDOWN_SCHEDULE_S) - 1)
            cooldown = self.COOLDOWN_SCHEDULE_S[level]
            self._lockout_until_monotonic = time.monotonic() + cooldown
            self._lockout_level = min(level + 1, len(self.COOLDOWN_SCHEDULE_S) - 1)
            self._health_check_pending = True  # Schedule health check after lockout expires
            LOGGER.warning("%s - Entering cooldown for %ss due to repeated failures", DOMAIN, cooldown)

    def _reset_failures(self) -> None:
        """Reset failure counters."""
        self._failure_count = 0
        self._lockout_level = 0
        self._lockout_until_monotonic = None
        self._health_check_pending = False

    async def _health_check_after_lockout(self) -> bool:
        """Perform health check probe after lockout expires. Returns True if device is responsive."""
        try:
            # Perform lightweight getInfo.cgi request to verify device responsiveness
            info = await self._enqueue_request("getInfo.cgi", op_id="health-check", control_seq_id="health_check")
            if isinstance(info, Mapping) and info.get("mode") is not None:
                LOGGER.info("%s - Health check passed; device is responsive", DOMAIN)
                self._reset_failures()
                self._endpoint_health["info"] = True
                return True
            LOGGER.warning("%s - Health check failed; device returned invalid response", DOMAIN)
            return False
        except Exception as err:
            LOGGER.warning("%s - Health check failed: %s; extending lockout", DOMAIN, err)
            # Extend lockout by one level if health check fails
            if self._lockout_level < len(self.COOLDOWN_SCHEDULE_S) - 1:
                self._lockout_level += 1
            cooldown = self.COOLDOWN_SCHEDULE_S[min(self._lockout_level, len(self.COOLDOWN_SCHEDULE_S) - 1)]
            self._lockout_until_monotonic = time.monotonic() + cooldown
            self._health_check_pending = True
            return False

    # ---------------------------------------------------------
    # Public data fetch
    # ---------------------------------------------------------
    async def async_get_data(self) -> dict[str, Any]:  # noqa: C901 - orchestrates poll/refresh/session handling
        """Fetch and merge info/params/version with caching."""
        now = time.monotonic()
        if self._lockout_until_monotonic and now < self._lockout_until_monotonic:
            raise HomeAssistantError(
                f"{DOMAIN} - Device lockout suspected; backing off until {self._lockout_until_monotonic}"
            )
        # Perform health check if lockout just expired
        if self._health_check_pending and (not self._lockout_until_monotonic or now >= self._lockout_until_monotonic):
            health_ok = await self._health_check_after_lockout()
            if not health_ok:
                # Health check failed, lockout was extended
                if self._lockout_until_monotonic and now < self._lockout_until_monotonic:
                    raise HomeAssistantError(
                        f"{DOMAIN} - Device lockout suspected; backing off until {self._lockout_until_monotonic}"
                    )

        if self._write_lock.locked() and self._last_info is not None:
            # A control sequence is in-flight; avoid adding extra requests—return last known data.
            merged: dict[str, Any] = {}
            for src in (self._last_version or {}, self._last_params or {}, self._last_info):
                merged.update(src)
            return self._with_transient_fields(merged)

        merged: dict[str, Any] = {}

        info: dict[str, Any] | None = None
        owned_sid = self._owned_sid
        if owned_sid:
            try:
                info = await self._enqueue_request(f"getInfo.cgi?sid={owned_sid}", op_id="data-poll", control_seq_id="poll")
            except Exception as err:  # pylint: disable=broad-except
                LOGGER.debug("%s debug getInfo sid=%s failed; falling back to sidless (%s)", DOMAIN, owned_sid, err)

        def _valid_info(payload: dict[str, Any] | None) -> bool:
            if not isinstance(payload, Mapping):
                return False
            # Require at least mode; prefer having temp/flow too.
            return payload.get("mode") is not None

        if info is None or not _valid_info(info):
            if info is not None and not _valid_info(info):
                LOGGER.debug("%s debug sid getInfo incomplete; falling back to sidless", DOMAIN)
            info = await self._enqueue_request("getInfo.cgi", op_id="data-poll", control_seq_id="poll")

        if not _valid_info(info):
            self._endpoint_health["info"] = False
            self._endpoint_failure_count["info"] += 1
            if self._last_info:
                LOGGER.warning("%s - getInfo payload invalid; keeping last known state", DOMAIN)
                # Only enter lockout if critical endpoint (info) fails repeatedly
                if self._endpoint_failure_count["info"] >= self.LOCKOUT_CONSEC_FAILURES:
                    await self._record_failure()
                return self._with_transient_fields(self._last_info)
            raise HomeAssistantError(
                f"{DOMAIN} - Device returned invalid data. "
                f"This may indicate a communication problem. "
                f"Check your network connection and try again in a few moments."
            )

        # Success: reset failure count and mark endpoint as healthy
        self._endpoint_health["info"] = True
        self._endpoint_failure_count["info"] = 0
        self._last_info = info

        # params caching (fail-open) - non-critical endpoint
        params: dict[str, Any] | None = self._last_params
        if (now >= self._params_expires or params is None) and now >= self._params_retry_at:
            try:
                params = await self._enqueue_request("getParams.cgi")
                self._last_params = params
                self._endpoint_health["params"] = True
                self._endpoint_failure_count["params"] = 0
                self._params_expires = now + 60 * 60  # 60 minutes
                self._params_retry_at = 0.0
            except Exception as err:  # pylint: disable=broad-except
                # Non-critical endpoint failure: log warning but don't enter lockout
                self._endpoint_health["params"] = False
                self._endpoint_failure_count["params"] += 1
                LOGGER.warning("%s - getParams.cgi failed (non-critical); using cached params (%s)", DOMAIN, type(err).__name__)
                self._params_retry_at = now + 300  # retry in 5 minutes

        # version caching (fail-open, low frequency)
        version: dict[str, Any] | None = self._last_version
        if (now >= self._version_expires or version is None) and now >= self._version_retry_at:
            try:
                version = await self._enqueue_request("version.cgi")
                self._last_version = version
                self._version_expires = now + 24 * 60 * 60  # 24 hours
                self._version_retry_at = 0.0
            except Exception as err:  # pylint: disable=broad-except
                # Non-critical endpoint failure: log warning but don't enter lockout
                self._endpoint_health["version"] = False
                self._endpoint_failure_count["version"] += 1
                LOGGER.warning("%s - version.cgi failed (non-critical); using cached version (%s)", DOMAIN, type(err).__name__)
                self._version_retry_at = now + 3600  # retry in 1 hour

        # merge precedence: version -> params -> info
        for src in (version or {}, params or {}, info or {}):
            merged.update(src)
        
        # Log partial failure state if any non-critical endpoints are failing
        degraded_endpoints = [ep for ep, healthy in self._endpoint_health.items() if not healthy and ep != "info"]
        if degraded_endpoints:
            LOGGER.info("%s - Operating in degraded mode: endpoints %s are failing but using cached data", DOMAIN, degraded_endpoints)

        # If we still "own" a control session, ensure we release it even if a prior release timed out.
        if self._owned_sid and not self._write_lock.locked():
            sid_from_info = to_int(merged.get("sid"))
            s_timeout_val = to_int(merged.get("sTimeout"))
            mode_val = to_int(merged.get("mode"))
            bathfill_active = self._bathfill_active(merged)
            now_mono = time.monotonic()
            if not bathfill_active and now_mono - self._last_release_attempt > 1.0:
                if mode_val == 10 or s_timeout_val not in (0, None) or sid_from_info in (self._owned_sid, 0, None):
                    self._last_release_attempt = now_mono
                    resp: dict[str, Any] | None = None
                    with contextlib.suppress(Exception):
                        resp = await self._enqueue_request(
                            f"ctrl.cgi?sid={self._owned_sid}&heatingCtrl=0",
                            control_seq_id="release_retry",
                            allow_read_timeout=True,
                        )
                    if resp is not None and not resp.get("_timeout"):
                        self._owned_sid = None

        # If bath fill not active anymore, allow temp changes again.
        if not self._bathfill_active(info):
            self._cancel_user_settemp = False
            # Try pending restore if idle/flow=0
            if self._pending_restore_temp is not None:
                mode_val = to_int(info.get("mode"))
                flow_val = to_float(info.get("flow"))
                if mode_val in (5, None) and flow_val in (0, None):
                    pending_temp = self._pending_restore_temp
                    if pending_temp is not None:
                        if "restore_temp" not in self._pending_writes:
                            self._log_action(
                                "restore_temp",
                                "enqueue",
                                control_seq_id="bathfill_restore",
                                extra={"temp": pending_temp},
                            )
                        self._enqueue_write(
                            "restore_temp",
                            {
                                "temp": pending_temp,
                                "entity_id": None,
                            },
                        )

        mode_val = to_int(merged.get("mode"))
        state_val = to_int(merged.get("state"))
        sid_val = to_int(merged.get("sid"))
        fill_percent = to_float(merged.get("fillPercent"))
        flow_val = to_float(merged.get("flow"))
        device_bathfill_active = self._bathfill_active(merged)
        if device_bathfill_active:
            self._bathfill_latched = True
            self._ignore_completion_state = False

        completion = (not self._ignore_completion_state) and (mode_val == 35 or state_val == 3)
        if completion:
            if fill_percent is None or fill_percent < 100:
                merged["fillPercent"] = 100.0
                fill_percent = 100.0
            self._completion_latched = True
        elif not device_bathfill_active and self._completion_latched:
            if mode_val in (5, None) and flow_val in (0, None):
                merged["fillPercent"] = 0.0
                self._completion_latched = False
                self._bathfill_latched = False

        # If completion has been explicitly cleared (exit pressed), hide stale completion state
        # and reset progress back to 0 for UI consistency.
        if self._ignore_completion_state and not device_bathfill_active and flow_val in (0, None):
            merged["fillPercent"] = 0.0
            fill_percent = 0.0

        self._schedule_drain()
        return self._with_transient_fields(merged)

    async def async_get_info_only(self) -> dict[str, Any]:
        """Fetch only getInfo (sid-aware) and merge with cached params/version without refetching them."""
        now = time.monotonic()
        if self._lockout_until_monotonic and now < self._lockout_until_monotonic:
            raise HomeAssistantError(
                f"{DOMAIN} - Device lockout suspected; backing off until {self._lockout_until_monotonic}"
            )

        if self._write_lock.locked() and self._last_info is not None:
            merged: dict[str, Any] = {}
            for src in (self._last_version or {}, self._last_params or {}, self._last_info):
                merged.update(src)
            return self._with_transient_fields(merged)

        merged: dict[str, Any] = {}
        info: dict[str, Any] | None = None
        owned_sid = self._owned_sid
        if owned_sid:
            try:
                info = await self._enqueue_request(f"getInfo.cgi?sid={owned_sid}", op_id="data-poll", control_seq_id="poll")
            except Exception as err:  # pylint: disable=broad-except
                LOGGER.debug("%s debug getInfo sid=%s failed; falling back to sidless (%s)", DOMAIN, owned_sid, err)
        if info is None or info.get("mode") is None:
            info = await self._enqueue_request("getInfo.cgi", op_id="data-poll", control_seq_id="poll")
        self._last_info = info

        for src in (self._last_version or {}, self._last_params or {}, info or {}):
            merged.update(src)
        return self._with_transient_fields(merged)

    # ---------------------------------------------------------
    # Control helpers
    # ---------------------------------------------------------
    async def _check_control_issues(
        self,
        reset_attribute_owner: object | None = None,
        reset_attribute: str | None = None,
        *,
        require_idle: bool = True,
        allow_owned_session: bool = False,
        use_cached: bool = False,
    ) -> dict[str, Any]:
        """Check for control conflicts and return latest info."""
        info: dict[str, Any] | None = None
        if use_cached and self._last_info and self._last_info.get("mode") is not None:
            info = self._last_info
        if info is None:
            info = await self._enqueue_request("getInfo.cgi")
            self._last_info = info

        s_timeout_val = to_int(info.get("sTimeout"))
        sid_val = to_int(info.get("sid"))
        mode_val = to_int(info.get("mode"))
        flow_val = to_float(info.get("flow"))

        if s_timeout_val not in (0, None):
            if allow_owned_session and self._owned_sid and sid_val == self._owned_sid:
                pass
            else:
                if reset_attribute_owner and reset_attribute:
                    reset_attribute_owner.__dict__[reset_attribute] = None
                raise ConditionErrorMessage(
                    type="invalid_sTimeout",
                    message=f"""{DOMAIN} - Couldn't take control - it appears another user has control.
                    Got this response: {info}""",
                )

        if require_idle:
            if mode_val is not None and mode_val != 5:
                if reset_attribute_owner and reset_attribute:
                    reset_attribute_owner.__dict__[reset_attribute] = None
                raise ConditionErrorMessage(
                    type="invalid_mode",
                    message=f"""{DOMAIN} - Couldn't take control - it appears that the water_heater is in use.
                    Got this response: {info}""",
                )
            if flow_val is not None and flow_val != 0:
                if reset_attribute_owner and reset_attribute:
                    reset_attribute_owner.__dict__[reset_attribute] = None
                raise ConditionErrorMessage(
                    type="invalid_flow",
                    message=f"""{DOMAIN} - Couldn't take control - it appears that the water_heater is in use.
                    Got this response: {info}""",
                )
        return info

    async def _async_set_param(self, param: str, response_param: str, response_check: str | int | float, *, control_seq_id: str | None = None) -> None:
        """Set parameters on Rheem."""
        async with self._write_lock:
            sid: int | None = None
            try:
                attempts = 2
                last_err: Exception | None = None
                for attempt in range(attempts):
                    try:
                        ctrl_resp = await self._enqueue_request(
                            f"ctrl.cgi?sid=0&heatingCtrl=1&sessionTimer={self._control_session_timer}",
                            control_seq_id=control_seq_id,
                        )
                        sid_raw = ctrl_resp.get("sid")
                        sid = to_int(sid_raw)
                        if sid:
                            self._owned_sid = sid
                        if not sid or not is_one(ctrl_resp.get("heatingCtrl")):
                            LOGGER.error("%s - Error when retrieving ctrl for set_param. Result: %s", DOMAIN, ctrl_resp)
                            if sid_raw is not None:
                                with contextlib.suppress(Exception):
                                    await self._enqueue_request(
                                        f"ctrl.cgi?sid={sid_raw}&heatingCtrl=0",
                                        control_seq_id=control_seq_id,
                                        allow_read_timeout=True,
                                    )
                            raise HomeAssistantError(f"{DOMAIN} - Unable to take control to set parameter {param}")

                        set_resp = await self._enqueue_request(f"set.cgi?sid={sid}&{param}", control_seq_id=control_seq_id)
                        resp_val = set_resp.get(response_param)
                        ok = False
                        if isinstance(response_check, int):
                            ok = to_int(resp_val) == response_check
                        elif isinstance(response_check, float):
                            ok = to_float(resp_val) == response_check
                        else:
                            ok = str(resp_val) == str(response_check)
                        if not ok:
                            LOGGER.error("%s - Error when setting %s. Response: %s", DOMAIN, param, set_resp)
                            raise HomeAssistantError(f"{DOMAIN} - Unable to set {param}")
                        last_err = None
                        break
                    except HomeAssistantError as err:
                        last_err = err
                        is_timeout = isinstance(err.__cause__, asyncio.TimeoutError) or "TimeoutError" in str(err.__cause__ or err)
                        if is_timeout and attempt < attempts - 1:
                            self._apply_control_cooldown("ctrl_timeout", base=12.0)
                            await asyncio.sleep(self._effective_min_request_gap + 0.5)
                            continue
                        raise
                if last_err:
                    raise last_err
            finally:
                if sid:
                    resp: dict[str, Any] | None = None
                    with contextlib.suppress(Exception):
                        resp = await self._enqueue_request(
                            f"ctrl.cgi?sid={sid}&heatingCtrl=0",
                            control_seq_id=control_seq_id,
                            allow_read_timeout=True,
                        )
                    if resp is not None and not resp.get("_timeout"):
                        self._owned_sid = None

    # ---------------------------------------------------------
    # Public control APIs (existing features)
    # ---------------------------------------------------------
    def _bathfill_active(self, data: Mapping[str, Any] | None = None) -> bool:
        data = data or self._last_info or {}
        mode_val = to_int(data.get("mode"))
        bathfill_ctrl = data.get("bathfillCtrl")
        return bool((mode_val in {20, 25, 30, 35}) or is_one(bathfill_ctrl))

    def _bathfill_engaged(self, data: Mapping[str, Any] | None = None) -> bool:
        """Return True when bath fill should be considered engaged (including completion until exit)."""
        data = data or self._last_info or {}
        if self._bathfill_active(data):
            return True
        # Latched from a known start/active session.
        if self._bathfill_latched or self._completion_latched:
            return True
        # Completion markers: treat as engaged unless explicitly ignored after an exit.
        if self._ignore_completion_state:
            return False
        mode_val = to_int(data.get("mode"))
        state_val = to_int(data.get("state"))
        return bool(mode_val == 35 or state_val == 3)

    async def async_set_temp(
        self,
        water_heater: WaterHeaterEntity | None,
        temp: int,
        *,
        allow_bathfill_override: bool = False,
        control_seq_id: str | None = None,
        origin: str | None = None,
        entity_id: str | None = None,
    ) -> None:
        """Set temperature (queued, direction-aware)."""
        info = self._last_info or {}
        if water_heater:
            mintemp = to_int(water_heater.min_temp)
            maxtemp = to_int(water_heater.max_temp)
        else:
            mintemp = to_int(info.get("tempMin"))
            maxtemp = to_int(info.get("tempMax"))

        # min/max can be unavailable early in startup or if getParams.cgi isn't cached yet
        if mintemp is None:
            mintemp = to_int(info.get("tempMin")) or to_int((self._last_params or {}).get("tempMin")) or 0
        if maxtemp is None:
            maxtemp = to_int(info.get("tempMax")) or to_int((self._last_params or {}).get("tempMax")) or 100

        if temp is None:
            if water_heater:
                water_heater.rheem_target_temperature = None
            raise ConditionErrorMessage(type="invalid_temperature", message=f"{DOMAIN} - No temperature was set. Ignoring call to set temperature.")
        try:
            temp_int = int(round(float(temp)))
        except (TypeError, ValueError):
            if water_heater:
                water_heater.rheem_target_temperature = None
            raise ConditionErrorMessage(type="invalid_temperature", message=f"{DOMAIN} - Invalid temperature value {temp!r}.")

        current_setpoint = to_int(info.get("setTemp") or info.get("temp"))
        direction = "up" if current_setpoint is None or temp_int >= int(current_setpoint) else "down"
        requested_at = time.monotonic()
        requested_ts_ms = int(time.time() * 1000)

        if self._cancel_user_settemp and not allow_bathfill_override:
            mode_val = to_int(info.get("mode"))
            flow_val = to_float(info.get("flow"))
            s_timeout = to_int(info.get("sTimeout"))
            if mode_val in (5, None) and flow_val in (0, None) and s_timeout in (0, None):
                self._cancel_user_settemp = False
            else:
                if water_heater:
                    water_heater.rheem_target_temperature = None
                raise ConditionErrorMessage(type="bathfill_active", message=f"{DOMAIN} - Bath fill active; temperature changes are locked.")

        if temp_int < mintemp:
            if water_heater:
                water_heater.rheem_target_temperature = None
            raise ConditionErrorMessage(
                type="minimum_temperature",
                message=f"""{DOMAIN} - Temperature {temp_int}°C is below the device minimum of {mintemp}°C.
                Please set a temperature between {mintemp}°C and {maxtemp}°C.""",
            )
        if temp_int > maxtemp:
            if water_heater:
                water_heater.rheem_target_temperature = None
            raise ConditionErrorMessage(
                type="maximum_temperature",
                message=f"""{DOMAIN} - Temperature {temp_int}°C is above the device maximum of {maxtemp}°C.
                Please set a temperature between {mintemp}°C and {maxtemp}°C.""",
            )

        if not allow_bathfill_override:
            # User override cancels any pending restore.
            self._pending_restore_temp = None
            if "restore_temp" in self._pending_writes:
                self._pending_writes.pop("restore_temp", None)
                self._log_action(
                    "restore_temp",
                    "cancel_user_override",
                    control_seq_id=control_seq_id,
                    extra={"origin": origin or "user", "entity_id": entity_id},
                )

        # Route to bath fill target-temp update if bath fill is active.
        if self._bathfill_active(info):
            # Bath fill temp updates must use current requested volume to avoid resetting.
            req_vol = to_int(info.get("reqbathvol")) or to_int(info.get("bathvol"))
            self._enqueue_write(
                "bathfill_set_temp",
                {
                    "temp": temp_int,
                    "vol": req_vol,
                    "direction": direction,
                    "requested_at": requested_at,
                    "requested_ts_ms": requested_ts_ms,
                    "control_seq_id": control_seq_id,
                    "origin": origin or "user",
                    "entity_id": entity_id,
                },
            )
            return None

        action_ctx = {"origin": origin or ("system" if allow_bathfill_override else "user"), "entity_id": entity_id, "temp": temp_int}
        self._enqueue_write(
            "set_temp",
            {
                "water_heater": water_heater,
                "temp": temp_int,
                "allow_bathfill_override": allow_bathfill_override,
                "control_seq_id": control_seq_id,
                "origin": action_ctx["origin"],
                "entity_id": entity_id,
                "direction": direction,
            },
        )
        return None

    async def _do_set_temp(
        self,
        *,
        water_heater: WaterHeaterEntity | None,
        temp: int,
        allow_bathfill_override: bool,
        control_seq_id: str | None,
        origin: str | None,
        entity_id: str | None,
        direction: str | None = None,
    ) -> None:
        """Execute set_temp (assumes validation already done)."""
        action_ctx = {"origin": origin or ("system" if allow_bathfill_override else "user"), "entity_id": entity_id, "temp": temp}
        self._log_action("set_temp", "start", control_seq_id=control_seq_id, extra=action_ctx)
        try:
            await self._check_control_issues(
                reset_attribute_owner=water_heater,
                reset_attribute="rheem_current_temperature",
                require_idle=(direction or "up") == "up",
                allow_owned_session=(direction or "up") == "down",
                use_cached=True,
            )
            await self._async_set_param(param=f"setTemp={temp}", response_param="reqtemp", response_check=temp, control_seq_id=control_seq_id)
            self._log_action("set_temp", "ok", control_seq_id=control_seq_id, extra=action_ctx | {"post": self._snapshot_state()})
            if control_seq_id == "bathfill_restore":
                self._pending_restore_temp = None
        except Exception as err:
            reason = str(err)
            self._log_action(
                "set_temp",
                "error",
                control_seq_id=control_seq_id,
                extra=action_ctx | {"error": type(err).__name__, "reason": reason, "snapshot": self._snapshot_state()},
            )
            raise

    async def _do_restore_temp(
        self,
        *,
        temp: int,
        entity_id: str | None,
    ) -> None:
        """Execute restore temp as its own queued op."""
        # Use system origin and allow override to bypass user lock.
        await self._do_set_temp(
            water_heater=None,
            temp=temp,
            allow_bathfill_override=True,
            control_seq_id="bathfill_restore",
            origin="system",
            entity_id=entity_id,
            direction="up",
        )

    async def _do_bathfill_set_temp(
        self,
        *,
        temp: int,
        vol: int | None,
        direction: str,
        requested_at: float | None,
        requested_ts_ms: int | None,
        control_seq_id: str | None = None,
        origin: str | None = None,
        entity_id: str | None = None,
    ) -> None:
        """Update bath fill target temp/vol using existing session."""
        try:
            temp = int(round(float(temp)))
        except (TypeError, ValueError):
            raise ConditionErrorMessage(type="invalid_temperature", message=f"{DOMAIN} - Invalid bath fill temp value {temp!r}.")
        info = self._last_info or {}
        if not self._bathfill_active(info):
            raise ConditionErrorMessage(type="bathfill_inactive", message=f"{DOMAIN} - Bath fill inactive; cannot update bath temp.")

        # Ensure volume present
        vol_safe = vol or to_int(info.get("reqbathvol")) or to_int(info.get("bathvol"))
        if vol_safe in (None, 0):
            raise ConditionErrorMessage(type="missing_volume", message=f"{DOMAIN} - Missing bath volume for temp update; will retry.")

        sid = self._owned_sid or to_int(info.get("sid"))
        s_timeout = to_int(info.get("sTimeout"))
        flow_val = to_float(info.get("flow"))

        # Allow a single reacquire if bath fill active, sTimeout==0, and sid missing.
        can_reacquire = sid in (None, 0) and s_timeout in (0, None)

        # For direction up, require flow==0.
        if direction == "up" and flow_val not in (0, None):
            raise ConditionErrorMessage(type="flow_active", message=f"{DOMAIN} - Flow active; will retry bath temp increase.")

        async def _do_request(sid_arg: int | None) -> dict[str, Any]:
            sid_q = sid_arg or 0
            resp = await self._enqueue_request(
                f"ctrl.cgi?sid={sid_q}&bathfillCtrl=1&setBathTemp={temp}&setBathVol={vol_safe}",
                control_seq_id=control_seq_id or "bathfill_set_temp",
            )
            return resp

        try:
            if sid in (None, 0) and can_reacquire:
                resp = await _do_request(None)
            else:
                if sid in (None, 0):
                    raise ConditionErrorMessage(type="invalid_session_id", message=f"{DOMAIN} - Missing bath fill session; will retry.")
                resp = await _do_request(sid)

            resp_sid = to_int(resp.get("sid"))
            resp_ctrl = resp.get("bathfillCtrl")
            if not resp_sid or not is_one(resp_ctrl):
                raise ConditionErrorMessage(type="bathfill_set_temp_retry", message=f"{DOMAIN} - Bath temp update not accepted; will retry.")
            try:
                self._owned_sid = int(resp_sid)
            except Exception:
                self._owned_sid = None
            self._last_info = resp
            self._bathfill_target_temp = temp
            self._log_action(
                "bathfill_set_temp",
                "ok",
                control_seq_id=control_seq_id or "bathfill_set_temp",
                extra={
                    "origin": origin or "user",
                    "entity_id": entity_id,
                    "temp": temp,
                    "vol": vol_safe,
                    "direction": direction,
                    "requested_at": requested_at,
                    "requested_ts_ms": requested_ts_ms,
                },
            )
        except ConditionErrorMessage:
            raise
        except Exception as err:
            self._log_action(
                "bathfill_set_temp",
                "error",
                control_seq_id=control_seq_id or "bathfill_set_temp",
                extra={
                    "origin": origin or "user",
                    "entity_id": entity_id,
                    "temp": temp,
                    "vol": vol_safe,
                    "direction": direction,
                    "error": type(err).__name__,
                    "reason": str(err),
                },
            )
            raise

    async def async_set_session_timer(self, number: NumberEntity, session_timer: float) -> None:
        """Set session timer."""
        if session_timer is None:
            number.rheem_session_timer = None
            raise ConditionErrorMessage(type="invalid_session_timer", message=f"{DOMAIN} - No session timer was set. Ignoring call to set session timer.")
        session_timer_int = int(session_timer)
        if session_timer_int < 60:
            number.rheem_session_timer = None
            raise ConditionErrorMessage(
                type="minimum_session_timer",
                message=f"""{DOMAIN} - An invalid session timer ({session_timer}) was attempted to be set.
                This is below the minimum session timer (60).""",
            )
        if session_timer_int > 900:
            number.rheem_session_timer = None
            raise ConditionErrorMessage(
                type="maximum_session_timer",
                message=f"""{DOMAIN} - An invalid session timer ({session_timer}) was attempted to be set.
                This is above the maximum temperature ({900}).""",
            )

        number.rheem_session_timer = session_timer
        self._enqueue_write(
            "set_session_timer",
            {
                "number": number,
                "session_timer": session_timer_int,
            },
        )
        return None

    async def _do_set_session_timer(self, *, number: NumberEntity, session_timer: int) -> None:
        """Execute session timer update."""
        action_ctx = {"session_timer": session_timer, "entity_id": getattr(number, "entity_id", None)}
        self._log_action("set_session_timer", "start", control_seq_id="set_session_timer", extra=action_ctx)
        try:
            await self._check_control_issues(reset_attribute_owner=number, reset_attribute="rheem_session_timer")
            await self._async_set_param(
                param=f"setSessionTimer={session_timer}",
                response_param="sessionTimer",
                response_check=session_timer,
                control_seq_id="set_session_timer",
            )
            self._control_session_timer = session_timer
            self._log_action("set_session_timer", "ok", control_seq_id="set_session_timer", extra=action_ctx | {"post": self._snapshot_state()})
        except Exception as err:
            reason = str(err)
            self._log_action(
                "set_session_timer",
                "error",
                control_seq_id="set_session_timer",
                extra=action_ctx | {"error": type(err).__name__, "reason": reason, "snapshot": self._snapshot_state()},
            )
            raise
        finally:
            number.rheem_session_timer = None

    # ---------------------------------------------------------
    # Bath fill control
    # ---------------------------------------------------------
    async def async_start_bath_fill(self, temp: int, vol: int, *, origin: str | None = None, entity_id: str | None = None) -> None:
        """Start bath fill with requested temp/volume."""
        try:
            temp = int(round(float(temp)))
            vol = int(round(float(vol)))
        except (TypeError, ValueError):
            raise ConditionErrorMessage(type="invalid_temperature", message=f"{DOMAIN} - Invalid bath fill temp/vol request.")
        data = self._last_info or await self.async_get_data()
        bathtemp_min = to_int(data.get("bathtempMin"))
        bathtemp_max = to_int(data.get("bathtempMax"))
        bathvol_min = to_int(data.get("bathvolMin"))
        bathvol_max = to_int(data.get("bathvolMax"))

        if bathtemp_min is not None and temp < bathtemp_min:
            raise ConditionErrorMessage(
                type="minimum_temperature",
                message=f"{DOMAIN} - Requested bath temp {temp} is below device minimum {bathtemp_min}",
            )
        if bathtemp_max is not None and temp > bathtemp_max:
            raise ConditionErrorMessage(
                type="maximum_temperature",
                message=f"{DOMAIN} - Requested bath temp {temp} is above device maximum {bathtemp_max}",
            )
        if bathvol_min is not None and vol < bathvol_min:
            raise ConditionErrorMessage(
                type="minimum_volume",
                message=f"{DOMAIN} - Requested bath volume {vol} is below device minimum {bathvol_min}",
            )
        if bathvol_max is not None and vol > bathvol_max:
            raise ConditionErrorMessage(
                type="maximum_volume",
                message=f"{DOMAIN} - Requested bath volume {vol} is above device maximum {bathvol_max}",
            )

        self._enqueue_write(
            "bathfill_start",
            {
                "temp": temp,
                "vol": vol,
                "origin": origin or "user",
                "entity_id": entity_id,
            },
        )
        return None

    async def async_cancel_bath_fill(self, *, origin: str | None = None, entity_id: str | None = None) -> None:
        """Cancel bath fill if active."""
        # Clear any pending start/temp updates so cancel truly wins.
        cleared = []
        for op in ("bathfill_start", "bathfill_set_temp"):
            if op in self._pending_writes:
                self._pending_writes.pop(op, None)
                cleared.append(op)
        if cleared:
            self._queue_log("cancel_clears_start", "bathfill_cancel", extra={"cleared": cleared})
        self._enqueue_write(
            "bathfill_cancel",
            {
                "origin": origin or "user",
                "entity_id": entity_id,
            },
        )
        return None

    async def _do_start_bath_fill(self, *, temp: int, vol: int, origin: str | None, entity_id: str | None) -> None:
        """Execute bath fill start (single bathfillCtrl request, no pre-sync)."""
        self._log_action(
            "bathfill_start",
            "start",
            control_seq_id="bathfill_start",
            extra={"origin": origin or "user", "entity_id": entity_id, "temp": temp, "vol": vol},
        )

        now = time.monotonic()
        if self._lockout_until_monotonic and now < self._lockout_until_monotonic:
            raise HomeAssistantError(
                f"{DOMAIN} - Device lockout suspected; backing off until {self._lockout_until_monotonic}"
            )

        try:
            temp = int(round(float(temp)))
            vol = int(round(float(vol)))
        except (TypeError, ValueError):
            raise ConditionErrorMessage(type="invalid_temperature", message=f"{DOMAIN} - Invalid bath fill temp/vol request.")

        # Use cached info; one lightweight refresh if missing.
        data = self._last_info or {}
        if not data:
            try:
                data = await self._enqueue_request("getInfo.cgi")
                self._last_info = data
            except Exception:
                data = {}

        bathtemp_min = to_int(data.get("bathtempMin"))
        bathtemp_max = to_int(data.get("bathtempMax"))
        bathvol_min = to_int(data.get("bathvolMin"))
        bathvol_max = to_int(data.get("bathvolMax"))

        if bathtemp_min is not None and temp < bathtemp_min:
            raise ConditionErrorMessage(
                type="minimum_temperature",
                message=f"{DOMAIN} - Requested bath temp {temp} is below device minimum {bathtemp_min}",
            )
        if bathtemp_max is not None and temp > bathtemp_max:
            raise ConditionErrorMessage(
                type="maximum_temperature",
                message=f"{DOMAIN} - Requested bath temp {temp} is above device maximum {bathtemp_max}",
            )
        if bathvol_min is not None and vol < bathvol_min:
            raise ConditionErrorMessage(
                type="minimum_volume",
                message=f"{DOMAIN} - Requested bath volume {vol} is below device minimum {bathvol_min}",
            )
        if bathvol_max is not None and vol > bathvol_max:
            raise ConditionErrorMessage(
                type="maximum_volume",
                message=f"{DOMAIN} - Requested bath volume {vol} is above device maximum {bathvol_max}",
            )

        mode_val = to_int(data.get("mode"))
        flow_val = to_float(data.get("flow"))
        s_timeout = to_int(data.get("sTimeout"))
        session_timer = to_int(data.get("sessionTimer"))

        if flow_val not in (0, None):
            raise ConditionErrorMessage(type="invalid_flow", message=f"{DOMAIN} - Heater is in use (flow > 0).")
        if mode_val not in (5, None):
            raise ConditionErrorMessage(type="invalid_mode", message=f"{DOMAIN} - Heater not idle (mode={mode_val}).")
        if s_timeout not in (0, None):
            raise ConditionErrorMessage(type="invalid_sTimeout", message=f"{DOMAIN} - Another controller holds session.")

        # Ensure a sufficiently long session timer so bath fill doesn't time out
        # before the user opens/closes the tap.
        desired_session_timer = max(600, int(self._control_session_timer or 0))
        if session_timer in (None, 0) or session_timer < desired_session_timer:
            self._log_action(
                "set_session_timer",
                "enqueue",
                control_seq_id="bathfill_start",
                extra={"session_timer": desired_session_timer, "reason": "bathfill_default"},
            )
            try:
                await self._async_set_param(
                    param=f"setSessionTimer={desired_session_timer}",
                    response_param="sessionTimer",
                    response_check=desired_session_timer,
                    control_seq_id="bathfill_start",
                )
                self._control_session_timer = desired_session_timer
            except Exception as err:
                self._log_action(
                    "set_session_timer",
                    "error",
                    control_seq_id="bathfill_start",
                    extra={
                        "session_timer": desired_session_timer,
                        "reason": "bathfill_default",
                        "error": type(err).__name__,
                        "detail": str(err),
                    },
                )
                raise ConditionErrorMessage(
                    type="session_timer_set_failed",
                    message=f"{DOMAIN} - Failed to set Session Timeout to {desired_session_timer}s. Try again, or set 'number.rheem_session_timeout' to {desired_session_timer} and retry.",
                ) from err

            # Best-effort refresh to ensure session released before starting.
            for _ in range(3):
                refreshed = await self._enqueue_request("getInfo.cgi")
                if refreshed:
                    self._last_info = refreshed
                    data = refreshed
                if to_int((data or {}).get("sTimeout")) in (0, None):
                    break
                await asyncio.sleep(self._effective_min_request_gap)
        current_sid = to_int(data.get("sid"))
        if self._owned_sid and not self._bathfill_active(data) and current_sid in (0, None):
            self._owned_sid = None
        if self._owned_sid and current_sid not in (0, None, self._owned_sid):
            raise HomeAssistantError(f"{DOMAIN} - Another session is active (sid={current_sid}); cannot start a new bath fill.")

        prev_cancel = self._cancel_user_settemp
        prev_prev_temp = self._prev_temp_for_bathfill
        prev_pending_restore = self._pending_restore_temp
        prev_owned_sid = self._owned_sid
        prev_bathfill_latched = self._bathfill_latched
        prev_completion_latched = self._completion_latched
        prev_ignore_completion = self._ignore_completion_state
        try:
            self._cancel_user_settemp = True
            self._bathfill_latched = True
            self._completion_latched = False
            self._ignore_completion_state = False
            self._prev_temp_for_bathfill = to_int(data.get("temp"))
            self._pending_restore_temp = self._prev_temp_for_bathfill

            # Single bathfillCtrl call; no heater control pre-sync.
            async with self._write_lock:
                resp = await self._enqueue_request(f"ctrl.cgi?sid=0&bathfillCtrl=1&setBathTemp={temp}&setBathVol={vol}")
                sid_raw = resp.get("sid")
                sid = to_int(sid_raw)
                if not sid or not is_one(resp.get("bathfillCtrl")):
                    LOGGER.error("%s - Failed to start bath fill, response: %s", DOMAIN, resp)
                    raise ConditionErrorMessage(
                        type="bathfill_start_busy",
                        message=f"{DOMAIN} - Failed to start bath fill; will retry when device free (resp={resp})",
                    )
                try:
                    self._owned_sid = int(sid)
                except Exception:
                    self._owned_sid = None
                # Capture immediate response and, if possible, a follow-up getInfo to include fillPercent.
                self._last_info = resp
                self._bathfill_target_temp = temp
            self._log_action(
                "bathfill_start",
                "ok",
                control_seq_id="bathfill_start",
                extra={"origin": origin or "user", "entity_id": entity_id, "temp": temp, "vol": vol, "post": self._snapshot_state()},
            )
            try:
                refreshed = await self._enqueue_request("getInfo.cgi")
                if refreshed:
                    self._last_info = refreshed
            except Exception:
                # Best-effort; continue even if refresh fails.
                pass
        except Exception:
            self._cancel_user_settemp = prev_cancel
            self._prev_temp_for_bathfill = prev_prev_temp
            self._pending_restore_temp = prev_pending_restore
            self._owned_sid = prev_owned_sid
            self._bathfill_latched = prev_bathfill_latched
            self._completion_latched = prev_completion_latched
            self._ignore_completion_state = prev_ignore_completion
            self._log_action(
                "bathfill_start",
                "error",
                control_seq_id="bathfill_start",
                extra={"origin": origin or "user", "entity_id": entity_id, "temp": temp, "vol": vol},
            )
            raise

    async def _do_cancel_bath_fill(self, *, origin: str | None, entity_id: str | None) -> None:
        """Execute bath fill cancel."""
        now = time.monotonic()
        if self._lockout_until_monotonic and now < self._lockout_until_monotonic:
            raise HomeAssistantError(
                f"{DOMAIN} - Device lockout suspected; backing off until {self._lockout_until_monotonic}"
            )

        info = await self._enqueue_request("getInfo.cgi")
        self._last_info = info

        # Poll flow every 30 seconds until tap is closed (flow = 0)
        # This matches Rheem mobile app behavior - it detects tap closure quickly
        flow_val = to_float(info.get("flow"))
        max_wait_s = 3600  # Maximum 1 hour wait
        poll_interval_s = 30.0  # Check every 30 seconds
        start_time = time.monotonic()

        while flow_val not in (0, None):
            elapsed = time.monotonic() - start_time
            if elapsed >= max_wait_s:
                raise ConditionErrorMessage(
                    type="flow_active_timeout",
                    message=f"{DOMAIN} - Flow still active after {max_wait_s}s; cannot cancel bath fill until tap is closed (flow={flow_val} L/min).",
                )

            # Wait 30 seconds before next check
            await asyncio.sleep(poll_interval_s)

            # Get fresh device state to check flow
            info = await self._enqueue_request("getInfo.cgi", control_seq_id="bathfill_exit_flow_check")
            self._last_info = info
            flow_val = to_float(info.get("flow"))

            # Log progress
            self._log_action(
                "bathfill_cancel",
                "waiting_for_flow",
                control_seq_id="bathfill_exit",
                extra={"flow": flow_val, "elapsed_s": int(elapsed)},
            )

        engaged = self._bathfill_engaged(info)
        if not engaged:
            return

        sid = self._owned_sid
        if not sid:
            sid = to_int(info.get("sid"))
            if sid == 0:
                sid = None

        if not sid:
            s_timeout = to_int(info.get("sTimeout"))
            if s_timeout not in (0, None):
                raise ConditionErrorMessage(
                    type="invalid_sTimeout",
                    message=f"{DOMAIN} - Cannot cancel bath fill: another controller holds session (sTimeout={s_timeout}).",
                )
            try:
                reacquire = await self._enqueue_request("ctrl.cgi?sid=0&bathfillCtrl=0", control_seq_id="bathfill_exit_reacquire")
            except Exception as err:
                raise ConditionErrorMessage(
                    type="invalid_session_id",
                    message=f"{DOMAIN} - Cannot cancel bath fill yet (sid missing); will retry.",
                ) from err
            # Some firmware returns sid=0 for this "no-sid" cancel path; treat as best-effort
            # and confirm via getInfo after sending the command.
            sid = to_int(reacquire.get("sid"))
            if sid in (None, 0):
                sid = 0

        self._log_action(
            "bathfill_cancel",
            "start",
            control_seq_id="bathfill_exit",
            extra={"origin": origin or "user", "entity_id": entity_id, "sid": sid, "owned_sid": self._owned_sid},
        )
        try:
            async with self._write_lock:
                await self._enqueue_request(f"ctrl.cgi?sid={sid}&bathfillCtrl=0", control_seq_id="bathfill_exit")
                self._owned_sid = None
                self._cancel_user_settemp = False
            # Single confirmation poll; if still active, retry later.
            safe_info = await self._enqueue_request("getInfo.cgi", control_seq_id="post_exit_info")
            self._last_info = safe_info
            if self._bathfill_active(safe_info):
                raise ConditionErrorMessage(
                    type="bathfill_exit_retry",
                    message=f"{DOMAIN} - Bath fill still active after cancel; will retry.",
                )
            self._bathfill_latched = False
            self._completion_latched = False
            self._ignore_completion_state = True
            self._bathfill_target_temp = None

            if self._pending_restore_temp is not None:
                mode_val = to_int(safe_info.get("mode"))
                flow_val = to_float(safe_info.get("flow"))
                if mode_val in (5, None) and flow_val in (0, None):
                    await self.async_set_temp(
                        water_heater=None,
                        temp=self._pending_restore_temp,
                        allow_bathfill_override=True,
                        control_seq_id="bathfill_restore",
                        origin="system",
                        entity_id=entity_id,
                    )
                    self._pending_restore_temp = None
            self._log_action(
                "bathfill_cancel",
                "ok",
                control_seq_id="bathfill_exit",
                extra={"origin": origin or "user", "entity_id": entity_id, "sid": sid, "owned_sid": self._owned_sid, "post": self._snapshot_state()},
            )
        except Exception as err:
            self._log_action(
                "bathfill_cancel",
                "error",
                control_seq_id="bathfill_exit",
                extra={"origin": origin or "user", "entity_id": entity_id, "sid": sid, "owned_sid": self._owned_sid, "error": type(err).__name__},
            )
            raise ConditionErrorMessage(
                type="bathfill_cancel_retry",
                message=f"{DOMAIN} - Cancel bath fill will retry ({type(err).__name__})",
            ) from err
