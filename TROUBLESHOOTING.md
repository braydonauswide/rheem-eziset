# Troubleshooting Guide

This guide helps you diagnose and resolve common issues with the Rheem EziSET HomeKit integration.

## Common Error Messages

### "Device is temporarily unavailable due to repeated connection failures"

**What it means:** The device has entered a lockout state after 3 consecutive connection failures.

**Solutions:**
1. Wait for the lockout period to expire (10s, 30s, 60s, or 180s depending on failure count)
2. Check your network connection - ensure the device and Home Assistant are on the same network
3. Verify the device is powered on and responsive
4. Check the device's IP address is correct in the integration configuration
5. If the issue persists, power-cycle the water heater

**How to check:** Look at the "Health Status" sensor - it will show "unavailable" during lockout and "recovering" when health check is in progress.

### "Cannot change temperature while bath fill is active"

**What it means:** You're trying to change the water heater temperature while a bath fill operation is in progress.

**Solutions:**
1. Cancel the active bath fill operation first
2. Wait for the device to return to idle mode
3. Then try setting the temperature again

**How to check:** The "Bath Fill Active" binary sensor will be "on" when bath fill is active.

### "Temperature X°C is below/above the device minimum/maximum"

**What it means:** You're trying to set a temperature outside the device's allowed range.

**Solutions:**
1. Check the device's min/max temperature limits (shown in the water heater entity attributes)
2. Set a temperature within the allowed range
3. The integration enforces these limits to prevent device errors

### "Device returned invalid data"

**What it means:** The device responded but the data format was unexpected or invalid.

**Solutions:**
1. Check your network connection
2. Wait a few moments and try again
3. If it persists, check the debug log for more details
4. Power-cycle the device if the issue continues

## Device Lockout Recovery

If the device enters lockout:

1. **Wait for cooldown:** The integration automatically backs off for 10s, 30s, 60s, or 180s
2. **Health check:** After lockout expires, a health check probe is performed
3. **Recovery:** If health check succeeds, normal operations resume
4. **Extension:** If health check fails, lockout is extended by one level

**Monitor recovery:**
- Check the "Health Status" sensor - it will show "recovering" during health check
- Check entity attributes for "device_lockout_remaining_s" to see time remaining

## Connectivity Issues

### Symptoms
- Entities show as "unavailable"
- Frequent timeouts
- "Connection error" messages in logs

### Solutions
1. **Network connectivity:**
   - Ensure device and Home Assistant are on the same network
   - Check network cables and Wi-Fi signal strength
   - Verify device IP address hasn't changed (use static IP if possible)

2. **Powerline communication:**
   - Ensure heater and powerline adapter are on the same electrical circuit
   - Check for electrical interference (large appliances, dimmers)
   - Try different power outlets

3. **Device power:**
   - Verify device is powered on
   - Check for power outages or circuit breaker trips
   - Power-cycle the device if needed

## Bath Fill Issues

### Bath fill won't start

**Check:**
1. Device is idle (mode = 5, flow = 0, sTimeout = 0)
2. No active session held by another controller
3. Preset is properly configured (temp/vol within device limits)
4. Device is not in lockout

**Solutions:**
1. Wait for device to be idle
2. Check preset configuration in integration options
3. Try the "Exit Bath Fill" switch if device appears stuck
4. Check debug log for specific error messages

### Bath fill won't cancel

**Check:**
1. Bath fill is actually active (check "Bath Fill Active" sensor)
2. Device is responsive (check "Health Status" sensor)

**Solutions:**
1. Wait a moment and try again (device may be processing)
2. Use the "Exit Bath Fill" switch
3. Check debug log for session reacquisition errors
4. If stuck, power-cycle the device (last resort)

### Bath fill auto-exits unexpectedly

**Check:**
1. Auto-exit is enabled (check "Auto Exit Bath Fill" switch)
2. Fill percent reached 80% AND flow is 0
3. OR completion reached (mode=35 or state=3) AND flow is 0

**Solutions:**
1. Disable auto-exit if you want manual control
2. Ensure tap is closed when auto-exit triggers
3. Check that fill percent is accurately reported

## Debug Logs

The integration writes detailed debug logs to:
`/config/custom_components/rheem_eziset/debug.log`

**Log format:** NDJSON (newline-delimited JSON)

**Key log messages:**
- `HTTP start`: Request being sent
- `HTTP ok`: Request succeeded
- `HTTP error`: Request failed
- `lockout`: Device entered lockout
- `health_check`: Health check performed

**How to check logs:**
1. In Home Assistant, go to Developer Tools > Logs
2. Filter for "rheem_eziset"
3. Or access the file directly in the config directory

**Log retention:** Logs are automatically pruned to 24 hours and cleared on integration startup.

## When to Power-Cycle the Device

Power-cycle (turn off and on) the water heater if:
1. Device is stuck in lockout and health checks keep failing
2. Bath fill is stuck and won't cancel
3. Device is completely unresponsive
4. After network configuration changes

**Warning:** Power-cycling will interrupt any active operations (heating, bath fill).

## Health Status Sensor

The "Health Status" sensor provides real-time device health information:

**States:**
- `healthy`: All endpoints responding normally
- `degraded`: Some non-critical endpoints failing (using cached data)
- `unavailable`: Device lockout or critical failures
- `recovering`: Lockout expired, performing health check

**Attributes:**
- `endpoint_info`: Health of getInfo.cgi endpoint
- `endpoint_params`: Health of getParams.cgi endpoint
- `endpoint_version`: Health of version.cgi endpoint
- `lockout_until`: Timestamp when lockout expires (if in lockout)
- `lockout_remaining_s`: Seconds remaining in lockout (if in lockout)

## Getting Help

If issues persist:
1. Check the debug log for detailed error information
2. Review entity states and attributes
3. Check the Health Status sensor
4. Open an issue on GitHub with:
   - Error messages
   - Relevant log excerpts
   - Device state information
   - Steps to reproduce
