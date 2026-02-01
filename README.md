# Rheem EziSET HomeKit Custom Component for Home Assistant

_Community-maintained fork focused on HomeKit-friendly Bath Fill preset controls (display name: Rheem EziSET HomeKit; domain remains `rheem_eziset`)._

## Fork notice

This repository is a **community fork** and is **not maintained by the original developer**.

- **Maintained by**: `braydonauswide` (community fork)
- **What this fork adds**: Bath Fill preset switches + an “Exit Bath Fill” control designed to bridge cleanly to Apple Home via HomeKit Bridge.
- **Upstream project (original integration)**: [illuzn/rheem-eziset][upstream]
- **Support**: please open issues/PRs in **this** repository; upstream is not responsible for this fork.

## Important

- Targets **Home Assistant ≥ 2026.1.1** (domain remains `rheem_eziset`; display name is **Rheem EziSET HomeKit**).
- Device has **DoS protection**: never exceed ~1 request/sec to the heater (polling + control combined). If the device stops responding, wait for cooldown; power-cycle if needed.

## Notice

This fork should be considered **beta**. Test carefully before relying on it (especially with bath fill), and prefer the upstream project if you want the original maintainer’s baseline behavior.

## Protocol

Unofficial protocol documentation: [Rheem Eziset Protocol][protocol_docs]

## Warning

While this integration does not allow you to do anything which the app lets you do. Using this integration makes it easier to set your hot water temperature (and also inadvertently set it incorrectly). While 50C meets Australian Standards for hot water, you should be aware of the following:

- Hot water temperature will vary depending on how close/ far the outlet you are using is. Your installer should have tested the temperature at the closest outlet to the heater (but my installer didn't do this).
- There are internal dip switches inside your water heater that offset the read temperature by +/-3C. This means that a setting of 50C may actually be 53C (which is outside the guidelines). The installer manual is available online.
- I strongly recommend setting up an automation to restore your water heater to a default low setting to avoid the risk of inadvertent burns/scalding. Remember, not everyone in your house knows that you have set the hot water to piping hot (and this may especially be an issue with young children or the elderly).

**This integration will set up the following entities.**

| Entity                                | Enabled by Default | Description                                                                                                                                                                                                                                                                                                  |
| ------------------------------------- | ------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| entity name prefixes                  |                    | All entity names will be prefixed with the heaterName read from your device. This defaults to "Rheem" however you can use the app to change it to any 8 character alphanumeric identifier you want.                                                                                                          |
| `water_heater.water_heater`           | True               | Controls the water heater. It reads the min, current and max temps from your water heater. It also supports setting the target temperature to your desired value.                                                                                                                                            |
| `binary_sensor.heater_error`          | True               | Will be off for no error and on for an error. The error code will be provided in sensor.error_code (it is not known at this time what the possible codes are)                                                                                                                                                |
| `binary_sensory.connectivity_problem` | True               | Because of the noisy nature of powerline comms, connectivity between the unit and powerline adapter can be easily degraded. For best results, ensure that the heater and powerline unit are plugged into the same power circuit. A connectivity problem is only raised in the event 5 successive requests fail.                                                                   |
| `sensor.flow`                         | True               | The current flow rate of the water heater in L/min as reported by the water heater.                                                                                                                                                                                                                          |
| `sensor.status`                       | True               | The current status of the water heater. Possible modes are: Idle, Heating, Bath Fill Complete (Off)                                                                                                                                                                                                          |
| `sensor.mode`                         | True               | The current mode of the water heater. Possible modes are: Idle, Heating Control Mode, Heating (Conventional Mode), Idle (Bath Fill Mode Waiting for Tap), Heating (Bath Fill Mode), Heating (Bath Fill Mode Flowing), Idle (Bath Fill Mode Complete)                                                         |
| `sensor.session_timeout`              | True               | The time in seconds until the current user's session times out. This will only apply if there is a communication error with the water heater or if somebody is using the app or a physical device in the house to control the water heater. You are locked out of controls while someone else is in control. |
| `number.session_timeout`              | True               | Configures the default session timeout. It is recommended to set the session time out to the lowest permitted value of 60. Control grabs (`heatingCtrl=1`) now request 60s by default; adjust this number entity if you need longer. Bath fill control uses the device's own timeout (unchanged).               |
| `sensor.status_raw`                   | False              | The raw status code provided by the water heater. Known status codes are 1, 2, 3.                                                                                                                                                                                                                            |
| `sensor.mode_raw`                     | False              | The raw mode code provided by the water heater. Known mode codes are 5, 10, 15, 20, 25, 30, 35.                                                                                                                                                                                                              |
| `sensor.heater_error_raw`             | False              | The raw heater error code provided by the water heater. 0 is normal but the other codes are unknown.                                                                                                                                                                                                         |
| `sensor.current_temperature`          | False              | Useful for setting up a safety automation.                                                                                                                                                                                                                                                                   |
| `sensor.heater_model`                 | False              | Reports the heater model. The only known model at this stage is "1".                                                                                                                                                                                                                                         |
| `sensor.heater_name`                  | False              | Reports the internal heater name (configurable via the app).                                                                                                                                                                                                                                                 |

## Installation

### HACS

[![Open your Home Assistant instance and open a repository inside the Home Assistant Community Store.](https://my.home-assistant.io/badges/hacs_repository.svg)](https://my.home-assistant.io/redirect/hacs_repository/?owner=braydonauswide&repository=rheem-eziset&category=integration)

Notes on versions in HACS:
- HACS lists the **default branch** plus (up to) the **5 latest GitHub releases** for version selection.
- If you see a message like "Commit `<hash>` will be downloaded", it usually means this repository has **no GitHub releases** (or HACS hasn’t refreshed its cache yet).

### Manual Installation

1. Using the tool of choice open the directory (folder) for your HA configuration (where you find `configuration.yaml`).
1. If you do not have a `custom_components` directory (folder) there, you need to create it.
1. In the `custom_components` directory (folder) create a new folder called `rheem_eziset`.
1. Download _all_ the files from the `custom_components/rheem_eziset/` directory (folder) in this repository.
1. Place the files you downloaded in the new directory (folder) you created.
1. Restart Home Assistant
1. In the HA UI go to **Settings → Devices & Services → Integrations** click **Add Integration** and search for **Rheem EziSET HomeKit**

### Upgrade (existing users)

- Domain stays `rheem_eziset`; only the display name changed to “Rheem EziSET HomeKit”.
- HACS upgrade:
  - Update the integration in HACS, then restart Home Assistant.
  - Open **Settings → Devices & Services → Rheem EziSET HomeKit → Configure**, set scan interval (≥2s; recommend 5s), and configure bath-fill presets.
- Manual upgrade:
  - Stop HA, replace `config/custom_components/rheem_eziset/` with the updated version, start HA, and configure presets via Options.

### HomeKit Bridge exposure (Apple Home/Siri)

Recommend a dedicated HomeKit Bridge instance filtered to only the Bath Fill entities to avoid clutter and stay under the 150 accessory limit:
- Bath Profile input_select (set `type: faucet`)
- Bath Fill switch (set `type: faucet`)
- Exit Bath Fill switch
- Bath Fill Status sensor
- Bath Fill Progress (humidity-style % sensor)

Example YAML (replace entity_ids with your own from HA → Entities):

```yaml
homekit:
  - name: "Rheem Bath"
    mode: bridge
    filter:
      include_entities:
        - input_select.rheem_<entryIdPrefix>_bath_profile
        - switch.<your_entity_id_for_bath_fill>
        - switch.<your_entity_id_for_exit_bath_fill>
        - sensor.<your_entity_id_for_bath_fill_status>
        - sensor.<your_entity_id_for_bath_fill_progress>
    entity_config:
      input_select.rheem_<entryIdPrefix>_bath_profile:
        type: faucet
      switch.<your_entity_id_for_bath_fill>:
        type: faucet
      switch.<your_entity_id_for_exit_bath_fill>:
        type: switch
      sensor.<your_entity_id_for_bath_fill_progress>:
        # humidity device class gives a native % tile in Apple Home
        # no extra config needed
```

Notes:
- Do not rename entity_ids after pairing to HomeKit; it can create duplicate accessories.
- The Bath Profile entity is an `input_select` entity (HomeKit-friendly); its options show labels with temperature and volume (e.g., "Children (39C, 110L)").
- **Bath Profile entity_id format**: `input_select.rheem_<entryIdPrefix>_bath_profile` where `<entryIdPrefix>` is the first 8 characters of your config entry ID (lowercase, stable even if you rename the device). Find your entry ID in Settings → Devices & Services → Rheem EziSET HomeKit → Configure, or check HA logs during integration setup.
- **Note**: If you see an entity with a suffix like `_2` (e.g., `input_select.rheem_<entryIdPrefix>_bath_profile_2`), this is normal and the integration handles it automatically. The entity name is "Bath Profile" (no IP address) and options don't include slot numbers for cleaner HomeKit display.
- If the device stops responding after rapid toggles, it may be in DoS lockout; wait for cooldown or power-cycle the heater/bridge.
- UI alternative: in the HomeKit Bridge integration (UI), add only the Bath Profile selector, Bath Fill switch, and Exit Bath Fill to the include filter. Per-entity `entity_config` (e.g., `type: faucet`) may require YAML depending on HA version.
- When starting Bath Fill, the integration aligns the heater’s target temperature to the Bath Fill temperature and may cancel any pending/optimistic temp change; after Bath Fill ends, the previous temp is restored when the heater is idle.

## Configuration (UI Options)

Options UI has been unreliable on some HA versions. Prefer managing Bath Fill presets via services (see below). Scan interval remains in Options; defaults to 5s (min 2s).

## How to use

### Set water heater temperature
- Use the `water_heater` entity; adjust target temperature in HA UI.
- Safety: consider an automation to restore a safe default temperature after use.

### Start a bath fill (from HA)
Preconditions (enforced by the integration):
- Tap closed / no hot water flow.
- Heater idle (mode 5).
- No other controller session (`sTimeout` is 0).

Steps:
1. Select the desired bath profile using the **Bath Profile** dropdown.
2. Turn **ON** the **Bath Fill** switch.
3. Open the bath hot tap.
4. When filling completes (mode 35), **close the tap**, then **turn the Bath Fill switch OFF** (or use Exit Bath Fill).
5. After starting, the `water_heater` entity exposes `bathfill_target_temperature` so you can confirm the requested bath fill setpoint.

### Cancel / Exit
- Turn **OFF** the **Bath Fill** switch, or use **Exit Bath Fill** (momentary switch).
- Safety: exiting bath fill may allow hot water to resume if a tap is open—close the tap first.

### End bath (with power fallback)

If the heater is unreachable or times out, the API cancel may fail and the bath keeps running. The **End Bath (with fallback)** script tries the API cancel first, then after a delay falls back to cutting power via your hot water switch (e.g. `switch.rheem_hot_water`) and optionally restores power.

- **Script**: `script.end_bath_with_fallback` (from `config/scripts.yaml`; add `script: !include scripts.yaml` to `configuration.yaml` if you use the sample config).
- **Variables**: `bath_fill_entity_id` (your Bath Fill switch, e.g. `switch.rheem_eziset_xxxx_bath_fill`), `hot_water_switch_entity_id` (your hot water power switch, e.g. `switch.rheem_hot_water`). Leave `hot_water_switch_entity_id` empty to skip the power fallback.
- **UI button**: Call `script.turn_on` with `entity_id: script.end_bath_with_fallback` and `variables`: `bath_fill_entity_id: <your_bath_fill_entity>`, `hot_water_switch_entity_id: switch.rheem_hot_water` (or leave empty to skip power fallback). Find your Bath Fill entity_id in **Developer Tools → States** or on the device card.

## Manage bath fill presets via services (recommended)

Use HA Services (Developer Tools → Services) instead of Options UI:

- `rheem_eziset.set_bathfill_preset` (preferred)
  - Target: `device_id` or any `entity_id` from this integration.
  - Fields: `slot` (1–6), `enabled`, `name`, `temp` (°C), `vol` (L).
  - Validation: slot in range; temp/vol required when enabling; enabled presets must have unique `(temp, vol)` pairs.
- `rheem_eziset.disable_bathfill_preset`
  - Target: `device_id` or `entity_id`.
  - Field: `slot`.
- `rheem_eziset.reset_bathfill_presets`
  - Target: `device_id` or `entity_id`.
  - Resets to defaults: Preset 1 Children 110L@39C; Preset 2 Adults 140L@43C; others disabled.

After calling a service, the integration reloads automatically; preset switches update accordingly. Expose only the preset switches + Exit Bath Fill to HomeKit as before.

## Troubleshooting

- **Another controller holds session (`sTimeout` ≠ 0)**  
  Wait until the session times out or the other controller releases control, then try again.

- **Device lockout (DoS protection)**  
  Avoid rapid toggling. If commands stop working, wait for cooldown; if it doesn’t recover, power-cycle the heater/bridge.

- **Device state unknown / values look wrong**  
  Ensure scan interval is not too low (recommend 5s). Wait for the coordinator to refresh; check network connectivity.

## Contributions are welcome!

If you want to contribute to this please read the [Contribution guidelines](CONTRIBUTING.md)

## Credits

[illuzn](https://github.com/illuzn) for the original integration and protocol documentation work this fork is based on.

[ludeeus](https://github.com/ludeeus) for the amazing [Integration Blueprint](https://github.com/ludeeus/integration_blueprint)

[bajarrr](https://github.com/bajarrr) for his work in deciphering the api.

[dymondj](https://github.com/dymondj) for correcting my erroneous knowledge regarding how the powerline unit works.

## Intellectual Property

Rheem and EZiSET are trademarks of Rheem Australia Pty Ltd in Australia and their respective owners worldwide. These trademarks are used on these pages under fair use and no affiliation or association with Rheem Australia Pty Ltd or any of its group companies is intended or to be inferred by the use of these marks.

---

[upstream]: https://github.com/illuzn/rheem-eziset
[protocol_docs]: https://illuzn.github.io/Rheem-Eziset-Protocol/
