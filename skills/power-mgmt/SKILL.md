# power-mgmt skill

CPU and WiFi power management for battery life on Raspberry Pi.

## Tools

### power_status

Reports the current power state of the system:

- **CPU governor** — reads `/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor`
- **CPU frequency** — current clock speed in MHz
- **WiFi power management** — whether WiFi power saving is on or off (via `iwconfig`)
- **Power state flag** — the value in `~/.zeroclaw/power_state` (`active`, `saving`, or `deep_sleep`)
- **Estimated state** — derived from the governor and flag file

No sudo required.

### power_save

Enters power-saving mode:

1. Sets CPU governor to `powersave` via `sudo cpufreq-set -g powersave`
2. Enables WiFi power management via `sudo iwconfig wlan0 power on`
3. Writes `saving` to `~/.zeroclaw/power_state` — the main agent should check this file and reduce the Telegram poll interval accordingly

**Requires sudo.** The script will report which components succeeded and which were skipped.

### power_wake

Restores full power:

1. Sets CPU governor to `ondemand` via `sudo cpufreq-set -g ondemand`
2. Disables WiFi power management via `sudo iwconfig wlan0 power off`
3. Writes `active` to `~/.zeroclaw/power_state` — the main agent restores the normal Telegram poll interval

**Requires sudo.** The script will report which components succeeded and which were skipped.

## Dependencies

| Package | Debian install | Used by |
|---------|---------------|---------|
| `cpufrequtils` | `sudo apt install cpufrequtils` | `power_save`, `power_wake` |
| `wireless-tools` | `sudo apt install wireless-tools` | all three tools (for `iwconfig`) |

Both tools fail gracefully if these packages are missing — they print install instructions and continue with the remaining steps.

## HEARTBEAT.md integration

Add the following lines to `HEARTBEAT.md` for automatic day/night power cycling:

```
- Every day at 23:00 → run power_save
- Every day at 07:00 → run power_wake
```

This gives ~8 hours of reduced power draw overnight when the Pi is unlikely to receive user messages.

## Auto-trigger suggestion

For more adaptive power management, the agent can implement this logic:

- **If no user messages for 30 minutes** → consider running `power_save` to conserve battery
- **On next user message after idle** → run `power_wake` to restore responsiveness before processing the message

This is optional and should be implemented in the agent's message loop, not in these scripts.

## Platform

Designed for **Raspberry Pi OS** (Debian-based, ARM64). Should work on any Debian/Ubuntu system with the listed dependencies installed.
