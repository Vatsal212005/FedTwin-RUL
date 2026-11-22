# Hardware implementation plan (not implemented in this bundle)

The dashboard currently uses saved inference results. Preserve this truthful
demonstration: historical model output controls real indicator LEDs. Without
new compatible sensors and a validated acquisition pipeline, do not call it
live physical engine prognostics.

## Recommended sequence

1. Dashboard -> local bridge -> USB serial -> Arduino Uno -> LEDs.
2. Dashboard -> same bridge on PC LAN -> ESP8266 Wi-Fi -> Uno -> LEDs.

The USB version proves the dashboard/command/firmware flow with fewer variables.
For Wi-Fi, identify the exact ESP8266 module first. An ESP-01, a NodeMCU and a
Wemos/D1 board have different power/programming/pin arrangements. Document both
development-board alternatives and the chosen wiring. No hardware firmware is
present in the source snapshot.

## Single source of truth

In `app/frontend/src/main.jsx`, the selected frame is
`engine.timeline[cursor]`. Send `frame.predictedRul` and validated thresholds to
the bridge, including engine ID/cycle, a controller session and sequence number.
Do not let the bridge advance its own separate engine replay. Otherwise the
LEDs can disagree with what the presenter is showing. Use versioned payloads.

Suggested local API responsibilities, to implement after choosing a stack:

- POST /api/hardware/state: validate and accept current dashboard state.
- GET /api/hardware/state: latest command for the Wi-Fi client, with age/status.
- POST /api/hardware/ack: device identity, accepted session/sequence and LED state.
- GET /api/hardware/status: bridge/device liveness and last acknowledgment.

These routes are a design proposal, not existing endpoints. Serial mode can use
the same internal state and a bounded newline-delimited protocol. Uno has limited
memory: avoid large JSON documents/dynamic unbounded String buffers on firmware.
Treat replay pause separately from transport loss. A paused frame may remain on
display under a documented hold policy, but its original measurement time must
remain visible. Device heartbeats alone do not make a sensor reading new.

## LEDs and validation

Default policy: RUL > 50 green, 20 < RUL <= 50 amber, RUL <= 20 red.
Missing, out-of-range, non-finite or expired data use a distinct UNKNOWN/FAULT
indication. Define the failure pattern explicitly. Do not leave green illuminated
after losing the controlling state. Send heartbeats when the value is unchanged.
Show requested state and acknowledged device state separately. Debounce should
not overwrite the raw model value. Add a labeled self-test to exercise all LEDs.

## Electrical checks before wiring

Confirm the board revision and available resistors, cables, breadboard,
USB/programming adapter, stable power supply and any required level shifter.
The Uno R3 operates at 5 V; ESP8266 chip power and logic require appropriate
lower-voltage handling. A development board may regulate its power input without
making its GPIO 5 V tolerant. Do not treat an unspecified board as pin-compatible
with another variant. Use current-limiting resistors and a common ground under a
verified wiring plan. The bundled RELAY_ON string is only historical UI text;
no physical relay/load is in scope unless separately requested.

Consult the actual board schematic and these official sources before choosing
pins, power arrangements and level shifting:

- [Arduino Uno Rev3 specifications](https://store.arduino.cc/products/arduino-uno-rev3)
- [Espressif ESP8266EX datasheet](https://documentation.espressif.com/0a-esp8266ex_datasheet_en.html)
- [Espressif hardware design guidelines](https://documentation.espressif.com/esp8266_hardware_design_guidelines_en.html)

## Completion evidence

Build the frontend; test mapping boundaries, validation and API/transport flow;
compile board-specific firmware; exercise simulated disconnects; then test the
physical LEDs on the actual boards. State clearly which of those checks actually
ran. Never treat an API response, simulator or generated sketch as proof that a
physical device was connected.
