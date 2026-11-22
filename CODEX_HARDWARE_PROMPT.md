# Paste this prompt into Codex with the extracted FedTwin-RUL folder open

You are working in the extracted FedTwin-RUL repository. Implement a working
hardware demonstration connecting the existing dashboard's selected engine and
predicted remaining useful life to physical green, amber and red LEDs. Complete
and verify the software, firmware and setup instructions. Preserve the existing
dashboard design, real replay data, trained model and research results.

First read README.md and docs/HARDWARE_PLAN.md, then inspect these actual files:
app/frontend/src/main.jsx, app/frontend/package.json,
app/frontend/public/demo-data.json, scripts/generate_dashboard_data.py,
src/twinfed_moe/models/tcm_model.py and configs/fedtwin_final.yaml.
Read any applicable AGENTS.md. Verify the existing project before changing it.

Project facts:
- The final model is FedTwin-RUL: a 64-D temporal Transformer representation and
  a 16-D operational twin representation fused with concat and trained with
  FedAvg. The Python package retains the older twinfed_moe name. Do not replace
  the final model with the earlier sparse MoE/personalized architecture.
- FD004 historical replay is precomputed from a real checkpoint. The browser
  fetches /demo-data.json; it does not currently call a live inference backend.
- The selected engine, cursor, replay speed and thresholds are React state.
  `frame.predictedRul` is the source for LED commands. Never use trueRul to
  control the hardware or fabricate predictions that make a smoother demo.
- Hardware commands in the UI are previews only. No backend/firmware/transport
  exists yet. Do not claim physical connection or validation until observed.
- Available hardware: Arduino Uno, LEDs, wires and an ESP8266 Wi-Fi thing.
  Exact Uno revision and ESP8266 variant, adapters, resistors and power are
  unconfirmed. Ask the owner for board markings/model and available components
  before final wiring or flashing. Continue independent software work meanwhile.

Implement in phases:
1. Get the existing dashboard running. Use npm ci and the included lockfile.
   Do not run the entire training study. Add a small local API/bridge, using a
   practical stack for this repository, with explicit start commands. Publish
   the exact currently selected frame and validated thresholds from React.
   Send updates on engine/cycle/threshold changes, reset/seek and replay events.
   Keep transport heartbeat distinct from measurement time and replay pause.
   Define an explicit pause/hold policy; never silently label old data live.
2. Implement USB serial to the Uno first as a simple end-to-end path. Add a
   browser-visible connect/disconnect control or local bridge port selection,
   device acknowledgments and bounded timeouts. If Python handles serial,
   provide a separate hardware requirements file including pyserial. Do not
   silently choose an arbitrary COM port or auto-flash a connected device.
3. Once the ESP8266 variant is identified, implement the Wi-Fi path. Prefer
   ESP8266 polling the same local bridge and forwarding compact commands to the
   Uno if both boards must participate. If a development board alone can drive
   the LEDs, describe that as an optional simpler alternative. Confirm with the
   owner whether using both boards is required. Avoid installing AT firmware
   or replacing existing firmware unless needed and agreed with the owner.
4. Provide board-specific sketches, an exact wiring table, dependency versions,
   upload steps, local-network instructions and troubleshooting. Verify electrical
   requirements using official Arduino/Espressif and actual board documentation.
   Account for Uno/ESP voltage differences, proper regulation and level shifting,
   common ground, LED polarity and current-limiting resistors. Do not assume a
   bare ESP-01 module has USB or a regulator. Default to LEDs only; any displayed
   RELAY_ON text is legacy preview and is not permission to control a real load.

Required behavior:
- Defaults: predictedRul > 50 = GREEN, 20 < predictedRul <= 50 = AMBER,
  predictedRul <= 20 = RED. Support fractional RUL. Validate finite values in
  the supported 0–125 range and enforce 0 <= critical < healthy <= 125.
- Transmit a version, session/sequence, engine ID, cycle, raw predicted RUL,
  thresholds, state, replay/transport status and timestamps. Define structured
  JSON for the API and a short bounded serial message suitable for Uno memory.
- Device acknowledgments must identify the accepted sequence and applied LED.
  Show requested versus confirmed state. API success alone is not physical
  device confirmation. An ACK confirms the firmware-applied output, not that
  a broken or unplugged LED actually illuminated.
- Handle malformed payloads, missing data, serial resets, Wi-Fi loss, bridge
  restarts, stale/out-of-order commands and session changes. Emit heartbeats
  even if RUL stays constant. A disconnected/stale state must never appear
  healthy. Use a clearly documented timeout fault pattern with no false claim
  that a communications fault means the engine is failing.
- If hysteresis/debounce is useful, make it configurable and keep raw RUL,
  threshold state and applied LED state distinguishable. Do not edit predictions.
- Scope one authoritative dashboard controller initially, or explicitly handle
  multiple browser tabs. ESP8266 uses the PC's LAN address, never localhost.
  Keep CORS/origins and network binding intentional. Do not publish the bridge
  publicly. Keep Wi-Fi credentials out of source control and examples.
- Keep hardware access optional so the website runs with no board attached.
  Simulator/mock transport must be labeled. Do not use mock acknowledgments to
  show a real device connected.

Validation and delivery:
- Run frontend production build, relevant existing tests and new meaningful
  integration tests. Test RUL 0,20,20.01,50,50.01,125 and invalid/null/non-finite
  values; threshold edits; engine switches; seek/reset; pause; end of replay;
  USB disconnect/reconnect; stale heartbeat; wrong sequence; bridge restart.
- Compile firmware for the identified board where tooling is available. Use
  mocked serial/network transports to test software without a physical board.
  Report what was compiled, simulated, physically flashed and physically tested
  separately. Do not call a hardware test complete without observing it.
- Provide a deterministic LED self-test mode, clearly separate from model-driven
  mode, because a historical replay may not cross every threshold naturally.
- Update README and docs with commands for two terminals, LAN/firewall setup,
  board package/library versions, safe wiring and a short demo procedure.
  Finish with working files, test evidence, remaining physical steps and known
  limitations. Do not stop at a proposal or supply placeholder firmware.
