# FedTwin-RUL ESP32 Health Indicator

A physical alarm panel for the FedTwin-RUL engine-life predictor. The laptop runs
the dashboard with the trained FedTwin-RUL model's predictions. An ESP32 board
shows the **predicted Remaining Useful Life (RUL)** of the selected engine as a
coloured light and a buzzer, the same way an industrial machine-health indicator would.

```
┌──────────────────────────┐   HTTP (local)   ┌───────────────────┐   USB serial   ┌──────────────────────────┐
│  FedTwin-RUL dashboard   │ ───────────────► │  rul_bridge.py    │ ─────────────► │  ESP32                   │
│  (browser, React)        │   current frame  │  (Python, laptop) │   JSON lines   │  🔵 D21  HEALTHY         │
│  engine + cycle +        │ ◄─────────────── │                   │ ◄───────────── │  🟡 D22  WARNING         │
│  predicted RUL           │   board's reply  │                   │   ACK replies  │  🔴 D23  CRITICAL        │
└──────────────────────────┘                  └───────────────────┘                │  🔊 D18  buzzer          │
                                                                                   └──────────────────────────┘
```

> **What this prototype is:** a prediction-to-hardware demonstration. The model's
> predictions come from real NASA C-MAPSS FD004 test engines and a trained
> FedTwin-RUL checkpoint. The ESP32 displays that prediction.
> **What it is not:** an engine sensor. Nothing on the board measures a real engine.

---

## Contents

1. [Quick start](#1-quick-start)
2. [Indicator behaviour](#2-indicator-behaviour)
3. [Using the dashboard](#3-using-the-dashboard)
4. [Presentation script](#4-presentation-script)
5. [Hardware and wiring](#5-hardware-and-wiring)
6. [One-time software setup](#6-one-time-software-setup)
7. [How it works](#7-how-it-works)
8. [About the colour-changing LEDs](#8-about-the-colour-changing-leds)
9. [Configuration reference](#9-configuration-reference)
10. [Troubleshooting](#10-troubleshooting)
11. [What was tested](#11-what-was-tested)
12. [Files in this folder](#12-files-in-this-folder)

---

## 1. Quick start

The firmware is already on the ESP32 and stays there after unplugging, so you
don't need to upload it again.

1. **Plug the ESP32 into the laptop** with a *data* USB cable (not charge-only).
   The board runs a self-test: yellow, red, blue, then a short beep.
2. **Close the Arduino IDE** (or at least its Serial Monitor). Only one program can use the USB port.
3. In the `FedTwin-RUL` folder, **double-click `START_DEMO.bat`**.
4. Two black windows open. **Keep both open.**
   - *FedTwin-RUL bridge*: talks to the ESP32. It should say `connected to ESP32 on COM4`.
   - *FedTwin-RUL dashboard*: the web server.
5. The browser opens **http://127.0.0.1:5173/**. On the **ESP32 indicator command**
   card, check for 🟢 **Bridge running** and 🟢 **ESP32 confirmed …**.

**To stop:** close both black windows. The board starts blinking yellow (OFFLINE), and you can unplug it.

> Keep **only one** dashboard tab open. If several are open, the tab you're using
> (the focused one) controls the board, and the others show
> "Another tab controls the board".

---

## 2. Indicator behaviour

| Predicted RUL | State | Light | Buzzer (D18) |
|---|---|---|---|
| **> 50** cycles | HEALTHY | 🔵 steady blue (D21) | silent |
| **20 – 50** cycles | WARNING | 🟡 yellow (D22) | short chirp every 2 s |
| **≤ 20** cycles | CRITICAL | 🔴🔵 flashing red/blue (D23) | fast beeping (250 ms on/off) |
| no data for 5 s, invalid data, or bridge stopped | OFFLINE | 🟡 yellow blinks slowly (1 s) | silent |

- The decision uses **only the model's predicted RUL**, never the NASA "true" RUL.
- The thresholds (20 and 50) follow the **threshold controls on the dashboard**. If
  you change them there, the board uses the new values immediately.
- **OFFLINE is a safety feature:** if the laptop, bridge or cable stops, the board
  never keeps showing an old "healthy" state.

---

## 3. Using the dashboard

### Control bar (top of the page)
| Control | What it does |
|---|---|
| **Test engine** | Choose engine 1, 25, 50 or 100: real FD004 test engines with saved model predictions. |
| **Start replay / Pause** | Play the engine cycle by cycle. The board follows each cycle. |
| **Reset** | Back to cycle 1. |
| **Replay speed** | 1× – 8× (cycles per second). |
| **−10 / −1 / +1 / +10** | Step through cycles (pauses the replay). |
| **Cycle number box** | Type an exact cycle to jump to (pauses the replay). |
| **Slider** | Drag to any cycle. |
| Text next to the buttons | Current RUL and state, e.g. `RUL 120.1 · HEALTHY`. |

### Cards
| Card | Meaning |
|---|---|
| **Predicted remaining life** | Model's RUL in cycles. "NASA reference" is the true value, shown for comparison only. |
| **Machine health** | HEALTHY / WARNING / CRITICAL with three lamps. |
| **ESP32 indicator command** | What is being sent to the board, plus two status dots: **Bridge running** and **ESP32 confirmed <STATE>**. The second dot only turns green when the board itself replies. |

### Hardware test buttons (on the ESP32 card)
**Test HEALTHY**, **Test WARNING** and **Test CRITICAL** put a fixed test value on
the board. The card then shows "Hardware TEST value … not a model prediction".
Press **Follow model** to go back to the model's predictions.

### Other pages
**Federation**, **Results** and **How it works** explain the federated training
(10 clients), the accuracy comparison and the model design.

---

## 4. Presentation script

### Where the model's predictions reach each state

| State | Engine | Cycle | Predicted RUL |
|---|---|---|---|
| HEALTHY | 100 | 50 | 120.1 |
| WARNING | 100 | 1 | 24.6 |
| CRITICAL | **25** | **483** | 19.5 |

The model rarely predicts RUL ≤ 20 for these four engines. **Only engine 25,
cycles 482, 483 and 485** go below 20. During a normal replay they pass in under
a second, and the last cycle (486, RUL 20.15) is WARNING. To show CRITICAL from the
model, **type the cycle number** so the replay stops there.

### Suggested demo (about 3 minutes)

| Step | Action | Board | What to say |
|---|---|---|---|
| 1 | Show the page header | – | "FedTwin-RUL is trained with federated learning across 10 clients on NASA C-MAPSS turbofan data." |
| 2 | Engine 100 → type cycle **50** | 🔵 blue, silent | "Predicted RUL is 120 cycles, so the engine is healthy." |
| 3 | Type cycle **1** | 🟡 yellow + chirp | "RUL 24.6: inside the warning zone, so maintenance is approaching." |
| 4 | Engine 25 → type cycle **483** | 🔴🔵 flashing + beeping | "RUL 19.5 cycles: critical, so the alarm goes on." |
| 5 | Press **Start replay** | changes with the prediction | "The hardware follows the prediction cycle by cycle." |
| 6 | Close the browser tab, wait 5 s | 🟡 slow blink | "If data stops, the board shows OFFLINE instead of a stale 'healthy'." |

The test buttons are a quick backup. Say "this is a test value" when you use them.

### Likely questions
- **"Is this a real engine?"** No. The laptop shows the trained model's predictions
  on saved NASA test-engine data, and the ESP32 is the indicator. It shows how a
  prediction reaches physical hardware.
- **"How does the board know the state?"** The dashboard sends the current frame
  to a Python bridge. The bridge sends a JSON line over USB. The ESP32 decides the
  state from the predicted RUL and replies with an ACK, which the dashboard displays.
- **"Why does CRITICAL flash red and blue?"** The LEDs have a built-in
  colour-changing chip. See [section 8](#8-about-the-colour-changing-leds).
  A plain red LED would give solid red.

---

## 5. Hardware and wiring

### Components used
| Item | Notes |
|---|---|
| ESP32 development board, 30-pin, CP2102 USB (shows as **COM4**) | Any "ESP32 Dev Module" board works. |
| 2 × red/blue **colour-changing** LEDs (2 legs) | One for HEALTHY (D21), one for CRITICAL (D23). |
| 1 × yellow LED | WARNING (D22). |
| 3 × 220 Ω resistors | **One per LED.** |
| Active buzzer (beeps by itself on 3.3 V) | D18. |
| Breadboard, jumper wires, data USB cable | |

No OLED display is used in this version.

### Wiring

**Unplug the USB cable before changing any wire.**

| Part | Leg | Connect to |
|---|---|---|
| 🔵 HEALTHY LED (colour-changing) | long leg (+) | 220 Ω → **D21** |
| | short leg (−) | **GND** |
| 🟡 WARNING LED (yellow) | long leg (+) | 220 Ω → **D22** |
| | short leg (−) | **GND** |
| 🔴 CRITICAL LED (colour-changing) | long leg (+) | 220 Ω → **D23** |
| | short leg (−) | **GND** |
| 🔊 Buzzer | + (or module I/O, IN, S) | **D18** |
| | − (module GND) | **GND** |
| | VCC (modules with 3 pins only) | **3V3** |

```
D21 ──[220Ω]──►|── HEALTHY LED ── GND
D22 ──[220Ω]──►|── YELLOW LED ─── GND
D23 ──[220Ω]──►|── CRITICAL LED ─ GND
D18 ───────────── buzzer (+)    buzzer (−) ── GND
```

### Safety rules
- **Every LED needs its own 220 Ω resistor.** Without one, the LED or the ESP32 pin can be damaged.
- **One LED per pin.** Two LEDs on the same pin always switch together.
- ESP32 pins are **3.3 V**. Never connect 5 V to a GPIO pin.
- The long leg (+) goes towards the pin. The short leg (−), on the flat side of the LED rim, goes to GND.
- If a buzzer sounds when it should be silent, it is a *low-level trigger* module.
  Set `BUZZER_ACTIVE_LOW = true` in the sketch and upload again.

---

## 6. One-time software setup

Already done on this laptop. Repeat these steps on a new computer.

| Tool | Version used | Install |
|---|---|---|
| Python | 3.11 | `pip install -r hardware/requirements.txt` (installs `pyserial`) |
| Node.js | 22 LTS | `cd app/frontend` then `npm ci` |
| Arduino IDE | 2.3.4 | with the **esp32 by Espressif** board package **3.3.12** |

### Uploading the firmware
**Arduino IDE:** open `hardware/esp32_rul_indicator/esp32_rul_indicator.ino`,
choose board **ESP32 Dev Module**, port **COM4**, then click **Upload**.

**Or from PowerShell** (from the `FedTwin-RUL` folder):
```powershell
$cli = "C:\Program Files\Arduino IDE\resources\app\lib\backend\resources\arduino-cli.exe"
& $cli compile --fqbn esp32:esp32:esp32 hardware/esp32_rul_indicator
& $cli upload  --fqbn esp32:esp32:esp32 -p COM4 hardware/esp32_rul_indicator
```
- If the upload stalls at `Connecting....`, hold the **BOOT** button on the board until it starts.
- Close the Serial Monitor and any bridge window first; otherwise you'll get "port busy".

### Running the parts by hand (instead of START_DEMO.bat)
```powershell
# Terminal 1: bridge (auto-detects the ESP32 port)
python hardware/rul_bridge.py --port auto

# Terminal 2: dashboard
cd app/frontend
npm run dev -- --host 127.0.0.1 --port 5173 --strictPort
```
Then open http://127.0.0.1:5173/.

### Other bridge modes
```powershell
python hardware/rul_bridge.py --port auto --mode test                  # labelled TEST values 75 / 35 / 15 / invalid
python hardware/rul_bridge.py --port auto --mode replay --engine 25    # replay saved predictions without the browser
python hardware/rul_bridge.py --list-ports                             # show serial ports
python hardware/rul_bridge.py --dry-run                                # no board: print messages only
```

---

## 7. How it works

### Data path
1. The dashboard shows a frame: engine, cycle and **predicted RUL**, the saved
   output of the FedTwin-RUL checkpoint (alpha 0.1, seed 42, round 33).
2. Whenever the frame or thresholds change, and every 1.5 s as a heartbeat, the
   dashboard POSTs the frame to the bridge at `http://127.0.0.1:8765/state`.
3. The bridge checks it (RUL a finite number from 0 to 125, thresholds
   0 ≤ critical < healthy ≤ 125) and writes one JSON line to the ESP32. It also
   resends the latest frame every second.
4. The ESP32 decides the state, drives the LEDs and buzzer, and replies with an ACK.
5. The bridge passes the last ACK back to the dashboard, which shows
   "ESP32 confirmed <STATE>".

### Messages (USB serial, 115200 baud, one line each)
Laptop → ESP32:
```json
{"seq":12,"engine_id":25,"cycle":483,"rul":19.47,"critical":20,"healthy":50,"valid":true,"mode":"dashboard"}
```
ESP32 → laptop:
```
ACK seq=12 state=CRITICAL rul=19.47        reply to every valid message
STATE CRITICAL engine=25 cycle=483 ...      printed when the state changes
STATE OFFLINE reason=timeout                no valid data for 5 s
ERR seq=13 bad_rul                          rejected message
SELFTEST start / READY ...                  after power-up
```
`mode` is `dashboard` (model prediction) or `test` (test button).

### Safety and reliability features
| Situation | What happens |
|---|---|
| Dashboard closed or frozen for 4 s | Bridge sends `valid:false` → board OFFLINE |
| Bridge or cable stops for 5 s | Board times out on its own → OFFLINE |
| Invalid RUL (NaN, < 0, > 125) | Rejected (`ERR bad_rul`) → OFFLINE |
| USB unplugged or glitch | Bridge prints `ESP32 link lost … reconnecting` and reconnects automatically (writes time out after 1 s) |
| Bridge program stops | `run_bridge.bat` restarts it after 3 s |
| Two bridges started | The second one exits with "port 8765 is already in use" |
| Two dashboard tabs open | Only the focused tab controls the board; others are ignored |
| Arduino Serial Monitor open | Bridge says "port busy" and keeps retrying |

---

## 8. About the colour-changing LEDs

The red/blue LEDs have a tiny chip inside. While powered, they cycle on their own:
```
power on → 🔵 BLUE (first) → 🔴 RED (second) → 🔵 BLUE → 🔴 RED → ...
```
Cutting the power **restarts the sequence from blue**.

**HEALTHY (D21): steady blue.** The firmware turns D21 off for **60 ms every
300 ms**. The LED restarts before it reaches red, so it stays blue. The short
gaps are too quick to see.

**CRITICAL (D23): flashing red/blue.** Red is the *second* colour, so every
restart passes through blue first, and no timing trick can show red alone. We
tested this: short power bursts gave a red/blue mixture. D23 therefore gets steady
power, and the LED flashes red/blue, which works well as an emergency light.

**Ways to get solid red (optional):**
- **Plain red LED on D23 (recommended):** solid red with no code change, as long as
  `RED_IS_COLOR_CHANGING_LED = false`.
- **Lower voltage on D25:** red needs about 2 V and blue about 3 V. Pin D25 can
  output a lower voltage (DAC), so at the right level only red can light. Wire the
  LED to D25, set `RED_USE_LOW_VOLTAGE = true`, upload, then tune while the bridge runs:
  `http://127.0.0.1:8765/redlevel?value=180` (lower = less blue, higher = brighter).
  This depends on the LED's chip and hasn't been tested on this board, because D25 is
  on the other side of the breadboard.

---

## 9. Configuration reference

### Firmware (`esp32_rul_indicator.ino`, top of file)
| Setting | Value | Meaning |
|---|---|---|
| `HEALTHY_LED_PIN` | 21 | Blue / HEALTHY LED |
| `YELLOW_LED_PIN` | 22 | WARNING LED |
| `RED_LED_PIN` | 23 | CRITICAL LED |
| `BUZZER_PIN` | 18 | Buzzer |
| `BLUE_IS_COLOR_CHANGING_LED` | true | Hold D21's LED on its first colour (blue) |
| `BLUE_HOLD_MS` / `BLUE_RESET_MS` | 300 / 60 | On/off times for that trick |
| `RED_IS_COLOR_CHANGING_LED` | false | Steady power on D23 |
| `RED_USE_LOW_VOLTAGE` | false | Use the D25 low-voltage red trick |
| `BUZZER_ACTIVE_LOW` | false | Set true for low-level-trigger buzzer modules |
| `HEALTHY_ABOVE` / `CRITICAL_AT` | 50 / 20 | Default thresholds (the dashboard overrides them) |
| `STALE_TIMEOUT_MS` | 5000 | No valid data → OFFLINE |

After changing a setting, upload the firmware again ([section 6](#6-one-time-software-setup)).

### Bridge (`rul_bridge.py`)
| Option | Default | Meaning |
|---|---|---|
| `--port` | – | `auto` (detect CP210x/CH340) or e.g. `COM4` |
| `--mode` | `dashboard` | `dashboard`, `replay` or `test` |
| `--engine`, `--interval`, `--loop` | 100, 0.25 s | Replay mode options |
| `--http-port` | 8765 | Port the dashboard sends to |
| `--red-level` | – | Initial D25 voltage level (0–255) |
| `--dry-run` | off | Run without a board |

---

## 10. Troubleshooting

| Problem | Check | Fix |
|---|---|---|
| Dashboard shows "Bridge not running" | Is the bridge window open? | Run `START_DEMO.bat` again. |
| Bridge says `ESP32 not found` | No serial port | Plug in the board. Use a **data** cable, not charge-only. |
| Bridge says `cannot open COM4 (port busy…)` | Another program has the port | Close the Arduino IDE / Serial Monitor and other bridge windows. |
| Bridge says `port 8765 is already in use` | A bridge is already running | Close the old bridge window first. |
| "No ESP32 confirmation" but bridge is running | Bridge window shows `link lost`? | Wait ~2 s for it to reconnect, or replug the USB cable. |
| Board flips between states by itself | Several dashboard tabs open | Close the extra tabs. The bridge window shows which tab controls the board. |
| Board blinks yellow | It is OFFLINE (no data) | Check the bridge and dashboard are running; refresh the page. |
| Never turns red | The model's prediction is above 20 | Use engine 25, cycle 483, or **Test CRITICAL**. |
| HEALTHY shows red and blue | D21 colour-changing trick off | `BLUE_IS_COLOR_CHANGING_LED = true`, upload again. |
| An LED never lights | Check its legs | Turn it around; check its resistor and GND. |
| Buzzer sounds all the time | Low-level-trigger module | `BUZZER_ACTIVE_LOW = true`, upload again. |
| Upload fails "port busy" | Bridge or Serial Monitor open | Close them, then upload. |
| Upload stuck at `Connecting....` | | Hold the **BOOT** button during the upload. |
| Page opens on the wrong port | Old dashboard still running | Close all black windows and run `START_DEMO.bat` again. |

---

## 11. What was tested

On the real board (ESP32 on COM4):
- Firmware compiled with esp32 core 3.3.12 and uploaded.
- From the dashboard, the board confirmed **HEALTHY** (engine 100, cycle 50),
  **WARNING** (engine 100, cycle 1 and cycle 196) and **CRITICAL** (engine 25, cycle 483).
- The test buttons drove all three states.
- The owner confirmed by eye: D21 shows **steady blue**, and CRITICAL on D23 flashes red/blue.
- 28 s stress check in WARNING and CRITICAL with the buzzer running: 55 ACKs, no resets, no write timeouts.
- With only one dashboard tab open, the board stayed HEALTHY for 12 s with no flicker.

Simulated (no board needed):
- Bridge accepts valid frames and rejects an RUL of 300 (HTTP 400).
- USB unplug/replug and "Write timeout": the bridge reconnects, never closes the port
  from the wrong thread, and doesn't crash.
- Two tabs: only the focused tab is accepted, and focusing the other tab hands over control.
- Starting a second bridge exits with a clear message.

Not tested: the D25 low-voltage red mode on real hardware.

---

## 12. Files in this folder

| File | Purpose |
|---|---|
| [`esp32_rul_indicator/esp32_rul_indicator.ino`](esp32_rul_indicator/esp32_rul_indicator.ino) | ESP32 firmware: message parsing, state logic, LEDs, buzzer, self-test, timeouts. |
| [`rul_bridge.py`](rul_bridge.py) | Laptop bridge: local HTTP server for the dashboard, USB serial to the ESP32, reconnect, tab control. |
| [`run_bridge.bat`](run_bridge.bat) | Runs the bridge and restarts it if it stops. |
| [`requirements.txt`](requirements.txt) | Python dependency (`pyserial`). |
| [`../START_DEMO.bat`](../START_DEMO.bat) | One-click start: bridge + dashboard + browser. |

Dashboard changes for the hardware are in
[`../app/frontend/src/main.jsx`](../app/frontend/src/main.jsx): sending frames to
the bridge, the ESP32 status card, the test buttons, the cycle step buttons, the
cycle number box, and single-tab control.
