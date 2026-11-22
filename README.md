# FedTwin-RUL

Start here. This bundle contains the newer dashboard, the Python research code,
the local C-MAPSS dataset files, processed FD004 data, trained checkpoints and
saved experimental results. No retraining is needed to run the demonstration.

## Windows setup for the presentation

Install Python 3.11 or 3.12, Git and Node.js 22.12 or newer supported LTS.
Open PowerShell and run:

```powershell
git clone https://github.com/Vatsal212005/FedTwin-RUL.git
cd FedTwin-RUL
python -m pip install --user -r hardware/requirements.txt
npm.cmd --prefix app/frontend ci
npm.cmd --prefix app/frontend run build
```

Do this while internet access is available. After setup, the local replay and
USB hardware demonstration work offline with the installed dependencies.
The dataset, trained checkpoints, results and replay JSON are included.

### Start the demo

1. Plug the programmed ESP32 into the laptop using a data USB cable.
2. Close the Arduino Serial Monitor and any previous dashboard or bridge.
3. Double-click `START_DEMO.bat` in the repository root.
4. Keep both command windows open and use one dashboard tab at
   http://127.0.0.1:5173/.
5. Check for **Bridge running** and a fresh **ESP32 confirmed** state.

The launcher finds the serial port automatically. For a different laptop or a
board that is not detected, follow the manual port commands below.
The dashboard can run without the ESP32 using the dashboard-only commands.

### Presentation sequence

Use the cycle number field to pause on these saved model predictions:

| State | Engine | Cycle | Predicted RUL |
|---|---|---|---|
| Healthy | 100 | 50 | 120.13 cycles |
| Warning | 100 | 1 | 24.61 cycles |
| Critical | 25 | 483 | 19.47 cycles |

Start replay to show the changing prediction and sensor traces. The critical
frames are brief, so enter cycle 483 directly for that example. Hardware test
buttons use fixed values and display TEST mode. Select **Follow model** to
resume prediction-driven output.

### Manual bridge startup and troubleshooting

From the repository root:

```powershell
python hardware/rul_bridge.py --list-ports
python hardware/rul_bridge.py --port auto
```

If necessary, replace `auto` with the port printed for your ESP32, for example
`COM4`. Keep this terminal open. In another terminal run:

```powershell
npm.cmd --prefix app/frontend run dev -- --host 127.0.0.1 --port 5173 --strictPort
```

If port 5173 or bridge port 8765 is already in use, close the previous demo
process and restart the launcher. If the bridge finds no device, check the USB
cable and install the board's CP2102 USB driver if Windows requires it.
Firmware stays on a previously programmed board. For a fresh ESP32, use the
upload and wiring instructions in [hardware/README.md](hardware/README.md).

## Hardware demo (ESP32 health indicator)

The dashboard drives a working ESP32 indicator panel: 🔵 blue = HEALTHY,
🟡 yellow + chirp = WARNING, flashing red/blue + fast beeping = CRITICAL, based on the
model's predicted RUL. To run it, plug in the ESP32 and **double-click
`START_DEMO.bat`**.

Wiring, setup, the demo script and troubleshooting are in
**[hardware/README.md](hardware/README.md)**.

## Run the dashboard

Install Node.js 22 LTS or newer supported LTS (at least 22.12). From this folder:

```powershell
cd app/frontend
npm.cmd ci
npm.cmd run dev -- --host 127.0.0.1 --port 5173 --strictPort
```

Open the URL Vite prints, normally http://127.0.0.1:5173/.
Internet is needed for the first dependency installation. Run `npm run build`
in the same folder to make a production build. The included lockfile preserves
the versions used by this dashboard. Python and CUDA are not needed for replay.

## What currently works

- The React/Vite frontend loads `app/frontend/public/demo-data.json`.
- That file contains saved real model predictions and historical FD004 signals
  for engines 1, 25, 50 and 100, not real-time hardware sensor measurements.
- Engine selection, cycle replay, speed and threshold controls operate locally.
- The dashboard sends the frame on screen (engine, cycle, predicted RUL,
  thresholds) to a local Python bridge (`hardware/rul_bridge.py`), which forwards
  it over USB serial to the ESP32 firmware (`hardware/esp32_rul_indicator/`).
  The board's acknowledgement is shown on the dashboard. The dashboard still
  works without the board connected.

`CODEX_HARDWARE_PROMPT.md` and `docs/HARDWARE_PLAN.md` are the original planning
notes (Arduino Uno / ESP8266). The implemented version uses a single ESP32 over
USB; see [hardware/README.md](hardware/README.md).

## Final model versus historical repository name

The folder/package name `twinfed_moe` comes from earlier experiments. The final
paper model is **FedTwin-RUL**, not the earlier personalized MoE architecture.
Its actual implementation is `TwinFedTCM` in
`src/twinfed_moe/models/tcm_model.py` with `fusion_mode="concat"` and twin
reconstruction enabled. It joins a 64-D temporal Transformer feature to a 16-D
twin feature, then predicts normalized RUL. Multiply that output by 125 for cycles.
The twin encoder consumes a sensor summary and five auxiliaries:
normalized operating settings o1/o2/o3, health proxy q and within-window drift d.
The final model has 60,159 parameters and trains with ordinary FedAvg.

`FEDTWIN_FINAL_README.md` and `configs/fedtwin_final.yaml` describe the final
experiment. Other configs, code and READMEs document earlier experiments and
remain available for reference. The original repository README is preserved at
`docs/README_REPOSITORY_ORIGINAL.md`; its initial architecture and TODO wording
do not describe the final paper model.

## Data and results

The included FD004 files contain 249 training engines and 248 test engines,
six operating conditions and two fault modes. The upstream dataset readme has
inconsistent FD004 counts: use the actual data files and final study metadata.
The pipeline retains 14 sensors, uses 40-cycle windows, training stride 5 and a
125-cycle RUL cap. Training-only statistics define preprocessing and the twin
reference. Whole-engine client/validation splits avoid trajectory leakage.

The final study uses ten simulated clients, alpha 1.0/0.5/0.1 and seeds
42/123/2027. Predictions in the dashboard come from alpha 0.1, seed 42,
validation-selected round 33 of 40. The three-seed comparison table is a separate
aggregate result, not a score calculated over the four replay engines.

Key paths:

```text
app/frontend/src/main.jsx                dashboard, replay state, thresholds
app/frontend/src/styles.css              dashboard styling
app/frontend/public/demo-data.json       ready-to-use replay and comparison data
scripts/generate_dashboard_data.py        regenerates predictions from checkpoint
scripts/preprocess_cmapss.py              training-only preprocessing and twin auxiliaries
scripts/run_fedtwin_final_study.py         final training harness
scripts/run_published_baselines.py        published baseline reproduction harness
src/twinfed_moe/models/tcm_model.py        final model (concat mode)
data/raw/cmapss/                          local NASA data and original attribution
data/processed/                          processed FD004 and preprocessing metadata
outputs/fedtwin_final_FD004_20260827_211745/
  final_with_published_baselines.csv      aggregate paper comparison
  runs/alpha_0.1/seed_42/fedtwin_rul/best_model.pt
```

## Python research setup (optional)

Use Python 3.11 or 3.12 and install a PyTorch build appropriate to your machine.
The repository requirements specify compatible minimum versions, not a complete
historical environment lock. CUDA is optional for inference and tests.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m pytest tests -q
.\.venv\Scripts\python.exe scripts/generate_dashboard_data.py
```

The last command overwrites the bundled replay JSON using the included checkpoint,
results CSV, raw test data and processed training data. Run from the repository
root. Do not regenerate preprocessing when merely adding LEDs, because the
checkpoint depends on the original preprocessing. Torch checkpoint files can
execute deserialization code: load only trusted project artifacts.

To run the final training study deliberately:

```powershell
.\.venv\Scripts\python.exe scripts/run_fedtwin_final_study.py --config configs/fedtwin_final.yaml
```

## Implemented hardware

The completed prototype uses a 30-pin ESP32 board with a CP2102 USB interface,
two red/blue colour-changing LEDs, one yellow LED, three 220 Ω resistors, an
active buzzer, a breadboard and jumper wires. A laptop sends the selected model
prediction through the Python bridge over USB serial at 115200 baud.

Default mapping uses the **predicted** RUL:

| Condition | State | LED | Buzzer on D18 |
|---|---|---|---|
| RUL > 50 | HEALTHY | Steady blue, D21 | Silent |
| 20 < RUL <= 50 | WARNING | Yellow, D22 | Chirp every 2 seconds |
| RUL <= 20 | CRITICAL | Flashing red/blue, D23 | Rapid beeping |
| No valid frame for 5 seconds | OFFLINE | Slow yellow blink, D22 | Silent |

The board returns acknowledgements that appear in the dashboard. Each LED uses
a 220 Ω series resistor and shares ground with the ESP32. The project hardware
README records the bench checks. Physical output follows historical model
predictions from the simulated NASA dataset.

Raw estimates can increase as new windows arrive. Preserve them; any optional
LED hysteresis must be disclosed separately from the model prediction.

## Packaging notes

The GitHub upload uses the Desktop `FedTwin-RUL.zip` updated on 11 November 2026.
It includes the original source scripts, configs and tests, local data, experiment
outputs, dashboard and completed ESP32 integration. Dependency folders, virtual
environments, builds and caches are regenerated locally.

The root README and ignore rules were updated for this repository.
`BUNDLE_MANIFEST.json`, `PACKAGING_REPORT.json` and `VERIFY_RESULTS.json` are
historical records from the earlier handoff package. They describe that snapshot
and retain its hashes. The current hardware implementation and recorded checks
are documented in [hardware/README.md](hardware/README.md).

External software and datasets retain their original terms. This handoff does
not grant a new license or change dataset ownership.
