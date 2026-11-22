"""FedTwin-RUL laptop -> ESP32 serial bridge.

Sends the model's PREDICTED RUL to the ESP32 as one JSON line per update and
prints the ESP32's acknowledgements.

Modes
-----
dashboard  (default) Runs a tiny local HTTP server on 127.0.0.1:8765. The React
           dashboard POSTs the frame it is currently showing; the bridge
           forwards it to the ESP32, so LEDs/buzzer follow the dashboard.
replay     No dashboard needed. Replays the saved FedTwin-RUL predictions from
           app/frontend/public/demo-data.json for one engine.
test       Clearly labelled TEST values (75, 35, 15, invalid) to check that the
           LEDs and buzzer react. Not model output.

Examples
--------
    python hardware/rul_bridge.py --list-ports
    python hardware/rul_bridge.py --port COM4                       # dashboard mode
    python hardware/rul_bridge.py --port COM4 --mode replay --engine 100 --interval 0.25
    python hardware/rul_bridge.py --port COM4 --mode test
    python hardware/rul_bridge.py --dry-run --mode replay            # no board, print only
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEMO_DATA = ROOT / "app" / "frontend" / "public" / "demo-data.json"
DASHBOARD_TIMEOUT_S = 4.0   # no dashboard update for this long -> send valid:false
HEARTBEAT_S = 1.0           # resend latest state so the ESP32 does not go stale


def classify(rul: float, critical: float = 20.0, healthy: float = 50.0) -> str:
    if rul <= critical:
        return "CRITICAL"
    if rul <= healthy:
        return "WARNING"
    return "HEALTHY"


def valid_rul(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) \
        and math.isfinite(value) and 0.0 <= value <= 125.0


class Link:
    """Serial (or dry-run) link to the ESP32 with a background reader."""

    def __init__(self, port: str | None, baud: int, dry_run: bool):
        self.dry_run = dry_run
        self.seq = 0
        self.lock = threading.Lock()
        self.last_ack = None
        self.last_ack_time = 0.0
        self.ser = None
        self.port_arg = port
        self.baud = baud
        self.last_problem = None
        self.io_lock = threading.Lock()   # serialises write() and close()
        self.broken = threading.Event()
        self.broken_why = ""
        if dry_run:
            print("[bridge] DRY RUN: no serial port opened, messages are only printed")
            return
        try:
            import serial  # noqa: F401  (pyserial)
        except ImportError:
            sys.exit("pyserial is missing. Run:  pip install -r hardware/requirements.txt")
        if not self._open():
            print("[bridge] will keep retrying every 2 s - plug in the ESP32 / close the Serial Monitor")
        threading.Thread(target=self._reader, daemon=True).start()
        time.sleep(1.5)

    def _say_once(self, msg: str):
        if msg != self.last_problem:
            print(msg)
            self.last_problem = msg

    def _open(self) -> bool:
        """Open the serial port. Returns False (without exiting) if not possible yet."""
        import serial
        port = self.port_arg
        if port.lower() == "auto":
            port = find_esp32_port(quiet=True)
            if not port:
                self._say_once("[bridge] ESP32 not found - is the USB cable plugged in?")
                return False
        ser = serial.Serial()
        ser.port = port
        ser.baudrate = self.baud
        ser.timeout = 0.2
        ser.write_timeout = 1.0   # never block forever if the board stops reading
        ser.dtr = False   # do not hold the ESP32 in reset / boot mode
        ser.rts = False
        try:
            ser.open()
        except serial.SerialException as exc:
            hint = " (port busy: close the Arduino Serial Monitor or another bridge window)" \
                if "denied" in str(exc).lower() or "busy" in str(exc).lower() else ""
            self._say_once(f"[bridge] cannot open {port}{hint}")
            return False
        self.ser = ser
        self.last_problem = None
        print(f"[bridge] connected to ESP32 on {port} @ {self.baud}")
        return True

    # Threading rule: ONLY the reader thread closes or reopens the port. Other
    # threads just flag a problem (self.broken). Closing a Windows serial handle
    # while another thread is inside ReadFile crashes Python with no message.
    def _close_port(self, why: str):
        with self.io_lock:
            ser, self.ser = self.ser, None
            if ser:
                try:
                    ser.close()
                except Exception:
                    pass
        self.last_ack = None
        self.broken.clear()
        print(f"[bridge] ESP32 link lost ({why}); reconnecting...")

    def _reader(self):
        while True:
            try:
                if self.ser is None:
                    time.sleep(2)
                    self._open()
                    continue
                if self.broken.is_set():
                    self._close_port(self.broken_why)
                    continue
                try:
                    raw = self.ser.readline()
                except Exception as exc:  # unplugged
                    self._close_port(f"read error: {exc}")
                    continue
                if not raw:
                    continue
                text = raw.decode("utf-8", errors="replace").strip()
                if not text:
                    continue
                if text.startswith("ACK"):
                    self.last_ack = text
                    self.last_ack_time = time.time()
                    continue  # ACKs are frequent; state changes are printed below
                print(f"[esp32] {text}")
            except Exception as exc:  # never let the reader thread die
                print(f"[bridge] reader problem: {exc}")
                time.sleep(1)

    def _write(self, data: bytes):
        with self.io_lock:
            ser = self.ser
            if ser is None or self.broken.is_set():
                return
            try:
                ser.write(data)
            except Exception as exc:
                self.broken_why = f"write error: {exc}"
                self.broken.set()   # reader thread will close + reconnect

    def send(self, payload: dict) -> dict:
        with self.lock:
            self.seq += 1
            msg = {"seq": self.seq, **payload}
        line = json.dumps(msg, separators=(",", ":")) + "\n"
        if self.dry_run:
            print(f"[tx] {line.strip()}")
        else:
            self._write(line.encode("ascii"))
        return msg

    def send_raw(self, text: str):
        if self.dry_run:
            print(f"[tx] {text}")
        else:
            self._write((text + "\n").encode("ascii"))

    def close(self):
        try:
            self.send({"valid": False, "reason": "bridge_stopped"})
            time.sleep(0.2)
        finally:
            with self.io_lock:
                if self.ser:
                    try:
                        self.ser.close()
                    except Exception:
                        pass


def frame_payload(engine_id, cycle, rul, mode, critical=20.0, healthy=50.0) -> dict:
    return {"engine_id": engine_id, "cycle": cycle, "rul": round(float(rul), 2),
            "critical": critical, "healthy": healthy, "valid": True, "mode": mode}


def log_frame(msg: dict, link: Link):
    state = classify(msg["rul"], msg.get("critical", 20), msg.get("healthy", 50))
    ack = link.last_ack or "-"
    tag = "TEST  " if msg.get("mode") == "test" else ""
    print(f"[tx] {tag}engine {msg['engine_id']:>3} cycle {msg['cycle']:>3}  "
          f"predicted RUL {msg['rul']:6.2f} -> {state:<8}  | last {ack}")


# ---------------------------------------------------------------- replay mode
def run_replay(link: Link, engine_id: int, interval: float, loop: bool):
    data = json.loads(DEMO_DATA.read_text(encoding="utf-8"))
    engines = {e["id"]: e for e in data["engines"]}
    if engine_id not in engines:
        sys.exit(f"Engine {engine_id} not in replay data. Available: {sorted(engines)}")
    engine = engines[engine_id]
    print(f"[bridge] REPLAY of saved {data['model']['name']} predictions, engine {engine_id}, "
          f"{len(engine['timeline'])} cycles")
    while True:
        for frame in engine["timeline"]:
            if not valid_rul(frame["predictedRul"]):
                link.send({"engine_id": engine_id, "cycle": frame["cycle"], "valid": False})
                continue
            msg = link.send(frame_payload(engine_id, frame["cycle"], frame["predictedRul"], "replay"))
            log_frame(msg, link)
            time.sleep(interval)
        if not loop:
            break
    last = engine["timeline"][-1]
    print("[bridge] replay finished; holding last frame (Ctrl+C to stop)")
    while True:
        link.send(frame_payload(engine_id, last["cycle"], last["predictedRul"], "replay_hold"))
        time.sleep(HEARTBEAT_S)


# ------------------------------------------------------------------ test mode
def run_test(link: Link):
    print("[bridge] TEST MODE - these are hand-entered test values, NOT model predictions")
    steps = [(75.0, "expect HEALTHY: blue on"),
             (35.0, "expect WARNING: yellow on + chirp every 2 s"),
             (15.0, "expect CRITICAL: red on + fast beeping"),
             (None, "expect OFFLINE: yellow slow blink, buzzer silent")]
    for rul, note in steps:
        print(f"\n[test] {note}")
        end = time.time() + 4
        while time.time() < end:
            if rul is None:
                link.send({"engine_id": 0, "valid": False, "mode": "test"})
            else:
                link.send(frame_payload(0, 0, rul, "test"))
            time.sleep(0.5)
        print(f"[test] last {link.last_ack}")
    print("\n[test] done")


# ------------------------------------------------------------- dashboard mode
class DashboardState:
    def __init__(self):
        self.lock = threading.Lock()
        self.latest: dict | None = None
        self.received_at = 0.0
        self.controller: str | None = None   # dashboard tab allowed to drive the board


def make_handler(state: DashboardState, link: Link):
    class Handler(BaseHTTPRequestHandler):
        def _cors(self):
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")

        def _json(self, code: int, body: dict):
            data = json.dumps(body).encode()
            self.send_response(code)
            self._cors()
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_OPTIONS(self):
            self.send_response(204)
            self._cors()
            self.end_headers()

        def do_GET(self):
            if self.path.startswith("/redlevel"):
                from urllib.parse import urlparse, parse_qs
                try:
                    v = int(parse_qs(urlparse(self.path).query)["value"][0])
                    assert 0 <= v <= 255
                except Exception:
                    return self._json(400, {"error": "use /redlevel?value=0..255"})
                link.send_raw(f"REDLEVEL {v}")
                return self._json(200, {"redlevel": v})
            if self.path.startswith("/status"):
                age = time.time() - link.last_ack_time if link.last_ack_time else None
                self._json(200, {"bridge": "ok", "dry_run": link.dry_run, "last_ack": link.last_ack,
                                 "ack_age_s": age})
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self):
            if not self.path.startswith("/state"):
                return self._json(404, {"error": "not found"})
            try:
                length = min(int(self.headers.get("Content-Length", 0)), 4096)
                body = json.loads(self.rfile.read(length) or b"{}")
                rul = body.get("predictedRul")
                critical = float(body.get("critical", 20))
                healthy = float(body.get("healthy", 50))
                if not valid_rul(rul) or not (0 <= critical < healthy <= 125):
                    raise ValueError("invalid predictedRul or thresholds")
                mode = "test" if body.get("mode") == "test" else "dashboard"
                payload = frame_payload(int(body["engineId"]), int(body["cycle"]), rul,
                                        mode, critical, healthy)
            except Exception as exc:
                return self._json(400, {"error": str(exc)})
            session = str(body.get("session", ""))
            with state.lock:
                now = time.time()
                free = state.controller is None or now - state.received_at > DASHBOARD_TIMEOUT_S
                if body.get("claim") or free or session == state.controller:
                    if session != state.controller:
                        print(f"[bridge] dashboard tab {session or '?'} now controls the board")
                    state.controller = session
                else:
                    # another (unfocused) tab: ignore it so two tabs cannot fight
                    return self._json(200, {"accepted": False, "reason": "another_tab_in_control"})
                changed = state.latest is None or state.latest != payload
                state.latest = payload
                state.received_at = time.time()
            if changed:
                log_frame(link.send(payload), link)
            self._json(200, {"accepted": True, "last_ack": link.last_ack})

        def log_message(self, *args):  # keep the console readable
            pass

    return Handler


def run_dashboard(link: Link, host: str, port: int):
    state = DashboardState()
    class Server(ThreadingHTTPServer):
        allow_reuse_address = False   # a second bridge must fail, not silently share the port
    try:
        server = Server((host, port), make_handler(state, link))
    except OSError:
        sys.exit(f"[bridge] port {port} is already in use: another bridge is running. "
                 "Close the other bridge window first.")
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"[bridge] DASHBOARD mode: listening on http://{host}:{port}  "
          "(start the dashboard and press 'Start replay')")
    offline_sent = False
    while True:
        time.sleep(HEARTBEAT_S)
        with state.lock:
            latest, age = state.latest, time.time() - state.received_at
        if latest is None:
            link.send({"valid": False, "reason": "waiting_for_dashboard"})
        elif age > DASHBOARD_TIMEOUT_S:
            if not offline_sent:
                print("[bridge] dashboard stopped sending -> ESP32 OFFLINE")
                offline_sent = True
            link.send({"valid": False, "reason": "dashboard_silent"})
        else:
            offline_sent = False
            link.send(latest)  # heartbeat: same frame, new seq


def find_esp32_port(quiet: bool = False) -> str | None:
    """Pick the ESP32's USB-serial port (CP210x / CH340 adapters)."""
    try:
        from serial.tools import list_ports as lp
    except ImportError:
        sys.exit("pyserial is missing. Run:  pip install -r hardware/requirements.txt")
    ports = list(lp.comports())
    usb = [p for p in ports if any(k in (p.description or "").upper()
                                   for k in ("CP210", "CH340", "CH910", "USB SERIAL", "UART"))]
    if len(usb) == 1:
        return usb[0].device
    if quiet:
        return None
    if not ports:
        sys.exit("No serial ports found. Plug in the ESP32 with a DATA USB cable and try again.")
    listing = "\n".join(f"  {p.device:8} {p.description}" for p in ports)
    sys.exit(f"Could not pick the ESP32 port automatically. Ports found:\n{listing}\n"
             "Run again with --port COMx")


def list_ports():
    try:
        from serial.tools import list_ports as lp
    except ImportError:
        sys.exit("pyserial is missing. Run:  pip install -r hardware/requirements.txt")
    ports = list(lp.comports())
    if not ports:
        print("No serial ports found. Check the USB cable (must be a data cable).")
    for p in ports:
        print(f"{p.device:8} {p.description}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", help="ESP32 serial port, e.g. COM4")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--mode", choices=["dashboard", "replay", "test"], default="dashboard")
    ap.add_argument("--engine", type=int, default=100, help="replay mode: engine 1, 25, 50 or 100")
    ap.add_argument("--interval", type=float, default=0.25, help="replay mode: seconds per cycle")
    ap.add_argument("--loop", action="store_true", help="replay mode: repeat forever")
    ap.add_argument("--http-host", default="127.0.0.1")
    ap.add_argument("--http-port", type=int, default=8765)
    ap.add_argument("--red-level", type=int, help="D25 red LED voltage level 0-255 (colour-changing LED trick)")
    ap.add_argument("--dry-run", action="store_true", help="print messages instead of using a board")
    ap.add_argument("--list-ports", action="store_true")
    args = ap.parse_args()
    sys.stdout.reconfigure(line_buffering=True)

    if args.list_ports:
        return list_ports()
    if not args.port and not args.dry_run:
        ap.error("--port is required (see --list-ports), or use --dry-run")

    link = Link(args.port, args.baud, args.dry_run)
    if args.red_level is not None:
        link.send_raw(f"REDLEVEL {args.red_level}")
    try:
        if args.mode == "replay":
            run_replay(link, args.engine, args.interval, args.loop)
        elif args.mode == "test":
            run_test(link)
        else:
            run_dashboard(link, args.http_host, args.http_port)
    except KeyboardInterrupt:
        print("\n[bridge] stopping")
    finally:
        link.close()


if __name__ == "__main__":
    main()
