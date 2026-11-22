// FedTwin-RUL ESP32 indicator (no OLED version)
//
// The laptop runs the model / replay. It sends one JSON line per update over
// USB serial (115200 baud), for example:
//   {"seq":12,"engine_id":100,"cycle":57,"rul":34.80,"valid":true,"mode":"dashboard"}
// The ESP32 maps the PREDICTED RUL to an output state and replies with:
//   ACK seq=12 state=WARNING rul=34.80
//
// Wiring (your board):
//   D22 (GPIO22) -> 220 ohm -> YELLOW LED long leg (+), short leg (-) -> GND
//   D23 (GPIO23) -> 220 ohm -> CRITICAL LED (red/blue colour-changing) (+), (-) -> GND
//   D21 (GPIO21) -> 220 ohm -> HEALTHY LED  (red/blue colour-changing) (+), (-) -> GND
//   D18 (GPIO18) -> buzzer + / signal, buzzer - -> GND
//
// States:
//   HEALTHY  (RUL > 50)        : D21 steady BLUE (power-pulse trick), buzzer silent
//   WARNING  (20 < RUL <= 50)  : YELLOW ON + short beep every 2 s
//   CRITICAL (RUL <= 20)       : D23 ON (flashes red/blue by itself) + fast beeping
//
// Full guide: hardware/README.md
//   OFFLINE  (no/invalid data) : yellow blinks slowly, buzzer OFF
//   On boot                    : self-test (yellow, red, blue, short beep)

#include <Arduino.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

// ---------------- configuration ----------------
const int YELLOW_LED_PIN  = 22;
const int RED_LED_PIN     = 23;
const int BUZZER_PIN      = 18;
const int HEALTHY_LED_PIN = 21;     // BLUE LED (HEALTHY)

// Keep the CRITICAL LED on D23 one colour.
// true  = D23 LED is a self colour-changing (red<->blue) LED: power it in short
//         bursts so it stays on its FIRST colour. If that first colour is blue
//         instead of red, make RED_HOLD_MS longer/shorter or use a plain red LED.
// false = plain red LED, steady on.
const bool RED_IS_COLOR_CHANGING_LED  = false;  // D23 (CRITICAL): steady power, LED flashes red/blue by itself
const bool BLUE_IS_COLOR_CHANGING_LED = true;   // D21 (HEALTHY)
// on time must be shorter than the LED's own colour step; off time must be long
// enough for its built-in chip to fully restart.
const unsigned long BLUE_HOLD_MS  = 300, BLUE_RESET_MS = 60;    // D21: steady blue (confirmed)
const unsigned long RED_HOLD_MS   = 200, RED_RESET_MS  = 150;   // D23: red blink, in time with buzzer

// RED-ONLY trick for a red/blue colour-changing LED: feed it a LOWER voltage
// from the ESP32's analog output (DAC, only on GPIO25/26). Red lights at ~1.8-2.2 V,
// blue needs ~2.7-3 V, so at the right level only the red part can glow and the
// blue phase stays dark. Wire: D25 -> 220 ohm -> LED long leg, short leg -> GND.
// Tune live from a serial monitor with:  REDLEVEL 190   (0-255, ~0-3.3 V)
//   blue still visible -> lower the number;  LED dark -> raise it.
const bool RED_USE_LOW_VOLTAGE = false;  // true only if the LED is wired to D25
const int  RED_DAC_PIN = 25;
int redDacLevel = 200;                 // ~2.6 V at the pin

// Many cheap buzzer modules are "low level trigger" (they sound when IN = LOW).
// If your buzzer sounds when it should be silent, change this to true.
const bool BUZZER_ACTIVE_LOW = false;

const float HEALTHY_ABOVE = 50.0f;  // RUL > 50          -> HEALTHY
const float CRITICAL_AT   = 20.0f;  // RUL <= 20         -> CRITICAL
const unsigned long STALE_TIMEOUT_MS = 5000;   // no valid update -> OFFLINE
// ------------------------------------------------

enum State { OFFLINE, HEALTHY, WARNING, CRITICAL };
const char* stateName(State s) {
  switch (s) {
    case HEALTHY: return "HEALTHY";
    case WARNING: return "WARNING";
    case CRITICAL: return "CRITICAL";
    default: return "OFFLINE";
  }
}

State state = OFFLINE;
unsigned long lastValidMs = 0;
float criticalAt = CRITICAL_AT;
float healthyAbove = HEALTHY_ABOVE;

char lineBuf[256];
size_t lineLen = 0;
bool lineOverflow = false;

void buzzer(bool on) {
  digitalWrite(BUZZER_PIN, (on != BUZZER_ACTIVE_LOW) ? HIGH : LOW);
}

// A self colour-changing LED cycles red/blue on its own while powered.
// Cutting power for a moment restarts its built-in sequence, so it never
// reaches the second colour. With a plain LED this is just a tiny flicker.
void driveLed(int pin, bool on, bool colorChanging, unsigned long holdMs, unsigned long resetMs) {
  if (pin < 0) return;
  bool power = on;
  if (on && colorChanging) {
    power = (millis() % (holdMs + resetMs)) < holdMs;
  }
  digitalWrite(pin, power ? HIGH : LOW);
}

void setLeds(bool yellow, bool red, bool healthyLed) {
  digitalWrite(YELLOW_LED_PIN, yellow ? HIGH : LOW);
  if (RED_USE_LOW_VOLTAGE) {
    // Steady low voltage: the LED keeps cycling internally, but only red can light.
    static int lastLevel = -1;
    int level = red ? redDacLevel : 0;
    if (level != lastLevel) { dacWrite(RED_DAC_PIN, level); lastLevel = level; }
    digitalWrite(RED_LED_PIN, LOW);
  } else {
    driveLed(RED_LED_PIN, red, RED_IS_COLOR_CHANGING_LED, RED_HOLD_MS, RED_RESET_MS);
  }
  driveLed(HEALTHY_LED_PIN, healthyLed, BLUE_IS_COLOR_CHANGING_LED, BLUE_HOLD_MS, BLUE_RESET_MS);
}

// Find "key": in a flat JSON object and return pointer to the value text.
const char* findKey(const char* json, const char* key) {
  char pattern[32];
  snprintf(pattern, sizeof(pattern), "\"%s\"", key);
  const char* p = strstr(json, pattern);
  if (!p) return nullptr;
  p += strlen(pattern);
  while (*p == ' ' || *p == '\t') p++;
  if (*p != ':') return nullptr;
  p++;
  while (*p == ' ' || *p == '\t') p++;
  return p;
}

bool readNumber(const char* json, const char* key, double& out) {
  const char* p = findKey(json, key);
  if (!p) return false;
  char* end = nullptr;
  double v = strtod(p, &end);
  if (end == p) return false;
  out = v;
  return true;
}

bool readBool(const char* json, const char* key, bool& out) {
  const char* p = findKey(json, key);
  if (!p) return false;
  if (strncmp(p, "true", 4) == 0) { out = true; return true; }
  if (strncmp(p, "false", 5) == 0) { out = false; return true; }
  return false;
}

void goOffline(const char* reason) {
  if (state != OFFLINE) {
    Serial.printf("STATE OFFLINE reason=%s\n", reason);
  }
  state = OFFLINE;
  buzzer(false);
}

void handleLine(const char* line) {
  if (strncmp(line, "REDLEVEL", 8) == 0) {
    int v = atoi(line + 8);
    if (v >= 0 && v <= 255) { redDacLevel = v; Serial.printf("REDLEVEL %d (~%.2f V)\n", v, v * 3.3f / 255.0f); }
    else Serial.println("ERR redlevel_range");
    return;
  }
  if (line[0] != '{') {
    Serial.println("ERR not_json");
    return;
  }
  double seq = -1, rul = NAN, engine = -1, cycle = -1, crit = NAN, healthy = NAN;
  bool valid = false;
  readNumber(line, "seq", seq);
  readNumber(line, "engine_id", engine);
  readNumber(line, "cycle", cycle);
  if (!readBool(line, "valid", valid)) valid = false;

  if (!valid) {
    Serial.printf("ACK seq=%ld state=OFFLINE reason=invalid_flag\n", (long)seq);
    goOffline("invalid_flag");
    return;
  }
  if (!readNumber(line, "rul", rul) || !isfinite(rul) || rul < 0.0 || rul > 125.0) {
    Serial.printf("ERR seq=%ld bad_rul\n", (long)seq);
    goOffline("bad_rul");
    return;
  }
  // Optional thresholds from the dashboard; accept only a sane pair.
  if (readNumber(line, "critical", crit) && readNumber(line, "healthy", healthy) &&
      isfinite(crit) && isfinite(healthy) && crit >= 0 && crit < healthy && healthy <= 125) {
    criticalAt = (float)crit;
    healthyAbove = (float)healthy;
  }

  State next = (rul <= criticalAt) ? CRITICAL : (rul <= healthyAbove) ? WARNING : HEALTHY;
  if (next != state) {
    Serial.printf("STATE %s engine=%ld cycle=%ld rul=%.2f\n", stateName(next), (long)engine, (long)cycle, rul);
  }
  state = next;
  lastValidMs = millis();
  Serial.printf("ACK seq=%ld state=%s rul=%.2f\n", (long)seq, stateName(state), rul);
}

void pollSerial() {
  while (Serial.available()) {
    char c = (char)Serial.read();
    if (c == '\r') continue;
    if (c == '\n') {
      lineBuf[lineLen] = '\0';
      if (lineOverflow) Serial.println("ERR line_too_long");
      else if (lineLen > 0) handleLine(lineBuf);
      lineLen = 0;
      lineOverflow = false;
    } else if (lineLen < sizeof(lineBuf) - 1) {
      lineBuf[lineLen++] = c;
    } else {
      lineOverflow = true;
    }
  }
}

void renderOutputs() {
  unsigned long now = millis();
  switch (state) {
    case HEALTHY:
      setLeds(false, false, true);
      buzzer(false);
      break;
    case WARNING:
      setLeds(true, false, false);
      buzzer((now % 2000) < 120);            // short chirp every 2 s
      break;
    case CRITICAL:
      setLeds(false, true, false);
      buzzer((now / 250) % 2 == 0);          // beep 250 ms on / 250 ms off
      break;
    default: {                               // OFFLINE: slow yellow blink
      bool blink = (now / 1000) % 2 == 0;
      setLeds(blink, false, false);
      buzzer(false);
    }
  }
}

void selfTest() {
  Serial.println("SELFTEST start");
  setLeds(true, false, false);  delay(400);
  setLeds(false, true, false);  delay(400);
  setLeds(false, false, true);  delay(400);
  setLeds(false, false, false);
  buzzer(true);  delay(150);  buzzer(false);
  Serial.println("SELFTEST done");
}

void setup() {
  Serial.begin(115200);
  pinMode(YELLOW_LED_PIN, OUTPUT);
  pinMode(RED_LED_PIN, OUTPUT);
  pinMode(BUZZER_PIN, OUTPUT);
  if (HEALTHY_LED_PIN >= 0) pinMode(HEALTHY_LED_PIN, OUTPUT);
  buzzer(false);
  setLeds(false, false, false);
  delay(300);
  selfTest();
  Serial.println("READY FedTwin-RUL ESP32 indicator (blue=21 yellow=22 red=23 buzzer=18)");
}

void loop() {
  pollSerial();
  if (state != OFFLINE && millis() - lastValidMs > STALE_TIMEOUT_MS) {
    goOffline("timeout");
  }
  renderOutputs();
  delay(5);
}
