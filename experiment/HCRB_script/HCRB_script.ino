#if defined(ESP32)
  #include <WiFiMulti.h>
  WiFiMulti wifiMulti;
  #define DEVICE "ESP32"
#elif defined(ESP8266)
  #include <ESP8266WiFiMulti.h>
  ESP8266WiFiMulti wifiMulti;
  #define DEVICE "ESP8266"
#endif

#include <InfluxDbClient.h>
#include <InfluxDbCloud.h>

// -------------------- USER SETTINGS --------------------

// WiFi credentials
#define WIFI_SSID "Ståle  sin iPhone"
#define WIFI_PASSWORD "secret"

// InfluxDB settings
#define INFLUXDB_URL "https://eu-central-1-1.aws.cloud2.influxdata.com"
#define INFLUXDB_TOKEN "secret"
#define INFLUXDB_ORG "f6eea422dd9af5e4"
#define INFLUXDB_BUCKET "Temperature"

#define TZ_INFO "UTC1"

// -------------------- PIN ASSIGNMENTS --------------------
// Adjust these to your actual wiring
#define TONGUE_BTN_PIN  4
#define PIPE_BTN_PIN    5
#define TONGUE_LED_PIN  6
#define PIPE_LED_PIN    7

// -------------------- GLOBAL STATE --------------------
InfluxDBClient client(INFLUXDB_URL, INFLUXDB_ORG, INFLUXDB_BUCKET, INFLUXDB_TOKEN, InfluxDbCloud2CACert);

// We'll write temperature data to this measurement
Point sensor("recording");

// For toggling each process
bool tongueActive = false;
bool pipeActive   = false;

// Track last button states (for detecting presses)
int lastTongueBtnState = HIGH;
int lastPipeBtnState   = HIGH;

// -------------------------------------------------------
// Blinks both LEDs in sync every 500ms (for WiFi error)
void blinkSync() {
  while (true) {
    digitalWrite(TONGUE_LED_PIN, HIGH);
    digitalWrite(PIPE_LED_PIN, HIGH);
    delay(500);
    digitalWrite(TONGUE_LED_PIN, LOW);
    digitalWrite(PIPE_LED_PIN, LOW);
    delay(500);
  }
}

// Blinks both LEDs alternately every 500ms (for InfluxDB error)
void blinkAsync() {
  while (true) {
    digitalWrite(TONGUE_LED_PIN, HIGH);
    digitalWrite(PIPE_LED_PIN, LOW);
    delay(500);
    digitalWrite(TONGUE_LED_PIN, LOW);
    digitalWrite(PIPE_LED_PIN, HIGH);
    delay(500);
  }
}

void setup() {
  Serial.begin(115200);
  Serial.println("Starting up recording system");

  // -------------------- PIN MODES --------------------
  pinMode(TONGUE_BTN_PIN, INPUT_PULLUP);
  pinMode(PIPE_BTN_PIN,   INPUT_PULLUP);
  pinMode(TONGUE_LED_PIN, OUTPUT);
  pinMode(PIPE_LED_PIN,   OUTPUT);

  digitalWrite(TONGUE_LED_PIN, LOW);
  digitalWrite(PIPE_LED_PIN,   LOW);

  // -------------------- WIFI --------------------
  Serial.print("Connecting to WiFi: ");
  Serial.println(WIFI_SSID);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 30) {
    Serial.print(".");
    delay(500);
    attempts++;
  }

  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("\n✅ Connected to WiFi!");
    Serial.print("IP Address: ");
    Serial.println(WiFi.localIP());
  } else {
    Serial.println("\n❌ Failed to connect to WiFi.");
    // Blink both LEDs in sync every 500ms for debugging.
    blinkSync();
  }

  // -------------------- TIME SYNC --------------------
  Serial.println("Syncing time...");
  timeSync("UTC", "pool.ntp.org", "time.google.com");
  Serial.println("Waiting 5 seconds for time sync to stabilize...");
  delay(5000);

  // -------------------- INFLUXDB --------------------
  if (client.validateConnection()) {
    Serial.print("Connected to InfluxDB: ");
    Serial.println(client.getServerUrl());
  } else {
    Serial.print("InfluxDB connection failed: ");
    Serial.println(client.getLastErrorMessage());
    // Blink both LEDs alternately every 500ms for debugging.
    blinkAsync();
  }

  // Tag the measurement with some device info
  sensor.addTag("device", "ESP32-C6");
  sensor.addTag("SSID", WiFi.SSID());
}

void loop() {
  // Check the tongue button
  checkTongueButton();

  // Check the pipe button
  checkPipeButton();

  delay(10); // Some loop delay
}

// -------------------------------------------------------
// Check the tongue button for a press and toggle start/stop
void checkTongueButton() {
  int currentState = digitalRead(TONGUE_BTN_PIN);
  // Detect a press from HIGH to LOW
  if (currentState == LOW && lastTongueBtnState == HIGH) {
    if (!tongueActive) {
      handleTongueStart();
    } else {
      handleTongueStop();
    }
  }
  lastTongueBtnState = currentState;
}

void handleTongueStart() {
  Serial.println("Tongue recording START");
  tongueActive = true;
  digitalWrite(TONGUE_LED_PIN, HIGH);

  // Build an Influx point with event = start
  Point eventPoint("recording_event");
  eventPoint.addTag("device", "ESP32-C6");
  eventPoint.addTag("object", "tongue");
  eventPoint.addField("status", "start");

  if (!client.writePoint(eventPoint)) {
    Serial.print("InfluxDB write failed (tongue start): ");
    Serial.println(client.getLastErrorMessage());
  }
}

void handleTongueStop() {
  Serial.println("Tongue recording STOP");
  tongueActive = false;
  digitalWrite(TONGUE_LED_PIN, LOW);

  Point eventPoint("recording_event");
  eventPoint.addTag("device", "ESP32-C6");
  eventPoint.addTag("object", "tongue");
  eventPoint.addField("status", "stop");

  if (!client.writePoint(eventPoint)) {
    Serial.print("InfluxDB write failed (tongue stop): ");
    Serial.println(client.getLastErrorMessage());
  }
}

// -------------------------------------------------------
// Check the pipe button for a press and toggle start/stop
void checkPipeButton() {
  int currentState = digitalRead(PIPE_BTN_PIN);
  // Detect a press from HIGH to LOW
  if (currentState == LOW && lastPipeBtnState == HIGH) {
    if (!pipeActive) {
      handlePipeStart();
    } else {
      handlePipeStop();
    }
  }
  lastPipeBtnState = currentState;
}

void handlePipeStart() {
  Serial.println("Pipe recording START");
  pipeActive = true;
  digitalWrite(PIPE_LED_PIN, HIGH);

  Point eventPoint("recording_event");
  eventPoint.addTag("device", "ESP32-C6");
  eventPoint.addTag("object", "pipe");
  eventPoint.addField("status", "start");

  if (!client.writePoint(eventPoint)) {
    Serial.print("InfluxDB write failed (pipe start): ");
    Serial.println(client.getLastErrorMessage());
  }
}

void handlePipeStop() {
  Serial.println("Pipe recording STOP");
  pipeActive = false;
  digitalWrite(PIPE_LED_PIN, LOW);

  Point eventPoint("recording_event");
  eventPoint.addTag("device", "ESP32-C6");
  eventPoint.addTag("object", "pipe");
  eventPoint.addField("status", "stop");

  if (!client.writePoint(eventPoint)) {
    Serial.print("InfluxDB write failed (pipe stop): ");
    Serial.println(client.getLastErrorMessage());
  }
}
