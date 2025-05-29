#include <Arduino.h>
#include <SPIFFS.h>
#include <SD.h>
#include <WiFi.h>
#include <InfluxDbClient.h>
#include <InfluxDbCloud.h>
#include <time.h>
#include <SPI.h>
#include "HX711.h"

// -------------------- CONFIGURATION --------------------
const char* WIFI_SSID = "Ståle  sin iPhone";
const char* WIFI_PASSWORD = "secret";

const char* INFLUXDB_URL = "https://eu-central-1-1.aws.cloud2.influxdata.com";
const char* INFLUXDB_TOKEN = "secret";
const char* INFLUXDB_ORG = "f6eea422dd9af5e4";
const char* INFLUXDB_BUCKET = "MainReading";
const char* MEASUREMENT_NAME = "load_cell_calibrated";

// -------------------- HARDWARE PINS --------------------
#define SD_CS        14
#define SD_MOSI       7
#define SD_SCK        6
#define SD_MISO       8
#define DIR_PIN       1
#define STEP_PIN      0
#define BUTTON_UP    22
#define BUTTON_DOWN  21
#define BUTTON_RECORD 19
#define LED_PIN      20
#define HX711_DOUT_PIN 5
#define HX711_SCK_PIN  4

// -------------------- NTP SETTINGS --------------------
const char* NTP_SERVER       = "pool.ntp.org";
const long  GMT_OFFSET_SEC   = 0;
const int   DAYLIGHT_OFFSET_SEC = 0;

// -------------------- SAMPLING PARAMETERS --------------------
const unsigned long SAMPLING_INTERVAL_US = 12500;  // 80 Hz
const int   CHUNK_SIZE              = 50;
const unsigned long RECORD_DURATION = 15000;     // ms

// -------------------- LOAD CELL CALIBRATION --------------------
float calibrationFactor = 433.387299;  // raw units per gram

HX711 scale(HX711_DOUT_PIN, HX711_SCK_PIN);

// -------------------- INFLUXDB CLIENT --------------------
InfluxDBClient client(INFLUXDB_URL, INFLUXDB_ORG, INFLUXDB_BUCKET,
                      INFLUXDB_TOKEN, InfluxDbCloud2CACert);
Point sensor(MEASUREMENT_NAME);

// -------------------- GLOBAL STATE --------------------
volatile bool isSampling     = false;
unsigned long recordingStart = 0;

// Motor control flags
volatile bool motorRunning = false;
volatile bool motorDir     = false;

// Sampling structures
struct Sample {
  unsigned long long timestamp; 
  float value;
  bool motorActive;
};

struct SampleChunk {
  Sample samples[CHUNK_SIZE];
  int count; // -1 signals termination
};

QueueHandle_t sampleQueue = NULL;

// Uncomment to enable Influx logging:  #define ENABLE_INFLUX
#ifdef ENABLE_INFLUX
void sendInfluxEvent(String object, String status) {
  Point eventPoint("recording_event");
  eventPoint.addTag("device", "ESP32-C6");
  eventPoint.addTag("object", object);
  eventPoint.addField("status", status);
  if (!client.writePoint(eventPoint)) {
    Serial.print("InfluxDB write failed (");
    Serial.print(object);
    Serial.println("): ");
    Serial.println(client.getLastErrorMessage());
  }
}
#else
void sendInfluxEvent(String object, String status) {}
#endif

void syncTime();
String getDateTimeString();
void samplingTask(void* parameter);
void sdWriteTask(void* parameter);
void motorTask(void* parameter);

// -------------------- TIME FUNCTIONS --------------------
void syncTime() {
  configTime(GMT_OFFSET_SEC, DAYLIGHT_OFFSET_SEC, NTP_SERVER);
  struct tm timeinfo;
  unsigned long start = millis();
  while (!getLocalTime(&timeinfo)) {
    if (millis() - start > 10000) {
      Serial.println("Failed to get NTP time!");
      return;
    }
    delay(100);
  }
  time_t now;
  time(&now);
  Serial.printf("Time synchronized: %s", ctime(&now));
}

String getDateTimeString() {
  struct tm timeinfo;
  if (!getLocalTime(&timeinfo)) {
    return "error";
  }
  char buffer[20];
  strftime(buffer, sizeof(buffer), "%d.%m.%y %H-%M-%S", &timeinfo);
  return String(buffer);
}

// -------------------- SETUP --------------------
void setup() {
  Serial.begin(115200);
  while (!Serial);

  // Pins
  pinMode(DIR_PIN, OUTPUT);
  pinMode(STEP_PIN, OUTPUT);
  pinMode(BUTTON_UP, INPUT_PULLUP);
  pinMode(BUTTON_DOWN, INPUT_PULLUP);
  pinMode(BUTTON_RECORD, INPUT_PULLUP);
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(DIR_PIN, LOW);
  digitalWrite(STEP_PIN, LOW);
  digitalWrite(LED_PIN, LOW);

  // SPIFFS
  if (!SPIFFS.begin(true)) {
    Serial.println("SPIFFS Mount Failed");
    while (1);
  }

  // SD card
  SPI.begin(SD_SCK, SD_MISO, SD_MOSI, SD_CS);
  if (!SD.begin(SD_CS)) {
    Serial.println("SD Card Mount Failed");
  } else {
    Serial.println("SD Card initialized");
  }

  // WiFi
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  Serial.print("Connecting to WiFi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nConnected!");

  // Time Sync
  syncTime();

  // InfluxDB tags
  sensor.addTag("device", "ESP32-C6");
  sensor.addTag("SSID", WiFi.SSID());

  // HX711
  scale.begin();
  scale.set_scale(calibrationFactor);
  scale.tare();  // reset the scale to 0
  Serial.println("HX711 Initialized.\nSystem ready!");

  // Queue
  sampleQueue = xQueueCreate(10, sizeof(SampleChunk));
  if (!sampleQueue) {
    Serial.println("Error creating sample queue!");
  }

  // SD Write Task
  xTaskCreatePinnedToCore(sdWriteTask, "SDWriteTask", 4096, NULL, 1, NULL, 0);

  // Motor Task
  xTaskCreatePinnedToCore(motorTask, "MotorTask", 2048, NULL, 2, NULL, 0);
}

void loop() {
  bool upPressed     = (digitalRead(BUTTON_UP) == LOW);
  bool downPressed   = (digitalRead(BUTTON_DOWN) == LOW);
  bool recordPressed = (digitalRead(BUTTON_RECORD) == LOW);

  if (!isSampling) {
    digitalWrite(LED_PIN, (upPressed || downPressed || recordPressed));
  }

  if (upPressed)     Serial.println("Forward (Up) button pressed");
  if (downPressed)   Serial.println("Backward (Down) button pressed");
  if (recordPressed) Serial.println("Record button pressed");

  // Motor logic
  if (upPressed && !downPressed) {
    motorRunning = true;
    motorDir     = true;
  } else if (downPressed && !upPressed) {
    motorRunning = true;
    motorDir     = false;
  } else {
    motorRunning = false;
  }

  // Start Sampling
  if (recordPressed && !isSampling) {
    isSampling     = true;
    recordingStart = millis();
    xTaskCreatePinnedToCore(samplingTask, "SamplingTask", 4096, NULL, 1, NULL, 0);
  }

  // Manage LED & stop
  if (isSampling) {
    unsigned long elapsed = millis() - recordingStart;
    if (elapsed >= RECORD_DURATION) {
      isSampling = false;
      digitalWrite(LED_PIN, LOW);
      Serial.println("Recording duration reached. Stopping sampling...");
    } else {
      if (RECORD_DURATION - elapsed < 3000) {
        digitalWrite(LED_PIN, (millis() / 500) % 2 == 0);
      } else {
        digitalWrite(LED_PIN, HIGH);
      }
    }
  }

  delay(10);
}

// -------------------- MOTOR TASK --------------------
void motorTask(void* parameter) {
  const unsigned int STEP_DELAY_US = 100;
  for (;;) {
    if (motorRunning) {
      digitalWrite(DIR_PIN, motorDir);
      digitalWrite(STEP_PIN, HIGH);
      delayMicroseconds(STEP_DELAY_US);
      digitalWrite(STEP_PIN, LOW);
      delayMicroseconds(STEP_DELAY_US);
      vTaskDelay(1 / portTICK_PERIOD_MS);
    } else {
      vTaskDelay(10 / portTICK_PERIOD_MS);
    }
  }
}

// -------------------- SAMPLING TASK --------------------
void samplingTask(void* parameter) {
  // Re-tare at start of each recording
  Serial.println("Taring scale...");
  scale.tare(10);
  Serial.println("Tare complete.");

  SampleChunk chunk;
  chunk.count = 0;
  unsigned long startTime      = millis();
  unsigned long lastSampleTime = micros();

  while ((millis() - startTime < RECORD_DURATION) && isSampling) {
    unsigned long currentMicros = micros();
    if (currentMicros - lastSampleTime >= SAMPLING_INTERVAL_US) {
      lastSampleTime = currentMicros;

      Sample s;
      struct timeval tv;
      gettimeofday(&tv, NULL);
      s.timestamp   = (tv.tv_sec * 1000000ULL + tv.tv_usec) * 1000ULL;
      s.value       = scale.get_units(1);  // weight in grams
      s.motorActive = motorRunning;

      chunk.samples[chunk.count++] = s;
      if (chunk.count >= CHUNK_SIZE) {
        xQueueSend(sampleQueue, &chunk, portMAX_DELAY);
        chunk.count = 0;
      }
    }
    vTaskDelay(1 / portTICK_PERIOD_MS);
  }

  // Flush remainder
  if (chunk.count > 0) {
    xQueueSend(sampleQueue, &chunk, portMAX_DELAY);
  }

  // Signal termination
  chunk.count = -1;
  xQueueSend(sampleQueue, &chunk, portMAX_DELAY);

  Serial.println("Sampling task complete.");
  vTaskDelete(NULL);
}

// -------------------- SD WRITE TASK --------------------
void sdWriteTask(void* parameter) {
  File dataFile;
  bool fileOpen = false;

  while (true) {
    SampleChunk chunk;
    if (xQueueReceive(sampleQueue, &chunk, portMAX_DELAY) == pdTRUE) {
      if (chunk.count == -1) {
        if (fileOpen) {
          dataFile.close();
          fileOpen = false;
          Serial.println("Recording file closed.");
        }
        continue;
      }

      if (!fileOpen) {
        String filename = "/" + getDateTimeString() + ".csv";
        dataFile = SD.open(filename.c_str(), FILE_WRITE);
        if (!dataFile) {
          Serial.println("Failed to open SD file for writing.");
          continue;
        }
        dataFile.println("Timestamp(ns),Value,MotorActive");
        fileOpen = true;
        Serial.print("Opened SD file: ");
        Serial.println(filename);
      }

      for (int i = 0; i < chunk.count; i++) {
        dataFile.print(chunk.samples[i].timestamp);
        dataFile.print(",");
        dataFile.print(chunk.samples[i].value);
        dataFile.print(",");
        dataFile.println(chunk.samples[i].motorActive ? "true" : "false");
      }
      dataFile.flush();
    }
  }
}
