/*********************************************************************************
 * ESP32 Data Logging:
 *   - Uses Preferences (NVS) to keep an incrementing file index
 *   - Creates a new CSV file on SD each startup (LOG_XXXX.csv)
 *   - Reads DS18B20 (up to 4 sensors), HX711 load cell
 *   - Logs data every 0.1 s in format:
 *
 *     time, temp1, temp2, temp3, temp4, force
 *
 *     For "time" we just use millis() 
 *   - Drives a stepper motor continuously
 *********************************************************************************/

#include <Preferences.h>        // For storing file index in NVS
#include <OneWire.h>            // DS18B20
#include <DallasTemperature.h>  // DS18B20
#include <HX711.h>              // Load cell
#include <SPI.h>                // SPI (for SD)
#include <SD.h>                 // SD card

// --------------------- Stepper Motor Pins ---------------------
const int STEP_PIN    = 21;
const int DIR_PIN     = 20;
const int MICRO_DELAY = 800;  // microseconds between steps
const int END_STOP_UPPER = 42;
const int END_STOP_LOWER = 39;
// --------------------- DS18B20 (OneWire) ---------------------
#define ONE_WIRE_BUS 4
OneWire oneWire(ONE_WIRE_BUS);
DallasTemperature sensors(&oneWire);

const int NUMBER_OF_SENSORS = 4; // up to 4 sensors
float temperatures[NUMBER_OF_SENSORS];

// --------------------- HX711 Load Cell ---------------------
const int LOADCELL_DOUT_PIN = 35;
const int LOADCELL_SCK_PIN  = 36;
HX711 scale;
float loadCellValue = 0.0;

// --------------------- SD Card  ---------------------
#define MISO_PIN 11
#define SCK_PIN  12
#define MOSI_PIN 13
#define SD_CS    14

// --------------------- Logging Interval ---------------------
const unsigned long SAMPLE_INTERVAL = 100; // 0.1 seconds
unsigned long lastSampleTime = 0;

// --------------------- File Index + Filename ---------------------
Preferences preferences;  // NVS object
uint32_t fileIndex = 0;   // We'll store/read this from Preferences
File logFile;
char filename[32];        // e.g., "/LOG_0001.csv"

// -----------------------------------------------------------------------------
void setup() {
  Serial.begin(115200);
  delay(1000);
  Serial.println("ESP32 Logging with unique filenames");

  // 1) Initialize Preferences (NVS)
  preferences.begin("logfiles", false); // "logfiles" is the namespace, RW mode
  fileIndex = preferences.getUInt("fileIndex", 0); // If not found, defaults to 0
  fileIndex++; // Increment for this new session
  preferences.putUInt("fileIndex", fileIndex); // Save updated value back to NVS
  preferences.end(); // Close Preferences

  // 2) Build a filename: /LOG_####.csv  (zero-padded to 4 digits, for example)
  snprintf(filename, sizeof(filename), "/LOG_%04u.csv", fileIndex);
  Serial.print("This session's filename: ");
  Serial.println(filename);

  // 3) Initialize DS18B20
  sensors.begin();

  // 4) Initialize HX711
  scale.begin(LOADCELL_DOUT_PIN, LOADCELL_SCK_PIN);
  scale.set_scale(0); 
  scale.tare();

  // 5) Initialize Stepper and End stop pins
  pinMode(STEP_PIN, OUTPUT);
  pinMode(DIR_PIN,  OUTPUT);
  pinMode(END_STOP_UPPER, INPUT_PULLUP);
  pinMode(END_STOP_LOWER, INPUT_PULLUP);

  // 6) Initialize SD
  SPI.begin(SCK_PIN, MISO_PIN, MOSI_PIN, SD_CS);
  if (!SD.begin(SD_CS, SPI, 4000000)) {
    Serial.println("SD card initialization failed!");
    while (true) {
      delay(1000);
    }
  }
  Serial.println("SD card initialized.");

  // 7) Create/open the file for this session
  logFile = SD.open(filename, FILE_WRITE);
  if (!logFile) {
    Serial.print("Error creating file: ");
    Serial.println(filename);
  } else {
    // Write CSV header
    logFile.println("time, temp1, temp2, temp3, temp4, force");
    logFile.flush();
    Serial.print("Logging to file: ");
    Serial.println(filename);
  }

  Serial.println("Setup complete. Sampling every 0.1 s...");
}

// -----------------------------------------------------------------------------
void loop() {
  unsigned long currentTime = millis();

  // 1) Check if it's time to sample
  if (currentTime - lastSampleTime >= SAMPLE_INTERVAL) {
    lastSampleTime = currentTime;

    // Read sensors
    readTemperatures();
    readLoadCell();

    // Log to SD
    logData();
  }

  // 2) run stepper motor continuously
  runStepperMotor();
}

// --------------------- Read DS18B20 Temperatures ---------------------
void readTemperatures() {
  sensors.requestTemperatures();
  for (int i = 0; i < NUMBER_OF_SENSORS; i++) {
    float tempC = sensors.getTempCByIndex(i);
    temperatures[i] = tempC; // store reading
  }
}

// --------------------- Read Load Cell (HX711) ---------------------
void readLoadCell() {
  if (scale.is_ready()) {
    loadCellValue = scale.get_units(1); // single reading
  } else {
    loadCellValue = 0.0; // or keep last value if you prefer
    Serial.println("HX711 not ready!");
  }
}

// --------------------- Log Data to CSV ---------------------
void logData() {
  if (!logFile) {
    // If file didn't open in setup, attempt again
    logFile = SD.open(filename, FILE_APPEND);
    if (!logFile) {
      Serial.println("Cannot write data; file not open.");
      return;
    }
  }

  unsigned long t = millis();

  // Format: time, temp1, temp2, temp3, temp4, force
  // Example row: "12345, 23.50, 24.10, 23.90, 24.00, 345.67"
  logFile.print(t);
  logFile.print(", ");
  logFile.print(temperatures[0], 2);
  logFile.print(", ");
  logFile.print(temperatures[1], 2);
  logFile.print(", ");
  logFile.print(temperatures[2], 2);
  logFile.print(", ");
  logFile.print(temperatures[3], 2);
  logFile.print(", ");
  logFile.println(loadCellValue, 2);

  logFile.flush(); // ensure it's physically written to the card

  // print to Serial for debugging
  Serial.print("LOG: ");
  Serial.print(t);
  Serial.print(", ");
  Serial.print(temperatures[0], 2);
  Serial.print(", ");
  Serial.print(temperatures[1], 2);
  Serial.print(", ");
  Serial.print(temperatures[2], 2);
  Serial.print(", ");
  Serial.print(temperatures[3], 2);
  Serial.print(", ");
  Serial.println(loadCellValue, 2);
}

// --------------------- Stepper Motor Control ---------------------
void runStepperMotor() {
  // Check end stops – don't run if either is triggered
  bool upperPressed = digitalRead(END_STOP_UPPER) == LOW;
  bool lowerPressed = digitalRead(END_STOP_LOWER) == LOW;

  if (upperPressed || lowerPressed) {
    Serial.println("End stop pressed. Motor stopped.");
    return; // Exit function early
  }

  // Run motor clockwise for 5000 second
  digitalWrite(DIR_PIN, HIGH);
  unsigned long startTime = millis();
  while (millis() - startTime < 5000) {
    digitalWrite(STEP_PIN, HIGH);
    delayMicroseconds(MICRO_DELAY);
    digitalWrite(STEP_PIN, LOW);
    delayMicroseconds(MICRO_DELAY);
  }
}

