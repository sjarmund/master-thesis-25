#include <Wire.h>
#include <Adafruit_NAU7802.h>

Adafruit_NAU7802 nau7802;

// How many samples to average for tare and calibration
const int numSamples = 20;

// GPIO pins for your ESP32-C6 I2C (adjust if needed)
#define SDA_PIN 2
#define SCL_PIN 3

void setup() {
  Serial.begin(115200);
  delay(1000); // Give time for Serial to initialize

  Serial.println("Initializing NAU7802...");
  Wire.begin(SDA_PIN, SCL_PIN);
  if (!nau7802.begin()) {
    Serial.println("Failed to find NAU7802. Check wiring!");
    while (1);
  }

  nau7802.setGain(NAU7802_GAIN_64); // Same as used in your experiment
  nau7802.setRate(NAU7802_RATE_320SPS); // Match your sampling rate

  Serial.println("NAU7802 initialized.");
  delay(500);

  // ------------------------ TARE ------------------------
  Serial.println("Taring... Make sure NOTHING is on the load cell. (but have the setup");
  long tare = averageRawReading();
  Serial.print("Tare offset (raw ADC): ");
  Serial.println(tare);

  // -------------------- CALIBRATION ---------------------
  Serial.println("Now place a known weight on the load cell.");
  Serial.println("Enter the known weight in grams (e.g., 1000): ");
  while (!Serial.available()); // Wait for input
  float knownWeight = Serial.parseFloat();
  Serial.print("Using known weight: ");
  Serial.print(knownWeight);
  Serial.println(" g");

  delay(2000); // Wait for weight to settle

  long withWeight = averageRawReading();
  Serial.print("Raw reading with weight: ");
  Serial.println(withWeight);

  // Calculate calibration factor
  float calibrationFactor = (float)(tare - withWeight) / knownWeight;
  Serial.println("---------------------------------------------");
  Serial.print("Your calibration factor is: ");
  Serial.println(calibrationFactor, 6);
  Serial.print("Use this with:  value = (raw - ");
  Serial.print(tare);
  Serial.print(") / ");
  Serial.println(calibrationFactor, 6);
  Serial.println("---------------------------------------------");
}

void loop() {
  // Nothing here
}

long averageRawReading() {
  long sum = 0;
  for (int i = 0; i < numSamples; i++) {
    long reading = nau7802.read();
    sum += reading;
    delay(100); // ~10 Hz sample rate for clean averaging
  }
  return sum / numSamples;
}
