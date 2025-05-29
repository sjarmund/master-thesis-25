#include <HX711.h>

// Pin definitions
#define TOP_DOUT_PIN 21
#define BOTTOM_DOUT_PIN 1
#define SCK_PIN 4

// Create HX711 instances
HX711 topLoadCell;
HX711 bottomLoadCell;

void setup() {
  Serial.begin(115200);
  while (!Serial);
  
  // Initialize load cells
  topLoadCell.begin(TOP_DOUT_PIN, SCK_PIN);
  bottomLoadCell.begin(BOTTOM_DOUT_PIN, SCK_PIN);

  Serial.println("Initialization complete");
  Serial.println("Starting calibration...\n");

  // Calibrate top load cell
  calibrateLoadCell(topLoadCell, "Top");
  
  // Calibrate bottom load cell
  calibrateLoadCell(bottomLoadCell, "Bottom");

  Serial.println("\nCalibration complete!");
  Serial.println("Note down these values for your main program:");
  Serial.print("Top load cell offset: "); Serial.println(topLoadCell.get_offset());
  Serial.print("Top load cell scale: "); Serial.println(topLoadCell.get_scale());
  Serial.print("Bottom load cell offset: "); Serial.println(bottomLoadCell.get_offset());
  Serial.print("Bottom load cell scale: "); Serial.println(bottomLoadCell.get_scale());
}

void loop() {
  // Display live readings after calibration
  Serial.print("Top: "); 
  Serial.print(topLoadCell.get_units(), 1);
  Serial.print(" g\t");
  
  Serial.print("Bottom: ");
  Serial.print(bottomLoadCell.get_units(), 1);
  Serial.println(" g");
  
  delay(1000);
}

void calibrateLoadCell(HX711 &loadCell, const String &name) {
  Serial.println("=== " + name + " Load Cell Calibration ===");
  
  // Taring process
  Serial.println("1. Remove all weight from the " + name + " load cell");
  Serial.println("2. Press any key to tare...");
  waitForInput();
  
  long offset = readAverage(10, loadCell);
  loadCell.set_offset(offset);
  Serial.println("Tare complete. Offset: " + String(offset));
  
  // Calibration process
  Serial.println("\n3. Place a known weight on the " + name + " load cell");
  Serial.println("4. Enter the weight in grams and press send");
  float knownWeight = getFloatInput();
  
  long rawValue = readAverage(10, loadCell);
  float calibrationFactor = (rawValue - offset) / knownWeight;
  loadCell.set_scale(calibrationFactor);
  
  Serial.print("Calibration complete! Factor: ");
  Serial.println(calibrationFactor, 6);
  Serial.println("====================================\n");
}

long readAverage(int readings, HX711 &cell) {
  long total = 0;
  for (int i = 0; i < readings; i++) {
    total += cell.read();
    delay(10);
  }
  return total / readings;
}

void waitForInput() {
  while (!Serial.available());
  while (Serial.available()) Serial.read(); // Clear buffer
}

float getFloatInput() {
  while (!Serial.available());
  String input = Serial.readStringUntil('\n');
  return input.toFloat();
}