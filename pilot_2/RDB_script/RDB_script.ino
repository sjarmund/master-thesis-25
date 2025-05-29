#if defined(ESP32)
  #include <WiFi.h>
  #define DEVICE "ESP32"
#elif defined(ESP8266)
  #include <ESP8266WiFi.h>
  #define DEVICE "ESP8266"
#else
  #include <WiFi.h>
  #define DEVICE "Unknown"
#endif

#include <InfluxDbClient.h>
#include <InfluxDbCloud.h>
#include <Wire.h>
#include <HX711.h>
#include <Adafruit_BME680.h>
#include <OneWire.h>         // For DS18B20
#include <DallasTemperature.h>  // For DS18B20

/* ------------------ Wi-Fi Credentials ------------------ */
#define WIFI_SSID      "Ståle  sin iPhone"
#define WIFI_PASSWORD  "secret"

/* ------------------ InfluxDB Settings ------------------ */
// InfluxDB 2.x (Cloud)
#define INFLUXDB_URL    "https://eu-central-1-1.aws.cloud2.influxdata.com"
#define INFLUXDB_TOKEN  "secret"
#define INFLUXDB_ORG    "f6eea422dd9af5e4"
#define INFLUXDB_BUCKET "MainReading"

// For time zone: UTC+1 (Oslo). Adjust if needed for DST, e.g. "CET-1CEST,M3.5.0,M10.5.0/3"
#define TZ_INFO "UTC1"

/* ------------------ Sensor Pin Definitions ------------------ */
// --- HX711 (Load Cells) ---
#define TOP_HX711_DOUT      21
#define HX711_SCK            4

// --- I2C for BME680 Sensor ---
#define I2C_SDA 42
#define I2C_SCL 41

// --- DS18B20 Sensor ---
#define DS18B20_PIN 17  // Data pin for DS18B20 on ESP32

/* ------------------ Calibration Factors ------------------ */
#define TOP_CALIBRATION     1057.352295

/* ------------------ Global Objects ------------------ */
// InfluxDB client instance
InfluxDBClient client(
  INFLUXDB_URL, 
  INFLUXDB_ORG, 
  INFLUXDB_BUCKET, 
  INFLUXDB_TOKEN, 
  InfluxDbCloud2CACert
);

// Measurement point for sensor data
Point sensorPoint("sensor_data");

// HX711 instance for load cell
HX711 hx711_top;

// BME680 sensor instance
Adafruit_BME680 bme;

// DS18B20 setup
OneWire oneWire(DS18B20_PIN);
DallasTemperature ds18b20(&oneWire);

/* ------------------ Function Declarations ------------------ */
void connectWiFi();
void setupInflux();

void setup() {
  Serial.begin(115200);
  delay(1000);

  // Connect to Wi-Fi
  connectWiFi();

  // Sync time for proper InfluxDB timestamps
  timeSync(TZ_INFO, "pool.ntp.org", "time.google.com");
  Serial.println("Waiting a few seconds for time sync...");
  delay(3000);

  // Initialize InfluxDB connection
  setupInflux();

  // ------------------ Initialize Load Cell ------------------
  hx711_top.begin(TOP_HX711_DOUT, HX711_SCK);
  hx711_top.set_scale(TOP_CALIBRATION);
  Serial.println("Taring load cells...");
  hx711_top.tare();

  // ------------------ Initialize BME680 Sensor ------------------
  Wire.begin(I2C_SDA, I2C_SCL);
  if (!bme.begin()) {
    Serial.println("❌ Could not find a valid BME680 sensor, check wiring!");
    while (1); // Halt execution if sensor is not found.
  }
  bme.setTemperatureOversampling(BME680_OS_8X);
  bme.setHumidityOversampling(BME680_OS_2X);
  bme.setPressureOversampling(BME680_OS_4X);
  bme.setIIRFilterSize(BME680_FILTER_SIZE_3);
  bme.setGasHeater(320, 150); // 320°C for 150 ms

  // ------------------ Initialize DS18B20 Sensor ------------------
  ds18b20.begin(); // Start the DS18B20 sensor

  // Add common tags to the InfluxDB measurement point
  sensorPoint.addTag("device", DEVICE);
  sensorPoint.addTag("SSID", WiFi.SSID());
}

void loop() {
  // ------------------ Read Load Cell Data ------------------
  float topWeight = hx711_top.get_units(10);

  // ------------------ Read BME680 Data ------------------
  if (!bme.performReading()) {
    Serial.println("Failed to perform reading from BME680 sensor!");
    delay(1000);
    return;
  }
  float temperature = bme.temperature;         // °C
  float humidity    = bme.humidity;              // %RH
  float pressure    = bme.pressure / 100.0;        // hPa
  float gas         = bme.gas_resistance;          // Ohms

  // ------------------ Read DS18B20 Data ------------------
  ds18b20.requestTemperatures();  // Request temperature reading
  float pipeTemp = ds18b20.getTempCByIndex(0); // Get temperature in °C

  // ------------------ Prepare Data Point for InfluxDB ------------------
  sensorPoint.clearFields();
  sensorPoint.addField("top_load", topWeight);
  sensorPoint.addField("temperature", temperature);
  sensorPoint.addField("humidity", humidity);
  sensorPoint.addField("pressure", pressure);
  sensorPoint.addField("gas_resistance", gas);
  sensorPoint.addField("pipeTemp", pipeTemp);  // DS18B20 temperature reading

  Serial.print("Writing sensor data to InfluxDB: ");
  Serial.println(sensorPoint.toLineProtocol());

  if (!client.writePoint(sensorPoint)) {
    Serial.print("❌ InfluxDB write failed: ");
    Serial.println(client.getLastErrorMessage());
  }

  // Wait a second before sending the next data point
  delay(300);
}

void connectWiFi() {
  Serial.print("Connecting to WiFi: ");
  Serial.println(WIFI_SSID);

  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  const unsigned long startAttemptTime = millis();
  
  while (WiFi.status() != WL_CONNECTED && (millis() - startAttemptTime) < 15000) {
    Serial.print(".");
    delay(500);
  }
  
  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("\n✅ WiFi connected!");
    Serial.print("IP address: ");
    Serial.println(WiFi.localIP());
  } else {
    Serial.println("\n❌ Failed to connect to WiFi.");
  }
}

void setupInflux() {
  if (client.validateConnection()) {
    Serial.print("Connected to InfluxDB: ");
    Serial.println(client.getServerUrl());
  } else {
    Serial.print("InfluxDB connection failed: ");
    Serial.println(client.getLastErrorMessage());
  }
}
