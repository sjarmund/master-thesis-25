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
#include <OneWire.h>
#include <DallasTemperature.h>

// DS18B20 Data pin
#define ONE_WIRE_BUS 4

OneWire oneWire(ONE_WIRE_BUS);
DallasTemperature dallasSensors(&oneWire);

// WiFi credentials
#define WIFI_SSID "Ståle  sin iPhone"
#define WIFI_PASSWORD "SECRET"

// InfluxDB settings
#define INFLUXDB_URL "https://eu-central-1-1.aws.cloud2.influxdata.com"
#define INFLUXDB_TOKEN "SECRET"
#define INFLUXDB_ORG "f6eea422dd9af5e4"
#define INFLUXDB_BUCKET "Temperature"

#define TZ_INFO "UTC1"

// Declare InfluxDB client instance
InfluxDBClient client(INFLUXDB_URL, INFLUXDB_ORG, INFLUXDB_BUCKET, INFLUXDB_TOKEN, InfluxDbCloud2CACert);
Point sensor("new_temp_sensors");
void setup() {
    Serial.begin(115200);
    Serial.println("Initializing DS18B20 Temperature Sensor...");

    dallasSensors.begin();

    Serial.print("Connecting to WiFi: ");
    Serial.println(WIFI_SSID);
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

    int attempts = 0;
    while (WiFi.status() != WL_CONNECTED && attempts < 30) { // Try for 15 seconds
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
        return; // Stop setup if no WiFi
    }

    // Sync time before InfluxDB connection
    Serial.println("Syncing time...");
    timeSync("UTC", "pool.ntp.org", "time.google.com");

    Serial.println("Waiting 5 seconds for time sync to stabilize...");
    delay(5000);  // Ensures time is synced before first temperature reading

    if (client.validateConnection()) {
        Serial.print("Connected to InfluxDB: ");
        Serial.println(client.getServerUrl());
    } else {
        Serial.print("InfluxDB connection failed: ");
        Serial.println(client.getLastErrorMessage());
    }

    // Ensure DS18B20 sensors are ready before taking readings
    Serial.println("Waiting 2 seconds for DS18B20 sensors to stabilize...");
    delay(2000);

    int numSensors = dallasSensors.getDeviceCount();
    Serial.print("Number of DS18B20 sensors detected: ");
    Serial.println(numSensors);

    if (numSensors == 0) {
        Serial.println("⚠️ ERROR: No DS18B20 sensors found! Check wiring.");
    }

    sensor.addTag("device", "ESP32-C6");
    sensor.addTag("SSID", WiFi.SSID());
}

void loop() {
    dallasSensors.requestTemperatures();
    delay(1000);  // Allow sensors time to respond

    float pipe = dallasSensors.getTempCByIndex(0);
    float water = dallasSensors.getTempCByIndex(1);
    float core = dallasSensors.getTempCByIndex(2);
    float core2 = dallasSensors.getTempCByIndex(3);

    // Avoid sending -127°C readings to InfluxDB
    if (pipe == -127.0 || water == -127.0 || core == -127.0 || core2 == -127.0) {
        Serial.println("⚠️ ERROR: Invalid DS18B20 reading (-127°C). Skipping this data point.");
        return;
    }

    Serial.println("Temperature Readings (°C):");
    Serial.print("Sensor 0: "); Serial.println(pipe);
    Serial.print("Sensor 1: "); Serial.println(water);
    Serial.print("Sensor 2: "); Serial.println(core);
    Serial.print("Sensor 3: "); Serial.println(core2);
    Serial.println("------------------------------");

    sensor.clearFields();
    sensor.addField("pipe", pipe);
    sensor.addField("water", water);
    sensor.addField("core", core);
    sensor.addField("core2", core2);

    Serial.print("Writing to InfluxDB: ");
    Serial.println(sensor.toLineProtocol());

    if (!client.writePoint(sensor)) {
        Serial.print("InfluxDB write failed: ");
        Serial.println(client.getLastErrorMessage());
    }

}
