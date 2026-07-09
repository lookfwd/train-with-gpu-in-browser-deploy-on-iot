// mqttio.js — thin wrapper over mqtt.js (bundled via npm) for the browser.
// The browser speaks MQTT over WebSockets; the ESP32 speaks plain TCP MQTT; same broker.
import mqtt from "mqtt";

export function connect(url, { subscribe = [], onStatus, onMessage } = {}) {
  const client = mqtt.connect(url, { reconnectPeriod: 2000, connectTimeout: 8000 });
  client.on("connect", () => {
    onStatus && onStatus("connected");
    for (const t of subscribe) client.subscribe(t, { qos: 1 });
  });
  client.on("reconnect", () => onStatus && onStatus("reconnecting…"));
  client.on("close", () => onStatus && onStatus("disconnected"));
  client.on("error", (e) => onStatus && onStatus("error: " + (e?.message || e)));
  client.on("message", (topic, payload) => onMessage && onMessage(topic, payload));
  return client;
}

export function publishJSON(client, topic, obj) {
  client.publish(topic, JSON.stringify(obj), { qos: 1 });
}

export function publishBytes(client, topic, bytes) {
  // bytes: Uint8Array — used to ship a .tflite from the browser (HTTP-converter mode).
  client.publish(topic, bytes, { qos: 1 });
}
