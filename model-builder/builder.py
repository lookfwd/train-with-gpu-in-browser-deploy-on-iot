"""MQTT model-builder (local mode).

Subscribes `train/weights`, converts the browser's trained weights to a `.tflite` via
`converter.convert_weights`, and publishes the flatbuffer on `model/flatbuffer` (+ metadata
on `status/builder`) for the ESP32-S3 to load. The same conversion is exposed over HTTP by
server.py (standalone) and lambda_function.py (AWS Lambda) — see the README.

Run:
    python builder.py                # connect to broker, serve forever
    python builder.py --selftest     # build+convert all archs with random weights (no MQTT)
    BROKER_HOST=192.168.86.50 python builder.py
"""
import os
import argparse
import json

import paho.mqtt.client as mqtt

import converter

BROKER_HOST = os.environ.get("BROKER_HOST", "192.168.86.50")
BROKER_PORT = int(os.environ.get("BROKER_PORT", "1883"))

TOPIC_WEIGHTS = "train/weights"     # in:  {"arch_id", "weights":[{"shape","data"},...]}
TOPIC_MODEL = "model/flatbuffer"    # out: raw .tflite bytes
TOPIC_STATUS = "status/builder"     # out: {"ok", "arch", "tflite_bytes", "convert_ms", "max_abs_err"}


def make_client():
    # paho-mqtt 2.x requires an explicit callback API version; 1.x does not have it.
    try:
        return mqtt.Client(mqtt.CallbackAPIVersion.VERSION1)
    except AttributeError:
        return mqtt.Client()


def handle_weights(client, payload):
    msg = json.loads(payload)
    tflite_bytes, meta = converter.convert_weights(
        msg["arch_id"], msg["weights"], quantize=msg.get("quantize", False), freqs=msg.get("freqs"))
    client.publish(TOPIC_MODEL, tflite_bytes, qos=1)
    client.publish(TOPIC_STATUS, json.dumps({"ok": True, **meta}), qos=1)
    print(f"[builder] {meta} -> published")


def main():
    client = make_client()

    def on_connect(c, userdata, flags, rc, *args):
        print(f"[builder] connected rc={rc}; subscribing to {TOPIC_WEIGHTS}")
        c.subscribe(TOPIC_WEIGHTS, qos=1)

    def on_message(c, userdata, m):
        try:
            handle_weights(c, m.payload)
        except Exception as e:  # keep the service alive; report the failure
            import traceback
            traceback.print_exc()
            c.publish(TOPIC_STATUS, json.dumps({"ok": False, "error": str(e)}), qos=1)

    client.on_connect = on_connect
    client.on_message = on_message
    print(f"[builder] connecting to {BROKER_HOST}:{BROKER_PORT} ...")
    client.connect(BROKER_HOST, BROKER_PORT, keepalive=60)
    client.loop_forever()


def selftest():
    """Round-trip every architecture (random weights -> convert -> validate). No MQTT."""
    s = converter.spec()
    print(f"input_len={s['input_len']} channels_out={s['channels_out']} fs={s['fs_hz']}Hz")
    for arch in s["architectures"]:
        model = converter.build_model(converter.get_arch(s, arch["id"]), s["input_len"], s["channels_out"])
        weights = [{"shape": list(w.shape), "data": w.flatten().tolist()} for w in model.get_weights()]
        _, meta = converter.convert_weights(arch["id"], weights)
        print(f"[selftest] {arch['id']:6s} params={model.count_params():6d} {meta}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true", help="build+convert all archs, no MQTT")
    args = ap.parse_args()
    if args.selftest:
        selftest()
    else:
        main()
