# Train with GPU in-Browser, Deploy to IoT

A small end-to-end demonstrator of a specific edge-AI architecture: **the browser does the
heavy, occasional analytics and training; a microcontroller (ESP32-S3) runs the cheap, real-time
forward pass; the model is shipped over MQTT.**

The concrete task is **blind separation of two frequencies**. A (simulated) ADC produces a signal
that is the sum of two sinusoids plus noise. The browser must figure out the two frequencies on its
own, train a small convolutional filter bank to separate them, and deploy that network to the
ESP32-S3, which then separates the live signal on-device and reports exactly how fast it does so.

[![AI Training and Inference at the Edge: WebGPU, TensorFlow.js, & ESP32-S3](https://img.youtube.com/vi/r_2JWi05fBw/0.jpg)](https://www.youtube.com/watch?v=r_2JWi05fBw)

---

## What it actually demonstrates

It is **not** a "real-time vs. laggy" side-by-side — both the device's output and the browser's own
output cross the network once before being drawn, so there is no visible jitter difference to show.
The honest payoff is:

1. **The full pipeline works end to end** — browser trains → model is built with real tooling →
   microcontroller loads and runs it.
2. **Hard timing numbers measured on the ESP32-S3**, reported back over MQTT:
   - `load_us` — time to build a `MicroInterpreter` and `AllocateTensors` for a freshly received
     `.tflite`.
   - `invoke_us` — time to push one window of samples through the model on the device.
3. **Agreement** — the device's separated waveforms match the browser's own inference on the same
   signal, confirming the deployed model is numerically correct.

---

## Architecture & data flow

```
        ┌─────────────────────────┐          MQTT broker (rumqttd, already running)
        │        Browser          │          192.168.86.50
        │  (Vite app + TF.js)     │            ├─ tcp   1883  (device)
        │                         │            └─ ws    8083  (browser)
        │  1. FFT → find 2 tones  │
        │  2. synth training data │   train/weights   ┌────────────────────────┐
        │  3. train CNN (TF.js)   │ ────────────────▶ │   model-builder        │
        │  4. run model locally   │                   │   (Python + TF)        │
        └───────────▲─────────────┘                   │  set_weights →         │
             adc/stream │ infer/device                │  TFLiteConverter →     │
             status/*   │                             │  self-validate         │
                        │                             └───────────┬────────────┘
                        │                          model/flatbuffer│ (.tflite bytes)
        ┌───────────────┴─────────────┐                           ▼
        │        ESP32-S3             │ ◀─────────────────────────┘
        │  (ESP-IDF + esp-tflite-micro)│
        │  • simulated ADC @ 1024 Hz  │   cmd/reshuffle (browser → device)
        │  • generic .tflite runner   │
        │  • esp_timer instrumentation│
        └─────────────────────────────┘
```

The device streams only the **raw mixed signal**. It never sends the true frequencies or the clean
components — the browser has to recover everything from the stream itself.

The **model-builder** is drawn above in its MQTT form (browser publishes `train/weights`; builder
publishes `model/flatbuffer`). It can equally run as an **HTTP converter** — standalone or an AWS
Lambda — in which case the browser POSTs the weights and publishes the returned `.tflite` itself
(select with `?converter=<url>`; see [Running it](#running-it)).

---

## The blind-separation idea

The ESP32 internally sums two sinusoids at secret frequencies `f1, f2` (plus noise). It streams that
mixture and nothing else. On the browser side:

1. **Identify** — run an FFT over the incoming windows, average the spectra, and pick the two
   dominant peaks. Those become the browser's estimate of `f1, f2`. This is the only "figuring out"
   step, and it uses nothing but the streamed signal.
2. **Synthesize** — knowing (its estimate of) the two frequencies, the browser generates its *own*
   labeled dataset: many windows of `random-amplitude sinusoid @ f1 + random-amplitude sinusoid @ f2
   + noise`, with the two clean components as labels. Because the browser generated these, it knows
   the ground truth **for its own synthetic data** — no labels ever come from the device.
3. **Train** — a fully-convolutional CNN learns to map the mixture to the two clean components. Its
   learned convolution kernels are, in effect, two band-pass filters.

When you press **Reshuffle**, the device picks new secret frequencies (it does *not* announce them);
the browser notices the FFT peaks have moved, re-identifies, retrains, and redeploys.

---

## The three model architectures

All three take a window of `N = 256` samples and output two length-`N` channels (the separated
components), using `same` padding so the output length is preserved. They are defined once in
[`web/architectures.json`](web/architectures.json) and built identically by the browser
(TF.js) and the builder (Keras).

| id     | structure                                             | params | ~tflite | role                                   |
|--------|-------------------------------------------------------|-------:|--------:|----------------------------------------|
| `fir1` | one `Conv1D(2, k=65)`                                  |    132 |  ~2.2 KB| two learned FIR filters; smallest      |
| `cnn3` | `Conv1D(8,15)→Conv1D(8,15)→Conv1D(2,15)` (ReLU)        |  1 338 |  ~8.5 KB| multi-level                            |
| `wide` | `Conv1D(8,33)→Conv1D(8,33)→Conv1D(2,33)` (ReLU)        |  2 922 | ~14.8 KB| large receptive field via wide kernels |

> The third model uses **wide kernels rather than dilation** to reach a large receptive field. A
> dilated TCN is the more elegant choice, but TensorFlow.js cannot backprop through dilated
> convolutions ("dilation rates greater than 1 are not yet supported in gradients"), and training
> happens in the browser — so dilation is off the table here.

Picking a different architecture from the dropdown is the "heavy" step; retraining and re-shipping
weights is the "light" step. Both are just "here is a new `.tflite`" as far as the device is
concerned — which is what makes the firmware a clean generic runner.

---

## MQTT protocol

All JSON except `model/flatbuffer` (raw `.tflite` bytes).

| topic              | dir              | payload                                                    |
|--------------------|------------------|------------------------------------------------------------|
| `adc/stream`       | device → browser | `{seq, fs, x:[N floats]}` — the raw mixed window            |
| `infer/device`     | device → browser | `{seq, invoke_us, y0:[N], y1:[N]}` — device separation      |
| `train/weights`    | browser → builder| `{arch_id, weights:[{shape,data}, …]}`                     |
| `model/flatbuffer` | builder → device | raw `.tflite` bytes                                        |
| `cmd/reshuffle`    | browser → device | `{band:[lo,hi]}` — pick new secret frequencies             |
| `status/device`    | device → browser | `{event, load_us, arena_bytes, tflite_bytes}` / `{invoke_us, free_heap}` |
| `status/builder`   | builder → browser| `{ok, arch, tflite_bytes, convert_ms, max_abs_err}`        |

---

## Repository layout

```
apps/ai-on-edges-demo/
├── model-builder/              # weights → .tflite (never trains). All 3 front-ends share converter.py:
│   ├── converter.py            #   pure convert_weights(arch_id, weights) -> (tflite, meta)
│   ├── architectures.py        #   Keras twin of the spec
│   ├── builder.py              #   front-end A: MQTT loop (+ --selftest)
│   ├── server.py               #   front-end B: standalone HTTP(S) POST /convert
│   ├── lambda_function.py      #   front-end C: AWS Lambda handler
│   ├── Dockerfile              #   container image for the Lambda (TF is too big for a zip)
│   ├── requirements.txt        #   local (tensorflow + paho-mqtt)
│   └── requirements-lambda.txt #   Lambda (tensorflow-cpu, no MQTT)
├── web/                        # Vite project → `npm run build` emits a self-contained web/dist/
│   ├── package.json  vite.config.js
│   ├── index.html              #   no CDN; TF.js + mqtt.js bundled from npm
│   ├── architectures.json      #   3 model defs (single source; imported by JS, read by Python)
│   └── src/{dsp,models,mqttio,viz,app}.js
├── infra/                      # AWS CDK (Python): deploy the converter as a Lambda + Function URL
│   ├── app.py  cdk.json  requirements.txt
├── esp-project/                # PlatformIO ESP-IDF firmware (esp32-s3-devkitc-1)
│   ├── platformio.ini  version.txt
│   ├── include/secrets.h       # <-- put your Wi-Fi credentials here
│   └── src/{main.cpp, idf_component.yml}
└── README.md
```

---

## Running it

A full loop needs the **web app** plus a **converter** (weights → `.tflite`). An MQTT broker is
assumed at `192.168.86.50`.

### Quick start (local, no hardware)

Prerequisites: Node, Python 3, and the broker reachable at `192.168.86.50`.

```bash
# one-time setup
( cd model-builder && python3 -m venv .venv && ./.venv/bin/pip install -r requirements.txt )
( cd web && npm install )
```

Then, in two terminals (leave both running):

```bash
# terminal 1 — converter (weights → .tflite)
cd model-builder && ./.venv/bin/python builder.py

# terminal 2 — web app (dev server)
cd web && npm run dev            # → http://localhost:5173
```

Open the URL, stay in **sim** mode (the browser generates the signal itself), and press
**Train & deploy** — the whole pipeline runs with no ESP32 attached. To bring in the board, flash the
firmware (below) and flip the page to **live** mode.

**Stop everything:** `Ctrl-C` both terminals. Nothing else persists — no daemons, no Docker, no
containers. (The MQTT broker is separate, and the ESP32 keeps running its last model until repowered.)

The rest of this section is reference: the deployable build, the HTTP/Lambda converter, and firmware.

### Web app (Vite)

```bash
cd web
npm install
npm run dev            # dev server (HMR)         → http://localhost:5173
npm run build          # deployable build         → web/dist/  (TF.js, mqtt.js, WASM all bundled in)
npm run preview        # serve the built dist/    → http://localhost:4173
```

> ⚠️ **Do not serve the `web/` source directly** (e.g. `python3 -m http.server` in `web/` and opening
> `index.html`). `src/*.js` uses bare imports like `import "@tensorflow/tfjs"` that only Vite resolves,
> so the browser will throw *"Failed to resolve module specifier @tensorflow/tfjs"*. Use `npm run dev`,
> or `npm run build` and serve `web/dist/` (`npm run preview`, or any static server pointed at `dist/`).

Start in **sim** mode (the browser generates the ADC signal itself) to exercise everything without
hardware: watch the FFT lock onto two peaks, press **Train & deploy**, and confirm a converted
`.tflite` comes back with a tiny `max_abs_err`.

The **Engine** dropdown lists the TF.js backends that initialize in your browser
(WebGPU / WebGL / WASM / CPU), defaults to **WebGPU**, and records the training wall-clock per backend
so you can compare them (e.g. WebGPU ~2.2 s vs software WebGL ~62 s).

**Choosing the converter:** by default the browser sends weights over MQTT to the Python builder
(mode A). To convert over HTTP instead (modes B/C), append `?converter=<url>` to the page URL:
`http://localhost:5173/?converter=https://localhost:8443/convert`.

### Converter — pick one

All three run the *same* `converter.py` (rebuild Keras arch → `set_weights` → `TFLiteConverter` →
self-validate). **None of them train** — training is 100 % in the browser (TF.js/WebGPU).

> ⚠️ **Restart `builder.py` / `server.py` after editing any Python here.** A long-running process keeps
> the code it was started with — a common surprise is a fresh request (e.g. `int8 quantize`) silently
> returning the old behaviour (a float model) because the running converter predates the change.

**A — MQTT builder** (default, simplest local):
```bash
cd model-builder
python3 -m venv .venv && ./.venv/bin/pip install -r requirements.txt   # one-time
./.venv/bin/python builder.py            # subscribes train/weights, publishes model/flatbuffer
./.venv/bin/python builder.py --selftest # convert all 3 archs offline (no MQTT/hardware)
```

**B — Standalone HTTPS server** (no auth):
```bash
cd model-builder
# self-signed TLS (browser must trust it once); omit for plain-HTTP local dev:
openssl req -x509 -newkey rsa:2048 -nodes -keyout key.pem -out cert.pem -days 365 -subj '/CN=localhost'
PORT=8443 ./.venv/bin/python server.py   # POST /convert -> .tflite ; GET / -> health
# open the web app with ?converter=https://localhost:8443/convert
```

**C — AWS Lambda** (container image behind an HTTPS Function URL, auth NONE), via CDK:
```bash
cd infra
python3 -m venv .venv && ./.venv/bin/pip install -r requirements.txt   # aws-cdk-lib
./.venv/bin/cdk deploy                    # builds the TF container, deploys, prints ConverterUrl
# open the web app with ?converter=<ConverterUrl>convert
```
TensorFlow is far too big for a zip Lambda, so this is a **container image** (`model-builder/Dockerfile`);
expect a slow first cold start. Needs AWS creds + Docker locally for `cdk deploy`.

### Firmware (ESP32-S3)

```bash
cd esp-project
cp include/secrets.h.example include/secrets.h   # then edit it — Wi-Fi SSID/password (git-ignored)
~/.platformio/penv/bin/pio run -e esp32-s3-devkitc-1              # build
~/.platformio/penv/bin/pio run -e esp32-s3-devkitc-1 -t upload    # flash over USB
~/.platformio/penv/bin/pio device monitor                        # serial log
```

Then switch the web app to **live** mode. The device streams `adc/stream`; the browser identifies,
trains, and deploys; the device loads the model (reporting `load_us`) and streams `infer/device`
(reporting `invoke_us`). The two separated components overlay the browser's own output.

---

## On-device timing methodology

Both numbers come from `esp_timer_get_time()` (microsecond monotonic clock) on the ESP32-S3:

- **`load_us`** brackets exactly: copy the flatbuffer into RAM → `tflite::GetModel` → construct the
  `MicroInterpreter` → `AllocateTensors()`. This is the whole cost of switching to a new model. For
  these few-KB models it is small, which is *why* shipping the whole flatbuffer on every update
  (instead of trying to patch weights in place) is the right call — see below. The swap holds a mutex
  so it cannot run while `pipeline_task` is mid-`Invoke()` — both touch the shared tensor arena, so a
  concurrent build would corrupt a running inference. That means `load_us` occasionally includes a
  wait for an in-flight `Invoke()` to finish (bounded by the current model's inference time).
- **`invoke_us`** brackets a single `Invoke()` on one 256-sample window (input copy excluded, so it
  is the pure inference cost). `arena_used_bytes()` is also reported so you can see the real memory
  footprint versus the 96 KB static arena.

---

## Why "ship the whole model", not "patch weights in place"

The original idea was to keep one model resident on the device and stream only new weights into it,
for instant updates. That was validated against the evidence and rejected as fragile **on the
ESP32-S3 specifically**:

- Espressif's optimized **ESP-NN kernels repack conv weights at `Prepare()`**, so overwriting the
  flatbuffer's weight bytes afterwards would not change what the kernel actually reads.
- Feeding weights as a runtime input tensor hits a known TFLite bug that silently serves **stale
  weights** ([tensorflow#31205](https://github.com/tensorflow/tensorflow/issues/31205)).
- TFLite Micro ships **no supported runtime weight-update API**
  ([tflite-micro#2475](https://github.com/tensorflow/tflite-micro/issues/2475)).

Building the whole `.tflite` with the real `TFLiteConverter` and doing a standard load each time
side-steps all of it, and (per the measured `load_us`) is cheap enough that the "instant update"
optimization buys nothing here. It also removes the Keras↔TFLite weight-layout transpose that
hand-patching would have required.

There is no pure-browser TF.js → `.tflite` path either, which is why the conversion lives in a small
Python service rather than in the page.

---

## Deploying

Four artefacts — only two are static files; the other two are running services.

| piece | what to ship | where it runs |
|-------|--------------|---------------|
| **Web** | `web/dist/` from `npm run build` (self-contained: HTML/JS/CSS/WASM, no CDN) | any static host — nginx, Netlify/Vercel, Cloudflare/GitHub Pages, S3+CloudFront |
| **Converter** | `converter.py` behind `server.py`, or the CDK Lambda | a small server/container, or AWS Lambda (`infra/` → `cdk deploy`) |
| **Broker** | an MQTT broker with a WebSocket listener | you already run one (rumqttd/mosquitto/EMQX…) |
| **Firmware** | `firmware.bin` (+ bootloader + partition table), or an OTA image | flashed to the ESP32-S3 |

The web tree and the firmware image are static; the converter and broker are processes that must be
up. (There is no pure-browser `.tflite` path, so the converter can't be "just static".)

---

## Limitations & next steps

- **Simulated ADC.** The signal is generated in firmware; wiring a real ADC (I2S/SAR) is the natural
  next step and changes nothing above the sample source.
- **Fixed two-tone model.** The task assumes exactly two frequencies below Nyquist (512 Hz at
  `fs = 1024`). Amplitudes/phases vary; the frequencies are fixed until you reshuffle.
- **Agreement is by latest-sample, not strict `seq` matching.** Good enough to eyeball; tightening it
  to match `infer/device.seq` against the browser's own run of the same window is a small change.
- **Float vs int8.** Models are float32 by default. Tick **int8 quantize** and the converter produces
  a full-int8 model **with float32 I/O** (a representative dataset — synthesized from the two tones —
  calibrates it; `QUANTIZE`/`DEQUANTIZE` wrap the int8 core, so the firmware I/O is unchanged). This
  lets the S3's **ESP-NN SIMD** kernels run: measured **`wide` 177 ms → 6.1 ms (≈29×)** and half the
  size. Caveat: it adds quantization error (reported as `max_abs_err`) — fine for `cnn3`/`wide`
  (~0.05–0.1), but poor for the tiny linear `fir1` (~0.6, and it's overhead-bound so int8 won't speed
  it up anyway).
