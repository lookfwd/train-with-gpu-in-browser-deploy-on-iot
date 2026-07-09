// app.js — orchestrates the whole browser side: identify → train → ship weights →
// overlay the device's separation against the browser's own.
import * as tf from "@tensorflow/tfjs";
import "@tensorflow/tfjs-backend-webgpu";
import { setWasmPaths } from "@tensorflow/tfjs-backend-wasm";
import wasmUrl from "@tensorflow/tfjs-backend-wasm/dist/tfjs-backend-wasm.wasm?url";
import wasmSimdUrl from "@tensorflow/tfjs-backend-wasm/dist/tfjs-backend-wasm-simd.wasm?url";
import wasmThreadedUrl from "@tensorflow/tfjs-backend-wasm/dist/tfjs-backend-wasm-threaded-simd.wasm?url";
import * as dsp from "./dsp.js";
import { loadSpec, buildModel, makeDataset, trainModel, serializeWeights, separate } from "./models.js";
import { connect, publishJSON, publishBytes } from "./mqttio.js";
import { plotSignal, plotSpectrum } from "./viz.js";

// tfjs-backend-wasm needs its .wasm files; Vite emits them as assets and hands us the URLs.
setWasmPaths({
  "tfjs-backend-wasm.wasm": wasmUrl,
  "tfjs-backend-wasm-simd.wasm": wasmSimdUrl,
  "tfjs-backend-wasm-threaded-simd.wasm": wasmThreadedUrl,
});

const BROKER_WS = "ws://192.168.86.50:8083";
// Model conversion. Empty -> send weights over MQTT to the Python builder (builder.py).
// Set to a converter URL (standalone HTTPS server or Lambda Function URL) to convert over HTTP
// and publish the .tflite directly. Overridable at runtime with ?converter=<url>.
const CONVERTER_URL = new URLSearchParams(location.search).get("converter") || "";
const BAND = [40, 200];            // Hz, kept well under Nyquist (fs/2 = 512) for clean sampling
const TRAIN_COUNT = 1200;
const EPOCHS = 25;

const T = {
  ADC: "adc/stream",
  INFER: "infer/device",
  STATUS_DEV: "status/device",
  STATUS_BUILDER: "status/builder",
  WEIGHTS: "train/weights",
  MODEL: "model/flatbuffer",
  RESHUFFLE: "cmd/reshuffle",
};

const state = {
  spec: null,
  archId: null,
  mode: "sim",              // "sim" | "live"
  model: null,
  trained: false,
  windowMs: 250,
  freqs: [80, 200],         // FFT-identified (what training uses)
  secret: [80, 200],        // sim-mode true freqs (never used for training)
  specEMA: null,            // smoothed magnitude spectrum
  lastMix: null,
  lastTruth: null,          // {c1, c2} in sim mode
  browserOut: null,         // [y0, y1]
  deviceOut: null,          // [y0, y1]
  deviceTiming: {},
  builderStatus: null,
  seq: 0,
  mqtt: null,
  simTimer: null,
  training: false,
  trainLoss: null,
  backend: null,        // active tfjs backend
  backends: [],         // backends that actually initialize in this browser
  trainTimes: {},       // backend -> last training wall-clock (ms)
  trainedFreqs: null,   // the freqs the current model was trained for
  freqHist: [],         // recent identified freqs, for drift-settle detection
  autoRetrain: true,    // auto-retrain once a drift settles
  quantize: false,      // ask the converter for an int8 model
};

const $ = (id) => document.getElementById(id);
function log(msg) {
  const el = $("log");
  const line = `${new Date().toLocaleTimeString()}  ${msg}\n`;
  el.textContent = (line + el.textContent).slice(0, 4000);
}

// ---- drift detection -----------------------------------------------------
function freqsClose(a, b, tol) {
  return a && b && Math.abs(a[0] - b[0]) < tol && Math.abs(a[1] - b[1]) < tol;
}

// Auto-retrain on drift, but ONLY once the identified freqs have settled (stopped moving)
// and differ from what the current model was trained on. The settle gate is what stops it
// retraining on the stale / half-updated peaks right after a reshuffle.
function maybeAutoRetrain() {
  if (!state.autoRetrain || state.training || !state.trained) return;
  state.freqHist.push([state.freqs[0], state.freqs[1]]);
  if (state.freqHist.length > 6) state.freqHist.shift();
  const settled =
    state.freqHist.length >= 6 && state.freqHist.every((h) => freqsClose(h, state.freqs, 4));
  if (!settled) return;
  if (freqsClose(state.freqs, state.trainedFreqs, 12)) return; // no meaningful drift
  log(`drift settled at [${state.freqs.map((f) => f.toFixed(0)).join(", ")}] Hz → auto-retraining`);
  onTrain();
}

// ---- data ingestion (one window) -----------------------------------------
function ingest(mix, truth, seq) {
  state.lastMix = mix;
  state.lastTruth = truth;
  state.seq = seq;

  const mag = dsp.magnitudeSpectrum(mix);
  if (!state.specEMA || state.specEMA.length !== mag.length) {
    state.specEMA = Float64Array.from(mag);
  } else {
    for (let i = 0; i < mag.length; i++) state.specEMA[i] = 0.8 * state.specEMA[i] + 0.2 * mag[i];
  }
  state.freqs = dsp.identifyTwoFreqs(state.specEMA, state.spec.fs_hz);
  maybeAutoRetrain();

  // Skip inference while training: running predict() 4×/s on the same WebGL context
  // starves model.fit() and can stall training.
  if (state.trained && state.model && !state.training) {
    state.browserOut = separate(state.model, state.spec, mix);
  }
  render();
}

// ---- simulated ADC (browser-local; used until a real device streams) -----
function startSim() {
  stopSim();
  state.simTimer = setInterval(() => {
    const { input_len: N, fs_hz: fs } = state.spec;
    const { mix, c1, c2 } = dsp.makeMixture(N, state.secret[0], state.secret[1], fs, { noise: 0.05 });
    ingest(mix, { c1, c2 }, state.seq + 1);
  }, state.windowMs);
}
function stopSim() {
  if (state.simTimer) { clearInterval(state.simTimer); state.simTimer = null; }
}

// ---- deploy: turn trained weights into a .tflite the device loads --------
// freqs (the two trained tones) are needed for int8 activation calibration.
async function deployModel(archId, weights, freqs) {
  const q = state.quantize;
  const payload = { arch_id: archId, weights, quantize: q, freqs };
  const tag = q ? " [int8]" : "";
  if (CONVERTER_URL) {
    // HTTP converter (standalone HTTPS server or Lambda): POST weights, get .tflite, publish it.
    const res = await fetch(CONVERTER_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!res.ok) throw new Error(`converter ${res.status}: ${(await res.text()).slice(0, 200)}`);
    const tflite = new Uint8Array(await res.arrayBuffer());
    publishBytes(state.mqtt, T.MODEL, tflite);
    state.builderStatus = {
      ok: true, arch: archId, tflite_bytes: tflite.length, quantized: q,
      convert_ms: Number(res.headers.get("X-Convert-Ms")) || 0,
      max_abs_err: Number(res.headers.get("X-Max-Abs-Err")) || 0,
    };
    log(`converted via HTTP${tag} (${tflite.length}B) → published model/flatbuffer → device`);
    render();
  } else {
    // MQTT path: publish weights; builder.py converts and publishes model/flatbuffer.
    publishJSON(state.mqtt, T.WEIGHTS, payload);
    log(`published weights (${weights.length} tensors)${tag} → MQTT builder`);
  }
}

// ---- training ------------------------------------------------------------
async function onTrain() {
  if (state.training) return;
  state.training = true;
  $("btnTrain").disabled = true;
  $("engine").disabled = true;
  try {
    log(`training ${state.archId} on ${state.backend} at f≈[${state.freqs.map((f) => f.toFixed(0)).join(", ")}] Hz …`);
    if (state.model) { state.model.dispose(); state.model = null; state.trained = false; }
    const trainFreqs = [state.freqs[0], state.freqs[1]];
    const model = buildModel(state.spec, state.archId);
    const ds = makeDataset(state.spec, trainFreqs[0], trainFreqs[1], TRAIN_COUNT);
    const t0 = performance.now();
    await trainModel(model, ds, {
      epochs: EPOCHS,
      onEpoch: (e, loss) => { state.trainLoss = loss; if (e % 5 === 0 || e === EPOCHS - 1) render(); },
    });
    const trainMs = performance.now() - t0;
    state.trainTimes[state.backend] = trainMs;
    ds.dispose();
    state.model = model;
    state.trained = true;
    state.trainedFreqs = trainFreqs;

    const weights = serializeWeights(model);
    log(`trained in ${(trainMs / 1000).toFixed(2)}s on ${state.backend}`);
    await deployModel(state.archId, weights, trainFreqs);
  } catch (e) {
    log("train error: " + e.message);
    console.error(e);
  } finally {
    state.training = false;
    $("btnTrain").disabled = false;
    $("engine").disabled = false;
    render();
  }
}

// ---- reshuffle -----------------------------------------------------------
function onReshuffle() {
  // Force a clean FFT re-lock and restart the settle timer, so drift is detected freshly
  // and auto-retrain waits for the new peaks to settle instead of firing on stale ones.
  state.specEMA = null;
  state.freqHist = [];
  if (state.mode === "live") {
    publishJSON(state.mqtt, T.RESHUFFLE, { band: BAND });
    log("sent cmd/reshuffle → device (FFT re-locks, then auto-retrains once settled)");
  } else {
    state.secret = dsp.randomFreqPair(BAND[0], BAND[1]);
    log("sim reshuffle → new secret freqs (hidden; FFT must re-find them)");
  }
}

// ---- MQTT ----------------------------------------------------------------
function onMessage(topic, payload) {
  try {
    if (topic === T.ADC) {
      const m = JSON.parse(payload.toString());
      if (state.mode === "live") ingest(Float32Array.from(m.x), null, m.seq ?? state.seq + 1);
    } else if (topic === T.INFER) {
      const m = JSON.parse(payload.toString());
      state.deviceOut = [Float32Array.from(m.y0), Float32Array.from(m.y1)];
      if (m.invoke_us != null) state.deviceTiming.invoke_us = m.invoke_us;
      render();
    } else if (topic === T.STATUS_DEV) {
      Object.assign(state.deviceTiming, JSON.parse(payload.toString()));
      render();
    } else if (topic === T.STATUS_BUILDER) {
      state.builderStatus = JSON.parse(payload.toString());
      log(`builder: ${payload.toString()}`);
      render();
    }
  } catch (e) {
    console.error("bad message on", topic, e);
  }
}

// ---- rendering -----------------------------------------------------------
function render() {
  if (state.lastMix) {
    plotSignal($("cMix"), [{ data: state.lastMix, color: "#ffd54f", label: "mixed ADC" }], { yrange: 2.5 });
  }
  if (state.specEMA) {
    plotSpectrum($("cSpec"), state.specEMA, state.spec.fs_hz, state.freqs);
  }
  const comp = (idx, canvasId, title) => {
    const series = [];
    if (state.lastTruth) series.push({ data: idx ? state.lastTruth.c2 : state.lastTruth.c1, color: "rgba(255,255,255,0.4)", dashed: true, label: "truth" });
    if (state.browserOut) series.push({ data: state.browserOut[idx], color: "#4fc3f7", label: "browser" });
    if (state.deviceOut) series.push({ data: state.deviceOut[idx], color: "#66bb6a", label: "device" });
    plotSignal($(canvasId), series, { yrange: 1.3 });
  };
  comp(0, "cComp1");
  comp(1, "cComp2");

  $("infoFreqs").textContent = state.freqs.map((f) => f.toFixed(1)).join(",  ") + " Hz";
  $("infoEngine").textContent = state.backend || "—";
  $("infoTrainTimes").textContent = state.backends.length
    ? state.backends
        .map((b) => `${b} ${state.trainTimes[b] ? (state.trainTimes[b] / 1000).toFixed(1) + "s" : "—"}`)
        .join("  ·  ")
    : "—";
  $("infoTrain").textContent = state.trained
    ? `trained (${state.archId}), loss=${state.trainLoss?.toExponential(2) ?? "?"}`
    : state.training ? `training… loss=${state.trainLoss?.toExponential(2) ?? "?"}` : "not trained";
  const b = state.builderStatus;
  $("infoBuilder").textContent = b
    ? (b.ok ? `${b.arch}${b.quantized ? " int8" : ""}: ${b.tflite_bytes}B, ${b.convert_ms}ms, err=${(+b.max_abs_err).toExponential(1)}` : `ERROR: ${b.error}`)
    : "—";
  const d = state.deviceTiming;
  $("infoDevice").textContent =
    (d.load_us != null ? `load=${(d.load_us / 1000).toFixed(1)}ms  ` : "") +
    (d.invoke_us != null ? `invoke=${(d.invoke_us).toFixed(0)}µs  ` : "") +
    (d.arena_bytes != null ? `arena=${(d.arena_bytes / 1024).toFixed(1)}KB  ` : "") +
    (d.arch ? `arch=${d.arch}` : "") || "—";
}

// ---- tfjs backends -------------------------------------------------------
async function setupEngines() {
  // (WASM paths are configured at module load via setWasmPaths above.)
  // Probe candidates in preference order; keep the ones that actually initialize here.
  const order = ["webgpu", "webgl", "wasm", "cpu"];
  const registered = tf.engine().registryFactory;
  const working = [];
  for (const b of order) {
    if (!(b in registered)) continue;
    try {
      if (await tf.setBackend(b)) { await tf.ready(); working.push(b); }
    } catch (e) { /* not usable in this browser */ }
  }
  state.backends = working;
  const def = working.includes("webgpu") ? "webgpu" : working.includes("webgl") ? "webgl" : working[0];
  await tf.setBackend(def);
  await tf.ready();
  state.backend = def;
}

async function switchEngine(name) {
  // The current model's tensors live on the old backend — drop it and require a retrain.
  if (state.model) { state.model.dispose(); state.model = null; }
  state.trained = false;
  state.browserOut = null;
  try {
    await tf.setBackend(name);
    await tf.ready();
    state.backend = name;
    log(`engine → ${name} (retrain to compare)`);
  } catch (e) {
    log(`engine ${name} failed: ${e.message}`);
    $("engine").value = state.backend;
  }
  render();
}

// ---- init ----------------------------------------------------------------
async function init() {
  await setupEngines();
  state.spec = await loadSpec();
  state.archId = state.spec.architectures[0].id;
  state.windowMs = Math.round((1000 * state.spec.input_len) / state.spec.fs_hz);
  state.secret = dsp.randomFreqPair(BAND[0], BAND[1]);

  const archSel = $("arch");
  for (const a of state.spec.architectures) {
    const opt = document.createElement("option");
    opt.value = a.id; opt.textContent = a.name;
    archSel.appendChild(opt);
  }
  archSel.value = state.archId;
  archSel.addEventListener("change", () => { state.archId = archSel.value; log(`arch → ${state.archId}`); });

  const engSel = $("engine");
  for (const b of state.backends) {
    const opt = document.createElement("option");
    opt.value = b; opt.textContent = b;
    engSel.appendChild(opt);
  }
  engSel.value = state.backend;
  engSel.addEventListener("change", () => switchEngine(engSel.value));

  $("mode").addEventListener("change", (e) => {
    state.mode = e.target.value;
    state.deviceOut = null;
    if (state.mode === "sim") { startSim(); log("mode → sim (browser-generated ADC)"); }
    else { stopSim(); log("mode → live (waiting for device adc/stream)"); }
  });
  $("btnTrain").addEventListener("click", onTrain);
  $("btnReshuffle").addEventListener("click", onReshuffle);
  const autoRe = $("autoRetrain");
  state.autoRetrain = autoRe.checked;
  autoRe.addEventListener("change", () => { state.autoRetrain = autoRe.checked; });
  const int8 = $("int8");
  state.quantize = int8.checked;
  int8.addEventListener("change", () => { state.quantize = int8.checked; log(`int8 quantize → ${state.quantize}`); });

  state.mqtt = connect(BROKER_WS, {
    subscribe: [T.ADC, T.INFER, T.STATUS_DEV, T.STATUS_BUILDER],
    onStatus: (s) => { $("mqttStatus").textContent = s; },
    onMessage,
  });

  startSim(); // default sim mode
  log(`ready. fs=${state.spec.fs_hz}Hz N=${state.spec.input_len} window=${state.windowMs}ms`);
}

init().catch((e) => { console.error(e); log("init error: " + e.message); });
