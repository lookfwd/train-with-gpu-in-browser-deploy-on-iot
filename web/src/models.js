// models.js — build/train the 3 architectures in TF.js from the shared spec.
// tf is bundled via npm; the spec is inlined at build time (Vite JSON import) —
// same file the Python builder reads, so there is no fetch and no 404 to get wrong.
import * as tf from "@tensorflow/tfjs";
import * as dsp from "./dsp.js";
import spec from "../architectures.json";

export function loadSpec() {
  return spec;
}

export function getArch(spec, archId) {
  const a = spec.architectures.find((x) => x.id === archId);
  if (!a) throw new Error(`unknown arch ${archId}`);
  return a;
}

// Build a Conv1D stack matching architectures.json. Input [N,1] -> output [N, channels_out].
// This mirrors the Python Keras builder exactly so weights transfer 1:1.
export function buildModel(spec, archId) {
  const arch = getArch(spec, archId);
  const model = tf.sequential();
  arch.layers.forEach((ly, i) => {
    if (ly.type !== "conv1d") throw new Error(`unsupported layer ${ly.type}`);
    const cfg = {
      filters: ly.filters,
      kernelSize: ly.kernel,
      dilationRate: ly.dilation ?? 1,
      padding: "same",
      activation: ly.activation === "linear" ? "linear" : ly.activation ?? "linear",
      name: `conv${i}`,
    };
    if (i === 0) cfg.inputShape = [spec.input_len, 1];
    model.add(tf.layers.conv1d(cfg));
  });
  return model;
}

// Build a training set at the identified frequencies. Input = noisy mix, labels = clean components.
export function makeDataset(spec, f1, f2, count) {
  const N = spec.input_len, fs = spec.fs_hz;
  const xs = new Float32Array(count * N);
  const ys = new Float32Array(count * N * 2);
  for (let b = 0; b < count; b++) {
    const { mix, c1, c2 } = dsp.makeMixture(N, f1, f2, fs, { noise: 0.05 });
    for (let i = 0; i < N; i++) {
      xs[b * N + i] = mix[i];
      ys[(b * N + i) * 2 + 0] = c1[i];
      ys[(b * N + i) * 2 + 1] = c2[i];
    }
  }
  return {
    xs: tf.tensor3d(xs, [count, N, 1]),
    ys: tf.tensor3d(ys, [count, N, 2]),
    dispose() { this.xs.dispose(); this.ys.dispose(); },
  };
}

export async function trainModel(model, ds, { epochs = 25, batchSize = 32, lr = 0.01, onEpoch } = {}) {
  model.compile({ optimizer: tf.train.adam(lr), loss: "meanSquaredError" });
  await model.fit(ds.xs, ds.ys, {
    epochs,
    batchSize,
    shuffle: true,
    // 'never' keeps fit from yielding to requestAnimationFrame between batches.
    // Default ('auto') stalls training whenever the tab isn't painting (backgrounded
    // / headless). The brief UI freeze during a deliberate "Train" click is fine.
    yieldEvery: "never",
    callbacks: { onEpochEnd: (e, logs) => onEpoch && onEpoch(e, logs.loss) },
  });
}

// Serialize weights for the builder: [{shape, data}, ...] in model weight order
// (matches Keras model.get_weights(): [conv0/kernel, conv0/bias, conv1/kernel, ...]).
export function serializeWeights(model) {
  return model.getWeights().map((w) => ({ shape: w.shape, data: Array.from(w.dataSync()) }));
}

// Run the model on one window (Float32Array length N). Returns [y0, y1] Float32Arrays.
export function separate(model, spec, mix) {
  const N = spec.input_len;
  return tf.tidy(() => {
    const x = tf.tensor3d(mix, [1, N, 1]);
    const out = model.predict(x);       // [1, N, 2]
    const data = out.dataSync();
    const y0 = new Float32Array(N), y1 = new Float32Array(N);
    for (let i = 0; i < N; i++) { y0[i] = data[i * 2]; y1[i] = data[i * 2 + 1]; }
    return [y0, y1];
  });
}
