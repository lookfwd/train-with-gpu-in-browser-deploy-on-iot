// viz.js — minimal canvas plotting for signals and spectra.

function prep(canvas) {
  const ctx = canvas.getContext("2d");
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);
  return { ctx, W, H };
}

// series: array of {data:Float32Array, color, dashed?, width?, label?}
export function plotSignal(canvas, series, { yrange = 2.0 } = {}) {
  const { ctx, W, H } = prep(canvas);
  ctx.strokeStyle = "rgba(255,255,255,0.12)";
  ctx.beginPath(); ctx.moveTo(0, H / 2); ctx.lineTo(W, H / 2); ctx.stroke();

  for (const s of series) {
    const d = s.data;
    if (!d || !d.length) continue;
    ctx.strokeStyle = s.color;
    ctx.lineWidth = s.width ?? 1.5;
    ctx.setLineDash(s.dashed ? [4, 3] : []);
    ctx.beginPath();
    for (let i = 0; i < d.length; i++) {
      const x = (i / (d.length - 1)) * W;
      const y = H / 2 - (d[i] / yrange) * (H / 2 - 3);
      i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
    }
    ctx.stroke();
  }
  ctx.setLineDash([]);

  // legend
  let lx = 8;
  ctx.font = "11px ui-monospace, monospace";
  for (const s of series) {
    if (!s.label) continue;
    ctx.fillStyle = s.color;
    ctx.fillRect(lx, 6, 10, 3);
    ctx.fillStyle = "rgba(255,255,255,0.75)";
    ctx.fillText(s.label, lx + 14, 10);
    lx += 14 + ctx.measureText(s.label).width + 14;
  }
}

export function plotSpectrum(canvas, mag, fs, peaks = []) {
  const { ctx, W, H } = prep(canvas);
  const n = mag.length;
  let max = 1e-9;
  for (let i = 0; i < n; i++) max = Math.max(max, mag[i]);

  // identified-peak markers behind the curve
  const binHz = fs / (n * 2);
  ctx.fillStyle = "rgba(255,112,67,0.85)";
  for (const f of peaks) {
    const x = (f / binHz / (n - 1)) * W;
    ctx.fillRect(x - 1, 0, 2, H);
  }

  ctx.strokeStyle = "#4fc3f7";
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  for (let k = 0; k < n; k++) {
    const x = (k / (n - 1)) * W;
    const y = H - (mag[k] / max) * (H - 4);
    k ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
  }
  ctx.stroke();

  ctx.fillStyle = "rgba(255,255,255,0.6)";
  ctx.font = "11px ui-monospace, monospace";
  ctx.fillText(`0 Hz`, 4, H - 4);
  ctx.fillText(`${Math.round(fs / 2)} Hz`, W - 52, H - 4);
}
