// Strumenti condivisi dagli script del catalogo (solo PC).
import path from 'node:path';
import ort from 'onnxruntime-node';
import sharp from 'sharp';
import { CARD_W, CARD_H, SIZE, DIM, artRect, toTensorData, normalize } from '../web/core.js';

export const ROOT = path.resolve(import.meta.dirname, '..');
export const CACHE = path.join(ROOT, 'catalogo/cache');
export const sleep = ms => new Promise(r => setTimeout(r, ms));

export async function fetchRetry(url, as = 'json') {
  for (let i = 0; ; i++) {
    try {
      const r = await fetch(url, { headers: { 'User-Agent': 'scanner-carte-catalogo/0.1' } });
      if (r.status === 404) return null;
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      return as === 'json' ? await r.json() : Buffer.from(await r.arrayBuffer());
    } catch (e) {
      if (i >= 3) throw new Error(`${url}: ${e.message}`);
      await sleep(1500 * (i + 1));
    }
  }
}

// Esegue fn su ogni elemento con al massimo n operazioni in parallelo.
export async function pool(items, n, fn) {
  let next = 0;
  await Promise.all(Array.from({ length: n }, async () => {
    while (next < items.length) { const i = next++; await fn(items[i], i); }
  }));
}

export async function loadSession(file = 'model.onnx', gpu = true) {
  const p = path.join(CACHE, 'model', file);
  if (gpu) {
    try { return await ort.InferenceSession.create(p, { executionProviders: ['dml'], logSeverityLevel: 3 }); }
    catch { /* niente GPU: si va di CPU */ }
  }
  return ort.InferenceSession.create(p, { logSeverityLevel: 3 });
}

// Immagine di una carta → i due ritagli 224×224 RGB: carta intera e finestra dell'illustrazione.
export async function crops(img) {
  const card = await sharp(img).resize(CARD_W, CARD_H, { fit: 'fill' }).removeAlpha().raw().toBuffer();
  const raw = { raw: { width: CARD_W, height: CARD_H, channels: 3 } };
  const fit = { fit: 'fill' };
  return {
    full: await sharp(card, raw).resize(SIZE, SIZE, fit).raw().toBuffer(),
    art: await sharp(card, raw).extract(artRect()).resize(SIZE, SIZE, fit).raw().toBuffer(),
  };
}

// Ritagli RGB → impronte normalizzate (token CLS di DINOv2).
export async function embed(session, raws) {
  const b = raws.length, per = 3 * SIZE * SIZE, data = new Float32Array(b * per);
  raws.forEach((px, i) => toTensorData(px, 3, data, i * per));
  const out = await session.run({ pixel_values: new ort.Tensor('float32', data, [b, 3, SIZE, SIZE]) });
  const h = out.last_hidden_state, tokens = h.dims[1];
  return raws.map((_, i) => normalize(h.data.subarray(i * tokens * DIM, i * tokens * DIM + DIM)));
}

// Una foto finta: carta ruotata e fuori centro, luce e colori diversi, sfocata, con riflesso, compressa.
// rnd = generatore casuale del test (con seme fisso, così il test si ripete uguale).
export async function photoLike(input, rnd, W = CARD_W, H = CARD_H) {
  const card = await sharp(input).resize(W, H, { fit: 'fill' }).toBuffer();
  const rot = await sharp(card).rotate((rnd() * 2 - 1) * 3, { background: '#1e1e1e' }).toBuffer();
  const m = await sharp(rot).metadata();
  const clamp = (v, max) => Math.max(0, Math.min(max, v));
  const left = clamp(Math.round((m.width - W) / 2 + (rnd() * 2 - 1) * 0.04 * W), m.width - W);
  const top = clamp(Math.round((m.height - H) / 2 + (rnd() * 2 - 1) * 0.04 * H), m.height - H);
  const shot = await sharp(rot).extract({ left, top, width: W, height: H })
    .modulate({ brightness: 0.75 + rnd() * 0.5, saturation: 0.8 + rnd() * 0.4 })
    .blur(0.3 + rnd()).toBuffer();
  const [gx, gy, r, op] = [rnd() * W, rnd() * H, (40 + rnd() * 80) * W / CARD_W, 0.25 + rnd() * 0.35];
  const glare = `<svg width="${W}" height="${H}"><defs><radialGradient id="g"><stop offset="0" stop-color="#fff" stop-opacity="${op}"/><stop offset="1" stop-color="#fff" stop-opacity="0"/></radialGradient></defs><ellipse cx="${gx}" cy="${gy}" rx="${r}" ry="${r * 0.7}" fill="url(#g)"/></svg>`;
  return sharp(shot).composite([{ input: Buffer.from(glare) }]).jpeg({ quality: 55 + Math.floor(rnd() * 30) }).toBuffer();
}
