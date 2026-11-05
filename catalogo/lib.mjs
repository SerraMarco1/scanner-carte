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
