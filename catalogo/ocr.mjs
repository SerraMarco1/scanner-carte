// Carica i modelli OCR sul PC e prepara immagini RGBA per web/testo.js.
import fs from 'node:fs/promises';
import path from 'node:path';
import ort from 'onnxruntime-node';
import sharp from 'sharp';
import { CACHE } from './lib.mjs';

export async function loadOcr() {
  const p = f => path.join(CACHE, 'ocr', f);
  const s = f => ort.InferenceSession.create(p(f), { logSeverityLevel: 3 });
  return {
    det: await s('PP-OCRv5_mobile_det.onnx'),
    latin: await s('latin_PP-OCRv5_mobile_rec.onnx'), latinDict: JSON.parse(await fs.readFile(p('latin_PP-OCRv5_mobile_rec.dict.json'), 'utf8')),
    cjk: await s('PP-OCRv5_mobile_rec.onnx'), cjkDict: JSON.parse(await fs.readFile(p('PP-OCRv5_mobile_rec.dict.json'), 'utf8')),
  };
}

// Immagine (file o buffer) → { data RGBA, width, height } alla risoluzione usata dal telefono per leggere.
export const OCR_W = 640, OCR_H = 892;
export async function rgba(input) {
  const { data, info } = await sharp(input).resize(OCR_W, OCR_H, { fit: 'fill' }).ensureAlpha().raw().toBuffer({ resolveWithObject: true });
  return { data, width: info.width, height: info.height };
}
export { ort };
