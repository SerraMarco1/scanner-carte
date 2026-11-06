// Codice condiviso tra PC (costruzione catalogo) e telefono (scanner):
// stessi ritagli e stessa matematica, altrimenti le impronte non sono confrontabili.

export const CARD_W = 320, CARD_H = 446;                     // carta raddrizzata, proporzioni 63×88
export const ART = { x: 0.08, y: 0.10, w: 0.84, h: 0.44 };   // finestra dell'illustrazione, in frazioni della carta
export const SIZE = 224, DIM = 384;                          // ingresso e uscita di DINOv2-small
const MEAN = [0.485, 0.456, 0.406], STD = [0.229, 0.224, 0.225];

export const artRect = () => ({
  left: Math.round(ART.x * CARD_W), top: Math.round(ART.y * CARD_H),
  width: Math.round(ART.w * CARD_W), height: Math.round(ART.h * CARD_H),
});

// Pixel SIZE×SIZE (RGB o RGBA) → tensore CHW normalizzato come in addestramento.
export function toTensorData(px, channels, out = new Float32Array(3 * SIZE * SIZE), offset = 0) {
  const n = SIZE * SIZE;
  for (let i = 0; i < n; i++)
    for (let c = 0; c < 3; c++)
      out[offset + c * n + i] = (px[i * channels + c] / 255 - MEAN[c]) / STD[c];
  return out;
}

export function normalize(v) {
  let s = 0;
  for (const x of v) s += x * x;
  s = Math.sqrt(s) || 1;
  return Float32Array.from(v, x => x / s);
}

// shortcut: vettori salvati in int8 (4× più piccoli); se la precisione ne risente passare a float16.
export const quantize = v => Int8Array.from(v, x => Math.max(-127, Math.min(127, Math.round(x * 127))));

// Similarità coseno tra la query (float, normalizzata) e ogni carta del catalogo (int8).
export function scoreAll(q, cat, dim = DIM) {
  const n = cat.length / dim, out = new Float32Array(n);
  for (let i = 0, o = 0; i < n; i++, o += dim) {
    let s = 0;
    for (let j = 0; j < dim; j++) s += q[j] * cat[o + j];
    out[i] = s / 127;
  }
  return out;
}

// Due stampe hanno lo stesso disegno se le loro illustrazioni nel catalogo sono quasi identiche.
// Misurato: disegni diversi stanno quasi sempre sotto 0,90; gemelle inglese/giapponese tra 0,90 e 0,97.
export const SAME_ART = 0.92;
export function artSim(cat, i, j, dim = DIM) {
  let s = 0;
  for (let k = 0; k < dim; k++) s += cat[i * dim + k] * cat[j * dim + k];
  return s / 127 / 127;
}

export function topK(scores, k = 5) {
  const best = []; // [punteggio, indice], dal migliore
  for (let i = 0; i < scores.length; i++) {
    const s = scores[i];
    if (best.length === k && s <= best[k - 1][0]) continue;
    let p = best.length;
    while (p > 0 && best[p - 1][0] < s) p--;
    best.splice(p, 0, [s, i]);
    if (best.length > k) best.pop();
  }
  return best;
}
