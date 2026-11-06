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

// Ingresso fisso per controllare che il motore del telefono calcoli come quello del PC (data/probe.json).
export function probeInput() {
  const out = new Float32Array(3 * SIZE * SIZE);
  for (let i = 0; i < out.length; i++) out[i] = 1.5 * Math.sin(i * 0.0011) * Math.cos(i * 0.00037 + (i % SIZE) * 0.05);
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

// Lingue internazionali in cui una carta inglese del catalogo esiste anche (campo `langs`, a bit).
export const LANG_BITS = { it: 1, fr: 2, de: 4, es: 8, pt: 16 };

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

// Distanza tra due scritte (quante lettere cambiare): accetta un numero letto con una cifra sbagliata.
export function distance(a, b) {
  const d = Array.from({ length: a.length + 1 }, (_, i) => [i]);
  for (let j = 1; j <= b.length; j++) d[0][j] = j;
  for (let i = 1; i <= a.length; i++)
    for (let j = 1; j <= b.length; j++) d[i][j] = Math.min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (a[i - 1] !== b[j - 1]));
  return d[a.length][b.length];
}
export const normNum = s => String(s ?? '').toUpperCase().replace(/[^A-Z0-9]/g, '').replace(/^0+(?=\d)/, '');

// Stampe candidate: le carte col disegno vicino al primo posto, più tutte quelle con lo stesso disegno
// (ristampe e gemelle giapponesi). sc = punteggi combinati (illustrazione + carta intera).
export function candidatePrints(sc, artCat, k = 10, gap = 0.12, same = 0.85) {
  const top = topK(sc, k), seeds = top.filter(([s]) => top[0][0] - s <= gap).map(([, i]) => i);
  const out = new Map(seeds.map(i => [i, sc[i]]));
  const n = artCat.length / DIM;
  for (const s of seeds.slice(0, 3)) for (let j = 0; j < n; j++) if (!out.has(j) && artSim(artCat, s, j) >= same) out.set(j, sc[j]);
  return [...out].map(([i, vis]) => ({ i, vis }));
}

// Sceglie la stampa: punti per lingua, numero e totale del set letti sulla carta, meno il distacco nel disegno.
// rows = righe del catalogo [id, nome, set, numero, lingua, immagine, totale, lingue]; read = esito di readCard.
// «sure» solo se stacca bene la seconda: altrimenti l'app chiede conferma.
export function choosePrint(cands, rows, read) {
  const want = read?.lang, n = read?.number, asian = want === 'zh' || want === 'ko';
  const top = Math.max(...cands.map(c => c.vis));
  const ranked = cands.map(({ i, vis }) => {
    const [, , , num, lang, , total, mask] = rows[i], jp = lang === 'ja';
    let s = (vis - top) * 10; // −1 ogni 0,1 di distacco dal disegno più simile
    if (want === 'ja' || asian) s += jp ? 3 : -3; // cinesi e coreane: la gemella giapponese è la più vicina
    else if (want) s += jp ? -3 : want === 'en' || mask & LANG_BITS[want] ? 3 : 1.5;
    const cn = normNum(String(num).split('/')[0]);
    if (n?.num) { const d = distance(normNum(n.num), cn); s += d === 0 ? 3 : d === 1 && cn.length > 1 ? 1.5 : 0; }
    else if (cn.length > 1 && normNum(n?.bottom).includes(cn)) s += 1.5;
    if (n?.total && total) { const d = distance(normNum(n.total), normNum(total)); s += d === 0 ? 1 : d === 1 ? 0.5 : 0; }
    return { i, s, lang: asian ? want : jp ? 'ja' : want && want in LANG_BITS ? want : 'en' };
  }).sort((a, b) => b.s - a.s);
  const [first, second] = ranked;
  return { ...first, sure: !!want && !asian && first.s >= 3 && (!second || first.s - second.s >= 1.5), ranked };
}
