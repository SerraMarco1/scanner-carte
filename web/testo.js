// Legge il testo della carta raddrizzata (PaddleOCR v5 mobile) e ne ricava lingua e numero.
// Gira uguale sul telefono (onnxruntime-web) e sul PC (onnxruntime-node): riceve pixel RGBA e sessioni.


const MEAN = [0.485, 0.456, 0.406], STD = [0.229, 0.224, 0.225];

// Ritaglio (x0, y0, w, h) dell'immagine RGBA ridimensionato a dw×dh, bilineare.
export function resizeRGBA(img, x0, y0, w, h, dw, dh) {
  const { data, width, height } = img, out = new Uint8ClampedArray(dw * dh * 4);
  const sx = w / dw, sy = h / dh;
  for (let y = 0; y < dh; y++) {
    const fy = Math.min(height - 1, Math.max(0, y0 + (y + 0.5) * sy - 0.5)), ya = Math.floor(fy), yb = Math.min(height - 1, ya + 1), ty = fy - ya;
    for (let x = 0; x < dw; x++) {
      const fx = Math.min(width - 1, Math.max(0, x0 + (x + 0.5) * sx - 0.5)), xa = Math.floor(fx), xb = Math.min(width - 1, xa + 1), tx = fx - xa;
      const o = (y * dw + x) * 4, a = (ya * width + xa) * 4, b = (ya * width + xb) * 4, c = (yb * width + xa) * 4, d = (yb * width + xb) * 4;
      for (let k = 0; k < 3; k++) out[o + k] = (data[a + k] * (1 - tx) + data[b + k] * tx) * (1 - ty) + (data[c + k] * (1 - tx) + data[d + k] * tx) * ty;
      out[o + 3] = 255;
    }
  }
  return out;
}

// Pixel RGBA → tensore BGR (PaddleOCR è addestrato su immagini lette in BGR).
function tensorData(px, n, mean, std) {
  const t = new Float32Array(3 * n);
  for (let i = 0; i < n; i++) for (let c = 0; c < 3; c++) t[c * n + i] = (px[i * 4 + 2 - c] / 255 - mean[c]) / std[c];
  return t;
}

// Righe di testo: zone collegate della mappa di probabilità, allargate come fa PaddleOCR ("unclip").
function boxes(prob, W, H, thr = 0.3, minScore = 0.6) {
  const seen = new Uint8Array(W * H), out = [], stack = [];
  for (let i = 0; i < W * H; i++) {
    if (prob[i] <= thr || seen[i]) continue;
    let x0 = W, y0 = H, x1 = 0, y1 = 0, sum = 0, cnt = 0;
    seen[i] = 1; stack.push(i);
    while (stack.length) {
      const p = stack.pop(), x = p % W, y = (p - x) / W;
      sum += prob[p]; cnt++;
      if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y;
      const near = [x > 0 ? p - 1 : -1, x < W - 1 ? p + 1 : -1, y > 0 ? p - W : -1, y < H - 1 ? p + W : -1];
      for (const q of near) if (q >= 0 && !seen[q] && prob[q] > thr) { seen[q] = 1; stack.push(q); }
    }
    const w = x1 - x0 + 1, h = y1 - y0 + 1;
    if (w < 4 || h < 3 || sum / cnt < minScore) continue;
    const d = (w * h * 1.5) / (2 * (w + h));
    const bx = Math.max(0, x0 - d), by = Math.max(0, y0 - d);
    out.push({ x: bx, y: by, w: Math.min(W, x1 + 1 + d) - bx, h: Math.min(H, y1 + 1 + d) - by });
  }
  return out;
}

export async function detect(ort, det, img) {
  const W = Math.max(32, Math.round(img.width / 32) * 32), H = Math.max(32, Math.round(img.height / 32) * 32);
  const px = resizeRGBA(img, 0, 0, img.width, img.height, W, H);
  const out = await det.run({ x: new ort.Tensor('float32', tensorData(px, W * H, MEAN, STD), [1, 3, H, W]) });
  const kx = img.width / W, ky = img.height / H;
  return boxes(Object.values(out)[0].data, W, H).map(b => ({ x: b.x * kx, y: b.y * ky, w: b.w * kx, h: b.h * ky }));
}

// Legge una riga: decodifica CTC (classe 0 = vuoto, k = dict[k-1], ultima = spazio).
export async function read(ort, rec, dict, img, b) {
  const h = 48, w = Math.max(16, Math.min(960, Math.round((b.w * h) / b.h)));
  const px = resizeRGBA(img, b.x, b.y, b.w, b.h, w, h);
  const o = Object.values(await rec.run({ x: new ort.Tensor('float32', tensorData(px, w * h, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]), [1, 3, h, w]) }))[0];
  const [, T, C] = o.dims, p = o.data;
  let text = '', last = 0, conf = 0, k = 0;
  for (let t = 0; t < T; t++) {
    let best = 0;
    for (let c = 1; c < C; c++) if (p[t * C + c] > p[t * C + best]) best = c;
    if (best && best !== last) { text += best === C - 1 ? ' ' : dict[best - 1]; conf += p[t * C + best]; k++; }
    last = best;
  }
  return { text: text.trim(), conf: k ? conf / k : 0 };
}

// Parole tipiche di ogni lingua, senza accenti e confrontate come parole intere
// (così «résistance» francese non conta come «resistance» inglese, né «attachez» come «attach»).
const WORDS = {
  en: ['weakness', 'retreat', 'evolves from', 'ability', 'your turn', 'opponent', 'damage', 'attach', 'discard', 'this pokemon',
    'basic', 'stage', 'supporter', 'stadium', 'lbs', 'ht', 'wt', 'your', 'card', 'cards'], // «trainer» e «item» anche in tedesco
  it: ['debolezza', 'ritirata', 'si evolve da', 'abilita', 'tuo turno', 'avversario', 'danni', 'assegna', 'scarta', 'questo pokemon',
    'resistenza', 'fase', 'allenatore', 'aiuto', 'strumento', 'stadio', 'altezza', 'tuo', 'carta', 'carte', 'energia'],
  fr: ['faiblesse', 'retraite', 'evolue de', 'talent', 'votre tour', 'adversaire', 'degats', 'attachez', 'defaussez', 'ce pokemon',
    'de base', 'niveau', 'dresseur', 'objet', 'stade', 'taille', 'poids', 'votre', 'carte'],
  de: ['schwache', 'ruckzug', 'entwickelt sich', 'fahigkeit', 'deines zuges', 'gegner', 'schadenspunkte', 'dieses pokemon',
    'resistenz', 'basis', 'phase', 'unterstutzer', 'stadion', 'grosse', 'gewicht', 'dein', 'deine', 'karte', 'karten'],
  es: ['debilidad', 'retirada', 'evoluciona de', 'habilidad', 'tu turno', 'rival', 'este pokemon', 'resistencia', 'basico',
    'entrenador', 'partidario', 'objeto', 'estadio', 'altura', 'carta', 'energia'],
  pt: ['fraqueza', 'recuo', 'evolui de', 'habilidade', 'seu turno', 'oponente', 'este pokemon', 'resistencia', 'basico',
    'estagio', 'treinador', 'apoiador', 'estadio', 'altura', 'seu', 'carta', 'energia'],
};
// Sigla dei punti salute accanto al numero: HP inglese, PS italiano/spagnolo/portoghese, PV francese, KP tedesco.
const HP = { HP: ['en'], PS: ['it', 'es', 'pt'], PV: ['fr'], KP: ['de'] };
const plain = s => s.normalize('NFD').replace(/[̀-ͯ]/g, '').toLowerCase();
const hasWord = (text, w) => new RegExp(`(^|[^a-z])${w}($|[^a-z])`).test(text);

// Lingua dalle righe lette con il lettore latino: codice lingua stampato (dalle Scarlatto e Violetto),
// sigla dei punti salute e parole tipiche.
export function latinLanguage(lines) {
  const code = lines.map(l => l.text.match(/\b[A-Z0-9]{2,4}\s?(EN|IT|FR|DE|ES|PT)\s?[A-Z]{0,3}\d{1,3}\//)).find(Boolean);
  if (code) return { lang: code[1].toLowerCase(), why: `codice ${code[0]}` };
  const raw = lines.map(l => l.text).join(' '), text = plain(raw), score = {}, why = {}, words = {};
  const add = (lang, n, w) => { score[lang] = (score[lang] || 0) + n; (why[lang] ||= []).push(w); };
  for (const [lang, ws] of Object.entries(WORDS)) for (const w of ws) if (hasWord(text, w)) { add(lang, 1, w); words[lang] = true; }
  for (const m of raw.matchAll(/(?:^|[^A-Z])(HP|PS|PV|KP)\s?\d{2,3}\b|\b\d{2,3}\s?(HP|PS|PV|KP)(?![A-Z])/g))
    for (const lang of HP[m[1] || m[2]]) add(lang, 2, m[1] || m[2]);
  // Serve almeno una parola: anche le carte giapponesi scrivono «HP».
  const [best, next] = Object.entries(score).filter(([lang]) => words[lang]).sort((a, b) => b[1] - a[1]);
  return best && (!next || best[1] > next[1]) ? { lang: best[0], why: why[best[0]].join(', ') } : null;
}

export const asianLanguage = text => /[぀-ヿ]/.test(text) ? 'ja' : /[一-鿿]/.test(text) ? 'zh' : /[가-힯]/.test(text) ? 'ko' : null;

// Numero di collezione dalle righe in fondo alla carta: «025/198», «TG05/TG30», o promo come «SWSH145».
// Insieme al numero tiene `bottom`, il testo grezzo in fondo: serve a confrontare con le stampe candidate
// anche quando una cifra è letta male.
export function cardNumber(lines, height) {
  const fix = s => s.replace(/[Oo]/g, '0').replace(/[Il|]/g, '1');
  const low = lines.filter(l => l.y + l.h / 2 > height * 0.8);
  const bottom = low.map(l => l.text).join(' ');
  for (const l of [...low].reverse()) {
    const m = l.text.replace(/\s+/g, '').match(/([A-Z]{0,3}[0-9OIl|]{1,3})\/([A-Z]{0,3}[0-9OIl|]{1,3})/);
    if (m) {
      const total = fix(m[2]), lettered = /[A-Z]/.test(total); // «TG05/TG30»: le lettere fanno parte del numero
      return { num: lettered ? fix(m[1]) : fix(m[1]).replace(/^[A-Z]+/, ''), total, bottom };
    }
    const promo = l.text.match(/\b(SWSH|SM|XY|BW|HGSS|DP)\s?([0-9OIl|]{2,3})\b/);
    if (promo) return { num: promo[1] + fix(promo[2]), total: null, bottom };
  }
  return bottom ? { num: null, total: null, bottom } : null;
}

// Tutta la lettura: ogni riga con il lettore latino; se la lingua non esce, il lettore asiatico
// sulle righe in alto (il nome) e su quelle più grandi.
export async function readCard(ort, m, img) {
  const t0 = Date.now();
  const found = (await detect(ort, m.det, img)).sort((a, b) => a.y - b.y).slice(0, 40);
  const lines = [];
  for (const b of found) lines.push({ ...b, ...(await read(ort, m.latin, m.latinDict, img, b)) });
  let lang = latinLanguage(lines), asian = '';
  if (!lang) {
    const pickB = [...found.slice(0, 3), ...[...found].sort((a, b) => b.w * b.h - a.w * a.h).slice(0, 5)];
    for (const b of new Set(pickB)) asian += (await read(ort, m.cjk, m.cjkDict, img, b)).text + ' ';
    const a = asianLanguage(asian);
    if (a) lang = { lang: a, why: 'caratteri ' + a };
  }
  return { lang: lang?.lang || null, why: lang?.why || '', number: cardNumber(lines, img.height), lines: lines.map(l => l.text), asian: asian.trim(), ms: Date.now() - t0 };
}
