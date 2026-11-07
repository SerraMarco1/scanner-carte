// Trova la carta nell'immagine (un contorno a 4 lati con le proporzioni di una carta) e la raddrizza.
const CV_URL = 'https://cdn.jsdelivr.net/npm/@techstark/opencv-js@4.12.0-release.1/dist/opencv.js';
let cv = null;

export async function loadCV() {
  await new Promise((ok, ko) => {
    const s = Object.assign(document.createElement('script'), { src: CV_URL, async: true });
    s.onload = ok;
    s.onerror = () => ko(new Error('rilevatore dei bordi non scaricato'));
    document.head.append(s);
  });
  // Il modulo ha un suo .then che passa sé stesso: con `await` si entra in un ciclo infinito.
  await new Promise(done => window.cv.then(m => { cv = m; done(); }));
}

const DET = 480; // lato lungo dell'immagine in cui si cerca la carta: basta e resta veloce
const det = document.createElement('canvas');
const detCtx = det.getContext('2d', { willReadFrequently: true });
const dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);

// Angoli in ordine: alto-sinistra, alto-destra, basso-destra, basso-sinistra.
function order(p) {
  const sum = p.map(([x, y]) => x + y), diff = p.map(([x, y]) => y - x);
  const pick = (v, f) => p[v.indexOf(f(...v))];
  return [pick(sum, Math.min), pick(diff, Math.min), pick(sum, Math.max), pick(diff, Math.max)];
}

// Carta in verticale: larghezza/altezza ≈ 63/88 = 0,72, con margine per la prospettiva.
function cardShaped([tl, tr, br, bl]) {
  const r = (dist(tl, tr) + dist(bl, br)) / (dist(tl, bl) + dist(tr, br));
  return r > 0.55 && r < 0.9;
}

// ---------- modello dei bordi (catalogo/bordi) ----------
// Dice per ogni punto se è carta; dal contorno della maschera i 4 angoli. Regge fondi chiari, venati, pieni di
// oggetti, pile e dita, dove le soglie di Canny sbagliano. Stessi passi in ScannerPlugin.java e catalogo/bordi/modello.py.
const IN_W = 192, IN_H = 256;
const inp = document.createElement('canvas');
inp.width = IN_W; inp.height = IN_H;
const inpCtx = inp.getContext('2d', { willReadFrequently: true });
let edges = null, ortLib = null;

export async function loadEdges(ort, bytes) {
  edges = await ort.InferenceSession.create(bytes, { executionProviders: ['wasm'] });
  ortLib = ort;
}

async function findByModel(src, w, h) {
  // Il modello vede il centro 3:4 verticale, come i fotogrammi della fotocamera nativa.
  const cw = Math.min(w, h * 0.75), ch = cw / 0.75, cx = (w - cw) / 2, cy = (h - ch) / 2;
  inpCtx.drawImage(src, cx, cy, cw, ch, 0, 0, IN_W, IN_H);
  const px = inpCtx.getImageData(0, 0, IN_W, IN_H).data, n = IN_W * IN_H, t = new Float32Array(3 * n);
  for (let i = 0; i < n; i++) for (let c = 0; c < 3; c++) t[c * n + i] = px[i * 4 + c];
  const out = await edges.run({ image: new ortLib.Tensor('float32', t, [1, 3, IN_H, IN_W]) });
  const q = quadFromMask(out.card.data);
  return q && q.map(([x, y]) => [cx + x * cw / IN_W, cy + y * ch / IN_H]);
}

// Maschera 96×128 → angoli in pixel dell'ingresso 192×256: il contorno più grande, il suo involucro convesso
// e il poligono a 4 lati che lo segue meglio (o il rettangolo ruotato che lo contiene).
function quadFromMask(prob) {
  const p = cv.matFromArray(IN_H / 2, IN_W / 2, cv.CV_32F, prob), big = new cv.Mat(), m = new cv.Mat();
  const cs = new cv.MatVector(), hier = new cv.Mat(), hull = new cv.Mat(), poly = new cv.Mat();
  try {
    cv.resize(p, big, new cv.Size(IN_W, IN_H), 0, 0, cv.INTER_LINEAR);
    cv.threshold(big, big, 0.5, 255, cv.THRESH_BINARY);
    big.convertTo(m, cv.CV_8U);
    cv.findContours(m, cs, hier, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE);
    let best = -1, bestArea = 0.06 * IN_W * IN_H;
    for (let i = 0; i < cs.size(); i++) { const c = cs.get(i), a = cv.contourArea(c); c.delete(); if (a > bestArea) { best = i; bestArea = a; } }
    if (best < 0) return null;
    const c = cs.get(best), pts = [];
    for (let i = 0; i < c.rows; i++) pts.push([c.data32S[2 * i], c.data32S[2 * i + 1]]);
    cv.convexHull(c, hull);
    c.delete();
    const peri = cv.arcLength(hull, true);
    let q = null;
    for (const eps of [0.02, 0.03, 0.04, 0.05, 0.06, 0.08]) {
      cv.approxPolyDP(hull, poly, eps * peri, true);
      if (poly.rows === 4) { q = order([0, 1, 2, 3].map(j => [poly.data32S[2 * j], poly.data32S[2 * j + 1]])); break; }
      if (poly.rows < 4) break;
    }
    q ||= order(cv.RotatedRect.points(cv.minAreaRect(hull)).map(pt => [pt.x, pt.y]));
    q = straighten(q, pts);
    return cardShaped(q) ? q : null;
  } finally {
    [p, big, m, cs, hier, hull, poly].forEach(x => x.delete());
  }
}

// Angoli più precisi: per ogni lato una retta sui punti del contorno lontani dagli angoli (arrotondati) e dalle dita,
// poi gli incroci delle rette. Come straighten in catalogo/bordi/modello.py.
function straighten(q, pts) {
  const lines = [];
  for (let k = 0; k < 4; k++) {
    const [ax, ay] = q[k], dx = q[(k + 1) % 4][0] - ax, dy = q[(k + 1) % 4][1] - ay, n = Math.hypot(dx, dy), ux = dx / n, uy = dy / n;
    const sel = pts.filter(([x, y]) => {
      const t = (x - ax) * ux + (y - ay) * uy, off = Math.abs((y - ay) * ux - (x - ax) * uy);
      return t > 0.15 * n && t < 0.85 * n && off < 0.04 * n + 2;
    });
    if (sel.length < 5) return q;
    const m = cv.matFromArray(sel.length, 1, cv.CV_32FC2, sel.flat()), l = new cv.Mat();
    cv.fitLine(m, l, cv.DIST_HUBER, 0, 0.01, 0.01);
    lines.push([...l.data32F]); // vx, vy, x0, y0
    m.delete(); l.delete();
  }
  const out = [];
  for (let k = 0; k < 4; k++) {
    const [ax, ay, px, py] = lines[(k + 3) % 4], [bx, by, rx, ry] = lines[k], det = bx * ay - ax * by;
    if (Math.abs(det) < 1e-6) return q;
    const s = (bx * (ry - py) - by * (rx - px)) / det; // incrocio: p + s·a = r + t·b
    out.push([px + s * ax, py + s * ay]);
  }
  return out.every(([x, y], k) => Math.max(Math.abs(x - q[k][0]), Math.abs(y - q[k][1])) < 0.1 * IN_W) ? out : q;
}

// Restituisce i 4 angoli in pixel della sorgente, oppure null se non c'è una carta.
export async function findCard(src, w, h) {
  return edges ? findByModel(src, w, h) : findByEdges(src, w, h);
}

// Metodo senza modello (soglie di Canny): resta per quando il modello non è disponibile.
function findByEdges(src, w, h) {
  const k = DET / Math.max(w, h);
  det.width = Math.round(w * k);
  det.height = Math.round(h * k);
  detCtx.drawImage(src, 0, 0, det.width, det.height);
  const img = cv.imread(det), gray = new cv.Mat(), edges = new cv.Mat(), kernel = cv.Mat.ones(3, 3, cv.CV_8U);
  const contours = new cv.MatVector(), hier = new cv.Mat();
  try {
    cv.cvtColor(img, gray, cv.COLOR_RGBA2GRAY);
    cv.GaussianBlur(gray, gray, new cv.Size(5, 5), 0);
    let best = null, bestArea = det.width * det.height * 0.06;
    // Tre sensibilità dei bordi e tutti i contorni, non solo quelli esterni: su fondi chiari, venati
    // o pieni di oggetti il bordo della carta si salda con quello che ha intorno (prova su scene
    // simulate: fondo bianco da 4 a 10 su 10, legno da 1 a 5; fondo scuro sempre 10 su 10).
    for (const [lo, hi] of [[20, 60], [40, 120], [80, 200]]) {
      cv.Canny(gray, edges, lo, hi);
      cv.dilate(edges, edges, kernel); // chiude i piccoli buchi nel bordo
      cv.findContours(edges, contours, hier, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE);
      for (let i = 0; i < contours.size(); i++) {
        const c = contours.get(i), area = cv.contourArea(c);
        if (area > bestArea) {
          const poly = new cv.Mat();
          cv.approxPolyDP(c, poly, 0.03 * cv.arcLength(c, true), true);
          let q = poly.rows === 4 && cv.isContourConvex(poly) ? order([0, 1, 2, 3].map(j => [poly.data32S[2 * j], poly.data32S[2 * j + 1]])) : null;
          if (!q) {
            // Contorno non a 4 lati ma che riempie bene un rettangolo ruotato: si prende quel rettangolo.
            const r = cv.minAreaRect(c);
            if (area > 0.85 * r.size.width * r.size.height) q = order(cv.RotatedRect.points(r).map(p => [p.x, p.y]));
          }
          if (q && cardShaped(q)) { best = q; bestArea = area; }
          poly.delete();
        }
        c.delete();
      }
    }
    return best && best.map(([x, y]) => [x / k, y / k]);
  } finally {
    [img, gray, edges, kernel, contours, hier].forEach(m => m.delete());
  }
}

// Raddrizza la carta indicata dagli angoli e la disegna in `out`, a piena risoluzione e poi ridotta.
const full = document.createElement('canvas'), flat = document.createElement('canvas');
const OUT_W = 640, OUT_H = 892;
export function warpCard(src, w, h, quad, out) {
  full.width = w;
  full.height = h;
  full.getContext('2d').drawImage(src, 0, 0, w, h);
  const m = cv.imread(full), dst = new cv.Mat();
  const from = cv.matFromArray(4, 1, cv.CV_32FC2, quad.flat());
  const to = cv.matFromArray(4, 1, cv.CV_32FC2, [0, 0, OUT_W, 0, OUT_W, OUT_H, 0, OUT_H]);
  const M = cv.getPerspectiveTransform(from, to);
  try {
    cv.warpPerspective(m, dst, M, new cv.Size(OUT_W, OUT_H), cv.INTER_LINEAR);
    cv.imshow(flat, dst);
    out.getContext('2d').drawImage(flat, 0, 0, out.width, out.height);
  } finally {
    [m, dst, from, to, M].forEach(x => x.delete());
  }
}
