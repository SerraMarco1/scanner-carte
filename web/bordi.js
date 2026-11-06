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

// Restituisce i 4 angoli in pixel della sorgente, oppure null se non c'è una carta.
export function findCard(src, w, h) {
  const k = DET / Math.max(w, h);
  det.width = Math.round(w * k);
  det.height = Math.round(h * k);
  detCtx.drawImage(src, 0, 0, det.width, det.height);
  const img = cv.imread(det), gray = new cv.Mat(), edges = new cv.Mat(), kernel = cv.Mat.ones(3, 3, cv.CV_8U);
  const contours = new cv.MatVector(), hier = new cv.Mat();
  try {
    cv.cvtColor(img, gray, cv.COLOR_RGBA2GRAY);
    cv.GaussianBlur(gray, gray, new cv.Size(5, 5), 0);
    cv.Canny(gray, edges, 40, 120);
    cv.dilate(edges, edges, kernel); // chiude i piccoli buchi nel bordo
    cv.findContours(edges, contours, hier, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);
    let best = null, bestArea = det.width * det.height * 0.06;
    for (let i = 0; i < contours.size(); i++) {
      const c = contours.get(i), area = cv.contourArea(c);
      if (area > bestArea) {
        const poly = new cv.Mat();
        cv.approxPolyDP(c, poly, 0.03 * cv.arcLength(c, true), true);
        if (poly.rows === 4 && cv.isContourConvex(poly)) {
          const q = order([0, 1, 2, 3].map(j => [poly.data32S[2 * j], poly.data32S[2 * j + 1]]));
          if (cardShaped(q)) { best = q; bestArea = area; }
        }
        poly.delete();
      }
      c.delete();
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
