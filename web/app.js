// Scanner Carte: inquadri una carta, l'app la riconosce e ti dice se ce l'hai (semaforo). Quando è sicura della
// stampa fa da sola (registra, con «Annulla»); nei dubbi ti chiede quale stampa è. Modalità: SPECIFICA.md §5-§9.
// La carta viene trovata e raddrizzata da sola; la scansione parte quando resta ferma.
// Disegno (impronte) + testo letto (lingua e numero) → stampa scelta.
// Nell'APK Android fotocamera, bordi, impronte e lettura sono nativi (ScannerPlugin.java); qui resta il resto.
import { CARD_W, CARD_H, SIZE, DIM, artRect, toTensorData, normalize, scoreAll, topK, artSim, probeInput, candidatePrints, choosePrint, artGap, CLEAR } from './core.js';
import { loadCV, loadEdges, findCard, warpCard } from './bordi.js';
import { readCard, latinLanguage, asianLanguage, cardNumber } from './testo.js';
import { OWNED_ART, isBasicEnergy, semaphore, pickVariant, variantLabel, load as loadCollection, save as saveCollection, newKey, toCSV } from './collezione.js';
// registerPlugin non è nel ponte nativo di Capacitor: sta nella sua libreria (copia di @capacitor/core 8.5.3).
import { registerPlugin } from './capacitor-core.js';

const NATIVE = !!window.Capacitor?.isNativePlatform?.();
const Scanner = NATIVE ? registerPlugin('Scanner') : null;
const ORT_DIST = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/';
let ort; // solo nella versione web: nell'APK il modello gira nativo
const MODEL_URL = 'https://huggingface.co/onnx-community/dinov2-small/resolve/8b1f705a3a7f6f062f6bdd21986c1583d3ef105d/onnx/';
const OCR_URL = 'https://huggingface.co/nakamura196/ppocrv5-mobile-web-onnx/resolve/dc7e2271c00d961a4c0a38dcebe23beb735ba33e/';
// fp16: impronte uguali a quelle del catalogo (somiglianza 0,9999); la versione int8 se ne allontana (0,985).
const MODEL_FILE = 'model_fp16.onnx';
const EDGES_V = 1; // versione di data/bordi.onnx: cambiarla quando si riaddestra, così il telefono lo riscarica
const OCR_W = 640, OCR_H = 892; // carta raddrizzata per leggere le scritte piccole
const STORE = 'scanner-proto-risultati', CAM = 'scanner-proto-obiettivo', LAST_LANG = 'scanner-proto-lingua';
const MODE = 'scanner-modo', COL = 'scanner-collezione-attiva', STICKY = 'scanner-varianti';
const LANGS = { it: 'italiano', en: 'inglese', fr: 'francese', de: 'tedesco', es: 'spagnolo', pt: 'portoghese', ja: 'giapponese', zh: 'cinese', ko: 'coreano' };
const $ = id => document.getElementById(id);
// Il catalogo non sta nell'APK: si prende dal sito e si tiene sul telefono, così si aggiorna senza reinstallare.
const DATA = window.Capacitor?.isNativePlatform?.() ? 'https://serramarco1.github.io/scanner-carte/data/' : 'data/';

let session, backend, ocr, cards, byId, idx, artCat, fullCat, track, shotInfo = '', nativeInfo = null, crash = '';
let pending = null, torchOn = false, cams = [], checks = [], pickLang = 'it', lastLang = 'it';
let data = loadCollection(), mode = 'inventario', col = data.cols[0], sticky = {}, retries = 0;
try {
  lastLang = localStorage.getItem(LAST_LANG) || 'it';
  mode = localStorage.getItem(MODE) || mode;
  col = localStorage.getItem(COL) || col;
  sticky = JSON.parse(localStorage.getItem(STICKY)) || {};
} catch { /* si parte dalle scelte predefinite */ }
if (!data.cols.includes(col)) col = data.cols[0];
const remember = (k, v) => { try { localStorage.setItem(k, typeof v === 'string' ? v : JSON.stringify(v)); } catch { /* resta in memoria */ } };
let results = [];
try { results = JSON.parse(localStorage.getItem(STORE)) || []; } catch { /* memoria non disponibile */ }
const save = () => remember(STORE, JSON.stringify(results));
function store() {
  try { saveCollection(data); renderStats(); } catch { alertLine('ATTENZIONE: memoria piena, la collezione non è stata salvata. Copia un backup dal Menu.'); }
}

// ---------- avvio ----------
function loading(msg, frac) {
  $('loadMsg').textContent = msg;
  if (frac !== undefined) $('loadBar').style.width = `${Math.round(frac * 100)}%`;
}

// Scarica un file mostrando l'avanzamento; keep = salvalo sul telefono (solo i modelli, che non cambiano mai).
async function getBytes(url, label, keep) {
  let cache = null;
  if (keep) {
    try { cache = await caches.open('scanner-proto-v1'); const hit = await cache.match(url); if (hit) return new Uint8Array(await hit.arrayBuffer()); }
    catch { cache = null; }
  }
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${label}: errore ${r.status}`);
  // Con un file compresso la dimensione dichiarata non corrisponde ai byte letti: niente totale.
  const total = r.headers.get('content-encoding') ? 0 : Number(r.headers.get('content-length')) || 0;
  const reader = r.body.getReader(), parts = [];
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    parts.push(value); got += value.length;
    loading(`Scarico ${label}: ${(got / 1e6).toFixed(1)}${total ? ` di ${(total / 1e6).toFixed(1)}` : ''} MB`, total ? got / total : undefined);
  }
  const bytes = new Uint8Array(got);
  let o = 0;
  for (const p of parts) { bytes.set(p, o); o += p.length; }
  try { await cache?.put(url, new Response(bytes)); } catch { /* spazio pieno: si riscarica la prossima volta */ }
  return bytes;
}

// Elenco carte: aggiornato se c'è rete, altrimenti l'ultimo salvato.
async function freshBytes(url, label) {
  try {
    const bytes = await getBytes(url, label);
    try { await (await caches.open('scanner-proto-v1')).put(url, new Response(bytes)); } catch { /* resta solo in memoria */ }
    return bytes;
  } catch (e) {
    const hit = await caches.open('scanner-proto-v1').then(c => c.match(url)).catch(() => null);
    if (hit) return new Uint8Array(await hit.arrayBuffer());
    throw new Error(`${e.message}. Serve internet al primo avvio`);
  }
}

// Impronte: salvate per versione del catalogo; le versioni vecchie si cancellano.
async function catalogBin(name, label, version) {
  const v = encodeURIComponent(version), bytes = await getBytes(`${DATA}${name}?v=${v}`, label, true);
  try {
    const c = await caches.open('scanner-proto-v1');
    for (const r of await c.keys()) if (r.url.includes(`/${name}?v=`) && !r.url.endsWith(v)) await c.delete(r);
  } catch { /* pulizia rimandata */ }
  return bytes;
}

async function startModel() {
  const gpu = navigator.gpu ? await navigator.gpu.requestAdapter().catch(() => null) : null;
  const probe = (await (await fetch('data/probe.json')).json()).v;
  // Ogni motore deve ridare l'impronta calcolata sul PC per l'ingresso fisso: alcune schede grafiche
  // di telefono sbagliano i calcoli a precisione ridotta. Allora si prova il modello completo,
  // e per ultimo il calcolo senza scheda grafica (più lento ma corretto).
  const tries = [['webgpu', 'model_fp16.onnx'], ['webgpu', 'model.onnx'], ['wasm', MODEL_FILE]].filter(([ep]) => gpu || ep === 'wasm');
  for (const [ep, file] of tries) {
    try {
      const bytes = await getBytes(MODEL_URL + file, file === 'model.onnx' ? 'modello completo' : 'modello', true);
      loading('Avvio il riconoscimento (la prima volta circa un minuto)…');
      const s = await ort.InferenceSession.create(bytes, { executionProviders: [ep] });
      const out = await s.run({ pixel_values: new ort.Tensor('float32', probeInput(), [1, 3, SIZE, SIZE]) });
      const v = normalize(out.last_hidden_state.data.subarray(0, DIM)), sim = v.reduce((a, x, i) => a + x * probe[i], 0);
      checks.push(`${ep}/${file.replace('.onnx', '')} ${sim.toFixed(3)}`);
      if (sim >= 0.98) { session = s; backend = `${ep} ${file}`; return; }
      s.release?.();
    } catch (e) { checks.push(`${ep}/${file} errore`); console.warn(ep, file, e); }
  }
  throw new Error(`il riconoscimento dà risultati sbagliati su questo telefono (${checks.join(', ')})`);
}

// Lettore di testo: sempre senza scheda grafica, che su alcuni telefoni sbaglia i calcoli.
async function startOcr() {
  const model = async (f, label) => ort.InferenceSession.create(await getBytes(OCR_URL + f, label, true), { executionProviders: ['wasm'] });
  const dict = async f => JSON.parse(new TextDecoder().decode(await getBytes(OCR_URL + f, 'dizionario', true)));
  ocr = {
    det: await model('PP-OCRv5_mobile_det.onnx', 'rilevatore del testo'),
    latin: await model('latin_PP-OCRv5_mobile_rec.onnx', 'lettore latino'), latinDict: await dict('latin_PP-OCRv5_mobile_rec.dict.json'),
    cjk: await model('PP-OCRv5_mobile_rec.onnx', 'lettore giapponese'), cjkDict: await dict('PP-OCRv5_mobile_rec.dict.json'),
  };
}

// Fotocamera posteriore; deviceId = un obiettivo preciso (scelto con «Obiettivo»).
async function startCamera(deviceId) {
  track?.stop();
  const video = { width: { ideal: 1920 }, height: { ideal: 1080 },
    ...(deviceId ? { deviceId: { exact: deviceId } } : { facingMode: { ideal: 'environment' } }) };
  const stream = await navigator.mediaDevices.getUserMedia({ audio: false, video });
  $('cam').srcObject = stream;
  await $('cam').play();
  track = stream.getVideoTracks()[0];
  const caps = track.getCapabilities?.() || {};
  if (caps.focusMode?.includes('continuous')) track.applyConstraints({ advanced: [{ focusMode: 'continuous' }] }).catch(() => {});
  $('torch').hidden = !caps.torch;
  if (!cams.length) cams = (await navigator.mediaDevices.enumerateDevices()).filter(d => d.kind === 'videoinput' && !/front|user|anterior/i.test(d.label));
  $('switchCam').hidden = cams.length < 2;
  try { localStorage.setItem(CAM, track.getSettings().deviceId); } catch { /* si riparte dall'obiettivo predefinito */ }
}

const camInfo = () => {
  if (NATIVE) return nativeInfo ? `analisi ${nativeInfo.frame}, obiettivo ${nativeInfo.index + 1} di ${nativeInfo.cameras}` : 'nessuna';
  const s = track?.getSettings?.() || {};
  return track ? `${s.width}×${s.height} «${track.label}»` : 'nessuna';
};

// APK: fotocamera nativa, poi modello verificato come nella versione web (deve ridare l'impronta del PC).
async function startNative() {
  document.documentElement.classList.add('native');
  crash = (await Scanner.lastCrash()).text || ''; // errore che ha chiuso l'app l'ultima volta
  loading('Accendo la fotocamera…');
  nativeInfo = await Scanner.start();
  loading('Avvio il riconoscimento…');
  const progress = await Scanner.addListener('download', ({ got, total }) =>
    loading(`Scarico il modello (solo la prima volta): ${Math.round(got / 1e6)} di ${Math.round(total / 1e6)} MB`, got / total));
  const { probe: pv } = await Scanner.prepare();
  progress.remove();
  const probe = (await (await fetch('data/probe.json')).json()).v;
  const sim = pv.reduce((a, x, i) => a + x * probe[i], 0);
  checks.push(`nativo ${sim.toFixed(3)}`);
  backend = 'nativo';
  if (sim < 0.98) throw new Error(`il riconoscimento dà risultati sbagliati su questo telefono (${checks.join(', ')})`);
  nativeInfo = await Scanner.info();
  $('switchCam').hidden = nativeInfo.cameras < 2;
  $('torch').hidden = !nativeInfo.torch;
  Scanner.addListener('frame', onNativeFrame);
  Scanner.addListener('error', e => { console.warn(e.message); if (!checks.includes(e.message)) checks.push(e.message); }); // finisce nei risultati da incollare
}

async function init() {
  try {
    const meta = JSON.parse(new TextDecoder().decode(await freshBytes(DATA + 'cards.json', 'elenco carte')));
    cards = meta.cards;
    byId = new Map(cards.map(c => [c[0], c]));
    idx = new Map(cards.map((c, i) => [c[0], i]));
    artCat = new Int8Array((await catalogBin('art.bin', 'impronte illustrazioni', meta.built)).buffer);
    fullCat = new Int8Array((await catalogBin('full.bin', 'impronte carte', meta.built)).buffer);
    if (NATIVE) await startNative();
    else {
      ort = await import(ORT_DIST + 'ort.webgpu.min.mjs');
      ort.env.wasm.wasmPaths = ORT_DIST;
      loading('Scarico il rilevatore dei bordi (circa 10 MB)…');
      await loadCV();
      try { await loadEdges(ort, await getBytes(`data/bordi.onnx?v=${EDGES_V}`, 'modello dei bordi', true)); }
      catch (e) { checks.push('bordi con Canny'); console.warn(e); } // senza modello si cercano i bordi alla vecchia maniera
      await startModel();
      await startOcr();
      loading('Accendo la fotocamera…');
      let saved = null;
      try { saved = localStorage.getItem(CAM); } catch { /* nessun obiettivo salvato */ }
      try { await startCamera(saved); }
      catch { try { await startCamera(); } catch { alertLine('Fotocamera non disponibile: puoi comunque usare «Da foto».'); } }
    }
    $('loading').remove();
    renderStats(); renderMode(); renderSticky();
    if (crash) alertLine("L'app si era chiusa per un errore: lo trovi in Risultati.");
    if (!NATIVE) tick();
  } catch (e) {
    loading(`Qualcosa non va: ${e.message}. Ricarica la pagina per riprovare.`);
  }
}

function alertLine(text) { $('stats').textContent = text; setTimeout(renderStats, 4000); }

// ---------- scansione automatica ----------
// Stati: 'cerca' (nessuna carta ferma), 'risultato' (scheda aperta),
// 'via' (aspetto che la carta esca, così una carta tenuta ferma non conta due volte).
let state = 'cerca', lastQuad = null, still = 0, gone = 0;
const STILL = 2, GONE = 4, MOVE = 0.025; // fotogrammi fermi, fotogrammi senza carta, spostamento tollerato
const HINT = { cerca: 'Inquadra una carta', ferma: 'Tieni ferma…', leggo: 'Riconosco…', via: 'Fatto: togli la carta e mettine un\'altra' };

const canvas = (w, h) => { const c = Object.assign(document.createElement('canvas'), { width: w, height: h }), x = c.getContext('2d', { willReadFrequently: true }); x.imageSmoothingQuality = 'high'; return [c, x]; };
const [cardHi, cardHiCtx] = canvas(OCR_W, OCR_H), [card, cardCtx] = canvas(CARD_W, CARD_H), [sq, sqCtx] = canvas(SIZE, SIZE);

// Con il menu o la scelta della modalità aperti non si scansiona.
const paused = () => $('menu').classList.contains('show') || $('modeSheet').classList.contains('show');
const moved = (a, b, size) => a.some(([x, y], i) => Math.hypot(x - b[i][0], y - b[i][1]) > MOVE * size);

// Raddrizza la carta nella versione grande (per leggere) e da lì quella piccola (per il disegno).
function capture(src, w, h, quad) {
  if (quad) warpCard(src, w, h, quad, cardHi);
  else cardHiCtx.drawImage(src, 0, 0, OCR_W, OCR_H); // nessun bordo trovato: la foto è già la carta
  cardCtx.drawImage(cardHi, 0, 0, CARD_W, CARD_H);
}

async function tick() {
  const video = $('cam'), w = video.videoWidth, h = video.videoHeight;
  if (state !== 'risultato' && w && !paused()) {
    let q = null;
    try { q = await findCard(video, w, h); } catch (e) { console.warn(e); }
    if (state === 'via') {
      gone = q ? 0 : gone + 1;
      if (gone >= GONE) state = 'cerca';
    } else {
      still = q && lastQuad && !moved(q, lastQuad, Math.max(w, h)) ? still + 1 : 0;
      if (still >= STILL) {
        state = 'risultato';
        // Fotogramma del video, non una foto: lo scatto vero (prova del 7/11) costava 1,5-2 s e usciva mosso.
        $('hint').textContent = HINT.leggo;
        capture(video, w, h, q);
        shotInfo = 'video';
        await recognize();
      }
    }
    lastQuad = q;
    drawQuad(q, w, h);
    if (state !== 'risultato') $('hint').textContent = state === 'via' ? HINT.via : q ? HINT.ferma : HINT.cerca;
  }
  setTimeout(tick, 100);
}

// APK: il modulo nativo manda ~10 volte al secondo se vede una carta e da quanti fotogrammi è ferma.
let refocus = false; // dopo «ci riprovo» si aspetta che il modulo nativo rimetta a fuoco (ferma torna a 0)
function onNativeFrame({ found, still: s }) {
  if (state === 'risultato' || paused()) return;
  if (refocus) { if (s) return; refocus = false; }
  if (state === 'via') {
    gone = found ? 0 : gone + 1;
    if (gone >= GONE) setState('cerca');
  } else if (found && s >= STILL) {
    setState('risultato');
    $('hint').textContent = HINT.leggo;
    shotInfo = 'nativo';
    recognize();
    return;
  }
  $('hint').textContent = state === 'via' ? HINT.via : found ? HINT.ferma : HINT.cerca;
}

function setState(s) {
  state = s;
  if (NATIVE) Scanner.setMode({ mode: s }); // colore del contorno disegnato dal modulo nativo
}

// Disegna il contorno trovato sopra il video (che è «cover»: va riproporzionato).
function drawQuad(q, w, h) {
  const o = $('overlay'), r = o.getBoundingClientRect(), dpr = devicePixelRatio || 1;
  if (o.width !== Math.round(r.width * dpr)) { o.width = Math.round(r.width * dpr); o.height = Math.round(r.height * dpr); }
  const ctx = o.getContext('2d');
  ctx.clearRect(0, 0, o.width, o.height);
  if (!q) return;
  const s = Math.max(o.width / w, o.height / h), ox = (o.width - w * s) / 2, oy = (o.height - h * s) / 2;
  ctx.lineWidth = 4 * dpr;
  ctx.strokeStyle = state === 'via' ? '#9aa1ae' : still > 0 ? '#3ccf7d' : '#ffd23f';
  ctx.beginPath();
  q.forEach(([x, y], i) => ctx[i ? 'lineTo' : 'moveTo'](ox + x * s, oy + y * s));
  ctx.closePath();
  ctx.stroke();
}

// ---------- riconoscimento ----------
function pixels(r) {
  sqCtx.drawImage(card, r.left, r.top, r.width, r.height, 0, 0, SIZE, SIZE);
  return sqCtx.getImageData(0, 0, SIZE, SIZE).data;
}

async function embed(px) {
  const out = await session.run({ pixel_values: new ort.Tensor('float32', toTensorData(px, 4), [1, 3, SIZE, SIZE]) });
  return normalize(out.last_hidden_state.data.subarray(0, DIM));
}

// Versione web: impronte e lettura qui nella pagina.
async function analyzeWeb() {
  const t0 = performance.now();
  const va = await embed(pixels(artRect()));
  const vf = await embed(pixels({ left: 0, top: 0, width: CARD_W, height: CARD_H }));
  const t1 = performance.now();
  const read = await readCard(ort, ocr, cardHiCtx.getImageData(0, 0, OCR_W, OCR_H));
  return { va, vf, read, msOcr: performance.now() - t1 };
}

// APK: impronte e lettura native; qui solo lingua e numero dalle righe lette (stesse regole della versione web).
async function analyzeNative() {
  const r = await Scanner.analyze(), t1 = performance.now();
  const im = new Image();
  im.src = r.seen;
  await im.decode();
  cardCtx.drawImage(im, 0, 0, CARD_W, CARD_H);
  let lang = latinLanguage(r.lines);
  if (!lang) {
    const text = async script => (await Scanner.readAsian({ script })).lines.map(l => l.text).join(' ');
    const ja = asianLanguage(await text('japanese'));
    const a = ja === 'ja' ? 'ja' : asianLanguage(await text('korean')) === 'ko' ? 'ko' : ja;
    if (a) lang = { lang: a };
  }
  const read = { lang: lang?.lang || null, number: cardNumber(r.lines, OCR_H) };
  return { va: Float32Array.from(r.art), vf: Float32Array.from(r.full), read, msOcr: r.msOcr + performance.now() - t1, msEmb: r.msEmbed, foil: r.foil };
}

async function recognize() {
  try {
    $('verdict').hidden = true;
    const t0 = performance.now();
    const { va, vf, read, msOcr, msEmb, foil } = NATIVE ? await analyzeNative() : await analyzeWeb();
    const sa = scoreAll(va, artCat), sf = scoreAll(vf, fullCat), sc = sa.map((s, i) => s + sf[i]);
    // Lingua non letta: si usa l'ultima confermata (una pila di carte è quasi sempre di una lingua), ma si chiede.
    const guessed = !read.lang;
    if (guessed) read.lang = lastLang;
    else remember(LAST_LANG, lastLang = read.lang);
    const print = choosePrint(candidatePrints(sc, artCat), cards, read);
    if (guessed) print.sure = false;
    const ids = t => t.map(([, i]) => cards[i][0]);
    const tc = topK(sc, 5), gap = artGap(sc, artCat), top = tc[0][0] / 2;
    pending = {
      t: Date.now(), m: mode, src: shotInfo, ms: Math.round(performance.now() - t0), msOcr: Math.round(msOcr), msEmb, foil,
      c: ids(tc), s: tc.map(([s]) => +(s / 2).toFixed(3)), a: ids(topK(sa, 5)), f: ids(topK(sf, 5)), px: lumaStats(),
      pi: cards[print.i][0], pl: print.lang, sure: print.sure ? 1 : 0, rl: guessed ? null : read.lang, rn: read.number?.num ?? null, rt: read.number?.total ?? null,
      ranked: [...new Set(print.ranked.map(r => cards[r.i][0]))].slice(0, 5), g: +gap.toFixed(3),
    };
    // Foto poco chiara (sfocata, riflesso, carta non in catalogo): si rifà da sola, con nuova messa a fuoco, fino a 2 volte.
    if ((top < CLEAR.score || (gap < CLEAR.gap && !print.sure)) && retries < 2) {
      retries++;
      results.push({ ...pending, p: null, l: null, auto: 0, rifai: 1 }); save();
      pending = null; still = gone = 0; refocus = NATIVE;
      alertLine('Non la vedo bene: tienila ferma, ci riprovo…');
      return setState('cerca');
    }
    retries = 0;
    if (top < CLEAR.score) print.sure = false;
    $('seen').getContext('2d').drawImage(card, 0, 0);
    if (isBasicEnergy(cards[print.i][1])) {
      record(null, true);
      verdict('grigio', 'Energia base', cards[print.i][1], ['Le energie base non si contano.']);
      return setState('via');
    }
    // Il colore è certo anche senza la stampa esatta se nessuna delle stampe possibili ha un disegno che hai già.
    const near = print.ranked.filter(r => print.ranked[0].s - r.s < 1.5);
    const greenSure = mode === 'controllo' && gap >= CLEAR.gap && near.every(r => !ownsArt(r.i));
    const scan = { pending: { ...pending }, best: { id: pending.pi, lang: print.lang } };
    if (print.sure || greenSure) act(scan.best, true, scan);
    else askPrint(scan);
  } catch (e) {
    alertLine(`Errore nel riconoscimento: ${e.message}`);
    setState('cerca');
  }
}

// Luminosità media, contrasto e nitidezza (varianza del laplaciano) della carta vista: per capire le foto venute male.
function lumaStats() {
  const d = cardCtx.getImageData(0, 0, CARD_W, CARD_H).data, n = d.length / 4, y = new Float32Array(n);
  let s = 0, s2 = 0, l = 0, l2 = 0, k = 0;
  for (let i = 0; i < n; i++) { y[i] = 0.299 * d[4 * i] + 0.587 * d[4 * i + 1] + 0.114 * d[4 * i + 2]; s += y[i]; s2 += y[i] * y[i]; }
  for (let r = 1; r < CARD_H - 1; r++) for (let c = 1; c < CARD_W - 1; c++) {
    const i = r * CARD_W + c, v = 4 * y[i] - y[i - 1] - y[i + 1] - y[i - CARD_W] - y[i + CARD_W];
    l += v; l2 += v * v; k++;
  }
  const m = s / n;
  return [Math.round(m), Math.round(Math.sqrt(Math.max(0, s2 / n - m * m))), Math.round(l2 / k - (l / k) ** 2)];
}

// ---------- decisione: semaforo e registrazione ----------
const numberOf = row => row[4] === 'ja' || !row[6] ? row[3] : `${row[3]}/${row[6]}`;
const sameArtId = (a, b) => idx.has(a) && idx.has(b) && artSim(artCat, idx.get(a), idx.get(b)) >= OWNED_ART;
const ownsArt = i => data.copies.some(c => idx.has(c.id) && artSim(artCat, idx.get(c.id), i) >= OWNED_ART);
const describe = c => { const row = byId.get(c.id) || []; return `${row[2]} · n. ${numberOf(row)} · ${c.l.toUpperCase()} · ${variantLabel(c.v)}`; };
const have = list => list.slice(0, 3).map(c => `Hai: ${describe(c)} ${c.s ? '(negli scarti)' : `in «${c.c}»`}`);

// Registra la scansione nelle prove (quanto spesso l'app fa da sola e quanto spesso la si corregge).
function record(p, auto) {
  const r = { ...pending, p: p?.id ?? null, l: p?.l ?? null, v: p?.v, auto: auto ? 1 : 0 };
  results.push(r); save(); pending = null; renderStats();
  return r;
}

// Dubbio: si chiede quale stampa è. «Rivedi dopo» registra la più probabile e la segna da rivedere.
function askPrint(scan) {
  chooseFrom(scan.pending.ranked, scan.best.lang, scan.pending, pick => {
    pending = { ...scan.pending };
    if (pick === 'later') act(scan.best, false, scan, true);
    else if (pick) act(pick, false, scan);
    else { record(null, false); verdict('grigio', 'Non registrata', '', ['Nessuna stampa scelta.']); setState('via'); }
  });
}

// Stampa decisa (pick = { id, lang }): semaforo e azione della modalità. v = variante scelta col tasto «È …?».
function act(pick, auto, scan, doubt = false, v = null) {
  const row = byId.get(pick.id), p = { id: pick.id, l: pick.lang, v: v || pickVariant(row[8], sticky) };
  const rec = record(p, auto), name = row[1], info = describe(p), s = semaphore(data.copies, p, sameArtId);
  const missing = row[8]?.length && (sticky.r && p.v[0] !== 'r' ? 'Reverse' : sticky.h && p.v[0] !== 'h' ? 'Holo' : sticky.first && !p.v.includes('1st-edition') ? '1ª edizione' : '');
  const note = missing ? `${missing} non esiste per questa carta` : '';
  // Annulla: toglie quello che è stato fatto e chiede la stampa giusta (o «Rifai»).
  const fix = undo => () => { undo?.(); rec.undo = 1; save(); renderStats(); setState('risultato'); askPrint(scan); };
  // Altra variante della stessa stampa con un tocco (tra quelle che esistono; giapponesi: normale, reverse, holo).
  const vars = row[8]?.length ? [...new Set(row[8])] : ['n', 'r', 'h'], nextV = vars[(vars.indexOf(p.v) + 1) % vars.length];
  const other = undo => vars.length > 1 && [`È ${variantLabel(nextV)}?`, () => { undo?.(); rec.vfix = nextV; save(); pending = { ...scan.pending }; act(pick, auto, scan, doubt, nextV); }];
  const add = (extra, kind, title, lines) => {
    const c = { k: newKey(), ...p, c: col, s: 0, t: Date.now(), ...(doubt ? { d: 1, alt: scan.pending.ranked } : {}), ...extra };
    data.copies.push(c); store();
    const remove = () => { data.copies = data.copies.filter(x => x.k !== c.k); store(); };
    verdict(kind, title, name, [...lines, info, note, doubt ? 'Stampa incerta: da rivedere nel Menu' : ''], fix(remove), 'Annulla', other(remove));
    setState('via');
  };
  if (mode === 'scarto') return add({ s: 1 }, 'viola', 'Negli scarti', []);
  if (mode === 'sistema') return system(p, name, info, fix);
  if (mode === 'controllo') {
    verdict(s.color, TITLE[s.color], name, [info, ...have(s.color === 'rosso' ? s.exact : s.same), note, 'Controllo: non salvata'], fix(), 'Sbagliata?', other());
    return setState('via');
  }
  if (s.color === 'verde') return add({}, 'verde', 'Nuova', [`Aggiunta a «${col}»`]);
  if (s.color === 'rosso') return add({ s: 1 }, 'rosso', 'Doppione', ['Copia identica: messa negli scarti']);
  feedback('giallo');
  ask(`Ce l'hai già: ${name}`, [`Questa: ${info}`, ...have(s.same), note], [
    ['Tieni', () => add({}, 'giallo', 'Tenuta', [`In «${col}»`])],
    ['Scarta', () => add({ s: 1 }, 'giallo', 'Scartata', ['Messa negli scarti'])],
  ]);
}
const TITLE = { verde: 'Nuova', giallo: 'Ce l\'hai già', rosso: 'Doppione' };

// Sistema: la copia che hai già passa nella collezione scelta; se ne hai più d'una, chiede quale.
function system(p, name, info, fix) {
  const mine = data.copies.filter(c => c.id === p.id && c.l === p.l && c.v === p.v);
  const move = c => {
    const from = c.c, was = c.s;
    c.c = col; c.s = 0; store();
    verdict('verde', 'Spostata', name, [`Da «${from}»${was ? ' (scarti)' : ''} a «${col}»`, info], fix(() => { c.c = from; c.s = was; store(); }));
    setState('via');
  };
  if (mine.length === 1) return move(mine[0]);
  if (mine.length) return ask(`Quale copia sposti in «${col}»?`, [info], [...mine.map(c => [`Quella in «${c.c}»${c.s ? ' (scarti)' : ''}`, () => move(c)]), ['Nessuna', () => setState('via')]]);
  ask(`Non hai questa carta: ${name}`, [info], [
    [`Aggiungila a «${col}»`, () => { const c = { k: newKey(), ...p, c: col, s: 0, t: Date.now() }; data.copies.push(c); store(); verdict('verde', 'Aggiunta', name, [info], fix(() => { data.copies = data.copies.filter(x => x.k !== c.k); store(); })); setState('via'); }],
    ['Lascia stare', () => setState('via')],
  ]);
}

// Colore a tutto schermo per un attimo, poi una striscia in alto finché non arriva la carta dopo.
let vTimer, undoFn = null, otherFn = null;
function verdict(kind, title, name, lines, undo = null, undoLabel = 'Annulla', other = null) {
  const v = $('verdict');
  v.className = `verdict ${kind}`;
  v.hidden = false;
  $('vTitle').textContent = title;
  $('vName').textContent = name;
  $('vLines').replaceChildren(...lines.filter(Boolean).map(t => Object.assign(document.createElement('div'), { className: 'vl', textContent: t })));
  undoFn = undo;
  $('undo').hidden = !undo;
  $('undo').textContent = undoLabel;
  otherFn = other?.[1];
  $('other').hidden = !other;
  $('other').textContent = other?.[0] || '';
  clearTimeout(vTimer);
  vTimer = setTimeout(() => v.classList.add('small'), 1600);
  feedback(kind);
}

// Suoni diversi per colore (anche su iPhone) e vibrazioni diverse (solo Android).
let audio = null;
const TONES = { verde: [[660, 0.1], [880, 0.15]], giallo: [[520, 0.12], [520, 0.12]], rosso: [[180, 0.35]], viola: [[440, 0.08]], grigio: [[300, 0.06]] };
const BUZZ = { verde: [40], giallo: [60, 80, 60], rosso: [350], viola: [30], grigio: [20] };
function feedback(kind) {
  try {
    audio ||= new AudioContext();
    let t = audio.currentTime;
    for (const [f, d] of TONES[kind] || []) {
      const o = audio.createOscillator(), g = audio.createGain();
      o.type = kind === 'rosso' ? 'square' : 'sine';
      o.frequency.value = f;
      g.gain.setValueAtTime(0.2, t);
      g.gain.exponentialRampToValueAtTime(0.001, t + d);
      o.connect(g).connect(audio.destination);
      o.start(t); o.stop(t + d);
      t += d + 0.03;
    }
  } catch { /* niente audio */ }
  if (NATIVE) Scanner.vibrate({ pattern: BUZZ[kind] || [] }).catch(() => {});
  else navigator.vibrate?.(BUZZ[kind] || []);
}
addEventListener('pointerdown', () => audio?.resume?.(), { passive: true });

// Domanda con pulsanti: buttons = [[testo, azione], …].
function ask(title, body, buttons) {
  $('askTitle').textContent = title;
  $('askBody').replaceChildren(...body.filter(Boolean).map(b => typeof b === 'string' ? Object.assign(document.createElement('p'), { textContent: b }) : b));
  $('askButtons').replaceChildren(...buttons.map(([label, fn]) => {
    const b = Object.assign(document.createElement('button'), { type: 'button', textContent: label });
    b.addEventListener('click', () => { $('ask').classList.remove('show'); fn(); });
    return b;
  }));
  $('ask').classList.add('show');
}

// ---------- scelta della stampa ----------
let onPick = null;
function setPickLang(lang) {
  pickLang = lang;
  for (const c of $('langPick').children) c.setAttribute('aria-pressed', c.dataset.lang === lang);
}

function cardRow(id, extra) {
  const row = byId.get(id), [, name, set, , l, img] = row;
  const b = Object.assign(document.createElement('button'), { type: 'button', className: 'cand' });
  const im = Object.assign(document.createElement('img'), { src: img, alt: '', loading: 'lazy', referrerPolicy: 'no-referrer' });
  const txt = document.createElement('div');
  txt.innerHTML = '<div class="n"></div><div class="s"></div>';
  txt.querySelector('.n').textContent = `${name}${l === 'ja' ? ' (giapponese)' : ''}`;
  txt.querySelector('.s').textContent = `${set} · ${numberOf(row)}`;
  if (extra) txt.append(extra);
  b.append(im, txt);
  return b;
}

// ids = stampe candidate, la prima è la proposta dell'app; p = cosa è stato letto (null quando si rivede dal Menu);
// done(pick | null = nessuna | 'later').
function chooseFrom(ids, lang, p, done) {
  onPick = done;
  const review = !p;
  $('seen').hidden = review;
  $('resultTitle').textContent = review ? 'Qual è la stampa giusta?' : 'Non sono sicuro: quale stampa è?';
  $('readInfo').textContent = review ? 'Tocca la stampa giusta.' :
    `Letto: ${p.rl ? LANGS[p.rl] : `lingua non letta (propongo ${LANGS[lang]})`}${p.rn ? ` · n. ${p.rn}${p.rt ? '/' + p.rt : ''}` : ' · numero non letto'}. Correggi la lingua se serve, poi tocca la stampa.`;
  setPickLang(lang);
  $('later').hidden = review || !(mode === 'inventario' || mode === 'scarto');
  $('redo').hidden = review;
  $('cands').replaceChildren(...ids.filter(id => byId.has(id)).map((id, k) => {
    const b = cardRow(id, k ? null : Object.assign(document.createElement('div'), { className: 'tag', textContent: 'Proposta dall\'app' }));
    if (!k) b.classList.add('chosen');
    // Cinesi e coreane non hanno stampe in catalogo: si sceglie la gemella giapponese e si tiene la lingua.
    b.addEventListener('click', () => decide({ id, lang: byId.get(id)[4] === 'ja' && !['zh', 'ko'].includes(pickLang) ? 'ja' : pickLang }));
    return b;
  }));
  $('result').classList.add('show');
}

// pick: { id, lang } scelti, null = nessuna, 'later' = rivedi dopo, undefined = rifai (riprova sulla stessa carta).
function decide(pick) {
  $('result').classList.remove('show');
  const done = onPick;
  onPick = null;
  if (pick === undefined) { pending = null; still = gone = 0; return setState('cerca'); }
  if (pick?.lang) remember(LAST_LANG, lastLang = pick.lang);
  done?.(pick);
}

// ---------- modalità, collezioni, varianti ----------
const MODES = { controllo: 'Controllo', inventario: 'Inventario', scarto: 'Scarti', sistema: 'Sistema' };
function renderMode() {
  $('modeBtn').textContent = mode === 'inventario' || mode === 'sistema' ? `${MODES[mode]} · ${col}` : MODES[mode];
  for (const b of $('modes').children) b.setAttribute('aria-pressed', b.dataset.mode === mode);
  $('colPick').replaceChildren(...data.cols.map(c => new Option(c, c, false, c === col)));
  delete $('delCol').dataset.sure;
  $('delCol').textContent = `Elimina «${col}»`;
}
function renderSticky() {
  for (const b of $('sticky').querySelectorAll('[data-v]')) b.setAttribute('aria-pressed', !!sticky[b.dataset.v]);
}

// ---------- menu: collezione ----------
function renderCollection() {
  const all = data.copies, count = f => all.filter(f).length;
  $('colInfo').textContent = `${all.length} carte · ${count(c => c.s)} negli scarti · ${count(c => c.d)} da rivedere`;
  const f = $('colFilter').value || '*';
  $('colFilter').replaceChildren(...[['*', 'Tutte'], ...data.cols.map(c => [c, c]), ['!s', 'Scarti'], ['!d', 'Da rivedere']].map(([v, t]) => new Option(t, v, false, v === f)));
  const q = $('colSearch').value.trim().toLowerCase();
  const list = all.filter(c => (f === '*' || (f === '!s' ? c.s : f === '!d' ? c.d : c.c === f && !c.s)) && (!q || (byId.get(c.id)?.[1] || '').toLowerCase().includes(q)))
    .sort((a, b) => b.t - a.t);
  $('colList').replaceChildren(...list.slice(0, 200).map(c => {
    const pills = document.createElement('div');
    pills.className = 's';
    pills.textContent = `${c.l.toUpperCase()} · ${variantLabel(c.v)} · ${c.c}`;
    if (c.s) pills.append(Object.assign(document.createElement('span'), { className: 'pill scarto', textContent: 'scarto' }));
    if (c.d) pills.append(Object.assign(document.createElement('span'), { className: 'pill dubbio', textContent: 'da rivedere' }));
    const b = byId.has(c.id) ? cardRow(c.id, pills) : Object.assign(document.createElement('button'), { className: 'cand', textContent: c.id });
    b.addEventListener('click', () => openCopy(c));
    return b;
  }), ...(list.length > 200 ? [Object.assign(document.createElement('p'), { textContent: `… e altre ${list.length - 200}: usa la ricerca.` })] : []));
}

// Scheda di una copia: collezione, lingua, variante, scarti, elimina; le incerte si rivedono qui.
function openCopy(c) {
  const row = byId.get(c.id) || [c.id, c.id, '', '', 'en', '', '', 0, []];
  const sel = (label, options, value, onChange) => {
    const s = document.createElement('select');
    s.setAttribute('aria-label', label);
    s.replaceChildren(...options.map(([v, t]) => new Option(`${label}: ${t}`, v, false, v === value)));
    s.addEventListener('change', () => { onChange(s.value); store(); renderCollection(); });
    const d = Object.assign(document.createElement('div'), { className: 'field' });
    d.append(s);
    return d;
  };
  const vars = row[8]?.length ? row[8] : ['n', 'r', 'h', 'n+1st-edition', 'h+1st-edition'];
  ask(row[1], [
    `${row[2]} · n. ${numberOf(row)}`,
    sel('Collezione', data.cols.map(x => [x, x]), c.c, v => { c.c = v; }),
    sel('Lingua', Object.entries(LANGS), c.l, v => { c.l = v; }),
    sel('Variante', [...new Set([...vars, c.v])].map(v => [v, variantLabel(v)]), c.v, v => { c.v = v; }),
  ], [
    ...(c.d ? [['Scegli la stampa giusta', () => chooseFrom(c.alt || [c.id], c.l, null, pick => {
      if (pick && pick !== 'later') Object.assign(c, { id: pick.id, l: pick.lang, v: pickVariant(byId.get(pick.id)[8], sticky) });
      if (pick) { delete c.d; delete c.alt; store(); }
      renderCollection();
    })]] : []),
    [c.s ? 'Togli dagli scarti' : 'Metti negli scarti', () => { c.s = c.s ? 0 : 1; store(); renderCollection(); }],
    ['Elimina', () => openCopyDelete(c)],
    ['Chiudi', () => {}],
  ]);
}
function openCopyDelete(c) {
  ask(`Eliminare ${byId.get(c.id)?.[1] || c.id}?`, ['La copia sparisce dalla collezione.'], [
    ['Sì, elimina', () => { data.copies = data.copies.filter(x => x.k !== c.k); store(); renderCollection(); }],
    ['No', () => openCopy(c)],
  ]);
}

async function copyOut(text, button, done) {
  try { await navigator.clipboard.writeText(text); button.textContent = done; }
  catch { const ta = $('copyText'); ta.hidden = false; ta.readOnly = true; ta.value = text; ta.focus(); ta.select(); }
}

// ---------- prove dello scanner ----------
// Stesso disegno: stessa carta, ristampa con lo stesso nome, o gemella in un'altra lingua.
const sameArt = (a, b) => a === b || (!!a && !!b && byId.has(a) && byId.has(b) && (byId.get(a)[1] === byId.get(b)[1] || artSim(artCat, idx.get(a), idx.get(b)) >= 0.85));
function summary(all) {
  const list = all.filter(r => !r.rifai), prints = list.filter(r => r.pi), auto = prints.filter(r => r.auto), asked = prints.filter(r => !r.auto);
  const n = list.length, hit = (k, m) => list.filter(r => r.p && r[m].slice(0, k).some(id => sameArt(id, r.p))).length;
  return {
    n, auto: auto.length, autoWrong: auto.filter(r => r.undo).length, asked: asked.length,
    askedRight: asked.filter(r => r.p === r.pi && r.l === r.pl).length, top1: hit(1, 'c'), top5: hit(5, 'c'),
    ms: n ? Math.round(list.reduce((s, r) => s + r.ms, 0) / n) : 0, redo: all.length - n, vfix: list.filter(r => r.vfix).length,
  };
}
const pct = (a, n) => n ? `${Math.round(100 * a / n)}%` : '—';

function renderStats() {
  const s = summary(results);
  $('stats').textContent = `${data.copies.length} carte in collezione${s.n ? ` · ${s.n} scansioni, da sola ${pct(s.auto, s.n)}` : ''}`;
}

function tableRows() {
  return [['Tutte', results], ...Object.entries(MODES).map(([m, name]) => [name, results.filter(r => r.m === m)])]
    .filter(([, l]) => l.length)
    .map(([name, l]) => { const s = summary(l); return [name, s.n, pct(s.auto, s.n), pct(s.autoWrong, s.auto), pct(s.askedRight, s.asked), pct(s.top1, s.n), pct(s.top5, s.n), s.ms, s.redo, s.vfix]; });
}

async function openMenu() {
  if (NATIVE) nativeInfo = await Scanner.info().catch(() => nativeInfo);
  renderCollection();
  const head = ['', 'n.', 'da sola', 'annullate tra le da sole', 'proposta giusta quando chiede', 'disegno 1°', 'prime 5', 'ms', 'rifatte (foto poco chiara)', 'variante cambiata'];
  const t = document.createElement('table');
  t.innerHTML = `<tr>${head.map(h => `<th>${h}</th>`).join('')}</tr>`;
  for (const r of tableRows()) { const tr = t.insertRow(); r.forEach(v => { tr.insertCell().textContent = v; }); }
  $('table').replaceChildren(results.length ? t : Object.assign(document.createElement('p'), { textContent: 'Ancora nessuna scansione.' }));
  $('tech').textContent = (crash ? `ULTIMO ERRORE: ${crash.slice(0, 400)} · ` : '') + `Catalogo: ${cards.length} carte · motore: ${backend} (verifica ${checks.join(', ')}, isolata ${self.crossOriginIsolated}) · fotocamera: ${camInfo()}`;
  $('copyText').hidden = $('restoreGo').hidden = true;
  $('menu').classList.add('show');
}

function copyResults() {
  const lines = ['SCANNER CARTE — risultati', `motore ${backend} · verifica ${checks.join(' ')} · isolata ${self.crossOriginIsolated} · catalogo ${cards.length} · fotocamera ${camInfo()} · ${navigator.userAgent}`,
    'gruppo | n | da sola | annullate tra le da sole | proposta giusta quando chiede | disegno 1° | prime 5 | ms | rifatte | variante cambiata', ...tableRows().map(r => r.join(' | ')),
    ...(crash ? ['ERRORE', crash.slice(0, 3000)] : []), 'DATI', ...results.map(r => JSON.stringify(r))];
  copyOut(lines.join('\n'), $('copy'), 'Copiati: incollali in chat');
}

// ---------- comandi ----------
$('none').addEventListener('click', () => decide(null));
$('later').addEventListener('click', () => decide('later'));
$('redo').addEventListener('click', () => decide(undefined));
$('undo').addEventListener('click', () => { const fn = undoFn; undoFn = null; $('verdict').hidden = true; fn?.(); });
$('other').addEventListener('click', () => { const fn = otherFn; otherFn = null; fn?.(); });
$('langPick').addEventListener('click', e => { const b = e.target.closest('[data-lang]'); if (b) setPickLang(b.dataset.lang); });
$('modeBtn').addEventListener('click', () => { renderMode(); $('modeSheet').classList.add('show'); });
$('modes').addEventListener('click', e => {
  const b = e.target.closest('[data-mode]');
  if (b) { mode = b.dataset.mode; remember(MODE, mode); renderMode(); }
});
$('colPick').addEventListener('change', () => { col = $('colPick').value; remember(COL, col); renderMode(); });
$('addCol').addEventListener('click', () => {
  const name = $('newCol').value.trim();
  if (!name) return;
  if (!data.cols.includes(name)) { data.cols.push(name); store(); }
  col = name; remember(COL, col); $('newCol').value = ''; renderMode();
});
$('delCol').addEventListener('click', () => {
  const b = $('delCol'), n = data.copies.filter(c => c.c === col).length;
  if (!b.dataset.sure) { b.dataset.sure = '1'; b.textContent = `Tocca di nuovo: via «${col}» e le sue ${n} carte`; return; }
  data.copies = data.copies.filter(c => c.c !== col);
  data.cols = data.cols.filter(c => c !== col);
  if (!data.cols.length) data.cols = ['Collezione'];
  col = data.cols[0]; remember(COL, col); store(); renderMode();
});
$('closeMode').addEventListener('click', () => $('modeSheet').classList.remove('show'));
$('sticky').addEventListener('click', e => {
  const b = e.target.closest('[data-v]');
  if (!b) return;
  const k = b.dataset.v;
  sticky[k] = !sticky[k];
  if (sticky[k] && k === 'r') sticky.h = false;
  if (sticky[k] && k === 'h') sticky.r = false;
  remember(STICKY, sticky);
  renderSticky();
});
$('openMenu').addEventListener('click', openMenu);
$('closeMenu').addEventListener('click', () => $('menu').classList.remove('show'));
$('colFilter').addEventListener('change', renderCollection);
$('colSearch').addEventListener('input', renderCollection);
$('copyCsv').addEventListener('click', () => copyOut(toCSV(data.copies, byId), $('copyCsv'), 'Elenco copiato: incollalo in Excel o in una nota'));
$('copyBackup').addEventListener('click', () => copyOut(JSON.stringify(data), $('copyBackup'), 'Backup copiato: salvalo in una nota o mandalo a te stesso'));
$('restore').addEventListener('click', () => {
  const ta = $('copyText');
  ta.hidden = $('restoreGo').hidden = false;
  ta.value = '';
  ta.readOnly = false;
  ta.placeholder = 'Incolla qui il testo del backup';
  ta.focus();
});
$('restoreGo').addEventListener('click', () => {
  let d;
  try { d = JSON.parse($('copyText').value); } catch { d = null; }
  if (!Array.isArray(d?.copies)) { $('restoreGo').textContent = 'Questo non è un backup: ricopialo tutto'; return; }
  remember('scanner-collezione-prima', JSON.stringify(data)); // la collezione di prima resta salvata a parte
  data = { cols: ['Collezione'], ...d };
  store(); renderMode(); renderCollection(); renderStats();
  $('restoreGo').textContent = `Ripristinate ${data.copies.length} carte`;
});
$('copy').addEventListener('click', copyResults);
$('reset').addEventListener('click', () => {
  if ($('reset').dataset.sure) { results = []; save(); renderStats(); openMenu(); delete $('reset').dataset.sure; $('reset').textContent = 'Azzera i risultati delle prove'; }
  else { $('reset').dataset.sure = '1'; $('reset').textContent = 'Tocca di nuovo per cancellare le prove (la collezione resta)'; }
});
$('torch').addEventListener('click', async () => {
  torchOn = !torchOn;
  try { await (NATIVE ? Scanner.torch({ on: torchOn }) : track.applyConstraints({ advanced: [{ torch: torchOn }] })); } catch { torchOn = false; }
  $('torch').setAttribute('aria-pressed', torchOn);
});
$('switchCam').addEventListener('click', async () => {
  if (NATIVE) {
    try { nativeInfo = await Scanner.switchCamera(); alertLine(`Obiettivo ${nativeInfo.index + 1} di ${nativeInfo.cameras}`); }
    catch { alertLine('Questo obiettivo non si apre: provane un altro.'); }
    return;
  }
  const now = cams.findIndex(d => d.deviceId === track?.getSettings().deviceId), next = (now + 1) % cams.length;
  try { await startCamera(cams[next].deviceId); alertLine(`Obiettivo ${next + 1} di ${cams.length}`); }
  catch { alertLine('Questo obiettivo non si apre: provane un altro.'); }
});
// Da foto: anche una foto fatta con la fotocamera del telefono; la carta viene cercata e raddrizzata.
$('file').addEventListener('change', async e => {
  const f = e.target.files[0];
  e.target.value = '';
  if (!f) return;
  const bmp = await createImageBitmap(f), k = Math.min(1, 1600 / Math.max(bmp.width, bmp.height));
  const pic = Object.assign(document.createElement('canvas'), { width: Math.round(bmp.width * k), height: Math.round(bmp.height * k) });
  pic.getContext('2d').drawImage(bmp, 0, 0, pic.width, pic.height);
  capture(pic, pic.width, pic.height, await findCard(pic, pic.width, pic.height));
  shotInfo = 'galleria';
  setState('risultato');
  await recognize();
});

init();
