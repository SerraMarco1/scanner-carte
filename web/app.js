// Prototipo dello scanner: misura quanto spesso l'app trova da sola la stampa esatta delle carte vere.
// La carta viene trovata e raddrizzata da sola; la scansione parte quando resta ferma.
// Disegno (impronte) + testo letto (lingua e numero) → stampa scelta.
// Nell'APK Android fotocamera, bordi, impronte e lettura sono nativi (ScannerPlugin.java); qui resta il resto.
import { CARD_W, CARD_H, SIZE, DIM, artRect, toTensorData, normalize, scoreAll, topK, artSim, probeInput, candidatePrints, choosePrint } from './core.js';
import { loadCV, findCard, warpCard } from './bordi.js';
import { readCard, latinLanguage, asianLanguage, cardNumber } from './testo.js';

const NATIVE = !!window.Capacitor?.isNativePlatform?.();
const Scanner = NATIVE ? window.Capacitor.registerPlugin('Scanner') : null;
const ORT_DIST = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/';
let ort; // solo nella versione web: nell'APK il modello gira nativo
const MODEL_URL = 'https://huggingface.co/onnx-community/dinov2-small/resolve/8b1f705a3a7f6f062f6bdd21986c1583d3ef105d/onnx/';
const OCR_URL = 'https://huggingface.co/nakamura196/ppocrv5-mobile-web-onnx/resolve/dc7e2271c00d961a4c0a38dcebe23beb735ba33e/';
// fp16: impronte uguali a quelle del catalogo (somiglianza 0,9999); la versione int8 se ne allontana (0,985).
const MODEL_FILE = 'model_fp16.onnx';
const OCR_W = 640, OCR_H = 892; // carta raddrizzata per leggere le scritte piccole
const STORE = 'scanner-proto-risultati', CAM = 'scanner-proto-obiettivo', LAST_LANG = 'scanner-proto-lingua';
const LANGS = { it: 'italiano', en: 'inglese', fr: 'francese', de: 'tedesco', es: 'spagnolo', pt: 'portoghese', ja: 'giapponese', zh: 'cinese', ko: 'coreano' };
const $ = id => document.getElementById(id);

let session, backend, ocr, cards, byId, idx, artCat, fullCat, track, shotInfo = '', nativeInfo = null, crash = '';
let sleeve = false, pending = null, torchOn = false, cams = [], checks = [], pickLang = 'it', lastLang = 'it';
try { lastLang = localStorage.getItem(LAST_LANG) || 'it'; } catch { /* si parte dall'italiano */ }
let results = [];
try { results = JSON.parse(localStorage.getItem(STORE)) || []; } catch { /* memoria non disponibile */ }
const save = () => { try { localStorage.setItem(STORE, JSON.stringify(results)); } catch { /* resta in memoria */ } };

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
  const { probe: pv } = await Scanner.prepare();
  const probe = (await (await fetch('data/probe.json')).json()).v;
  const sim = pv.reduce((a, x, i) => a + x * probe[i], 0);
  checks.push(`nativo ${sim.toFixed(3)}`);
  backend = 'nativo';
  if (sim < 0.98) throw new Error(`il riconoscimento dà risultati sbagliati su questo telefono (${checks.join(', ')})`);
  nativeInfo = await Scanner.info();
  $('switchCam').hidden = nativeInfo.cameras < 2;
  $('torch').hidden = !nativeInfo.torch;
  Scanner.addListener('frame', onNativeFrame);
  Scanner.addListener('error', e => console.warn(e.message));
}

async function init() {
  try {
    const meta = JSON.parse(new TextDecoder().decode(await getBytes('data/cards.json', 'elenco carte')));
    cards = meta.cards;
    byId = new Map(cards.map(c => [c[0], c]));
    idx = new Map(cards.map((c, i) => [c[0], i]));
    artCat = new Int8Array((await getBytes('data/art.bin', 'impronte illustrazioni')).buffer);
    fullCat = new Int8Array((await getBytes('data/full.bin', 'impronte carte')).buffer);
    if (NATIVE) await startNative();
    else {
      ort = await import(ORT_DIST + 'ort.webgpu.min.mjs');
      ort.env.wasm.wasmPaths = ORT_DIST;
      loading('Scarico il rilevatore dei bordi (circa 10 MB)…');
      await loadCV();
      await startModel();
      await startOcr();
      loading('Accendo la fotocamera…');
      let saved = null;
      try { saved = localStorage.getItem(CAM); } catch { /* nessun obiettivo salvato */ }
      try { await startCamera(saved); }
      catch { try { await startCamera(); } catch { alertLine('Fotocamera non disponibile: puoi comunque usare «Da foto».'); } }
    }
    $('loading').remove();
    renderStats();
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
const HINT = { cerca: 'Inquadra una carta su un fondo scuro', ferma: 'Tieni ferma…', leggo: 'Riconosco…', via: 'Fatto: togli la carta e mettine un\'altra' };

const canvas = (w, h) => { const c = Object.assign(document.createElement('canvas'), { width: w, height: h }), x = c.getContext('2d', { willReadFrequently: true }); x.imageSmoothingQuality = 'high'; return [c, x]; };
const [cardHi, cardHiCtx] = canvas(OCR_W, OCR_H), [card, cardCtx] = canvas(CARD_W, CARD_H), [sq, sqCtx] = canvas(SIZE, SIZE);

const moved = (a, b, size) => a.some(([x, y], i) => Math.hypot(x - b[i][0], y - b[i][1]) > MOVE * size);

// Raddrizza la carta nella versione grande (per leggere) e da lì quella piccola (per il disegno).
function capture(src, w, h, quad) {
  if (quad) warpCard(src, w, h, quad, cardHi);
  else cardHiCtx.drawImage(src, 0, 0, OCR_W, OCR_H); // nessun bordo trovato: la foto è già la carta
  cardCtx.drawImage(cardHi, 0, 0, CARD_W, CARD_H);
}

async function tick() {
  const video = $('cam'), w = video.videoWidth, h = video.videoHeight;
  if (state !== 'risultato' && w) {
    let q = null;
    try { q = findCard(video, w, h); } catch (e) { console.warn(e); }
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
function onNativeFrame({ found, still: s }) {
  if (state === 'risultato') return;
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
  return { va: Float32Array.from(r.art), vf: Float32Array.from(r.full), read, msOcr: r.msOcr + performance.now() - t1 };
}

async function recognize() {
  try {
    const t0 = performance.now();
    const { va, vf, read, msOcr } = NATIVE ? await analyzeNative() : await analyzeWeb();
    const sa = scoreAll(va, artCat), sf = scoreAll(vf, fullCat), sc = sa.map((s, i) => s + sf[i]);
    // Lingua non letta: si usa l'ultima confermata (una pila di carte è quasi sempre di una lingua), ma si chiede.
    const guessed = !read.lang;
    if (guessed) read.lang = lastLang;
    const print = choosePrint(candidatePrints(sc, artCat), cards, read);
    if (guessed) print.sure = false;
    const ids = t => t.map(([, i]) => cards[i][0]);
    const tc = topK(sc, 5);
    pending = {
      t: Date.now(), b: sleeve ? 1 : 0, src: shotInfo, ms: Math.round(performance.now() - t0), msOcr: Math.round(msOcr),
      c: ids(tc), s: tc.map(([s]) => +(s / 2).toFixed(3)), a: ids(topK(sa, 5)), f: ids(topK(sf, 5)), px: lumaStats(),
      pi: cards[print.i][0], pl: print.lang, sure: print.sure ? 1 : 0, rl: guessed ? null : read.lang, rn: read.number?.num ?? null, rt: read.number?.total ?? null,
      ranked: [...new Set(print.ranked.map(r => cards[r.i][0]))].slice(0, 5),
    };
    $('seen').getContext('2d').drawImage(card, 0, 0);
    showCandidates();
  } catch (e) {
    alertLine(`Errore nel riconoscimento: ${e.message}`);
    setState('cerca');
  }
}

// Luminosità media e contrasto della carta vista (0-255): un'immagine nera o piatta si riconosce subito.
function lumaStats() {
  const d = cardCtx.getImageData(0, 0, CARD_W, CARD_H).data, n = d.length / 4;
  let s = 0, s2 = 0;
  for (let i = 0; i < d.length; i += 4) { const y = 0.299 * d[i] + 0.587 * d[i + 1] + 0.114 * d[i + 2]; s += y; s2 += y * y; }
  const m = s / n;
  return [Math.round(m), Math.round(Math.sqrt(Math.max(0, s2 / n - m * m)))];
}

const numberOf = row => row[4] === 'ja' || !row[6] ? row[3] : `${row[3]}/${row[6]}`;

function setPickLang(lang) {
  pickLang = lang;
  for (const c of $('langPick').children) c.setAttribute('aria-pressed', c.dataset.lang === lang);
}

function showCandidates() {
  const p = pending;
  $('readInfo').textContent = `Letto: ${p.rl ? LANGS[p.rl] : `lingua non letta (propongo ${LANGS[p.pl]}, l'ultima confermata)`}${p.rn ? ` · n. ${p.rn}${p.rt ? '/' + p.rt : ''}` : ' · numero non letto'}. Correggi la lingua se serve, poi tocca la tua stampa.`;
  setPickLang(p.pl === 'ja' ? 'ja' : p.pl);
  $('cands').replaceChildren(...p.ranked.map((id, k) => {
    const row = byId.get(id), [, name, set, , l, img] = row;
    const b = document.createElement('button');
    b.type = 'button'; b.className = 'cand' + (k ? '' : ' chosen');
    const im = Object.assign(document.createElement('img'), { src: img, alt: '', referrerPolicy: 'no-referrer' });
    const txt = document.createElement('div');
    txt.innerHTML = '<div class="n"></div><div class="s"></div>';
    txt.querySelector('.n').textContent = `${k + 1}. ${name}${l === 'ja' ? ' (giapponese)' : ''}`;
    txt.querySelector('.s').textContent = `${set} · ${numberOf(row)}`;
    if (!k) txt.append(Object.assign(document.createElement('div'), { className: 'tag', textContent: p.sure ? 'Scelta dall\'app: sicura' : 'Scelta dall\'app: da confermare' }));
    b.append(im, txt);
    b.addEventListener('click', () => decide({ id, lang: l === 'ja' ? 'ja' : pickLang }));
    return b;
  }));
  $('result').classList.add('show');
}

// pick: { id, lang } scelti, null = nessuna, undefined = rifai (non conta e riprova subito sulla stessa carta).
function decide(pick) {
  if (pick !== undefined) { results.push({ ...pending, p: pick?.id ?? null, l: pick?.lang ?? null }); save(); }
  if (pick?.lang) { lastLang = pick.lang; try { localStorage.setItem(LAST_LANG, lastLang); } catch { /* resta in memoria */ } }
  pending = null;
  setState(pick === undefined ? 'cerca' : 'via');
  still = gone = 0;
  $('hint').textContent = HINT[state];
  $('result').classList.remove('show');
  renderStats();
}

// ---------- statistiche ----------
// Stesso disegno: stessa carta, ristampa con lo stesso nome, o gemella in un'altra lingua
// (0,85 come il gruppo usato per scegliere la stampa: le gemelle giapponesi scendono fino a lì).
const sameArt = (a, b) => a === b || (!!a && !!b && (byId.get(a)[1] === byId.get(b)[1] || artSim(artCat, idx.get(a), idx.get(b)) >= 0.85));
function summary(list) {
  // Cinesi e coreane non hanno stampe in catalogo: basta lingua giusta e stesso disegno.
  const right = r => r.l === r.pl && (r.p === r.pi || (['zh', 'ko'].includes(r.l) && sameArt(r.p, r.pi)));
  const prints = list.filter(r => r.pi), sure = prints.filter(r => r.sure);
  const n = list.length, hit = (k, m) => list.filter(r => r.p && r[m].slice(0, k).some(id => sameArt(id, r.p))).length;
  return {
    n, top1: hit(1, 'c'), top5: hit(5, 'c'), np: prints.length, ok: prints.filter(right).length,
    sure: sure.length, sureWrong: sure.filter(r => !right(r)).length,
    ms: n ? Math.round(list.reduce((s, r) => s + r.ms, 0) / n) : 0,
  };
}
const pct = (a, n) => n ? `${Math.round(100 * a / n)}%` : '—';

function renderStats() {
  const s = summary(results);
  $('stats').innerHTML = `<b>${s.n}</b> scansioni · stampa giusta <b>${pct(s.ok, s.np)}</b> · disegno ${pct(s.top1, s.n)}${s.n ? ` · ${s.ms} ms` : ''}`;
}

function tableRows() {
  return [['Tutte', results], ['In bustina', results.filter(r => r.b)], ['Senza bustina', results.filter(r => !r.b)]]
    .filter(([, l]) => l.length)
    .map(([name, l]) => { const s = summary(l); return [name, s.n, pct(s.ok, s.np), pct(s.sure, s.np), pct(s.sureWrong, s.sure), pct(s.top1, s.n), pct(s.top5, s.n), s.ms]; });
}

function openMenu() {
  const head = ['', 'n.', 'stampa giusta', 'da sole', 'errate tra le «da sole»', 'disegno 1°', 'prime 5', 'ms'];
  const t = document.createElement('table');
  t.innerHTML = `<tr>${head.map(h => `<th>${h}</th>`).join('')}</tr>`;
  for (const r of tableRows()) { const tr = t.insertRow(); r.forEach(v => { tr.insertCell().textContent = v; }); }
  $('table').replaceChildren(results.length ? t : Object.assign(document.createElement('p'), { textContent: 'Ancora nessuna scansione.' }));
  $('tech').textContent = (crash ? `ULTIMO ERRORE: ${crash.slice(0, 400)} · ` : '') + `Catalogo: ${cards.length} carte · motore: ${backend} (verifica ${checks.join(", ")}, isolata ${self.crossOriginIsolated}) · fotocamera: ${camInfo()}`;
  $('copyText').style.display = 'none';
  $('menu').classList.add('show');
}

async function copyResults() {
  const lines = ['SCANNER CARTE — risultati prototipo', `motore ${backend} · verifica ${checks.join(" ")} · isolata ${self.crossOriginIsolated} · catalogo ${cards.length} · fotocamera ${camInfo()} · ${navigator.userAgent}`,
    'gruppo | n | stampa giusta | da sole | errate tra le da sole | disegno 1° | prime 5 | ms', ...tableRows().map(r => r.join(' | ')),
    ...(crash ? ['ERRORE', crash.slice(0, 3000)] : []), 'DATI', ...results.map(r => JSON.stringify(r))];
  const text = lines.join('\n');
  try { await navigator.clipboard.writeText(text); $('copy').textContent = 'Copiati: incollali in chat'; }
  catch { const ta = $('copyText'); ta.value = text; ta.style.display = 'block'; ta.focus(); ta.select(); }
}

// ---------- comandi ----------
$('none').addEventListener('click', () => decide(null));
$('redo').addEventListener('click', () => decide(undefined));
$('langPick').addEventListener('click', e => { const b = e.target.closest('[data-lang]'); if (b) setPickLang(b.dataset.lang); });
$('openMenu').addEventListener('click', openMenu);
$('closeMenu').addEventListener('click', () => $('menu').classList.remove('show'));
$('copy').addEventListener('click', copyResults);
$('reset').addEventListener('click', () => {
  if ($('reset').dataset.sure) { results = []; save(); renderStats(); openMenu(); delete $('reset').dataset.sure; $('reset').textContent = 'Azzera i risultati'; }
  else { $('reset').dataset.sure = '1'; $('reset').textContent = 'Tocca di nuovo per cancellare tutto'; }
});
$('sleeve').addEventListener('click', e => { sleeve = !sleeve; e.currentTarget.setAttribute('aria-pressed', sleeve); });
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
  capture(pic, pic.width, pic.height, findCard(pic, pic.width, pic.height));
  shotInfo = 'galleria';
  setState('risultato');
  await recognize();
});

init();
