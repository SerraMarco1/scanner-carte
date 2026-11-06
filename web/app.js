// Prototipo dello scanner: misura quanto spesso il riconoscimento indovina sulle carte vere.
// La carta viene trovata e raddrizzata da sola; la scansione parte quando resta ferma.
import * as ort from 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/ort.webgpu.min.mjs';
import { CARD_W, CARD_H, SIZE, DIM, artRect, toTensorData, normalize, scoreAll, topK, artSim, SAME_ART } from './core.js';
import { loadCV, findCard, warpCard } from './bordi.js';

ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/';
const MODEL_URL = 'https://huggingface.co/onnx-community/dinov2-small/resolve/8b1f705a3a7f6f062f6bdd21986c1583d3ef105d/onnx/';
// fp16: impronte uguali a quelle del catalogo (somiglianza 0,9999); la versione int8 se ne allontana (0,985).
const MODEL_FILE = 'model_fp16.onnx';
const STORE = 'scanner-proto-risultati', CAM = 'scanner-proto-obiettivo';
const $ = id => document.getElementById(id);

let session, backend, cards, byId, idx, artCat, fullCat, track;
let sleeve = false, pending = null, torchOn = false, cams = [];
let results = [];
try { results = JSON.parse(localStorage.getItem(STORE)) || []; } catch { /* memoria non disponibile */ }
const save = () => { try { localStorage.setItem(STORE, JSON.stringify(results)); } catch { /* resta in memoria */ } };

// ---------- avvio ----------
function loading(msg, frac) {
  $('loadMsg').textContent = msg;
  if (frac !== undefined) $('loadBar').style.width = `${Math.round(frac * 100)}%`;
}

// Scarica un file mostrando l'avanzamento; keep = salvalo sul telefono (solo il modello, che non cambia mai).
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
  const bytes = await getBytes(MODEL_URL + MODEL_FILE, 'modello', true);
  loading('Avvio il riconoscimento (la prima volta circa un minuto)…');
  for (const ep of gpu ? ['webgpu', 'wasm'] : ['wasm']) {
    try { session = await ort.InferenceSession.create(bytes, { executionProviders: [ep] }); backend = ep; return; }
    catch (e) { console.warn(ep, e); }
  }
  throw new Error('Il modello non parte su questo telefono.');
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

const camInfo = () => { const s = track?.getSettings?.() || {}; return track ? `${s.width}×${s.height} «${track.label}»` : 'nessuna'; };

async function init() {
  try {
    const meta = JSON.parse(new TextDecoder().decode(await getBytes('data/cards.json', 'elenco carte')));
    cards = meta.cards;
    byId = new Map(cards.map(c => [c[0], c]));
    idx = new Map(cards.map((c, i) => [c[0], i]));
    artCat = new Int8Array((await getBytes('data/art.bin', 'impronte illustrazioni')).buffer);
    fullCat = new Int8Array((await getBytes('data/full.bin', 'impronte carte')).buffer);
    loading('Scarico il rilevatore dei bordi (circa 10 MB)…');
    await loadCV();
    await startModel();
    loading('Accendo la fotocamera…');
    let saved = null;
    try { saved = localStorage.getItem(CAM); } catch { /* nessun obiettivo salvato */ }
    try { await startCamera(saved); }
    catch { try { await startCamera(); } catch { alertLine('Fotocamera non disponibile: puoi comunque usare «Da foto».'); } }
    $('loading').remove();
    renderStats();
    tick();
  } catch (e) {
    loading(`Qualcosa non va: ${e.message}. Ricarica la pagina per riprovare.`);
  }
}

function alertLine(text) { $('stats').textContent = text; setTimeout(renderStats, 4000); }

// ---------- scansione automatica ----------
// Stati: 'cerca' (nessuna carta ferma), 'risultato' (scheda aperta),
// 'via' (aspetto che la carta esca, così una carta tenuta ferma non conta due volte).
let state = 'cerca', lastQuad = null, still = 0, gone = 0;
const STILL = 3, GONE = 4, MOVE = 0.015; // fotogrammi fermi, fotogrammi senza carta, spostamento tollerato
const HINT = { cerca: 'Inquadra una carta su un fondo scuro', ferma: 'Tieni ferma…', leggo: 'Riconosco…', via: 'Fatto: togli la carta e mettine un\'altra' };

const card = Object.assign(document.createElement('canvas'), { width: CARD_W, height: CARD_H });
const cardCtx = card.getContext('2d', { willReadFrequently: true });
const sq = Object.assign(document.createElement('canvas'), { width: SIZE, height: SIZE });
const sqCtx = sq.getContext('2d', { willReadFrequently: true });
cardCtx.imageSmoothingQuality = sqCtx.imageSmoothingQuality = 'high';

const moved = (a, b, size) => a.some(([x, y], i) => Math.hypot(x - b[i][0], y - b[i][1]) > MOVE * size);

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
        $('hint').textContent = HINT.leggo;
        warpCard(video, w, h, q, card);
        await recognize();
      }
    }
    lastQuad = q;
    drawQuad(q, w, h);
    if (state !== 'risultato') $('hint').textContent = state === 'via' ? HINT.via : q ? HINT.ferma : HINT.cerca;
  }
  setTimeout(tick, 100);
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

async function recognize() {
  try {
    const t0 = performance.now();
    const va = await embed(pixels(artRect()));
    const vf = await embed(pixels({ left: 0, top: 0, width: CARD_W, height: CARD_H }));
    const sa = scoreAll(va, artCat), sf = scoreAll(vf, fullCat), sc = sa.map((s, i) => s + sf[i]);
    const ms = Math.round(performance.now() - t0);
    const ids = t => t.map(([, i]) => cards[i][0]);
    const tc = topK(sc, 5);
    pending = { t: Date.now(), b: sleeve ? 1 : 0, ms, c: ids(tc), s: tc.map(([s]) => +(s / 2).toFixed(3)), a: ids(topK(sa, 5)), f: ids(topK(sf, 5)) };
    $('seen').getContext('2d').drawImage(card, 0, 0);
    showCandidates();
  } catch (e) {
    alertLine(`Errore nel riconoscimento: ${e.message}`);
    state = 'cerca';
  }
}

function showCandidates() {
  const box = $('cands');
  box.replaceChildren(...pending.c.map((id, k) => {
    const [, name, set, num, l, img] = byId.get(id);
    const b = document.createElement('button');
    b.type = 'button'; b.className = 'cand';
    const im = Object.assign(document.createElement('img'), { src: img, alt: '', loading: 'lazy', referrerPolicy: 'no-referrer' });
    const txt = document.createElement('div');
    txt.innerHTML = `<div class="n"></div><div class="s"><span class="lang">${l === 'ja' ? 'JP' : 'EN'}</span></div>`;
    txt.querySelector('.n').textContent = `${k + 1}. ${name}`;
    txt.querySelector('.s').append(`${set} · ${num}`);
    const sc = Object.assign(document.createElement('span'), { className: 'sc', textContent: `${Math.round(pending.s[k] * 100)}%` });
    b.append(im, txt, sc);
    b.addEventListener('click', () => decide(id));
    return b;
  }));
  $('result').classList.add('show');
}

// pick: id scelto, null = nessuna, undefined = rifai (non conta e riprova subito sulla stessa carta).
function decide(pick) {
  if (pick !== undefined) { results.push({ ...pending, p: pick }); save(); }
  pending = null;
  state = pick === undefined ? 'cerca' : 'via';
  still = gone = 0;
  $('hint').textContent = HINT[state];
  $('result').classList.remove('show');
  renderStats();
}

// ---------- statistiche ----------
// Stesso disegno: stessa carta, ristampa con lo stesso nome, o gemella in un'altra lingua.
const sameArt = (a, b) => a === b || (!!a && !!b && (byId.get(a)[1] === byId.get(b)[1] || artSim(artCat, idx.get(a), idx.get(b)) >= SAME_ART));
function summary(list) {
  const n = list.length, hit = (k, m) => list.filter(r => r.p && r[m].slice(0, k).some(id => sameArt(id, r.p))).length;
  return {
    n, top1: hit(1, 'c'), top5: hit(5, 'c'), art1: hit(1, 'a'), full1: hit(1, 'f'),
    ms: n ? Math.round(list.reduce((s, r) => s + r.ms, 0) / n) : 0,
  };
}
const pct = (a, n) => n ? `${Math.round(100 * a / n)}%` : '—';

function renderStats() {
  const s = summary(results);
  $('stats').innerHTML = `<b>${s.n}</b> scansioni · 1° posto <b>${pct(s.top1, s.n)}</b> · prime 5 ${pct(s.top5, s.n)}${s.n ? ` · ${s.ms} ms` : ''}`;
}

function tableRows() {
  return [['Tutte', results], ['In bustina', results.filter(r => r.b)], ['Senza bustina', results.filter(r => !r.b)]]
    .filter(([, l]) => l.length)
    .map(([name, l]) => { const s = summary(l); return [name, s.n, pct(s.top1, s.n), pct(s.top5, s.n), pct(s.art1, s.n), pct(s.full1, s.n), s.ms]; });
}

function openMenu() {
  const head = ['', 'n.', '1° posto', 'prime 5', 'solo disegno', 'carta intera', 'ms'];
  const t = document.createElement('table');
  t.innerHTML = `<tr>${head.map(h => `<th>${h}</th>`).join('')}</tr>`;
  for (const r of tableRows()) { const tr = t.insertRow(); r.forEach(v => { tr.insertCell().textContent = v; }); }
  $('table').replaceChildren(results.length ? t : Object.assign(document.createElement('p'), { textContent: 'Ancora nessuna scansione.' }));
  $('tech').textContent = `Catalogo: ${cards.length} carte · motore: ${backend} · fotocamera: ${camInfo()}`;
  $('copyText').style.display = 'none';
  $('menu').classList.add('show');
}

async function copyResults() {
  const lines = ['SCANNER CARTE — risultati prototipo', `motore ${backend} ${MODEL_FILE} · catalogo ${cards.length} · fotocamera ${camInfo()} · ${navigator.userAgent}`,
    'gruppo | n | 1° posto | prime 5 | solo disegno | carta intera | ms', ...tableRows().map(r => r.join(' | ')),
    'DATI', ...results.map(r => JSON.stringify(r))];
  const text = lines.join('\n');
  try { await navigator.clipboard.writeText(text); $('copy').textContent = 'Copiati: incollali in chat'; }
  catch { const ta = $('copyText'); ta.value = text; ta.style.display = 'block'; ta.focus(); ta.select(); }
}

// ---------- comandi ----------
$('none').addEventListener('click', () => decide(null));
$('redo').addEventListener('click', () => decide(undefined));
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
  try { await track.applyConstraints({ advanced: [{ torch: torchOn }] }); } catch { torchOn = false; }
  $('torch').setAttribute('aria-pressed', torchOn);
});
$('switchCam').addEventListener('click', async () => {
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
  const q = findCard(pic, pic.width, pic.height);
  if (q) warpCard(pic, pic.width, pic.height, q, card);
  else cardCtx.drawImage(pic, 0, 0, CARD_W, CARD_H); // nessun bordo trovato: la foto è già la carta
  state = 'risultato';
  await recognize();
});

init();
