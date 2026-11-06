// Prototipo dello scanner: misura quanto spesso il riconoscimento indovina sulle carte vere.
// La carta va allineata alla cornice gialla (il ritaglio automatico arriverà dopo).
import * as ort from 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/ort.webgpu.min.mjs';
import { CARD_W, CARD_H, SIZE, DIM, artRect, toTensorData, normalize, scoreAll, topK } from './core.js';

ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/';
const MODEL_URL = 'https://huggingface.co/onnx-community/dinov2-small/resolve/8b1f705a3a7f6f062f6bdd21986c1583d3ef105d/onnx/';
// fp16: impronte uguali a quelle del catalogo (somiglianza 0,9999); la versione int8 se ne allontana (0,985).
const MODEL_FILE = 'model_fp16.onnx';
const STORE = 'scanner-proto-risultati';
const $ = id => document.getElementById(id);

let session, backend, cards, byId, artCat, fullCat, track;
let lang = 'it', sleeve = false, pending = null, torchOn = false;
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
  const total = Number(r.headers.get('content-length')) || 0, reader = r.body.getReader(), parts = [];
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
  loading('Avvio il riconoscimento…');
  for (const ep of gpu ? ['webgpu', 'wasm'] : ['wasm']) {
    try { session = await ort.InferenceSession.create(bytes, { executionProviders: [ep] }); backend = ep; return; }
    catch (e) { console.warn(ep, e); }
  }
  throw new Error('Il modello non parte su questo telefono.');
}

async function startCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: false, video: { facingMode: { ideal: 'environment' }, width: { ideal: 1920 }, height: { ideal: 1080 } },
  });
  const video = $('cam');
  video.srcObject = stream;
  await video.play();
  track = stream.getVideoTracks()[0];
  if (track.getCapabilities?.().torch) $('torch').hidden = false;
}

async function init() {
  try {
    const meta = JSON.parse(new TextDecoder().decode(await getBytes('data/cards.json', 'elenco carte')));
    cards = meta.cards;
    byId = new Map(cards.map(c => [c[0], c]));
    artCat = new Int8Array((await getBytes('data/art.bin', 'impronte illustrazioni')).buffer);
    fullCat = new Int8Array((await getBytes('data/full.bin', 'impronte carte')).buffer);
    await startModel();
    loading('Accendo la fotocamera…');
    try { await startCamera(); }
    catch { alertLine('Fotocamera non disponibile: puoi comunque usare «Da foto».'); }
    $('loading').remove();
    $('shoot').disabled = false;
    renderStats();
  } catch (e) {
    loading(`Qualcosa non va: ${e.message}. Ricarica la pagina per riprovare.`);
  }
}

function alertLine(text) { $('stats').textContent = text; setTimeout(renderStats, 4000); }

// ---------- riconoscimento ----------
const card = Object.assign(document.createElement('canvas'), { width: CARD_W, height: CARD_H });
const cardCtx = card.getContext('2d', { willReadFrequently: true });
const sq = Object.assign(document.createElement('canvas'), { width: SIZE, height: SIZE });
const sqCtx = sq.getContext('2d', { willReadFrequently: true });
cardCtx.imageSmoothingQuality = sqCtx.imageSmoothingQuality = 'high';

// Copia la parte di video sotto la cornice gialla (il video è «cover»: va riproporzionato).
function grabFromVideo() {
  const video = $('cam'), vw = video.videoWidth, vh = video.videoHeight;
  const vr = video.getBoundingClientRect(), g = $('guide').getBoundingClientRect();
  const s = Math.max(vr.width / vw, vr.height / vh);
  const ox = vr.left + (vr.width - vw * s) / 2, oy = vr.top + (vr.height - vh * s) / 2;
  cardCtx.drawImage(video, (g.left - ox) / s, (g.top - oy) / s, g.width / s, g.height / s, 0, 0, CARD_W, CARD_H);
}

function pixels(r) {
  sqCtx.drawImage(card, r.left, r.top, r.width, r.height, 0, 0, SIZE, SIZE);
  return sqCtx.getImageData(0, 0, SIZE, SIZE).data;
}

async function embed(px) {
  const out = await session.run({ pixel_values: new ort.Tensor('float32', toTensorData(px, 4), [1, 3, SIZE, SIZE]) });
  return normalize(out.last_hidden_state.data.subarray(0, DIM));
}

async function recognize() {
  $('shoot').disabled = true;
  try {
    const t0 = performance.now();
    const va = await embed(pixels(artRect()));
    const vf = await embed(pixels({ left: 0, top: 0, width: CARD_W, height: CARD_H }));
    const sa = scoreAll(va, artCat), sf = scoreAll(vf, fullCat), sc = sa.map((s, i) => s + sf[i]);
    const ms = Math.round(performance.now() - t0);
    const ids = t => t.map(([, i]) => cards[i][0]);
    const tc = topK(sc, 5);
    pending = { t: Date.now(), l: lang, b: sleeve ? 1 : 0, ms, c: ids(tc), s: tc.map(([s]) => +(s / 2).toFixed(3)), a: ids(topK(sa, 5)), f: ids(topK(sf, 5)) };
    $('seen').getContext('2d').drawImage(card, 0, 0);
    showCandidates();
  } catch (e) {
    alertLine(`Errore nel riconoscimento: ${e.message}`);
  } finally {
    $('shoot').disabled = false;
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

function decide(pick) {
  if (pick !== undefined) { results.push({ ...pending, p: pick }); save(); }
  pending = null;
  $('result').classList.remove('show');
  renderStats();
}

// ---------- statistiche ----------
const sameArt = (a, b) => a === b || (!!a && !!b && byId.get(a)?.[1] === byId.get(b)?.[1]);
function summary(list) {
  const n = list.length, hit = (k, m) => list.filter(r => r.p && r[m].slice(0, k).some(id => sameArt(id, r.p))).length;
  return {
    n, top1: hit(1, 'c'), top5: hit(5, 'c'), art1: hit(1, 'a'), full1: hit(1, 'f'),
    ms: n ? Math.round(list.reduce((s, r) => s + r.ms, 0) / n) : 0,
  };
}
const pct = (a, n) => n ? `${Math.round(100 * a / n)}%` : '—';
const LANGS = { it: 'Italiane', en: 'Inglesi', ja: 'Giapponesi', altro: 'Altre' };

function renderStats() {
  const s = summary(results);
  $('stats').innerHTML = `<b>${s.n}</b> scansioni · 1° posto <b>${pct(s.top1, s.n)}</b> · prime 5 ${pct(s.top5, s.n)}${s.n ? ` · ${s.ms} ms` : ''}`;
}

function tableRows() {
  const rows = [['Tutte', results], ...Object.entries(LANGS).map(([k, v]) => [v, results.filter(r => r.l === k)]),
    ['In bustina', results.filter(r => r.b)]].filter(([, l]) => l.length);
  return rows.map(([name, l]) => { const s = summary(l); return [name, s.n, pct(s.top1, s.n), pct(s.top5, s.n), pct(s.art1, s.n), pct(s.full1, s.n), s.ms]; });
}

function openMenu() {
  const head = ['', 'n.', '1° posto', 'prime 5', 'solo disegno', 'carta intera', 'ms'];
  const t = document.createElement('table');
  t.innerHTML = `<tr>${head.map(h => `<th>${h}</th>`).join('')}</tr>`;
  for (const r of tableRows()) { const tr = t.insertRow(); r.forEach(v => { tr.insertCell().textContent = v; }); }
  $('table').replaceChildren(results.length ? t : Object.assign(document.createElement('p'), { textContent: 'Ancora nessuna scansione.' }));
  $('tech').textContent = `Catalogo: ${cards.length} carte · motore: ${backend} (${MODEL_FILE})`;
  $('copyText').style.display = 'none';
  $('menu').classList.add('show');
}

async function copyResults() {
  const lines = ['SCANNER CARTE — risultati prototipo', `motore ${backend} ${MODEL_FILE} · catalogo ${cards.length} · ${navigator.userAgent}`,
    'gruppo | n | 1° posto | prime 5 | solo disegno | carta intera | ms', ...tableRows().map(r => r.join(' | ')),
    'DATI', ...results.map(r => JSON.stringify(r))];
  const text = lines.join('\n');
  try { await navigator.clipboard.writeText(text); $('copy').textContent = 'Copiati: incollali in chat'; }
  catch { const ta = $('copyText'); ta.value = text; ta.style.display = 'block'; ta.focus(); ta.select(); }
}

// ---------- comandi ----------
$('shoot').addEventListener('click', () => { if (!$('cam').videoWidth) return alertLine('Fotocamera non pronta.'); grabFromVideo(); recognize(); });
$('none').addEventListener('click', () => decide(null));
$('redo').addEventListener('click', () => decide(undefined));
$('openMenu').addEventListener('click', openMenu);
$('closeMenu').addEventListener('click', () => $('menu').classList.remove('show'));
$('copy').addEventListener('click', copyResults);
$('reset').addEventListener('click', () => {
  if ($('reset').dataset.sure) { results = []; save(); renderStats(); openMenu(); delete $('reset').dataset.sure; $('reset').textContent = 'Azzera i risultati'; }
  else { $('reset').dataset.sure = '1'; $('reset').textContent = 'Tocca di nuovo per cancellare tutto'; }
});
$('langs').addEventListener('click', e => {
  const b = e.target.closest('[data-lang]'); if (!b) return;
  lang = b.dataset.lang;
  for (const c of $('langs').children) c.setAttribute('aria-pressed', c === b);
});
$('sleeve').addEventListener('click', e => { sleeve = !sleeve; e.currentTarget.setAttribute('aria-pressed', sleeve); });
$('torch').addEventListener('click', async () => {
  torchOn = !torchOn;
  try { await track.applyConstraints({ advanced: [{ torch: torchOn }] }); } catch { torchOn = false; }
  $('torch').setAttribute('aria-pressed', torchOn);
});
// Da foto: la foto deve contenere solo la carta (ritagliala prima con la galleria).
$('file').addEventListener('change', async e => {
  const f = e.target.files[0]; if (!f) return;
  const bmp = await createImageBitmap(f);
  cardCtx.drawImage(bmp, 0, 0, CARD_W, CARD_H);
  e.target.value = '';
  recognize();
});

init();
