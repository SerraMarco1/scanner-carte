// Test automatico del riconoscimento, senza telefono.
// Prende carte del catalogo (e carte ITALIANE da cercare tra le inglesi), le rovina come una foto
// (rotazione, carta storta, luce, sfocatura, riflesso, JPEG) e misura quante volte la prima
// risposta è giusta. Stampa anche quante carte entrerebbero da sole a seconda della soglia di sicurezza.
// Uso: node catalogo/test.mjs [--n=300] [--modello=model_quantized.onnx]
import fs from 'node:fs/promises';
import { existsSync } from 'node:fs';
import path from 'node:path';
import sharp from 'sharp';
import { CARD_W, CARD_H, scoreAll, topK, artSim, SAME_ART } from '../web/core.js';
import { ROOT, CACHE, fetchRetry, pool, loadSession, crops, embed, photoLike } from './lib.mjs';

const arg = (k, d) => process.argv.find(a => a.startsWith(`--${k}=`))?.split('=')[1] ?? d;
const N = Number(arg('n', 300)), MODEL = arg('modello', 'model.onnx');

// Generatore casuale con seme fisso: stesso test a ogni esecuzione.
let seed = 42;
const rnd = () => ((seed = (seed * 1664525 + 1013904223) >>> 0) / 2 ** 32);
const pick = (arr, n) => { const a = [...arr]; for (let i = a.length - 1; i > 0; i--) { const j = Math.floor(rnd() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; } return a.slice(0, n); };

const data = path.join(ROOT, 'web/data');
const cat = JSON.parse(await fs.readFile(path.join(data, 'cards.json'), 'utf8'));
const cards = cat.cards.map(([id, name, set, num, lang]) => ({ id, name, set, num, lang }));
const index = new Map(cards.map((c, i) => [c.id, i]));
const artCat = new Int8Array((await fs.readFile(path.join(data, 'art.bin'))).buffer.slice(0));
const fullCat = new Int8Array((await fs.readFile(path.join(data, 'full.bin'))).buffer.slice(0));
console.log(`Catalogo: ${cards.length} carte. Modello per le foto: ${MODEL}`);
if (cards.length < 1000) console.log('ATTENZIONE: catalogo piccolo, i numeri non sono significativi.');

const safe = s => s.replace(/[^A-Za-z0-9._-]/g, '_');
const fileOf = c => path.join(CACHE, 'img', c.lang === 'en' ? 'en' : 'ja', c.lang === 'en' ? `${safe(c.id)}.webp` : `${c.id.slice(3)}.webp`);

// Campioni: inglesi e giapponesi dal catalogo; italiane = stessa carta di una inglese del catalogo.
const groups = {
  'Inglesi': pick(cards.filter(c => c.lang === 'en'), N).map(c => ({ file: fileOf(c), expect: index.get(c.id) })),
  'Giapponesi': pick(cards.filter(c => c.lang === 'ja'), N).map(c => ({ file: fileOf(c), expect: index.get(c.id) })),
};
const itList = (await fetchRetry('https://api.tcgdex.net/v2/it/cards')).filter(c => c.image && index.has(c.id));
const itSample = pick(itList, N);
await pool(itSample, 8, async c => {
  const dest = path.join(CACHE, 'img/it', `${safe(c.id)}.webp`);
  if (existsSync(dest)) return;
  const buf = await fetchRetry(c.image + '/low.webp', 'buffer');
  if (!buf) return;
  await fs.mkdir(path.dirname(dest), { recursive: true });
  await sharp(buf).resize(CARD_W, CARD_H, { fit: 'fill' }).webp({ quality: 92 }).toFile(dest);
});
groups['Italiane → catalogo inglese'] = itSample
  .map(c => ({ file: path.join(CACHE, 'img/it', `${safe(c.id)}.webp`), expect: index.get(c.id) }))
  .filter(q => existsSync(q.file));

// Giusta = stesso disegno: stessa carta, ristampa con lo stesso nome, o gemella in un'altra lingua.
const sameArt = (i, j) => i === j || cards[i].name === cards[j].name || artSim(artCat, i, j) >= SAME_ART;

const session = await loadSession(MODEL, !MODEL.includes('quantized'));
const pct = (a, b) => b ? `${(100 * a / b).toFixed(1)}%` : '—';
const all = [];
for (const [name, qs] of Object.entries(groups)) {
  const res = { art: [0, 0], full: [0, 0], comb: [0, 0] };
  for (const q of qs) {
    const c = await crops(await photoLike(q.file, rnd));
    const [va] = await embed(session, [c.art]);
    const [vf] = await embed(session, [c.full]);
    const sa = scoreAll(va, artCat), sf = scoreAll(vf, fullCat), sc = sa.map((s, i) => s + sf[i]);
    const ok = i => sameArt(i, q.expect);
    for (const [k, s] of [['art', sa], ['full', sf], ['comb', sc]]) {
      const t = topK(s, 5);
      if (ok(t[0][1])) res[k][0]++;
      if (t.some(([, i]) => ok(i))) res[k][1]++;
      if (k === 'comb') {
        // Margine = distacco dalla prima candidata con un altro disegno.
        const t20 = topK(s, 20), other = t20.find(([, i]) => !sameArt(i, t[0][1]));
        all.push({ ok: ok(t[0][1]), margin: t[0][0] - (other ? other[0] : 0) });
      }
    }
  }
  console.log(`\n${name} (${qs.length} foto finte)`);
  for (const [k, label] of [['art', 'solo illustrazione'], ['full', 'carta intera'], ['comb', 'combinato']])
    console.log(`  ${label.padEnd(20)} giusta al 1° posto ${pct(res[k][0], qs.length).padStart(6)}   tra le prime 5 ${pct(res[k][1], qs.length).padStart(6)}`);
}

console.log('\nSoglia di sicurezza (combinato): quante entrano da sole e quante di quelle sono sbagliate');
for (const t of [0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.15]) {
  const auto = all.filter(a => a.margin >= t), wrong = auto.filter(a => !a.ok).length;
  console.log(`  margine ≥ ${t.toFixed(2)}: entrano da sole ${pct(auto.length, all.length).padStart(6)}, errori tra queste ${pct(wrong, auto.length).padStart(6)}`);
}
