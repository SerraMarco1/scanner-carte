// Test automatico della stampa esatta, senza telefono: foto finta → disegno + testo letto → stampa scelta.
// Carte inglesi, italiane, francesi, tedesche (TCGdex) e giapponesi (sito ufficiale), ad alta risoluzione.
// Misura anche la sola lettura: lingua e numero.
// Uso: node catalogo/test-stampa.mjs [--n=40]
import fs from 'node:fs/promises';
import { existsSync } from 'node:fs';
import path from 'node:path';
import { ROOT, CACHE, fetchRetry, pool, photoLike, loadSession, crops, embed } from './lib.mjs';
import { loadOcr, rgba, ort, OCR_W, OCR_H } from './ocr.mjs';
import { readCard } from '../web/testo.js';
import { distance, normNum, scoreAll, candidatePrints, choosePrint } from '../web/core.js';

const N = Number(process.argv.find(a => a.startsWith('--n='))?.slice(4)) || 40;
let seed = 7;
const rnd = () => ((seed = (seed * 1664525 + 1013904223) >>> 0) / 2 ** 32);
const pick = (arr, n) => { const a = [...arr]; for (let i = a.length - 1; i > 0; i--) { const j = Math.floor(rnd() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; } return a.slice(0, n); };
const safe = s => s.replace(/[^A-Za-z0-9._-]/g, '_');

const data = path.join(ROOT, 'web/data');
const cat = JSON.parse(await fs.readFile(path.join(data, 'cards.json'), 'utf8')).cards;
if (cat[0].length < 8) throw new Error('Catalogo senza totali e lingue: rilancia node catalogo/build.mjs --senza-scaricare');
const artCat = new Int8Array((await fs.readFile(path.join(data, 'art.bin'))).buffer.slice(0));
const fullCat = new Int8Array((await fs.readFile(path.join(data, 'full.bin'))).buffer.slice(0));
const inCatalog = new Set(cat.filter(c => c[4] === 'en').map(c => c[0]));

const groups = {};
for (const lang of ['en', 'it', 'fr', 'de']) {
  const list = (await fetchRetry(`https://api.tcgdex.net/v2/${lang}/cards`)).filter(c => c.image && inCatalog.has(c.id));
  groups[lang] = pick(list, N).map(c => ({ url: c.image + '/high.webp', file: path.join(CACHE, 'img-hi', lang, `${safe(c.id)}.webp`), num: c.localId, id: c.id }));
}
groups.ja = pick(cat.filter(c => c[4] === 'ja'), N).map(c => ({ url: c[5], file: path.join(CACHE, 'img-hi/ja', `${c[0]}.img`), num: c[3].split('/')[0], id: c[0] }));

await pool(Object.values(groups).flat(), 6, async q => {
  if (existsSync(q.file)) return;
  const buf = await fetchRetry(q.url, 'buffer').catch(() => null);
  if (!buf) { q.missing = true; return; }
  await fs.mkdir(path.dirname(q.file), { recursive: true });
  await fs.writeFile(q.file, buf);
});

const m = await loadOcr(), session = await loadSession();
const pct = (a, b) => b ? `${Math.round(100 * a / b)}%` : '—';
const total = { n: 0, ok: 0, sure: 0, sureWrong: 0 };
for (const [lang, qs] of Object.entries(groups)) {
  let okLang = 0, nearNum = 0, visOk = 0, ok = 0, sure = 0, sureWrong = 0, n = 0;
  const ms = [], misses = [];
  for (const q of qs.filter(q => !q.missing)) {
    const photo = await photoLike(q.file, rnd, OCR_W, OCR_H);
    const r = await readCard(ort, m, await rgba(photo));
    const c = await crops(photo);
    const [va] = await embed(session, [c.art]), [vf] = await embed(session, [c.full]);
    const sa = scoreAll(va, artCat), sf = scoreAll(vf, fullCat), sc = sa.map((s, i) => s + sf[i]);
    const t0 = Date.now(), p = choosePrint(candidatePrints(sc, artCat), cat, r);
    ms.push(r.ms + Date.now() - t0);
    const right = cat[p.i][0] === q.id && p.lang === lang;
    let best = 0; for (let i = 1; i < sc.length; i++) if (sc[i] > sc[best]) best = i;
    okLang += r.lang === lang; n++;
    nearNum += normNum(r.number?.num) === normNum(q.num) || (!!r.number?.num && distance(normNum(r.number.num), normNum(q.num)) <= 1);
    visOk += cat[best][0] === q.id; ok += right; sure += p.sure; sureWrong += p.sure && !right;
    if (!right && misses.length < 3)
      misses.push(`${q.id} ${lang}: scelta ${cat[p.i][0]} ${p.lang}${p.sure ? ' (SICURA)' : ''} · letto lingua ${r.lang}, numero ${r.number?.num ?? '-'}`);
  }
  ms.sort((a, b) => a - b);
  console.log(`\n${lang.toUpperCase()} (${n} foto finte)`);
  console.log(`  lettura: lingua giusta ${pct(okLang, n)}, numero giusto o a una cifra ${pct(nearNum, n)}, tempo mediano ${ms[n >> 1]} ms`);
  console.log(`  stampa giusta: ${pct(ok, n)} (solo col disegno: ${pct(visOk, n)}) · sicure ${pct(sure, n)}, sbagliate tra le sicure ${pct(sureWrong, sure)}`);
  misses.forEach(s => console.log('  ✗ ' + s));
  Object.assign(total, { n: total.n + n, ok: total.ok + ok, sure: total.sure + sure, sureWrong: total.sureWrong + sureWrong });
}
console.log(`\nTUTTE (${total.n}): stampa giusta ${pct(total.ok, total.n)} · entrano da sole ${pct(total.sure, total.n)}, sbagliate tra queste ${pct(total.sureWrong, total.sure)}`);
