// Test automatico della lettura del testo (lingua e numero) su foto finte, senza telefono.
// Carte inglesi, italiane, francesi, tedesche (TCGdex) e giapponesi (sito ufficiale), ad alta risoluzione.
// Uso: node catalogo/test-testo.mjs [--n=40]
import fs from 'node:fs/promises';
import { existsSync } from 'node:fs';
import path from 'node:path';
import { ROOT, CACHE, fetchRetry, pool, photoLike } from './lib.mjs';
import { loadOcr, rgba, ort, OCR_W, OCR_H } from './ocr.mjs';
import { readCard, distance } from '../web/testo.js';

const N = Number(process.argv.find(a => a.startsWith('--n='))?.slice(4)) || 40;
let seed = 7;
const rnd = () => ((seed = (seed * 1664525 + 1013904223) >>> 0) / 2 ** 32);
const pick = (arr, n) => { const a = [...arr]; for (let i = a.length - 1; i > 0; i--) { const j = Math.floor(rnd() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; } return a.slice(0, n); };
const safe = s => s.replace(/[^A-Za-z0-9._-]/g, '_');
const norm = s => String(s ?? '').toUpperCase().replace(/[^A-Z0-9]/g, '').replace(/^0+(?=\d)/, '');

const cat = JSON.parse(await fs.readFile(path.join(ROOT, 'web/data/cards.json'), 'utf8')).cards;
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

const m = await loadOcr();
const pct = (a, b) => b ? `${Math.round(100 * a / b)}%` : '—';
for (const [lang, qs] of Object.entries(groups)) {
  let okLang = 0, okNum = 0, nearNum = 0, n = 0;
  const ms = [], misses = [];
  for (const q of qs.filter(q => !q.missing)) {
    const r = await readCard(ort, m, await rgba(await photoLike(q.file, rnd, OCR_W, OCR_H)));
    const goodLang = r.lang === lang, goodNum = norm(r.number?.num) === norm(q.num);
    const near = goodNum || (!!r.number?.num && distance(norm(r.number.num), norm(q.num)) <= 1) || norm(r.number?.bottom).includes(norm(q.num));
    okLang += goodLang; okNum += goodNum; nearNum += near; n++; ms.push(r.ms);
    if ((!goodLang || !goodNum) && misses.length < 3)
      misses.push(`${q.id}: lingua ${r.lang} (${r.why || '-'}), numero ${r.number?.num ?? '-'} invece di ${q.num} · ${r.lines.slice(-3).join(' | ').slice(0, 90)}`);
  }
  ms.sort((a, b) => a - b);
  console.log(`\n${lang.toUpperCase()} (${n} foto finte): lingua giusta ${pct(okLang, n)}, numero giusto ${pct(okNum, n)} (giusto o a una cifra ${pct(nearNum, n)}), tempo mediano ${ms[n >> 1]} ms`);
  misses.forEach(s => console.log('  ✗ ' + s));
}
