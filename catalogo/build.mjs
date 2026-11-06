// Costruisce il catalogo per lo scanner:
// 1. elenco carte: inglesi da TCGdex, giapponesi da PTCG-database (sito ufficiale giapponese);
// 2. scarica ogni immagine una volta sola (catalogo/cache/img, già ridimensionata);
// 3. calcola le impronte (illustrazione e carta intera) e scrive web/data/.
// Uso: node catalogo/build.mjs [--limit=300] [--solo-scaricare | --senza-scaricare]
//   --limit            prova veloce su poche carte per lingua
//   --solo-scaricare   scarica le immagini mancanti e basta (lento: si può lasciare in sottofondo)
//   --senza-scaricare  calcola le impronte solo con le immagini già scaricate
import fs from 'node:fs/promises';
import { existsSync } from 'node:fs';
import path from 'node:path';
import sharp from 'sharp';
import ort from 'onnxruntime-node';
import { CARD_W, CARD_H, SIZE, DIM, ART, quantize, normalize, probeInput, LANG_BITS } from '../web/core.js';
import { ROOT, CACHE, fetchRetry, pool, loadSession, crops, embed } from './lib.mjs';

const LIMIT = Number(process.argv.find(a => a.startsWith('--limit='))?.slice(8)) || Infinity;
const ONLY_DL = process.argv.includes('--solo-scaricare'), NO_DL = process.argv.includes('--senza-scaricare');
const OUT = path.join(ROOT, 'web/data');
const safe = s => s.replace(/[^A-Za-z0-9._-]/g, '_');

async function englishCards() {
  const list = await fetchRetry('https://api.tcgdex.net/v2/en/cards');
  const sets = Object.fromEntries((await fetchRetry('https://api.tcgdex.net/v2/en/sets')).map(s => [s.id, s]));
  // In quali altre lingue TCGdex conosce la stessa carta (stesso id): aiuta a scegliere la stampa.
  const langs = {};
  for (const [lang, bit] of Object.entries(LANG_BITS))
    for (const c of await fetchRetry(`https://api.tcgdex.net/v2/${lang}/cards`)) langs[c.id] = (langs[c.id] || 0) | bit;
  // /tcgp/ = TCG Pocket: carte solo digitali, non esistono stampate.
  return list.filter(c => c.image && !c.image.includes('/tcgp/')).slice(0, LIMIT).map(c => {
    const set = sets[c.image.split('/').at(-2)];
    return { id: c.id, name: c.name, set: set?.name || c.image.split('/').at(-2), num: c.localId, lang: 'en',
      total: String(set?.cardCount?.official ?? ''), langs: langs[c.id] || 0,
      img: c.image + '/low.webp', file: `en/${safe(c.id)}.webp` };
  });
}

async function japaneseCards() {
  const dir = path.join(CACHE, 'ptcg-database/data_jp');
  if (!existsSync(dir)) throw new Error('Manca catalogo/cache/ptcg-database: vedi README.md, «Ricostruire il catalogo»');
  const out = [];
  for (const set of await fs.readdir(dir)) {
    for (const f of await fs.readdir(path.join(dir, set))) {
      if (!f.endsWith('.json')) continue;
      let c;
      try { c = JSON.parse(await fs.readFile(path.join(dir, set, f), 'utf8')); } catch { continue; }
      if (!c.img || !c.jp_id) continue;
      out.push({ id: `jp-${c.jp_id}`, name: c.name, set: c.set_name || set,
        num: c.set_total ? `${c.number}/${c.set_total}` : (c.number || ''), lang: 'ja', total: c.set_total || '', langs: 0,
        img: c.img, file: `ja/${c.jp_id}.webp`, jpId: c.jp_id });
    }
  }
  return out.sort((a, b) => b.jpId - a.jpId).slice(0, LIMIT); // id più alto = carta più recente
}

async function download(cards, parallel) {
  let done = 0, failed = 0;
  await pool(cards, parallel, async c => {
    const dest = path.join(CACHE, 'img', c.file);
    if (!existsSync(dest)) {
      try {
        const buf = await fetchRetry(c.img, 'buffer');
        if (!buf) throw new Error('404');
        await fs.mkdir(path.dirname(dest), { recursive: true });
        await sharp(buf).resize(CARD_W, CARD_H, { fit: 'fill' }).webp({ quality: 92 }).toFile(dest);
      } catch (e) { failed++; c.missing = true; }
    }
    if (++done % 1000 === 0) console.log(`  immagini ${done}/${cards.length} (fallite ${failed})`);
  });
  console.log(`  immagini pronte: ${cards.length - failed}, fallite: ${failed}`);
}

console.time('catalogo');
const en = await englishCards();
const ja = await japaneseCards();
console.log(`Carte: ${en.length} inglesi, ${ja.length} giapponesi`);
if (!NO_DL) {
  await download(en, 12);
  await download(ja, 8); // sito ufficiale (dietro CloudFront): poche richieste alla volta
}
if (ONLY_DL) process.exit(0);
const cards = [...en, ...ja].filter(c => !c.missing && existsSync(path.join(CACHE, 'img', c.file)));

const session = await loadSession();
const kept = [], artV = [], fullV = [];
const BATCH = 32;
for (let i = 0; i < cards.length; i += BATCH) {
  const part = cards.slice(i, i + BATCH);
  // Un'immagine rovinata (es. scaricamento interrotto) si scarta e si cancella: verrà riscaricata.
  const cs = await Promise.all(part.map(c => {
    const file = path.join(CACHE, 'img', c.file);
    return crops(file).catch(() => fs.rm(file, { force: true }).then(() => null));
  }));
  const ok = part.filter((_, j) => cs[j]), good = cs.filter(Boolean);
  if (!good.length) continue;
  const va = await embed(session, good.map(c => c.art));
  const vf = await embed(session, good.map(c => c.full));
  ok.forEach((c, j) => { kept.push(c); artV.push(quantize(va[j])); fullV.push(quantize(vf[j])); });
  if ((i / BATCH) % 100 === 0) console.log(`  impronte ${i}/${cards.length}`);
}
const art = new Int8Array(kept.length * DIM), full = new Int8Array(kept.length * DIM);
kept.forEach((_, i) => { art.set(artV[i], i * DIM); full.set(fullV[i], i * DIM); });
cards.length = 0;
cards.push(...kept);

await fs.mkdir(OUT, { recursive: true });
// Impronta di riferimento dell'ingresso fisso: il telefono la usa per verificare il suo motore.
const probe = await session.run({ pixel_values: new ort.Tensor('float32', probeInput(), [1, 3, SIZE, SIZE]) });
await fs.writeFile(path.join(OUT, 'probe.json'),
  JSON.stringify({ v: Array.from(normalize(probe.last_hidden_state.data.subarray(0, DIM)), x => +x.toFixed(6)) }));
await fs.writeFile(path.join(OUT, 'art.bin'), art);
await fs.writeFile(path.join(OUT, 'full.bin'), full);
await fs.writeFile(path.join(OUT, 'cards.json'), JSON.stringify({
  built: new Date().toISOString(), model: 'onnx-community/dinov2-small@8b1f705', dim: DIM, crop: ART,
  fields: ['id', 'name', 'set', 'num', 'lang', 'img', 'total', 'langs'],
  cards: cards.map(c => [c.id, c.name, c.set, c.num, c.lang, c.img, c.total, c.langs]),
}));
console.log(`Scritto web/data: ${cards.length} carte`);
console.timeEnd('catalogo');
