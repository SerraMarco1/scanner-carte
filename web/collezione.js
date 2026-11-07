// Collezione sul telefono e regole del semaforo (SPECIFICA.md §5, §7, §9). Niente interfaccia qui:
// la usa app.js e la prova catalogo/test-collezione.mjs.
//
// Una copia: { k: chiave, id: stampa del catalogo, l: lingua, v: variante, c: collezione, s: 1 se negli scarti,
//              t: quando, d: 1 se la stampa era incerta (da rivedere), alt: stampe alternative per rivederla }

// Stesso disegno: le gemelle giapponesi stanno tra 0,90 e 0,97, i disegni diversi quasi sempre sotto 0,90 (misurato).
export const OWNED_ART = 0.90;

// Energie base: lo scanner le riconosce e le ignora (grigio).
export const isBasicEnergy = name => /^(basic )?(grass|fire|water|lightning|psychic|fighting|darkness|metal|fairy) energy$/i.test(name) || /^基本.+エネルギー$/.test(name);

// Colore per la stampa vista (pick = { id, l, v }) rispetto alle copie possedute, scarti compresi.
// sameArt(idA, idB) dice se due stampe hanno lo stesso disegno.
export function semaphore(copies, pick, sameArt) {
  const same = copies.filter(c => c.id === pick.id || sameArt(c.id, pick.id));
  const exact = same.filter(c => c.id === pick.id && c.l === pick.l && c.v === pick.v);
  return { color: exact.length ? 'rosso' : same.length ? 'giallo' : 'verde', same, exact };
}

// ---------- varianti ----------
// Codici del catalogo (catalogo/build.mjs): n/r/h, «:sottotipo», «+timbro», «~lamina». Es. «h:shadowless+1st-edition».
const TYPE = { n: 'Normale', r: 'Reverse', h: 'Holo' };
const WORDS = { '1st-edition': '1ª edizione', shadowless: 'senza ombra', unlimited: 'unlimited', '1999-2000-copyright': '© 1999-2000',
  'no-e-reader': 'senza e-Reader', 'set-logo': 'logo del set', staff: 'Staff', 'player-rewards-program': 'Play! Pokémon',
  '30th-anniversary': '30° anniversario', '25th-celebration': '25° anniversario', 'pokemon-center': 'Pokémon Center',
  'pokemon-together': 'Pokémon Together', snowflake: 'fiocco di neve', cosmos: 'cosmo', rainbow: 'arcobaleno', gold: 'oro',
  'cracked-ice': 'ghiaccio', league: 'Lega', energy: 'energia', pokeball: 'Poké Ball', masterball: 'Master Ball',
  duskball: 'Dusk Ball', loveball: 'Love Ball', friendball: 'Friend Ball' };
const word = w => WORDS[w] || w.replace(/-/g, ' ');
export function variantLabel(code) {
  if (!code) return 'Normale';
  const [, t, sub, stamps, foil] = code.match(/^([nrh])(?::([^+~]+))?((?:\+[^+~]+)*)(?:~(.+))?$/) || [, 'n'];
  const parts = [TYPE[t] || code];
  if (sub) parts.push(word(sub));
  for (const s of (stamps || '').split('+').filter(Boolean)) parts.push(s === '1st-edition' ? word(s) : `timbro ${word(s)}`);
  if (foil) parts.push(`lamina ${word(foil)}`);
  return parts.join(', ');
}

// Variante da registrare: quella scelta con gli interruttori (sticky = { r, h, first }) se esiste per la stampa,
// altrimenti la sua prima (TCGdex mette prima la più comune). Senza dati (giapponesi): quella degli interruttori.
export function pickVariant(vars, sticky = {}) {
  const want = sticky.r ? 'r' : sticky.h ? 'h' : null;
  if (!vars?.length) return (want || 'n') + (sticky.first ? '+1st-edition' : '');
  const ok = v => (!want || v[0] === want) && v.includes('+1st-edition') === !!sticky.first;
  const plain = v => !/[:~]|\+(?!1st-edition)/.test(v); // senza timbri promo né lamine speciali
  return vars.find(v => ok(v) && plain(v)) || vars.find(ok) || vars[0];
}

// ---------- salvataggio ----------
const KEY = 'scanner-collezione';
export function load(storage = globalThis.localStorage) {
  try { const d = JSON.parse(storage.getItem(KEY)); if (d?.copies) return { cols: ['Collezione'], ...d }; } catch { /* prima volta */ }
  return { copies: [], cols: ['Collezione'] };
}
export function save(data, storage = globalThis.localStorage) {
  storage.setItem(KEY, JSON.stringify(data)); // se fallisce lo dice app.js: la collezione non deve sparire in silenzio
}

export const newKey = () => Date.now().toString(36) + Math.random().toString(36).slice(2, 7);

// Elenco per foglio di calcolo (separatore «;», come lo apre Excel in italiano).
export function toCSV(copies, byId) {
  const q = v => `"${String(v ?? '').replace(/"/g, '""')}"`;
  const rows = copies.map(c => { const r = byId.get(c.id) || []; return [r[1], r[2], r[3], c.l, variantLabel(c.v), c.c, c.s ? 'sì' : '', c.d ? 'sì' : '', new Date(c.t).toISOString().slice(0, 10), c.id]; });
  return [['Nome', 'Set', 'Numero', 'Lingua', 'Variante', 'Collezione', 'Scarto', 'Da rivedere', 'Data', 'Codice'], ...rows].map(r => r.map(q).join(';')).join('\n');
}
