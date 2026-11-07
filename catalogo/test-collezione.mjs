// Prova delle regole della collezione (semaforo, varianti, CSV): node catalogo/test-collezione.mjs
import assert from 'node:assert/strict';
import { semaphore, pickVariant, variantLabel, isBasicEnergy, toCSV, load, save } from '../web/collezione.js';

const twins = new Set(['sv01-001|jp-1']);
const sameArt = (a, b) => twins.has(`${a}|${b}`) || twins.has(`${b}|${a}`);
const mine = [{ id: 'sv01-001', l: 'it', v: 'n', c: 'Collezione', s: 0, t: 0 }];

assert.equal(semaphore([], { id: 'sv01-001', l: 'it', v: 'n' }, sameArt).color, 'verde');
assert.equal(semaphore(mine, { id: 'sv01-001', l: 'it', v: 'n' }, sameArt).color, 'rosso');
assert.equal(semaphore(mine, { id: 'sv01-001', l: 'en', v: 'n' }, sameArt).color, 'giallo');   // altra lingua
assert.equal(semaphore(mine, { id: 'sv01-001', l: 'it', v: 'r' }, sameArt).color, 'giallo');   // altra variante
assert.equal(semaphore(mine, { id: 'jp-1', l: 'ja', v: 'n' }, sameArt).color, 'giallo');       // gemella giapponese
assert.equal(semaphore(mine, { id: 'sv01-002', l: 'it', v: 'n' }, sameArt).color, 'verde');
assert.equal(semaphore([{ ...mine[0], s: 1 }], { id: 'sv01-001', l: 'it', v: 'n' }, sameArt).color, 'rosso'); // anche negli scarti

const charizard = ['h:unlimited', 'h:shadowless+1st-edition', 'h:shadowless', 'h:1999-2000-copyright'];
assert.equal(pickVariant(charizard, {}), 'h:unlimited');
assert.equal(pickVariant(charizard, { first: true }), 'h:shadowless+1st-edition');
const pikachu = ['n', 'r', 'n+pokemon-together', 'n+snowflake', 'r~cosmos'];
assert.equal(pickVariant(pikachu, {}), 'n');
assert.equal(pickVariant(pikachu, { r: true }), 'r');
assert.equal(pickVariant(['h'], { r: true }), 'h');      // la reverse non esiste: resta quella che c'è
assert.equal(pickVariant([], { r: true }), 'r');         // giapponesi: nessun dato, vale l'interruttore
assert.equal(variantLabel('h:shadowless+1st-edition'), 'Holo, senza ombra, 1ª edizione');
assert.equal(variantLabel('r~cosmos'), 'Reverse, lamina cosmo');
assert.equal(variantLabel('h+set-logo+staff'), 'Holo, timbro logo del set, timbro Staff');
assert.equal(variantLabel('r~masterball'), 'Reverse, lamina Master Ball');
assert.equal(variantLabel('n+jason-klaczynski'), 'Normale, timbro jason klaczynski');

assert.ok(isBasicEnergy('Fire Energy') && isBasicEnergy('Basic Psychic Energy') && isBasicEnergy('基本炎エネルギー'));
assert.ok(!isBasicEnergy('Double Turbo Energy') && !isBasicEnergy('Pikachu'));

const byId = new Map([['sv01-001', ['sv01-001', 'Sprigatito', 'Scarlatto e Violetto', '1', 'en', '', '198', 1, ['n', 'r']]]]);
assert.equal(toCSV(mine, byId).split('\n')[1], '"Sprigatito";"Scarlatto e Violetto";"1";"it";"Normale";"Collezione";"";"";"1970-01-01";"sv01-001"');

const mem = { v: null, getItem() { return this.v; }, setItem(_, v) { this.v = v; } };
assert.deepEqual(load(mem), { copies: [], cols: ['Collezione'] });
save({ copies: mine, cols: ['Collezione', 'Raccoglitore'] }, mem);
assert.equal(load(mem).cols[1], 'Raccoglitore');
console.log('collezione: tutto ok');
