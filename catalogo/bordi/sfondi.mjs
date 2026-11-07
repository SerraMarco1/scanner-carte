// Scarica le texture di superfici (tavoli, pavimenti, tessuti…) usate come sfondo delle scene finte:
// CC0 di Poly Haven, solo l'immagine del colore a 1k. Uso: node catalogo/bordi/sfondi.mjs
import fs from 'node:fs';
import path from 'node:path';

const out = path.join(import.meta.dirname, '../cache/sfondi'), ids = new Set();
fs.mkdirSync(out, { recursive: true });
for (const c of ['indoor', 'fabric', 'leather', 'carpet', 'tiles', 'clean', 'wood']) {
  const a = await (await fetch(`https://api.polyhaven.com/assets?t=textures&c=${encodeURIComponent(c)}`)).json();
  Object.keys(a).filter(id => c !== 'wood' || /table|floor|plank|laminate|parquet|board|veneer/.test(id)).forEach(id => ids.add(id));
}
const list = [...ids].slice(0, 160);
let i = 0;
await Promise.all(Array.from({ length: 6 }, async () => {
  while (i < list.length) {
    const id = list[i++], dest = path.join(out, `${id}.jpg`);
    if (fs.existsSync(dest)) continue;
    try {
      const url = (await (await fetch(`https://api.polyhaven.com/files/${id}`)).json()).Diffuse?.['1k']?.jpg?.url;
      if (url) fs.writeFileSync(dest, Buffer.from(await (await fetch(url)).arrayBuffer()));
    } catch (e) { console.log(id, e.message); }
  }
}));
console.log('texture:', fs.readdirSync(out).length);
