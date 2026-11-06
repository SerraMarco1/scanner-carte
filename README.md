# Scanner Carte

App privata tra amici per riconoscere carte Pokémon con la fotocamera e trovare i doppioni.
Specifica completa: [SPECIFICA.md](SPECIFICA.md). Stato: **prototipo dello scanner**.

## Struttura
- `web/` la pagina che gira sul telefono (pubblicata da GitHub Pages a ogni modifica).
  - `core.js` ritagli, confronto delle impronte e scelta della stampa, condiviso con gli script del PC.
  - `testo.js` legge lingua e numero sulla carta (PaddleOCR).
  - `bordi.js` trova la carta nell'inquadratura e la raddrizza (OpenCV).
  - `data/` catalogo: elenco carte e impronte (nessuna immagine).
- `catalogo/` script del PC che costruiscono `web/data/` e lo misurano.

## Ricostruire il catalogo
Serve Node.js. La prima volta:

```bash
npm install
mkdir -p catalogo/cache/model
curl -L -o catalogo/cache/model/model.onnx https://huggingface.co/onnx-community/dinov2-small/resolve/8b1f705a3a7f6f062f6bdd21986c1583d3ef105d/onnx/model.onnx
git clone --depth 1 --filter=blob:none --sparse https://github.com/type-null/PTCG-database catalogo/cache/ptcg-database
git -C catalogo/cache/ptcg-database sparse-checkout set data_jp
```

Poi `npm run catalogo` (scarica le immagini una volta sola in `catalogo/cache/`, poi calcola le impronte)
e `npm test` (riconoscimento del disegno su foto finte: inglesi, giapponesi, e italiane cercate tra le inglesi).

Per il test della stampa esatta (lingua e numero letti sulla carta) servono anche i modelli di lettura del testo
da [ppocrv5-mobile-web-onnx](https://huggingface.co/nakamura196/ppocrv5-mobile-web-onnx) in `catalogo/cache/ocr/`
(`PP-OCRv5_mobile_det.onnx`, `latin_PP-OCRv5_mobile_rec.onnx`, `PP-OCRv5_mobile_rec.onnx` e i due `.dict.json`),
poi `node catalogo/test-stampa.mjs`.

## Fonti
- [TCGdex](https://tcgdex.dev): carte internazionali e prezzi.
- [PTCG-database](https://github.com/type-null/PTCG-database): carte giapponesi dal sito ufficiale.
- Modello [DINOv2-small](https://huggingface.co/onnx-community/dinov2-small) per le impronte delle illustrazioni.

Le immagini delle carte appartengono a The Pokémon Company: qui non ce ne sono, l'app le mostra dai siti di origine.
