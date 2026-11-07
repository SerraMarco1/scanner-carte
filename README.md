# Scanner Carte

App privata tra amici per riconoscere carte Pokémon con la fotocamera e trovare i doppioni.
Specifica completa: [SPECIFICA.md](SPECIFICA.md). Stato: **prima versione (v1) in prova**: modalità Controllo,
Inventario, Scarti e Sistema, collezioni sul telefono, semaforo con suoni e vibrazione.

## Struttura
- `web/` la pagina che gira sul telefono (pubblicata da GitHub Pages a ogni modifica).
  - `core.js` ritagli, confronto delle impronte e scelta della stampa, condiviso con gli script del PC.
  - `collezione.js` copie possedute, semaforo, varianti, elenco per Excel (provato da `catalogo/test-collezione.mjs`).
  - `testo.js` legge lingua e numero sulla carta (PaddleOCR).
  - `bordi.js` trova la carta nell'inquadratura (modello `data/bordi.onnx`, o Canny se manca) e la raddrizza (OpenCV).
  - `data/` catalogo: elenco carte e impronte (nessuna immagine).
- `catalogo/` script del PC che costruiscono `web/data/` e lo misurano.
  - `bordi/` scene finte, addestramento e misura del modello che trova la carta (vedi sotto).
- `android/` app Android (Capacitor) con la pagina di `web/` dentro: `ScannerPlugin.java` fa in nativo
  fotocamera, bordi (OpenCV), impronte (ONNX Runtime) e lettura (ML Kit). GitHub la costruisce a ogni modifica
  (`.github/workflows/apk.yml`) e la pubblica in
  [releases/apk](https://github.com/SerraMarco1/scanner-carte/releases/tag/apk). Per firmarla sempre con la
  stessa chiave servono i segreti `ANDROID_KEYSTORE_B64` e `ANDROID_KEYSTORE_PASSWORD`.

## Ricostruire il catalogo
Serve Node.js. La prima volta:

```bash
npm install
mkdir -p catalogo/cache/model
curl -L -o catalogo/cache/model/model.onnx https://huggingface.co/onnx-community/dinov2-small/resolve/8b1f705a3a7f6f062f6bdd21986c1583d3ef105d/onnx/model.onnx
git clone --depth 1 --filter=blob:none --sparse https://github.com/type-null/PTCG-database catalogo/cache/ptcg-database
git -C catalogo/cache/ptcg-database sparse-checkout set data_jp data_en
```

`data_en` serve per le ~1.250 carte inglesi che TCGdex elenca senza immagine (Galleria GG, Trainer Gallery,
Shiny Vault, promo…). Poi `npm run catalogo` (scarica le immagini e le varianti di stampa una volta sola in
`catalogo/cache/`, poi calcola le impronte)
e `npm test` (riconoscimento del disegno su foto finte: inglesi, giapponesi, e italiane cercate tra le inglesi).

Per il test della stampa esatta (lingua e numero letti sulla carta) servono anche i modelli di lettura del testo
da [ppocrv5-mobile-web-onnx](https://huggingface.co/nakamura196/ppocrv5-mobile-web-onnx) in `catalogo/cache/ocr/`
(`PP-OCRv5_mobile_det.onnx`, `latin_PP-OCRv5_mobile_rec.onnx`, `PP-OCRv5_mobile_rec.onnx` e i due `.dict.json`),
poi `node catalogo/test-stampa.mjs`.

## Modello dei bordi
Trova la carta anche su fondi chiari, venati o pieni di oggetti, in una pila o tenuta in mano. Si addestra sul PC
con scene finte (carte del catalogo su texture vere CC0 di [Poly Haven](https://polyhaven.com) in `catalogo/cache/sfondi`):

```bash
py -m venv catalogo/cache/venv
catalogo/cache/venv/Scripts/python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
catalogo/cache/venv/Scripts/python -m pip install opencv-python-headless onnx onnxruntime onnxscript
node catalogo/bordi/sfondi.mjs
catalogo/cache/venv/Scripts/python catalogo/bordi/addestra.py
catalogo/cache/venv/Scripts/python catalogo/bordi/misura.py --modello=catalogo/cache/bordi.onnx
```

Poi copiare `catalogo/cache/bordi.onnx` in `web/data/` e aumentare `EDGES_V` in `web/app.js`.

## Fonti
- [TCGdex](https://tcgdex.dev): carte internazionali, varianti di stampa e prezzi.
- [PTCG-database](https://github.com/type-null/PTCG-database): carte giapponesi dal sito ufficiale; inglesi mancanti
  (immagini di pkmncards.com).
- Modello [DINOv2-small](https://huggingface.co/onnx-community/dinov2-small) per le impronte delle illustrazioni.

Le immagini delle carte appartengono a The Pokémon Company: qui non ce ne sono, l'app le mostra dai siti di origine.
