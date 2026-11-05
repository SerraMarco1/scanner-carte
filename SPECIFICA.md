# Scanner Carte — specifica

Nome provvisorio: **Scanner Carte**. Stato al 5 novembre 2026: requisiti chiusi, nessun codice scritto.
Fonte delle decisioni: questionario (93 risposte) + 4 domande di chiarimento.

## 1. Obiettivo
Inquadrando una carta Pokémon, l'app dice subito se ce l'hai già, in quale versione, o se serve a un amico. Serve a trovare doppioni e carte da dare via in una collezione di 500–2.000 carte.

## 2. Utenti e piattaforme
- 2–5 persone, di cui 1–2 con iPhone. Si entra solo con un invito tuo.
- **APK Android + webapp**, fuori da qualsiasi store. Un solo codice: la webapp è l'app; l'APK è la stessa app dentro un guscio Android (Capacitor), che aggiunge vibrazione, lettura del testo migliore e fotocamera nativa.
- Su iPhone la webapp ha lo scanner completo, ma senza vibrazione (solo suoni).
- APK su GitHub Releases: l'app avvisa quando c'è una versione nuova e si aggiorna con un tap.
- Telefono di riferimento per i test: Samsung Galaxy A52s 5G.
- Interfaccia in italiano. Nomi delle carte nella lingua della carta. Si apre sullo scanner.
- Stile: pulito e centrato sulle carte, ispirato a TCG Pocket (senza usarne grafica o marchi).

## 3. Costi
0 €. Firebase con piano a consumo e carta registrata (serve per le notifiche ad app chiusa), con avviso di spesa a 1 €. Catalogo, prezzi e APK su GitHub (gratis).

## 4. Catalogo
- **TCGdex**: lingue internazionali (IT, EN, FR, DE…) e prezzi Cardmarket.
- **PTCG-database** (github.com/type-null/PTCG-database): giapponesi (24.039 carte, sito ufficiale giapponese), coreane (20.323, sito ufficiale coreano), cinesi tradizionali (14.583, sito ufficiale asiatico), con immagini e collegamenti alle stesse carte in inglese.
- Un'automazione su GitHub rigenera ogni notte catalogo, impronte delle illustrazioni e prezzi; l'app li scarica da sola.
- Tutte le lingue scaricate per tutti.
- Carta non presente: l'app riconosce l'artwork e tu inserisci set e numero. La foto della tua copia diventa il riferimento per le scansioni successive. Le aggiunte a mano valgono per tutti gli amici (si condividono l'impronta e una miniatura).

## 5. Quando due carte sono «la stessa»
| Colore | Significato |
|---|---|
| Verde | Artwork che non hai. Un disegno ritagliato, ridisegnato o allargato conta come artwork diverso. |
| Giallo | Artwork che hai già, in un'altra versione: lingua, variante o set diversi (le ristampe in altri set comprese). |
| Rosso | Copia identica: stesso set, numero, lingua e variante. |
| Blu (in aggiunta) | La carta è nella lista desideri di un amico: colore normale + bordo blu + suono extra + nome dell'amico. |

- Anche le carte negli scarti contano come possedute.
- Energie base: escluse. Lo scanner le riconosce e le ignora (grigio, nessun conteggio).
- Varianti: tutte quelle che esistono. Siccome i nomi non sono noti, l'app mostra le varianti possibili di quella stampa con una descrizione e un esempio; se non c'è, «altra variante» con nota e foto.
- Copie da tenere: 1 per carta, modificabile carta per carta. Tra due copie identiche va negli scarti l'ultima scansionata.

## 6. Scanner
- Fotocamera sempre accesa, niente scatto. L'app trova la carta, la raddrizza, riconosce l'artwork dall'impronta e poi sceglie la stampa esatta tra i candidati con numero, lingua e simbolo del set.
- **Precisione**: entra da sola solo una carta di cui l'app è sicura; altrimenti ti mostra i candidati. Obiettivo: quasi zero errori silenziosi; stima di 10–20 domande ogni 100 carte all'inizio, da verificare col prototipo.
- Una carta si conta di nuovo solo quando esce dall'inquadratura e ne entra un'altra.
- Reverse/holo: interruttore che resta acceso finché non lo spegni.
- Schermo: colore a tutto schermo + nome della carta, niente prezzo. Il giallo mostra cosa cambia («Hai: IT, set X n. 12 normale — Questa: JP, set Y n. 3»).
- Quattro suoni semplici e distinti + vibrazioni diverse (vibrazioni solo su Android). Pulsante torcia.
- Foto delle scansioni sbagliate salvate sul telefono; tasto «Segnala un problema».
- Ritmo atteso: 20–30 carte al minuto.

## 7. Modalità
1. **Controllo**: solo semaforo, non registra niente.
2. **Inventario**: le carte entrano nella collezione scelta all'inizio della sessione; i dubbi finiscono in una lista da rivedere.
3. **Scarto**: tutto negli scarti, senza semaforo.
4. **Sistema** (nuova): scansioni carte che hai già e passano nella collezione scelta. Se hai più copie, chiede quale.

## 8. Collezioni
- Una copia sta in una sola collezione, come in un raccoglitore. Nuove collezioni visibili agli amici di default.
- Viste automatiche per artista, Pokémon, set e tipo: filtri che non spostano le carte. Le collezioni a tema («buffi», «cibo») le riempi tu con la modalità Sistema.
- Avanzamento: completamento dei set e Pokédex.
- Per ogni copia: condizione, note, posizione fisica (raccoglitore, pagina).
- Cronologia modifiche con annulla, ultimi 30 giorni.

## 9. Scarti e vendita
- Rossi negli scarti da soli (solo se l'app è sicura), con 3 secondi per annullare.
- Gialli: chiede ogni volta Tieni / Scarta, mostrando cosa cambia.
- Valore in euro solo degli scarti, con l'andamento (trend) Cardmarket. Prezzo a mano solo dove manca quello automatico (spesso per giapponesi e asiatiche).
- Stati: Disponibile, Riservata, In vendita. Una carta venduta si elimina (resta 30 giorni nella cronologia).
- Griglia o lista; ordina per valore, set, rarità; raggruppa per set, tipo, lingua.
- Export: lista stampabile per le fiere (PDF) e CSV. Nessun avviso sui prezzi.

## 10. Amici, desideri e scambi
- Amicizia reciproca con codice breve o QR. Gli amici vedono le collezioni visibili, mai i prezzi.
- Lista desideri: stampa esatta, artwork in qualsiasi versione, artwork solo in certe lingue, tutte le carte di un Pokémon. Priorità alta/media/bassa. Si aggiunge da catalogo, scansione o toccando una carta di un amico.
- «Le tue carte che servono a lui» pesca solo dagli scarti.
- Scambi: carte contro carte, oppure regali. Nessuna differenza di valore mostrata. Controproposte ammesse.
- Accettato lo scambio, le carte sono «in arrivo»; cambiano proprietario quando entrambi confermano la consegna. Le carte promesse sono bloccate. Quelle ricevute vanno in «Ricevute da scambi».
- Notifiche push anche ad app chiusa (su iPhone serve aggiungere la webapp alla schermata Home).
- Niente gruppi, niente chat.

## 11. Account e dati
- Accesso con Google. Stesso account su più dispositivi.
- Offline: scanner e collezioni. Online: amici e scambi.
- Export completo della collezione (backup).

## 12. Tappe
0. **Prototipo scanner** provato su circa 100 tue carte (molte italiane e giapponesi, varie epoche, holo, in bustina) per misurare la precisione.
1. **v1**: Controllo, Inventario, Scarto, Sistema, più collezioni, scarti con valore, export.
2. **v2**: amici, lista desideri, blu, scambi, notifiche.
3. **v3**: pagina di raccoglitore da 9 carte in un colpo, riconoscimento da foto della galleria.

## 13. Limiti accettati
- I prezzi sono indicativi: per prodotto, senza distinguere lingua e condizione.
- Reverse e holo si indicano con l'interruttore.
- Giapponesi vecchie senza numero e varianti rare possono richiedere l'inserimento a mano.
- App privata tra amici, anche per il copyright delle immagini.
- Le regole Google 2027 sugli sviluppatori verificati si affrontano se e quando arrivano in Italia.
