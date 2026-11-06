// Aggiunge alla pagina le intestazioni di «isolamento» che GitHub Pages non permette di impostare:
// senza, il riconoscimento può usare un solo core del processore.
self.addEventListener('install', () => self.skipWaiting());
self.addEventListener('activate', e => e.waitUntil(self.clients.claim()));
self.addEventListener('fetch', e => {
  if (e.request.mode !== 'navigate') return; // tutto il resto passa senza modifiche
  e.respondWith(fetch(e.request).then(r => {
    const headers = new Headers(r.headers);
    headers.set('Cross-Origin-Opener-Policy', 'same-origin');
    headers.set('Cross-Origin-Embedder-Policy', 'credentialless'); // immagini delle carte da altri siti ancora ammesse
    return new Response(r.body, { status: r.status, statusText: r.statusText, headers });
  }));
});
